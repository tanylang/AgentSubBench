"""
AutoCodeRover trace collection driver (v2: in-process run + tool-level interception).

Zero agent source changes -- the ACR source (source_code/swe_agents/autocoderover/)
is untouched:
- Task integration: use ACR's official swe-bench public entry point
  (make_swe_tasks → run_raw_task); the harness generates setup_map / tasks_map
  in the official format (repo points at the directory we prepared). v1 used
  run_task_groups, but it puts tasks into ProcessPoolExecutor subprocesses:
  under Windows spawn, config assignments and litellm global callbacks are all
  lost (trace llm=0), and subprocess exceptions are invisible to the parent
  (silent failure, no patch). Now the harness calls run_raw_task directly
  in-process -- still ACR's public function, with its internal logic unchanged.
- LLM: ACR's LiteLLMGeneric.call goes through litellm.completion and is
  intercepted in-process by swebench_common's global callbacks. The model uses
  ACR's official litellm-generic channel: litellm-generic-deepseek/deepseek-chat
  (verified: the cost table resolves and real calls work; the api_key is
  injected into DEEPSEEK_API_KEY by swebench_common).
- Tools: ACR's tools are in-process LLM-driven operations (search_* / run_script /
  write_patch / reproduce / evaluate_patch) with no external execution
  boundary -- the driver layer wraps each tool class at **runtime** (same idea
  as mini-swe-agent's TraceEnvironment: external wrapping, internal logic
  unchanged), uniformly recording canonical tool.execution spans.
- Patch landing: ACR only writes the patch as a diff file (extracted_patch_*.diff)
  without touching the repo. Trace evaluation replays the repo from
  env_changes, so after ACR finishes, the harness takes the final patch selected
  by the official organize_and_form_input, git-applies it into the repo, and
  records it as an acr.apply_patch span -- env_changes captures the patch content from that.
- Environment adaptation: no conda on the host, so setup_map's env_name is
  "local", and the run_script wrap maps "local" to this process's python (conda
  is still used when available).

Run:
    python scripts/run_swebench_autocoderover.py [--task ID | --tasks A,B] [--limit N]
        [--mirror https://ghfast.top/] [--out DIR]
        [--conv-round-limit 15] [--overall-retries 3]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
ACR_ROOT = ROOT / "source_code" / "swe_agents" / "autocoderover"
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Windows: stdout/stderr default to GBK after redirection; ACR's loguru/rich
# prints symbols like • that would crash the process (same convention as runners/run_task_cli.py)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from swebench_common import record_swe_tool, run_tasks  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL  # noqa: E402
from adapter.swebench import SWETask  # noqa: E402

# ACR's official litellm-generic model channel: set_model can resolve the
# cost_per_token table, and completion goes through litellm's deepseek provider
# (reads the DEEPSEEK_API_KEY environment variable).
ACR_MODEL = f"openai/{MAIN_MODEL_NAME}"

# Per-task run options (written by main(), read by launch -- run_tasks' launch signature is fixed)
_ACR_OPTS: dict[str, int] = {}

# ─────────────────────────── Tool-level interception ───────────────────────────
# Same pattern as swebench_common._CURRENT_COLLECTOR: the wrap happens only once
# after module load, and the current task's collector switches via the holder
# (avoiding cross-writes when collecting multiple tasks sequentially).

_CURRENT_COLLECTOR: dict[str, Any] = {}

# The 8 search APIs visible to the model in agent_search.py (tool convention matches the agent prompt)
_SEARCH_TOOLS = (
    "search_class",
    "search_class_in_file",
    "search_method_in_file",
    "search_method_in_class",
    "search_method",
    "search_code",
    "search_code_in_file",
    "get_code_around_line",
)


def _safe_str(x: Any, limit: int = 5000) -> str:
    """Normalize tool output: ACR tools return (text, results, ok) tuples; take the text part."""
    if isinstance(x, tuple):
        x = x[0] if x else ""
    s = str(x)
    return s[:limit] + ("..." if len(s) > limit else "")


def _clean_args(kwargs: dict) -> dict:
    return {k: str(v)[:300] for k, v in kwargs.items()}


def _wrap_method(cls, name: str, tool_name: str, out_of=None):
    """Class method → tool.execution span (original semantics unchanged: exceptions still propagate, return value unchanged)."""
    orig = getattr(cls, name)
    if getattr(orig, "_acr_trace_wrapped", False):
        return
    out_of = out_of or (lambda out: _safe_str(out))

    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = orig(self, *args, **kwargs)
            ok = True
        except Exception as e:
            record_swe_tool(_CURRENT_COLLECTOR["collector"], tool_name,
                            {"args": _clean_args(kwargs)},
                            f"Error: {type(e).__name__}: {e}",
                            (time.perf_counter() - t0) * 1000, False)
            raise
        record_swe_tool(_CURRENT_COLLECTOR["collector"], tool_name,
                        {"args": _clean_args(kwargs)}, out_of(out),
                        (time.perf_counter() - t0) * 1000, ok)
        return out

    wrapper._acr_trace_wrapped = True
    setattr(cls, name, wrapper)


def _wrap_acr_tools(collector) -> None:
    """Idempotent: wrap ACR's tool boundaries into tool.execution spans.

    Only patches runtime objects (module attributes/class attributes); no ACR
    source files are modified.
    """
    _CURRENT_COLLECTOR["collector"] = collector
    if getattr(_wrap_acr_tools, "_done", False):
        return
    _wrap_acr_tools._done = True

    # 1) Search APIs: SearchManager calls getattr(self.backend, func_name)(**kwargs)
    #    -- a class-attribute patch takes effect for all instances
    from app.search.search_backend import SearchBackend
    for name in _SEARCH_TOOLS:
        _wrap_method(SearchBackend, name, f"acr.{name}")

    # 2) Script execution (reproducer runs etc.). The "local" env maps to this
    #    process's python: setup_map's env_name is an environment name generated
    #    by the harness itself (no conda on this machine), and the agent's call
    #    semantics are unchanged. Patch both use sites: the app.utils module
    #    attribute + the from-import binding in app.task.
    import app.utils as acr_utils
    import app.task as acr_task
    _orig_run_script = acr_utils.run_script_in_conda
    if getattr(_orig_run_script, "_acr_trace_wrapped", False):
        return

    def run_script_wrapper(args, env_name, **kwargs):
        # Take the current task's collector from the holder at runtime (the wrap happens only once; multi-task runs must not capture a closure)
        col = _CURRENT_COLLECTOR["collector"]
        t0 = time.perf_counter()
        if env_name == "local":
            cmd = [sys.executable, *map(str, args)]
        else:
            cmd = ["conda", "run", "-n", env_name, "python", *map(str, args)]
        try:
            cp = subprocess.run(cmd, **kwargs)
            ok = cp.returncode == 0
            out = (cp.stdout or "") + "\n" + (cp.stderr or "")
        except Exception as e:
            record_swe_tool(col, "acr.run_script",
                            {"env_name": env_name, "args": [str(a) for a in args],
                             "cwd": kwargs.get("cwd")},
                            f"Error: {type(e).__name__}: {e}",
                            (time.perf_counter() - t0) * 1000, False)
            raise
        record_swe_tool(col, "acr.run_script",
                        {"env_name": env_name, "args": [str(a) for a in args],
                         "cwd": kwargs.get("cwd")},
                        _safe_str(out), (time.perf_counter() - t0) * 1000, ok)
        return cp

    run_script_wrapper._acr_trace_wrapped = True
    acr_utils.run_script_in_conda = run_script_wrapper
    acr_task.run_script_in_conda = run_script_wrapper

    # 3) Patch generation (PatchAgent: the LLM produces diff text, the repo is untouched)
    from app.agents.agent_write_patch import PatchAgent
    _wrap_method(PatchAgent, "write_applicable_patch_without_feedback",
                 "acr.write_patch",
                 out_of=lambda out: _safe_str(out[1]))  # (handle, patch_content)

    # 4) Reproducer generation (TestAgent: the LLM produces tests + its internal execution is already recorded by 2))
    from app.agents.agent_reproducer import TestAgent

    def _repro_out(out):
        handle, test_content, repro_result = out
        return (f"handle={handle} reproduced={getattr(repro_result, 'reproduced', None)} "
                f"returncode={getattr(repro_result, 'returncode', None)}\n"
                f"test_content:\n{_safe_str(test_content, 2000)}")

    _wrap_method(TestAgent, "write_reproducing_test_without_feedback",
                 "acr.write_reproducer", out_of=_repro_out)

    # 5) Patch evaluation (a no-op when validation is off; runs the test suite
    #    when on; wrap regardless). Patch both use sites: the validation module
    #    attribute + inference's from-import binding.
    import app.api.validation as acr_validation
    import app.inference as acr_inference
    _orig_eval = acr_validation.evaluate_patch
    if not getattr(_orig_eval, "_acr_trace_wrapped", False):

        def evaluate_patch_wrapper(task, patch_handle, patch_content, output_dir):
            col = _CURRENT_COLLECTOR["collector"]
            t0 = time.perf_counter()
            try:
                passed, msg = _orig_eval(task, patch_handle, patch_content, output_dir)
                ok = True
            except Exception as e:
                record_swe_tool(col, "acr.evaluate_patch",
                                {"patch_handle": str(patch_handle),
                                 "patch": _safe_str(patch_content, 2000)},
                                f"Error: {type(e).__name__}: {e}",
                                (time.perf_counter() - t0) * 1000, False)
                raise
            record_swe_tool(col, "acr.evaluate_patch",
                            {"patch_handle": str(patch_handle),
                             "patch": _safe_str(patch_content, 2000)},
                            f"passed={passed} {msg}",
                            (time.perf_counter() - t0) * 1000, ok)
            return passed, msg

        evaluate_patch_wrapper._acr_trace_wrapped = True
        acr_validation.evaluate_patch = evaluate_patch_wrapper
        acr_inference.evaluate_patch = evaluate_patch_wrapper

    # 6) SBFL fault localization (off by default; runs tests via sbfl.run when on. Module attribute access, one site suffices)
    import app.analysis.sbfl as acr_sbfl
    _orig_sbfl = acr_sbfl.run
    if not getattr(_orig_sbfl, "_acr_trace_wrapped", False):

        def sbfl_wrapper(task):
            col = _CURRENT_COLLECTOR["collector"]
            t0 = time.perf_counter()
            try:
                out = _orig_sbfl(task)
                ok = True
            except Exception as e:
                record_swe_tool(col, "acr.fault_localization",
                                {"task": getattr(task, "task_id", "")},
                                f"Error: {type(e).__name__}: {e}",
                                (time.perf_counter() - t0) * 1000, False)
                raise
            record_swe_tool(col, "acr.fault_localization",
                            {"task": getattr(task, "task_id", "")},
                            f"ranked_lines={len(out[1])} test_files={len(out[0])}",
                            (time.perf_counter() - t0) * 1000, ok)
            return out

        sbfl_wrapper._acr_trace_wrapped = True
        acr_sbfl.run = sbfl_wrapper


# ─────────────────────────── Task integration ───────────────────────────


def _make_maps(task: SWETask, repo_dir: Path, base_ref: str) -> tuple[Path, Path]:
    """Generate setup_map / tasks_map in ACR's official format.

    base_ref: the ref ACR uses internally for git reset/checkout (a short-named
    tag under the codeload channel, see _ensure_base_rev); the real base_commit
    in the trace is written by run_tasks from adapter.SWETask and is unaffected.
    The "patch" key (gold patch) is required by ACR's dump_meta_data; with
    angelic debugging off it is only archived and takes no part in any logic.
    """
    maps_dir = repo_dir.parent / f"acr_maps_{task.instance_id}"
    maps_dir.mkdir(parents=True, exist_ok=True)
    setup = {
        task.instance_id: {
            "repo_path": str(repo_dir),
            "env_name": "local",  # No conda on this machine; the run_script wrap maps it to the current python
            "pre_install": [],
            "install": "",
            "test_cmd": "python -m pytest " + " ".join(
                task.FAIL_TO_PASS[:1] + task.PASS_TO_PASS[:1]),
        }
    }
    tasks_map = {
        task.instance_id: {
            "base_commit": base_ref,
            "hints_text": "",
            "created_at": "",
            "test_patch": task.test_patch,
            "patch": task.patch,
            "repo": task.repo,
            "problem_statement": task.problem_statement,
            "version": task.base_commit[:8],
            "instance_id": task.instance_id,
            "FAIL_TO_PASS": task.FAIL_TO_PASS,
            "PASS_TO_PASS": task.PASS_TO_PASS,
            "environment_setup_commit": task.environment_setup_commit,
        }
    }
    setup_path = maps_dir / "setup_map.json"
    tasks_path = maps_dir / "tasks_map.json"
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    tasks_path.write_text(json.dumps(tasks_map), encoding="utf-8")
    return setup_path, tasks_path


def _patch_timeout_decorator() -> None:
    """Windows has no SIGALRM: ACR's search backend uses timeout_decorator.timeout(120),
    which by default takes the Unix signal implementation (use_signals=True),
    raising AttributeError on Windows and breaking all search tools; its
    use_signals=False path spawns a process per call, which is too slow on
    Windows. Replace it with the harness's own thread-based timeout decorator
    (same semantics: timeout raises TimeoutError, which ACR's catch_all_and_log
    catches as a tool error). Only the third-party library module attribute is
    replaced; ACR source is untouched. Must be called before the first import
    of app (the decoration happens at module import time)."""
    import functools
    import threading
    import timeout_decorator
    if getattr(timeout_decorator.timeout, "_acr_patched", False):
        return

    def timeout(seconds=None, use_signals=False, timeout_exception=TimeoutError,
                exception_message=None):
        def decorate(function):
            @functools.wraps(function)
            def new_function(*args, **kwargs):
                limit = kwargs.pop("timeout", seconds)
                if not limit:
                    return function(*args, **kwargs)
                result: list = []

                def runner():
                    try:
                        result.append(function(*args, **kwargs))
                    except BaseException as e:  # Propagate in-thread exceptions back to the calling thread
                        result.append(e)

                t = threading.Thread(target=runner, daemon=True)
                t.start()
                t.join(limit)
                if t.is_alive():
                    if exception_message:
                        raise timeout_exception(exception_message)
                    raise timeout_exception(
                        f"Function {function.__name__} timed out after "
                        f"{limit} seconds")
                res = result[0]
                if isinstance(res, BaseException):
                    raise res
                return res

            return new_function

        return decorate

    timeout._acr_patched = True
    timeout_decorator.timeout = timeout


def _ensure_base_rev(repo_dir: Path, base_commit: str) -> str:
    """Ensure ACR's git reset/checkout target is resolvable, returning the
    base_commit to write into tasks_map (no ACR logic touched; only the git
    state it needs is prepared).

    - Normal clone: the repo has real history → return base_commit unchanged
    - codeload channel: .git is rebuilt from a single commit with no history →
      tag HEAD with a short name and return that tag. Note that a 40-hex
      character hash cannot be used as the tag name: git resolves 40-hex
      arguments only as object names, ignoring same-named refs (ambiguity
      warning), and the reset would fail.
    """
    r = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--verify", "--quiet",
         base_commit + "^{commit}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode == 0:
        return base_commit
    # Clean up old failed 40-hex tags (invalid but they cause ambiguity warnings)
    subprocess.run(["git", "-C", str(repo_dir), "tag", "-d", base_commit],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
    tag = f"acr-base-{base_commit[:8]}"
    subprocess.run(["git", "-C", str(repo_dir), "tag", "-f", tag, "HEAD"],
                   capture_output=True, text=True, check=True,
                   encoding="utf-8", errors="replace")
    return tag


def _find_patch_fallback(out_dir: Path) -> str:
    """Degraded path when organize fails: prefer selected_patch.json, otherwise
    take the extracted_patch_*.diff with the highest number."""
    for sel in sorted(out_dir.rglob("selected_patch.json"), reverse=True):
        try:
            rel = json.loads(sel.read_text(encoding="utf-8")).get("selected_patch", "")
        except Exception:
            continue
        if rel:
            p = sel.parent / rel
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
    diffs = sorted(out_dir.rglob("extracted_patch_*.diff"))
    if diffs:
        return diffs[-1].read_text(encoding="utf-8", errors="replace")
    return ""


def _apply_patch(repo_dir: Path, patch_text: str) -> bool:
    for extra in ([], ["--3way"]):
        p = subprocess.run(["git", "-C", str(repo_dir), "apply", *extra],
                           input=patch_text, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if p.returncode == 0:
            return True
    return False


def launch(task: SWETask, repo_dir: Path, collector) -> dict | None:
    """Run ACR's official flow in-process (make_swe_tasks → run_raw_task).

    Running in-process lets the litellm global callbacks and the tool wraps take
    effect directly; ACR source is unchanged (only its public functions and
    config knobs are used).
    """
    base_ref = _ensure_base_rev(repo_dir, task.base_commit)
    setup_path, tasks_path = _make_maps(task, repo_dir, base_ref)
    out_dir = repo_dir.parent / f"acr_out_{task.instance_id}"
    # Clean up leftovers from previous runs (organize classification dirs etc.), so organize_and_form_input only sees this run's results
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(ACR_ROOT) not in sys.path:
        sys.path.insert(0, str(ACR_ROOT))
    prev_cwd = os.getcwd()
    os.chdir(str(ACR_ROOT))  # ACR has several relative-path assumptions rooted at its source root
    t0 = time.perf_counter()
    try:
        _patch_timeout_decorator()  # Must precede the app import (the decoration happens at import time)
        from app import main as acr_main
        from app import config as acr_config
        acr_main.register_all_models()
        acr_config.models = [ACR_MODEL]
        acr_config.output_dir = str(out_dir)
        acr_config.conv_round_limit = _ACR_OPTS.get("conv_round_limit", 15)
        acr_config.overall_retry_limit = _ACR_OPTS.get("overall_retries", 3)
        acr_config.test_exec_timeout = 300
        # Keep validation/sbfl off by default: avoid conda installation and
        # test suite execution (trace evaluation is done uniformly by
        # finalize_swe_traces, same convention as the official one)

        _wrap_acr_tools(collector)

        tasks = acr_main.make_swe_tasks(
            task.instance_id, None, str(setup_path), str(tasks_path))
        run_ok = acr_main.run_raw_task(tasks[0])
        stderr = ""
    except Exception:
        import traceback
        run_ok = False
        stderr = traceback.format_exc()
    finally:
        os.chdir(prev_cwd)
    duration = time.perf_counter() - t0

    # Take the official final patch (organize_and_form_input → predictions_for_swebench.json)
    patch_text = ""
    try:
        pred_file = acr_main.organize_and_form_input(str(out_dir))
        for p in json.loads(Path(pred_file).read_text(encoding="utf-8")):
            if p.get("instance_id") == task.instance_id and p.get("model_patch"):
                patch_text = p["model_patch"]
                break
    except Exception:
        patch_text = _find_patch_fallback(out_dir)

    # Patch landing: git apply into the repo (env_changes captures it from this; evaluation replay depends on it)
    if patch_text.strip():
        t_apply = time.perf_counter()
        ok = _apply_patch(repo_dir, patch_text)
        record_swe_tool(collector, "acr.apply_patch",
                        {"repo": str(repo_dir), "patch_size": len(patch_text)},
                        _safe_str(patch_text, 3000) if ok else "git apply failed",
                        (time.perf_counter() - t_apply) * 1000, ok)
    else:
        record_swe_tool(collector, "acr.apply_patch", {"repo": str(repo_dir)},
                        "ACR produced no patch", 0.0, False)

    return {"acr_output_dir": str(out_dir), "run_ok": bool(run_ok),
            "wall_seconds": round(duration, 1),
            "stderr_tail": stderr[-2000:] or None}


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoCodeRover trace collection")
    parser.add_argument("--task")
    parser.add_argument("--tasks", help="comma-separated multiple tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mirror", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--conv-round-limit", type=int, default=15,
                        help="ACR context retrieval round limit (official default 15, can be lowered for smoke tests)")
    parser.add_argument("--overall-retries", type=int, default=3,
                        help="ACR whole-task retry count (official default 3)")
    args = parser.parse_args()
    _ACR_OPTS["conv_round_limit"] = args.conv_round_limit
    _ACR_OPTS["overall_retries"] = args.overall_retries
    run_tasks("autocoderover", launch,
              ([args.task] if args.task else None)
              or (args.tasks.split(",") if args.tasks else None),
              args.limit, mirror_prefix=args.mirror, output_dir=args.out)


if __name__ == "__main__":
    main()
