"""
SWE-bench common driver layer -- collection infrastructure shared by the
per-agent driver scripts.

Responsibilities (no agent logic changes; everything happens outside):
1. LLM interception: register litellm global callbacks (these agents all call
   the LLM via litellm) and canonically record each call as an llm.call span
2. Tool recording: provide record_swe_tool(), called by each driver in **its
   own environment wrapper layer**, uniformly producing tool.execution spans
3. Collector/checkpoint assembly + task iteration (SKIP logic) + post-collection finalize
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from trace_collector import OrderedTraceCollector  # noqa: E402
from adapter.swebench import (  # noqa: E402
    SWETask, load_swebench_tasks, make_swe_checkpoint_provider, prepare_repo,
)

OUTPUT_ROOT = ROOT / "traces"

# API key environment injection (same convention as the AppWorld benchmark: config.json → environment variables)
try:
    _cfg = json.load(open(ROOT / "config.json", encoding="utf-8"))
    _key = _cfg.get("deepseek_api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
except Exception:
    _key = os.environ.get("DEEPSEEK_API_KEY", "")
if _key:
    os.environ["OPENAI_API_KEY"] = _key
    # Model switch: the MODEL_BASE_URL environment variable can override it (defaults to DeepSeek)
    os.environ["OPENAI_BASE_URL"] = os.environ.get(
        "MODEL_BASE_URL", "https://api.deepseek.com/v1")
    os.environ["DEEPSEEK_API_KEY"] = _key
    # ACR's agent_select hardcodes gpt-4-0125-preview; without OPENAI_KEY its
    # sys.exit(1) would kill the whole harness process (traces never saved).
    # With the key injected, that path fails fast with a 401 and is caught by
    # ACR's own except (agent logic unchanged)
    os.environ.setdefault("OPENAI_KEY", _key)
# mini-swe-agent's cost accounting errors on unregistered model names; this switch comes from the official docs
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

# LLM routing used by all SWE agents (normalized recording in the litellm callbacks)
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL, THINKING_MODE  # noqa: E402
AGENT_MODELS = {
    "mini-swe-agent": MAIN_MODEL_NAME,
    "swe-agent": MAIN_MODEL_NAME,
    "openhands": MAIN_MODEL_NAME,
    "autocoderover": MAIN_MODEL_NAME,
    "devika": MAIN_MODEL_NAME,
    "deepagents": MAIN_MODEL_NAME,
    "opengpts": MAIN_MODEL_NAME,
}


# litellm global callbacks are registered only once; the current task's
# collector is switched via the holder (previously one callback was appended
# per task, so global callbacks stacked up and cross-wrote between tasks)
_CURRENT_COLLECTOR: dict[str, Any] = {}


def install_litellm_hook(collector: OrderedTraceCollector, model: str) -> None:
    """Register litellm global success/failure callbacks → collector.log_llm of the current task.

    litellm's global callbacks take effect for every litellm.completion call
    inside the agent, with no agent source changes needed. Exceptions raised by
    the callbacks do not affect the agent's run.
    """
    _CURRENT_COLLECTOR["collector"] = collector
    _CURRENT_COLLECTOR["model"] = model
    try:
        import litellm
    except ImportError:
        print("[warn] litellm not installed; LLM interception unavailable")
        return
    if getattr(install_litellm_hook, "_registered", False):
        return
    install_litellm_hook._registered = True

    # Thinking mode switch: wrap litellm.completion to inject extra_body (when THINKING_MODE is non-empty)
    # -- applies uniformly to every litellm-based call (minisweagent/ACR/swe-agent etc.), zero agent source changes
    _orig_completion = litellm.completion

    def _completion_with_thinking(*args, **kwargs):
        if THINKING_MODE:
            kwargs["extra_body"] = dict(kwargs.get("extra_body") or {}, **THINKING_MODE)
        # seed/temperature same convention: all litellm-based SWE agents uniformly reproducible
        kwargs.setdefault("seed", 100)
        kwargs.setdefault("temperature", 0.0)
        return _orig_completion(*args, **kwargs)

    litellm.completion = _completion_with_thinking

    def _on_success(kwargs, completion, start_time, end_time):
        try:
            collector = _CURRENT_COLLECTOR.get("collector")
            if collector is None:
                return
            messages = kwargs.get("messages", [])
            usage = getattr(completion, "usage", None) or {}
            i_tok = getattr(usage, "prompt_tokens", 0) or 0
            o_tok = getattr(usage, "completion_tokens", 0) or 0
            out = completion
            tool_calls = None
            if hasattr(completion, "choices") and completion.choices:
                msg = completion.choices[0].message
                tcs = getattr(msg, "tool_calls", None)
                if tcs:
                    tool_calls = [{"id": t.id, "name": t.function.name,
                                   "arguments": t.function.arguments} for t in tcs]
                if getattr(msg, "content", ""):
                    out = msg.content
            collector.log_llm(
                model=kwargs.get("model") or _CURRENT_COLLECTOR.get("model"),
                messages=[{"role": str(m.get("role", "unk")),
                           "content": str(m.get("content") or "")}
                          for m in (messages or [])],
                output_text=str(out)[:5000],
                input_tokens=i_tok, output_tokens=o_tok,
                tools=kwargs.get("tools"), tool_calls=tool_calls,
            )
        except Exception:
            pass

    def _on_failure(kwargs, exception, start_time, end_time):
        try:
            collector = _CURRENT_COLLECTOR.get("collector")
            if collector is None:
                return
            collector.log_llm(
                model=kwargs.get("model") or _CURRENT_COLLECTOR.get("model"),
                messages=[{"role": str(m.get("role", "unk")),
                           "content": str(m.get("content") or "")}
                          for m in (kwargs.get("messages") or [])],
                output_text=f"[LLM ERROR] {type(exception).__name__}: {exception}",
                input_tokens=0, output_tokens=0,
            )
        except Exception:
            pass

    litellm.success_callback.append(_on_success)
    litellm.failure_callback.append(_on_failure)


def record_swe_tool(collector: OrderedTraceCollector, tool_name: str,
                    arguments: dict, result: Any, duration_ms: float = 0.0,
                    success: bool = True) -> None:
    """Unified tool recording entry point (called by each driver in its environment wrapper layer)."""
    collector.log_tool(tool_name, arguments, str(result)[:5000], duration_ms, success)


def make_collector(agent_name: str, task_id: str, model: str) -> OrderedTraceCollector:
    return OrderedTraceCollector(agent_name=agent_name, model=model,
                                 benchmark_task_id=task_id)


def run_tasks(
    agent_name: str,
    launch: Callable[[SWETask, Path, OrderedTraceCollector], dict | None],
    task_ids: list[str] | None = None,
    limit: int = 0,
    work_root: Path | None = None,
    mirror_prefix: str = "",
    output_dir: str = "",
) -> None:
    """Task iteration skeleton: repo preparation → collector/checkpoint assembly → launch callback → save.

    launch(task, repo_dir, collector) is implemented by each agent driver:
    - Builds the agent itself (can use the source under source_code/swe_agents)
    - Records tool calls via record_swe_tool
    - Returns the agent's exit info dict (optional, stored in the trace's top-level extra)
    """
    tasks = load_swebench_tasks(limit=limit, task_ids=task_ids)
    out_root = Path(output_dir) if output_dir else OUTPUT_ROOT / f"swebench_{agent_name}"
    out_root.mkdir(parents=True, exist_ok=True)
    # Workspaces are isolated per agent: concurrent runs of the same task by
    # multiple agents get separate repo directories, avoiding git operations
    # stepping on each other (previously caused intermittent checkpoint collection failures)
    work_root = work_root or (ROOT / "swe_workspaces" / agent_name)

    model = AGENT_MODELS.get(agent_name, "deepseek-chat")
    for i, task in enumerate(tasks, 1):
        out_path = out_root / f"{task.instance_id}.json"
        if out_path.exists():
            print(f"[{i}/{len(tasks)}] {task.instance_id}: [SKIP]")
            continue
        print(f"[{i}/{len(tasks)}] {task.instance_id} ({task.repo})")
        collector = make_collector(agent_name, task.instance_id, model)
        t0 = time.perf_counter()
        try:
            repo_dir = prepare_repo(task, work_root, mirror_prefix)
            collector.set_checkpoint_provider(
                make_swe_checkpoint_provider(repo_dir, task.instance_id))
            install_litellm_hook(collector, model)
            extra = launch(task, repo_dir, collector)
        except Exception as e:
            import traceback
            traceback.print_exc()
            extra = {"error": f"{type(e).__name__}: {e}"}
        finally:
            dt = time.perf_counter() - t0
        collector.save(str(out_path),
                       task_instruction=task.problem_statement[:500],
                       outcome="unknown")
        data = json.load(open(out_path, encoding="utf-8"))
        data["task"] = task.problem_statement[:500]
        data["swebench"] = {
            "instance_id": task.instance_id, "repo": task.repo,
            "base_commit": task.base_commit,
            "FAIL_TO_PASS": task.FAIL_TO_PASS, "PASS_TO_PASS": task.PASS_TO_PASS,
        }
        if extra:
            data["agent_extra"] = extra
        # Atomic write-back (same convention as trace_collector.save, guards against truncation)
        tmp_path = str(out_path) + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(out_path))
        print(f"    spans={collector._step} llm={collector.llm_count} "
              f"tools={collector.tool_count} ({dt:.0f}s)")

    # ── Required post-collection step: evaluation (pass) + event_type annotation ──
    try:
        from runners.finalize_swe_traces import finalize_swe_trace_dir
        finalize_swe_trace_dir(str(out_root), agent_name)
    except ImportError as e:
        print(f"[warn] finalize unavailable: {e}")
