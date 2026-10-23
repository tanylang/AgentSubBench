"""
SWE-agent trace collection driver (Windows local runtime shim).

Zero agent source changes -- the swe-agent source (source_code/swe_agents/swe-agent/)
is untouched:
- LLM: SWE-agent's GenericAPIModelConfig calls the model via litellm
  (model name deepseek/deepseek-chat) → intercepted by swebench_common's
  global callbacks
- Tools: all of SWE-agent's shell/file interaction goes through
  deployment.runtime (swerex). Its LocalDeployment depends on the Unix-only
  pexpect.spawn (pty) and fails at import time on Windows. The harness injects
  its own subprocess-based local runtime at get_deployment (bash -c + sentinel
  output keeps the session cwd/env state; the session "/" maps to the repo's
  parent dir, so /<repo_name> from the container's perspective is our repo),
  and records tool.execution spans in real time at DefaultAgent.handle_action
  (same convention as mini-swe-agent's TraceEnvironment: external wrapping,
  internal logic unchanged).
- Fallback: after the run, if no LLM/tool span was produced, replay records
  from the .traj trajectory file.

Run:
    python scripts/run_swebench_sweagent.py [--task ID | --limit N]
        [--mirror PREFIX] [--out DIR] [--cost-limit 3.0] [--call-limit 0]

Note: the local deployment runs bash directly on the host machine --
installing/testing SWE-bench repos carries arbitrary code execution risk. For
production collection switch to the docker deployment (change back
env.deployment in the config; the rest of the driver is unchanged).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
# The sweagent package lives at the repo root (swe-agent/sweagent/), not src/
AGENT_SRC = ROOT / "source_code" / "swe_agents" / "swe-agent"
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"), str(AGENT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Windows: default GBK after redirection; SWE-agent's rich/loguru logging would crash
# (same convention as runners/run_task_cli.py and the autocoderover driver)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from swebench_common import record_swe_tool, run_tasks  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL  # noqa: E402
from adapter.swebench import SWETask  # noqa: E402

# Per-task run options (written by main(), read by launch -- run_tasks' launch signature is fixed)
_OPTS: dict = {}

# Same pattern as swebench_common._CURRENT_COLLECTOR: the wrap happens only once, multiple tasks switch via the holder
_CURRENT_COLLECTOR: dict[str, Any] = {}
_RUNTIME_REPO: dict[str, Any] = {}


# ─────────────────── Windows replacement for swerex's local runtime ───────────────────


class _WindowsLocalRuntime:
    """subprocess-based equivalent of swerex's LocalDeployment (pexpect BashSession).

    All of SWE-agent's shell/file interaction goes through deployment.runtime's
    methods: create_session / run_in_session / execute / read_file / write_file.
    This implementation runs commands with bash -c and uses sentinel output to
    keep session state (cwd/env persist across commands), matching real
    persistent-shell semantics. The session's "/" maps to the deployment root
    (= repo_dir's parent); /<repo_name> from the container's perspective is our repo.
    """

    _X_EXIT = "__SWE_SHIM_EXIT__"
    _X_CWD = "__SWE_SHIM_CWD__"
    _X_ENV_B = "__SWE_SHIM_ENV_BEGIN__"
    _X_ENV_E = "__SWE_SHIM_ENV_END__"

    def __init__(self, root: Path, repo_name: str):
        self._root = Path(root).resolve()
        self._repo_name = repo_name
        self._cwd = str(self._root)  # Session's initial cwd = "/"
        self._env = dict(os.environ)
        self._alive = True

    def _translate(self, command: str) -> str:
        """Text translation from container paths to host paths (Git Bash's "/"
        points to its install dir, not the deployment root, so the container's
        / and /<repo_name> must be rewritten).

        - /<repo_name> (word boundary, may be followed by / to continue the path) → "deployment root/<repo_name>"
        - a standalone cd / → cd deployment root (/ must be followed by whitespace/end, avoiding collateral damage to cd /tmp)
        Other paths (e.g. /tmp) stay as-is (native MSYS semantics, which happen to work).
        """
        root_posix = str(self._root).replace("\\", "/")
        repo = self._repo_name
        s = re.sub(
            r"(?<![A-Za-z0-9_./-])/" + re.escape(repo) + r"(?![A-Za-z0-9_.-])",
            f'"{root_posix}/{repo}"', command)
        s = re.sub(r"\bcd\s+/(?=[\s;&;|]|$)", f'cd "{root_posix}"', s)
        # /root is the container's writable root dir (auto-submission writes
        # /root/model.patch etc.); Git Bash's /root lands in its install dir
        # with no write permission, so map it to the deployment root
        s = re.sub(r"(?<![A-Za-z0-9_./-])/root(?![A-Za-z0-9_.-])",
                   f'"{root_posix}"', s)
        # git fetch depends on github network access (often unreachable on this
        # machine); SWE-agent's task reset command chain starts with
        # `git fetch && ...`, and a network failure (exit 128) would abort the
        # task startup (observed as a 0-span empty trace). Local reset/checkout
        # already guarantees the baseline, and fetch is only an optional sync --
        # make it failure-tolerant; semantics are unchanged on success.
        s = re.sub(r"\bgit fetch(\s+[^\s&;|]*)?\s*&&",
                   r"(git fetch\1 || true) &&", s)
        return s

    def _host(self, path: str) -> Path:
        """Container path → host path: /x → deployment root/x; relative paths
        are also resolved from the deployment root per container semantics.
        Note that on Windows pathlib does not consider "/x" (no drive letter)
        an absolute path, so here any leading "/" is treated as the container root.

        /root collapses to the deployment root itself, consistent with the
        command translation rules (auto-submission writes /root/model.patch and
        the check reads the same path; otherwise reads and writes would mismatch).
        """
        s = str(path).replace("\\", "/")
        if s.startswith("/"):
            s = s.lstrip("/")
        if s == "root" or s.startswith("root/"):
            s = s[len("root"):].lstrip("/")
        return self._root / s

    def _run_bash(self, command: str, timeout: float | None) -> tuple[str, int]:
        """Run a bash command in the session, returning (observed output, exit code)."""
        command = self._translate(command)
        wrapper = (
            'eval "$SWE_SHIM_CMD"\n'
            '__shim_rc=$?\n'
            f'echo {self._X_EXIT}$__shim_rc\n'
            f'echo {self._X_CWD}$(cygpath -m "$PWD")\n'
            f'echo {self._X_ENV_B}\n'
            'env\n'
            f'echo {self._X_ENV_E}'
        )
        env = dict(self._env)
        env["SWE_SHIM_CMD"] = command
        # Popen rather than run: on timeout we need the pid to kill the whole
        # process tree (on Windows subprocess.run's kill only kills bash itself;
        # grandchildren like find get orphaned and keep scanning the disk for
        # hours, dragging down the whole batch)
        cp = subprocess.Popen(["bash", "-c", wrapper], cwd=self._cwd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            out_raw, _ = cp.communicate(timeout=timeout)
            out = out_raw.decode("utf-8", errors="replace") if isinstance(out_raw, bytes) else (out_raw or "")
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(cp.pid)],
                           capture_output=True)
            from swerex.exceptions import CommandTimeoutError
            raise CommandTimeoutError(
                f"Command timed out after {timeout}s") from None
        m_exit = re.search(re.escape(self._X_EXIT) + r"(-?\d+)", out)
        m_cwd = re.search(re.escape(self._X_CWD) + r"([^\r\n]+)", out)
        m_env = re.search(
            re.escape(self._X_ENV_B) + r"\n(.*?)\n" + re.escape(self._X_ENV_E),
            out, re.S)
        rc = int(m_exit.group(1)) if m_exit else cp.returncode
        if m_cwd:
            self._cwd = m_cwd.group(1).strip()
        if m_env:
            new_env = {}
            for line in m_env.group(1).splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    new_env[k] = v
            self._env.update(new_env)
        cut = out.find(self._X_EXIT)
        return (out[:cut] if cut >= 0 else out), rc

    # ── swerex runtime interface ──

    async def create_session(self, req):
        from swerex.runtime.abstract import CreateBashSessionResponse
        self._cwd = str(self._root)  # New sessions start from "/"
        return CreateBashSessionResponse()

    async def run_in_session(self, action):
        from swerex.runtime.abstract import BashObservation
        if getattr(action, "action_type", "bash") == "bash_interrupt":
            return BashObservation(output="", exit_code=0)
        obs, rc = self._run_bash(action.command, action.timeout)
        if action.check == "raise" and rc != 0:
            raise RuntimeError(action.error_msg or f"Command failed ({rc})")
        return BashObservation(output=obs, exit_code=rc)

    async def execute(self, command):
        from swerex.runtime.abstract import CommandResponse
        cmd = command.command
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        else:
            cmd = self._translate(str(cmd))
        cwd = self._host(str(command.cwd)) if command.cwd else Path(self._cwd)
        env = dict(self._env)
        if command.env:
            env.update(command.env)
        try:
            cp = subprocess.run(["bash", "-c", str(cmd)], cwd=str(cwd), env=env,
                                capture_output=True, text=True,
                                timeout=command.timeout,
                                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            from swerex.exceptions import CommandTimeoutError
            raise CommandTimeoutError(
                f"Command timed out after {command.timeout}s") from None
        resp = CommandResponse(stdout=cp.stdout or "", stderr=cp.stderr or "",
                               exit_code=cp.returncode)
        if command.check and cp.returncode != 0:
            raise RuntimeError(command.error_msg
                               or f"Command failed ({cp.returncode})")
        return resp

    async def read_file(self, req):
        from swerex.runtime.abstract import ReadFileResponse
        host = self._host(req.path)
        content = host.read_text(encoding=req.encoding or "utf-8",
                                 errors=req.errors or "replace")
        return ReadFileResponse(content=content)

    async def write_file(self, req):
        from swerex.runtime.abstract import WriteFileResponse
        host = self._host(req.path)
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_text(req.content, encoding="utf-8")
        return WriteFileResponse()

    async def upload(self, req):
        from swerex.runtime.abstract import UploadResponse
        src = Path(req.source_path).resolve()
        dst = self._host(req.target_path).resolve()
        if src == dst or dst in src.parents:
            # Under the local deployment, the upload target is the repo itself
            # (repo.copy → /<repo_name>), so treat it as already done
            return UploadResponse()
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return UploadResponse()

    async def close(self):
        self._alive = False


class _WindowsLocalDeployment:
    """Implement the deployment interface sweagent uses (start/stop/is_alive/runtime)."""

    def __init__(self, root: Path, repo_name: str):
        self._runtime = _WindowsLocalRuntime(root, repo_name)

    def add_hook(self, hook):
        pass  # Logging/monitoring hooks; not needed for trace collection

    async def is_alive(self, *, timeout: float | None = None):
        from swerex.runtime.abstract import IsAliveResponse
        return IsAliveResponse(is_alive=True)

    async def start(self, *args, **kwargs):
        pass

    async def stop(self, *args, **kwargs):
        pass

    @property
    def runtime(self):
        return self._runtime


def _install_runtime_shim(repo_dir: Path) -> None:
    """Inject the subprocess-based local deployment, replacing swerex's pexpect local runtime.

    Must be called before the first import of sweagent.* (swerex.runtime.local
    raises AttributeError at module import time because pexpect.spawn is missing).

    Note: get_deployment is patched only once (reused across tasks); the repo is
    read from the holder at runtime -- a closure capturing the first task's repo
    would make every subsequent task run in the wrong repository.
    """
    _RUNTIME_REPO["repo_dir"] = repo_dir
    import pexpect
    if not hasattr(pexpect, "spawn"):
        # Only satisfies swerex.runtime.local's type annotations (module-level evaluation); sessions are implemented by this shim
        pexpect.spawn = object
    import swerex.deployment.config as swx_cfg
    if getattr(swx_cfg.get_deployment, "_swe_shim_patched", False):
        return
    import sweagent.environment.swe_env as swe_env_mod

    def get_deployment(config):
        repo = _RUNTIME_REPO["repo_dir"]
        return _WindowsLocalDeployment(repo.parent, repo.name)

    get_deployment._swe_shim_patched = True
    # patch both use sites: the swerex module attribute + swe_env's from-import binding
    swx_cfg.get_deployment = get_deployment
    swe_env_mod.get_deployment = get_deployment


# ─────────────────── Tool-level interception (real time) ───────────────────


def _wrap_handle_action(collector) -> None:
    """Record tool.execution spans in real time at DefaultAgent.handle_action.

    Default tools = bash + submit: every model action goes through handle_action.
    """
    _CURRENT_COLLECTOR["collector"] = collector
    if getattr(_wrap_handle_action, "_done", False):
        return
    _wrap_handle_action._done = True
    from sweagent.agent.agents import DefaultAgent
    orig = DefaultAgent.handle_action

    def wrapper(self, step):
        col = _CURRENT_COLLECTOR["collector"]
        t0 = time.perf_counter()
        try:
            out = orig(self, step)
            ok = True
        except Exception as e:
            record_swe_tool(col, _tool_name(step), {"action": str(step.action)[:2000]},
                            f"Error: {type(e).__name__}: {e}",
                            (time.perf_counter() - t0) * 1000, False)
            raise
        record_swe_tool(col, _tool_name(step), {"action": str(step.action)[:2000]},
                        str(out.observation)[:2000],
                        (time.perf_counter() - t0) * 1000, ok)
        return out

    wrapper._swe_trace_wrapped = True
    DefaultAgent.handle_action = wrapper


def _tool_name(step) -> str:
    if str(step.action).strip() == "submit":
        return "submit"
    return "bash"


# ─────────────────── Task integration ───────────────────


def launch(task: SWETask, repo_dir: Path, collector) -> dict | None:
    """Run via SWE-agent 1.0's programmatic interface (RunSingleConfig → run_from_config).

    Tool execution goes through the harness's Windows local runtime (subprocess);
    the LLM goes through the litellm global callbacks; tool spans are recorded
    in real time at handle_action.
    """
    _install_runtime_shim(repo_dir)
    from sweagent.run.run_single import run_from_config, RunSingleConfig

    _wrap_handle_action(collector)

    output_dir = ROOT / "swe_workspaces" / f"sweagent_{task.instance_id}"
    # Clean up leftovers from previous runs: otherwise the fallback would read the last run's .traj when tool_count==0
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RunSingleConfig(
        # No docker on this machine: the harness's local runtime (shim) replaces
        # swerex LocalDeployment; for production collection with docker, just
        # change deployment.type back to docker
        env=dict(deployment=dict(type="local"),
                 repo=dict(type="local", path=str(repo_dir))),
        agent=dict(
            name="default",
            model=dict(name=f"openai/{MAIN_MODEL_NAME}",
                       per_instance_cost_limit=_OPTS.get("cost_limit", 3.0),
                       per_instance_call_limit=_OPTS.get("call_limit", 0)),
        ),
        problem_statement=dict(text=task.problem_statement),
        output_dir=output_dir,
    )
    run_from_config(cfg)
    # The trajectory file is kept under output_dir (<instance>.traj) for
    # debugging; if no tool span was recorded during the run (early model exit
    # etc.), replay-record from the traj as a fallback
    if collector.tool_count == 0:
        traj_path = output_dir / f"{task.instance_id}.traj"
        if not traj_path.exists():
            traj_path = next(output_dir.glob("*.traj"), None)
        if traj_path:
            _record_trajectory_tools(Path(traj_path), collector)
    return {"output_dir": str(output_dir)}


def _record_trajectory_tools(traj_path: Path, collector) -> None:
    """Fallback: parse action/observation from the .traj JSONL and replay-record tool.execution spans."""
    import json as _json
    if not traj_path.exists():
        return
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        try:
            e = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        a = e.get("action") or {}
        action_type = a.get("type", "")
        if not action_type or action_type == "message":
            continue
        record_swe_tool(
            collector, action_type,
            {k: v for k, v in a.items() if k not in ("type", "message")},
            str(e.get("observation", ""))[:2000], 0.0,
            not str(e.get("observation", "")).startswith("ERROR"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-agent trace collection")
    parser.add_argument("--task")
    parser.add_argument("--tasks", help="comma-separated multiple tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mirror", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--cost-limit", type=float, default=3.0,
                        help="per-task LLM cost limit (USD), can be lowered for smoke tests)")
    parser.add_argument("--call-limit", type=int, default=0,
                        help="per-task LLM call count limit (0=unlimited), for smoke tests)")
    args = parser.parse_args()
    _OPTS["cost_limit"] = args.cost_limit
    _OPTS["call_limit"] = args.call_limit
    run_tasks("swe-agent", launch,
              ([args.task] if args.task else None)
              or (args.tasks.split(",") if args.tasks else None),
              args.limit, mirror_prefix=args.mirror, output_dir=args.out)


if __name__ == "__main__":
    main()
