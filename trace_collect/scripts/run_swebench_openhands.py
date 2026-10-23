"""
OpenHands (1.x new architecture) trace collection driver -- agent_server HTTP route.

Zero agent internal changes:
- Start the official agent_server (local subprocess, uvicorn) pointed at the
  task repository directory
- The harness acts as an HTTP client: create a session → issue the task →
  consume the event stream
- LLM recording: llm_completion_log events in the stream (including usage);
  tool recording: acp_tool_call / command output events
- checkpoint: git state collection from swebench_common (independent of the
  agent architecture)

Run:
    python scripts/run_swebench_openhands.py [--task ID | --tasks A,B] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from swebench_common import record_swe_tool, run_tasks  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL  # noqa: E402
from adapter.swebench import SWETask  # noqa: E402

SERVER_PORT = int(os.environ.get("OPENHANDS_PORT", "38527"))


SESSION_KEY = "trace-collector-key"


class _OpenHandsServer:
    """agent_server subprocess manager (official entry: python -m openhands.agent_server).

    Presets the session key and workspace at startup via the server config file
    (skipping the /api/init orchestration flow); the harness calls directly with
    the X-Session-API-Key header.
    """

    def __init__(self, repo_dir: Path, model: str = MAIN_MODEL_NAME):
        self.repo_dir = repo_dir
        self.model = model
        self.proc: subprocess.Popen | None = None
        self.server_dir: Path | None = None

    def start(self) -> None:
        import tempfile
        # Server config: workspace/<config>.json (relative to its cwd), presetting the session key and workspace
        self.server_dir = Path(tempfile.mkdtemp(prefix="oh_server_"))
        config = {
            "workspace_path": str(self.repo_dir),
            "session_api_keys": [SESSION_KEY],
        }
        (self.server_dir / "workspace").mkdir(exist_ok=True)
        (self.server_dir / "workspace" / "openhands_agent_server_config.json").write_text(
            json.dumps(config), encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", ""),
        })
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "openhands.agent_server",
             "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
            env=env, cwd=str(self.server_dir),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 120.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                with urllib.request.urlopen(
                        f"http://localhost:{SERVER_PORT}/", timeout=2):
                    return
            except Exception:
                time.sleep(2)
        raise RuntimeError("agent_server startup timed out")

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _api(method: str, path: str, payload: dict | None = None) -> dict:
    """Minimal HTTP client (no extra dependencies)."""
    req = urllib.request.Request(
        f"http://localhost:{SERVER_PORT}{path}", method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-Session-API-Key", SESSION_KEY)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(req, data=data, timeout=60) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body) if body else {}


def launch(task: SWETask, repo_dir: Path, collector) -> dict | None:
    server = _OpenHandsServer(repo_dir)
    server.start()
    try:
        server.wait_ready()
        conv = _api("POST", "/api/conversations", {
            "workspace": {"working_dir": str(repo_dir)},
            "agent": {
                "llm": {
                    "model": f"openai/{MAIN_MODEL_NAME}",
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "base_url": MAIN_BASE_URL,
                    "temperature": 0.0,
                    # Force native function calling (non-whitelisted models default to prompt-mock)
                    "native_tool_calling": True,
                },
                # The default tool set must be given explicitly (same as the official examples);
                # tool_module_qualnames imports these modules into the server registry,
                # and only the two together let the agent call them
                "tools": [{"name": n} for n in
                          ("terminal", "file_editor", "task_tracker",
                           "apply_patch", "glob", "grep")],
            },
            # SWE tools are registered via their modules ({name: module} mapping; importing triggers registration)
            "tool_module_qualnames": {
                "terminal": "openhands.tools.terminal",
                "file_editor": "openhands.tools.file_editor",
                "apply_patch": "openhands.tools.apply_patch",
                "glob": "openhands.tools.glob",
                "grep": "openhands.tools.grep",
            },
            "initial_message": {"role": "user",
                                "content": [{"type": "text",
                                             "text": task.problem_statement}],
                                "run": True},
            "max_iterations": 30,
        })
        conv_id = conv.get("id") or conv.get("conversation_id")
        if not conv_id:
            raise RuntimeError(f"failed to create conversation: {conv}")
        try:
            _api("POST", f"/api/conversations/{conv_id}/run")
        except urllib.error.HTTPError as e:
            if e.code != 409:  # 409 = already running (initial_message.run=true already started it)
                raise

        # Poll the event stream: paginated pulls from /events/search (items format)
        deadline = time.time() + 1800
        seen_ids: set = set()
        pending: dict = {}
        done = False
        while time.time() < deadline and not done:
            page_id = None
            while True:
                q = f"/api/conversations/{conv_id}/events/search?limit=100"
                if page_id:
                    q += f"&page_id={page_id}"
                resp = _api("GET", q)
                items = resp.get("items", [])
                new = [ev for ev in items if ev.get("id") not in seen_ids]
                for ev in new:
                    seen_ids.add(ev.get("id"))
                    done = _record_event(ev, collector, pending) or done
                page_id = resp.get("next_page_id") or resp.get("nextPageId")
                if not page_id or not new:
                    break
            time.sleep(3)
        # Wrap-up: flush the last unmatched LLM span when the session ends
        if "_llm" in pending:
            llm = pending.pop("_llm")
            collector.log_llm(
                model=llm["model"],
                messages=[{"role": "system",
                           "content": "(OpenHands event stream; prompt in MessageEvent)"}],
                output_text="[end] conversation ended",
                input_tokens=llm["input_tokens"],
                output_tokens=llm["output_tokens"],
            )
        return {"conversation_id": str(conv_id)}
    finally:
        server.stop()


def _parse_value(v) -> Any:
    """Event values are repr-serialized dicts/strs; convert them back to Python objects."""
    import ast
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def _record_event(ev: dict, collector, pending: dict) -> bool:
    """Event → canonical span; returns True when the session has ended.

    Observed event shapes (openhands 1.11, calibrated with probes):
    - ConversationStateUpdateEvent key=stats: cumulative token usage
      (accumulated_token_usage); the LLM span is recorded as the delta between
      two stats, and the output text uses the summary of the ActionEvent that
      follows (guaranteeing the span has content and stays interleaved in order
      with the tool spans)
    - ActionEvent (agent): tool_name + tool_call_id + action → tool call
    - ObservationEvent (environment): tool_name + tool_call_id + observation → tool result
    - ConversationStateUpdateEvent key=execution_status: finished/error → ended
    """
    kind = ev.get("kind", "")
    source = ev.get("source", "")
    if kind == "ConversationStateUpdateEvent":
        key = ev.get("key", "")
        if key == "execution_status":
            return str(ev.get("value", "")) in (
                "completed", "finished", "stopped", "error")
        if key == "stats":
            value = _parse_value(ev.get("value")) or {}
            metrics = value.get("usage_to_metrics", {}).get("default", {})
            usage = metrics.get("accumulated_token_usage", {}) or {}
            prev = pending.get("_tokens", (0, 0))
            cur = (int(usage.get("prompt_tokens", 0) or 0),
                   int(usage.get("completion_tokens", 0) or 0))
            pending["_tokens"] = cur
            # Buffer first: emit together with the output text when the ActionEvent arrives
            pending["_llm"] = {
                "model": metrics.get("model_name", MAIN_MODEL_NAME),
                "input_tokens": max(cur[0] - prev[0], 0),
                "output_tokens": max(cur[1] - prev[1], 0),
            }
        return False
    if kind == "ActionEvent" and source == "agent":
        # Flush the corresponding LLM span first (output text = summary of this action, guaranteeing non-empty)
        if "_llm" in pending:
            llm = pending.pop("_llm")
            action = _parse_value(ev.get("action"))
            collector.log_llm(
                model=llm["model"],
                messages=[{"role": "system",
                           "content": "(OpenHands event stream; prompt in MessageEvent)"}],
                output_text=f"[action] {ev.get('tool_name', '')}: "
                            f"{str(action)[:300]}",
                input_tokens=llm["input_tokens"],
                output_tokens=llm["output_tokens"],
            )
        pending[ev.get("tool_call_id")] = (ev.get("tool_name", ""),
                                           _parse_value(ev.get("action")))
        return False
    if kind == "ObservationEvent" and source == "environment":
        tool_name = ev.get("tool_name", "")
        args = {}
        entry = pending.pop(ev.get("tool_call_id"), None)
        if entry:
            action = entry[1]
            if isinstance(action, dict):
                args = {k: v for k, v in action.items()
                        if k not in ("message", "thought")}
        record_swe_tool(collector, tool_name or "tool", args,
                        str(_parse_value(ev.get("observation")))[:2000],
                        0.0, True)
        return False
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenHands trace collection")
    parser.add_argument("--task")
    parser.add_argument("--tasks", help="comma-separated multiple tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mirror", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    run_tasks("openhands", launch,
              ([args.task] if args.task else None)
              or (args.tasks.split(",") if args.tasks else None),
              args.limit, mirror_prefix=args.mirror, output_dir=args.out)


if __name__ == "__main__":
    main()
