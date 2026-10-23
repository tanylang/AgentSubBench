"""
Shared ordered trace collector — produces traces matching the autogen format.

Every agent batch script should use this to ensure:
  invoke_agent -> llm.call -> tool.execution -> llm.call -> tool.execution -> ...
  agent.step: 1, 2, 3, 4... (strictly sequential)
  gen_ai.tool.name: extracted from code (e.g. supervisor.show_profile)
"""

import json, os, time, uuid
from typing import Any

# ── On-disk truncation switch (default 0 = no truncation, full dump) ────────────────────────
# If traces get too large, you can re-enable caps via env (character counts):
#   TRACE_CAP_MSG=500 TRACE_CAP_OUTPUT=5000 TRACE_CAP_REASONING=5000 TRACE_CAP_TOOL=5000
_CAP_MSG = int(os.environ.get("TRACE_CAP_MSG", "0") or "0")
_CAP_OUTPUT = int(os.environ.get("TRACE_CAP_OUTPUT", "0") or "0")
_CAP_REASONING = int(os.environ.get("TRACE_CAP_REASONING", "0") or "0")
_CAP_TOOL = int(os.environ.get("TRACE_CAP_TOOL", "0") or "0")


def _cap(s: str, cap: int) -> str:
    return s if cap <= 0 else s[:cap]


# freezegun-immune real clock (calls the kernel directly; inside AppWorld's frozen world, time.* is all simulated)
try:
    from adapter.clock import real_time_ns as _real_time_ns
except ImportError:  # fallback for the extreme case where adapter is not on sys.path (no span timestamp needed)
    def _real_time_ns() -> int:
        return int(time.time() * 1e9)


class OrderedTraceCollector:
    """Collects llm.call and tool.execution spans in strict call order."""

    def __init__(self, agent_name: str, model: str, benchmark_task_id: str,
                 benchmark: str = "AppWorld"):
        import threading
        self.agent_name = agent_name
        self.model = model
        self.benchmark_task_id = benchmark_task_id
        self.benchmark = benchmark
        self._checkpoint_provider = None  # registered by the benchmark script
        self._request_context_provider = None  # request protocol context (tools etc.), used by openai SDK streaming
        # litellm callbacks may fire on background threads (concurrent with tool
        # recording on the main thread) -- the lock keeps _next_step increment and append atomic, so step numbers never skip
        self._lock = threading.Lock()

        self.trace_id = uuid.uuid4().hex
        self.episode_id = str(uuid.uuid4())
        self._spans: list[dict] = []
        self._step = 0
        self._llm_count = 0
        self._tool_count = 0
        self._last_span_end: int | None = None  # timestamp of the previous span record (for computing span gaps)
        self._input_tokens = 0
        self._output_tokens = 0
        self._t0 = time.perf_counter()

        # Root span (will be finalized on save)
        self._root_span = {
            "type": "invoke_agent",
            "span_id": uuid.uuid4().hex[:16],
            "trace_id": self.trace_id,
            "parent_id": None,
            "start_time": self._ns_now(),
            "end_time": self._ns_now(),
            "attributes": {
                "agent.episode.id": self.episode_id,
                "gen_ai.agent.name": agent_name,
                "model": model,
                "benchmark": benchmark,
                "benchmark_task_id": benchmark_task_id,
            },
            "status": "OK",
            "kind": "SERVER",
        }

        # Track last LLM output for dedup
        self._last_llm_output: str | None = None

    def _ns_now(self) -> int:
        return _real_time_ns()

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def _add_span(self, span_type: str, **attrs) -> dict:
        # take-number + append both happen inside the lock: when the litellm callback
        # thread races the main thread, step increment and append must be atomic, otherwise steps skip/reorder (observed across 14 traces)
        with self._lock:
            now = self._ns_now()
            # gap from the previous span (record-point gap: previous span recorded -> this span recorded)
            gap_ms = 0.0
            if self._last_span_end is not None:
                gap_ms = round((now - self._last_span_end) / 1e6, 3)
            self._last_span_end = now
            s = {
                "type": span_type,
                "span_id": uuid.uuid4().hex[:16],
                "trace_id": self.trace_id,
                "parent_id": self._root_span["span_id"],
                "start_time": now,
                "end_time": now,
                # duration_ms is backfilled by log_llm/log_tool once the measured duration is known
                "duration_ms": 0.0,
                "gap_from_prev_ms": gap_ms,
                "attributes": {"agent.step": self._next_step(), **attrs},
                "status": "OK",
                "kind": "CLIENT" if span_type == "llm.call" else "INTERNAL",
            }
            # attach a check_point to every span (task_id / state hash or snapshot / env changes of this span)
            if self._checkpoint_provider is not None:
                try:
                    cp = self._checkpoint_provider(span_type, span_id=s["span_id"])
                    if cp:
                        s["check_point"] = cp
                except Exception:
                    pass  # checkpoint collection failure does not affect the main trace
            self._spans.append(s)
        return s

    # ── Public API ──

    def set_checkpoint_provider(self, provider) -> None:
        """Register a checkpoint collection callback: provider(span_type) -> dict.

        The dict should contain fields like task_id / application_state_hash /
        env_changes. Benchmark scripts build it from their own AppWorld world
        instance (see make_appworld_checkpoint_provider).
        """
        self._checkpoint_provider = provider

    def set_request_context_provider(self, provider) -> None:
        """Register a request protocol context callback: provider() -> {"tools": ..., "tool_calls": ...}.

        For SDK streaming frameworks (openai_agents): hooks only see the response; request
        parameters (tools) are stashed at the driver-layer client patch and retrieved at log_llm.
        """
        self._request_context_provider = provider

    def log_llm(self, model: str, messages: list[dict], output_text: str,
                input_tokens: int = 0, output_tokens: int = 0,
                tools: list | str | None = None,
                tool_calls: list[dict] | None = None,
                reasoning: str = "",
                duration_ms: float = 0.0) -> None:
        """Record an LLM call. Skips if output is identical to previous call.

        tools: tools parameter of this request (protocol-layer tool catalog, raw JSON)
        tool_calls: structured tool_calls array of this response [{id, name, arguments}]
        reasoning: thinking content of this response (reasoning_content field; truncated
            to 5000 in the attribute, full output_reasoning in the conversation)
        duration_ms: measured duration of this call (measured at the call site by hook/driver)
        """
        # request protocol context (provided via the provider for SDK streaming frameworks)
        if self._request_context_provider is not None:
            try:
                ctx = self._request_context_provider() or {}
            except Exception:
                ctx = {}
            if tools is None:
                tools = ctx.get("tools")
            if tool_calls is None:
                tool_calls = ctx.get("tool_calls")

        # Dedup: skip if output is same as last LLM call
        if output_text and output_text == self._last_llm_output:
            return
        self._last_llm_output = output_text

        self._llm_count += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

        # input message summary (default: full, no truncation; env TRACE_CAP_MSG can re-enable a cap)
        msgs_summary = json.dumps(
            [{"role": m.get("role", "unk"),
              "content": _cap(str(m.get("content") or ""), _CAP_MSG)}
             for m in (messages if isinstance(messages, list) else [])],
            ensure_ascii=False,
        )

        attrs = {"gen_ai.request.model": model,
                 "gen_ai.system": model.split("/")[0] if "/" in model else model.split("-")[0],
                 "gen_ai.input.messages": msgs_summary,
                 "gen_ai.output.messages": _cap(output_text, _CAP_OUTPUT),
                 "gen_ai.response.model": model,
                 "gen_ai.usage.input_tokens": input_tokens,
                 "gen_ai.usage.output_tokens": output_tokens}
        if tools:
            # protocol-layer tool catalog (the tools parameter the model actually sees, raw JSON on one line)
            attrs["gen_ai.request.tools"] = (
                tools if isinstance(tools, str)
                else json.dumps(tools, ensure_ascii=False, default=str))
        if reasoning:
            # thinking content (reasoning_content): trimmed in the attribute, full in the conversation
            attrs["gen_ai.output.reasoning"] = _cap(reasoning, _CAP_REASONING)
        if duration_ms:
            attrs["gen_ai.request.duration_ms"] = round(duration_ms, 3)

        span = self._add_span("llm.call", **attrs)
        span["duration_ms"] = round(duration_ms, 3)  # backfill measured duration (measured at the call site)

        # store the full conversation (untruncated) in check_point for prompt-level analysis
        if "check_point" in span:
            def _norm_msg(m: dict) -> dict:
                # keep extra keys (e.g. tool_results), normalize role/content to strings
                d = dict(m)
                d["role"] = str(d.get("role", "unk"))
                d["content"] = str(d.get("content") or "")
                return d

            conv = {
                "has_full_conversation": True,
                "messages": [
                    _norm_msg(m) for m in (messages if isinstance(messages, list) else [])
                ],
                "output_text": output_text,
                "output_reasoning": reasoning,
            }
            if tool_calls:
                # structured tool_calls (protocol level, distinct from text serialization)
                conv["tool_calls"] = tool_calls
            span["check_point"]["conversation"] = conv

    def log_tool(self, tool_name: str, arguments: dict, result: str,
                 duration_ms: float = 0.0, success: bool = True) -> None:
        """Record a tool execution. tool_name/arguments are already in canonical form (benchmark-agnostic)."""
        self._tool_count += 1

        span = self._add_span("tool.execution",
            **{"gen_ai.tool.name": tool_name,
               "gen_ai.tool.call.arguments": _cap(json.dumps(arguments, ensure_ascii=False), _CAP_TOOL),
               "gen_ai.tool.call.result": _cap(str(result), _CAP_TOOL),
               "tool.duration_ms": duration_ms,
               "tool.success": success,
            })
        span["duration_ms"] = round(duration_ms, 3)  # backfill measured duration (measured at the execution site)

    def save(self, output_path: str, task_instruction: str = "",
             outcome: str = "unknown") -> str:
        """Finalize and save the trace JSON. Returns the file path.

        If outcome is 'unknown', auto-detect: 'success' if complete_task was
        called in any tool span, 'failure' otherwise.
        """
        if outcome == "unknown" and self.benchmark == "AppWorld":
            # Auto-detect: check if supervisor.complete_task was called
            # (AppWorld-specific heuristic; other benchmarks keep the caller-provided value)
            for s in self._spans:
                if s["type"] == "tool.execution":
                    name = s["attributes"].get("gen_ai.tool.name", "")
                    if "complete_task" in name:
                        outcome = "success"
                        break
            else:
                outcome = "failure"

        wall_ms = (time.perf_counter() - self._t0) * 1000

        # Finalize root span
        self._root_span["end_time"] = self._ns_now()
        self._root_span["attributes"]["agent.outcome"] = outcome
        self._root_span["attributes"]["agent.total_steps"] = self._step
        self._root_span["attributes"]["agent.total_llm_calls"] = self._llm_count
        self._root_span["attributes"]["agent.total_tool_calls"] = self._tool_count
        self._root_span["attributes"]["agent.wall_time_ms"] = wall_ms

        # sort by agent.step before saving: a litellm callback thread may arrive after
        # later tool spans; the lock keeps steps contiguous and the sort keeps the list order aligned with step (semantically correct order)
        with self._lock:
            spans_sorted = sorted(self._spans, key=lambda s: s["attributes"]["agent.step"])

        data = {
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "agent_name": self.agent_name,
            "task": task_instruction or "(loaded from AppWorld task)",
            "total_steps": self._step,
            "total_llm_calls": self._llm_count,
            "total_tool_calls": self._tool_count,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "spans": [self._root_span] + spans_sorted,
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # atomic write: tmp + replace -- a force-killed process never leaves a truncated trace file
        # (a session-restart kill once produced a 3.4MB truncated JSON that crashed evaluation)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, output_path)
        return output_path

    @property
    def llm_count(self) -> int:
        return self._llm_count

    @property
    def tool_count(self) -> int:
        return self._tool_count

    @property
    def token_count(self) -> int:
        return self._input_tokens + self._output_tokens


def make_appworld_checkpoint_provider(world_or_getter, task_id: str,
                                      force_remote: bool = False):
    """Build a per-span AppWorld environment snapshot collector.

    Each span produces:
      - task_id: task location
      - application_state_hash: sha256 of the accumulated environment state
        at span start (the full chain of DB increments since the task began), the replay-consistency anchor
      - env_changes: increments produced by each app's DB during this span
        (full Change tuples in AppWorld changes format, collected read-only,
        not disturbing the world's own checkpoint semantics)

    Read-only access to AppWorld's db.tracker.changes increment log; the
    world can be an instance or a lambda (some agents create it only inside solve).

    Remote-execution agents (OpenAI/MCP): the API runs in a separate service
    process, so the client tracker sees no changes; we tail the server-side DB's jsonl change log instead
    (each line = [statement, parameters, executemany], the same format as
    the env_changes Change tuples). No appworld/SDK/agent source logic is modified.
    """
    import hashlib
    from appworld.common.imports import import_models_module
    from appworld.common.path_store import path_store

    _trackers: dict[str, Any] = {}
    _prev: dict[str, int] = {}
    _baseline: dict[str, Any] = {}
    _running = hashlib.sha256()
    _init_done = [False]
    # ── remote (server-side on-disk DB table-level diff) mode state ──
    _remote_db_home: str | None = None
    _remote_prev: dict[str, Any] = {}       # app -> {table: (cols, pks, {pk_key: row})}
    _remote_running: Any = None             # independent hash chain (same semantics as local mode)

    def _remote_read_app(app: str) -> dict:
        """Open the server-side on-disk sqlite read-only and read all user-table rows."""
        import sqlite3
        p = os.path.join(_remote_db_home, f"{app}.db")
        con = sqlite3.connect(f"file:{os.path.abspath(p)}?mode=ro", uri=True, timeout=10)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            snap: dict[str, Any] = {}
            for t in tables:
                info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
                cols = [r[1] for r in info]
                pks = [r[1] for r in info if r[5] > 0]
                rows: dict[Any, Any] = {}
                for row in con.execute(f'SELECT * FROM "{t}"'):
                    key = tuple(row[cols.index(p)] for p in pks) if pks else tuple(row)
                    rows[key] = row
                snap[t] = (cols, pks, rows)
            return snap
        finally:
            con.close()

    def _remote_diff_app(prev: dict, cur: dict) -> list:
        """Table-level diff -> list of Change tuples in AppWorld changes format."""
        changes: list = []
        for t, (cols, pks, prev_rows) in prev.items():
            cur_entry = cur.get(t)
            cur_rows = cur_entry[2] if cur_entry else {}
            ins_rows: list = []
            for k, row in cur_rows.items():
                p = prev_rows.get(k)
                if p == row:
                    continue
                if p is None:
                    if pks:
                        col_list = ", ".join(f'"{c}"' for c in cols)
                        ph = ", ".join("?" * len(cols))
                        changes.append(
                            (f'INSERT INTO "{t}" ({col_list}) VALUES ({ph})',
                             list(row), False))
                    else:
                        ins_rows.append(list(row))
                elif pks:
                    set_c = ", ".join(f'"{c}"=?' for c in cols)
                    where = " AND ".join(f'"{c}"=?' for c in pks)
                    changes.append(
                        (f'UPDATE "{t}" SET {set_c} WHERE {where}',
                         list(row) + [row[cols.index(c)] for c in pks], False))
            if ins_rows:
                col_list = ", ".join(f'"{c}"' for c in cols)
                ph = ", ".join("?" * len(cols))
                changes.append(
                    (f'INSERT INTO "{t}" ({col_list}) VALUES ({ph})', ins_rows, True))
            if pks:
                dels = [list(row) for k, row in prev_rows.items() if k not in cur_rows]
                if dels:
                    where = " AND ".join(f'"{c}"=?' for c in pks)
                    changes.append((f'DELETE FROM "{t}" WHERE {where}', dels, True))
        return changes

    def _ensure_init_remote(world) -> None:
        if _init_done[0]:
            return
        import hashlib
        from appworld.apps.lib.models.db import get_db_home_path
        # closure-trap fix: nonlocal is required, otherwise the assignments stay
        # local to this function and the outer _remote_db_home/_remote_running remain
        # None forever (_remote_read_app reads the outer ones -> the os.path.join(None, ...)
        # TypeError is swallowed -> the baseline stays empty; hash retrieval fails the same way)
        nonlocal _remote_db_home, _remote_running
        # on-disk location of the server side (apis_server_kwargs on_disk=True); in
        # in-memory mode this directory is empty/absent, so collection is naturally empty.
        _remote_db_home = get_db_home_path(
            storage_type="disk", type="base", temporary=True)
        _remote_running = hashlib.sha256()
        if os.path.isdir(_remote_db_home):
            app_names = sorted(set([*world.task.allowed_apps, "admin", "supervisor"]))
            for app in app_names:
                if os.path.exists(os.path.join(_remote_db_home, f"{app}.db")):
                    try:
                        _remote_prev[app] = _remote_read_app(app)
                    except Exception:
                        pass
        if os.environ.get("TRACE_DEBUG") == "1":
            print(f"[remote-checkpoint-debug] db_home={_remote_db_home} "
                  f"isdir={os.path.isdir(_remote_db_home)} "
                  f"snapped_apps={sorted(_remote_prev.keys())}", flush=True)
        _init_done[0] = True

    def provide_remote(world, span_id: str | None = None) -> dict:
        if world is None:
            return {"task_id": task_id, "application_state_hash": None, "env_changes": {}}
        first = not _init_done[0]
        _ensure_init_remote(world)
        # lazily backfill baselines: apps whose .db files were not yet on disk at
        # init get snapshotted when a later span discovers them (baseline only, no increments; normal diffing starts next span)
        if os.path.isdir(_remote_db_home):
            _app_names = sorted(set([*world.task.allowed_apps, "admin", "supervisor"]))
            for _app in _app_names:
                if (_app not in _remote_prev
                        and os.path.exists(os.path.join(_remote_db_home, f"{_app}.db"))):
                    try:
                        _remote_prev[_app] = _remote_read_app(_app)
                    except Exception:
                        pass
        start_hash = _remote_running.hexdigest()
        env_changes: dict[str, Any] = {}
        for app, prev in _remote_prev.items():
            try:
                cur = _remote_read_app(app)
                changes = _remote_diff_app(prev, cur)
            except Exception as _e:
                if os.environ.get("TRACE_DEBUG") == "1":
                    print(f"[remote-checkpoint-debug] app={app} read/diff failed: "
                          f"{type(_e).__name__}: {_e}", flush=True)
                continue
            if changes:
                env_changes[app] = changes
                _remote_running.update(
                    json.dumps(changes, default=str, ensure_ascii=False).encode("utf-8"))
            _remote_prev[app] = cur
        if os.environ.get("TRACE_DEBUG") == "1":
            print(f"[remote-checkpoint-debug] span={span_id} "
                  f"apps={len(_remote_prev)} changes={sum(len(v) for v in env_changes.values())}",
                  flush=True)
        cp = {
            "task_id": task_id,
            "application_state_hash": start_hash,
            "env_changes": env_changes,
        }
        if first:
            cp["env_changes_baseline"] = {}
        return cp

    def _is_remote_world(world) -> bool:
        # remote mode is determined by remote_apis_url (requester.client is the
        # local mode's TestClient; in remote mode it is actually None).
        if getattr(world, "remote_apis_url", None):
            return True
        req = getattr(world, "requester", None)
        return req is not None and bool(getattr(req, "remote_apis_url", None))

    def _ensure_init(world) -> None:
        if _init_done[0]:
            return
        app_names = sorted(set([*world.task.allowed_apps, "admin", "supervisor"]))
        for app in app_names:
            try:
                mod = import_models_module(app)
                _trackers[app] = mod.SQLModel.db.tracker
                # writes from the world-initialization phase are archived separately (skipped
                # during replay -- replay starts from the "already-initialized world"; AppWorld itself handles initialization)
                _baseline[app] = list(_trackers[app].changes)
                _prev[app] = len(_baseline[app])
                # the baseline joins the hash chain: application_state_hash covers
                # all state changes since world creation (a true full-chain hash, no placeholder segments)
                if _baseline[app]:
                    _running.update(
                        json.dumps(_baseline[app], default=str, ensure_ascii=False).encode("utf-8"))
            except Exception:
                _trackers[app] = None
                _prev[app] = 0
        _init_done[0] = True

    def provide(span_type: str, span_id: str | None = None) -> dict:
        world = world_or_getter() if callable(world_or_getter) else world_or_getter
        if world is None:
            return {"task_id": task_id, "application_state_hash": None, "env_changes": {}}
        # remote-execution agents: take the server-side DB snapshot path (same provider interface, no behavior difference)
        if not _init_done[0] and (force_remote or _is_remote_world(world)):
            return provide_remote(world, span_id)
        if _remote_db_home is not None:
            return provide_remote(world, span_id)
        first = not _init_done[0]
        _ensure_init(world)
        start_hash = _running.hexdigest()
        env_changes: dict[str, Any] = {}
        for app, tracker in _trackers.items():
            if tracker is None:
                continue
            try:
                new = list(tracker.changes[_prev[app]:])
                _prev[app] = len(tracker.changes)
            except Exception:
                continue
            if new:
                env_changes[app] = new  # full Change tuples (incremental diff)
                _running.update(
                    json.dumps(new, default=str, ensure_ascii=False).encode("utf-8"))
        cp = {
            "task_id": task_id,
            "application_state_hash": start_hash,
            "env_changes": env_changes,
        }
        if first:
            cp["env_changes_baseline"] = _baseline  # archived only, skipped by replay tools
        return cp

    return provide
