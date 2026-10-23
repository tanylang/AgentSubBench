"""SWE-bench (mini-swe-agent) live replay runtime.

Hard rule (same as the AppWorld runtime): the downstream continuation must be driven
by the REAL mini-swe-agent source (seeded message history + the real DefaultAgent
loop); report backend unavailability honestly, never degrade to an approximation
loop.

Key design: - SwebenchModelAdapter: a duck-typing implementation of the
minisweagent.Model Protocol; internally it is the span_offload ChatModelClient +
mini-swe-agent's own parse/format functions (parse_toolcall_actions /
format_toolcall_observation_messages / FormatError) -- every downstream step's LMResponse
is natively captured (itemized token accounting), zero litellm imports, and the
FormatError semantics are verbatim identical to the real agent. - restore = base_commit
checkout + env_delta_log delta replay + hash-chain verification
(swebench_env.restore_deltas); e3b0... is a legal empty state. - injected message shape:
{"role":"assistant","content":None, "tool_calls":[native], "extra":{"actions":[{command,
tool_call_id}]}} (consumed by default.py:154). - verify_local NEVER executes bash (an
arbitrary-code security red line); execution equivalence is completed by real execution
after injection. - final evaluation = apply_test_patch + pytest FAIL_TO_PASS ∪
PASS_TO_PASS (local basis).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import ChatModelClient, LMResponse
from ..runtime import (
    AgentRuntime, ParseResult, ResolvedRequest, RestoredSession, SpanContext,
    StopReplay,
)
from ..schemas import (
    DownstreamOutcome, FinalOutcome, SpanRecord, TraceRun,
)
from ..verifiers import VerifierVerdict
from .swebench_env import (
    SWE_COLLECT_ROOT, SWE_MIRROR, SWE_TEST_TIMEOUT, SWE_WORK_ROOT,
    apply_env_changes, ensure_workspace, reset_workspace, restore_deltas,
    run_pytest_eval,
)

#: mini-swe-agent source root (SWE_COLLECT_ROOT deploy package preferred, MSWEA_SRC overrides)
MSWEA_SRC = os.environ.get(
    "MSWEA_SRC",
    os.path.join(SWE_COLLECT_ROOT, "source_code", "swe_agents",
                 "mini-swe-agent"))
WORKSPACE_TOKEN = "<SWE_WORKSPACE>/"
# : the 27B collection batch's docker-style repo path convention (/testbed in commands -> the real workspace); the old 58-task batch has no such convention, a non-matching replace is
# harmless
TESTBED_TOKEN = "/testbed"
TRACE_RESULT_MAX_LEN = 2000

_msw = None  # minisweagent module cache


def _minisweagent():
    global _msw
    if _msw is not None:
        return _msw
    src = MSWEA_SRC
    if not os.path.isdir(src):
        raise RuntimeError(
            "backend unavailable: minisweagent source not found at %s; the "
            "downstream continuation must run the REAL agent source" % src)
    sys.path.insert(0, src)
    import minisweagent  # noqa: F401
    _msw = minisweagent
    return _msw


# ---------------------------------------------------------------------------
# SwebenchModelAdapter:duck-typed minisweagent.Model over ChatModelClient
# ---------------------------------------------------------------------------
class _ToolCall:
    def __init__(self, d: Dict[str, Any]):
        self.id = d.get("id")
        from types import SimpleNamespace
        self.function = SimpleNamespace(
            name=d.get("name", ""), arguments=d.get("arguments", "{}"))


class SwebenchModelAdapter:
    """Model Protocol over ChatModelClient; reuses minisweagent's real parse/format.

    Point-by-point aligned with litellm_model.py's query flow: prepare (drop
    extra) -> complete (tools=[BASH_TOOL]) -> usage accounting ->
    parse_toolcall_actions (on FormatError, pad cost/response extra then raise)
    -> return {role, content, tool_calls, extra:{actions, response, cost,
    timestamp}}
    """

    def __init__(self, client: ChatModelClient, usage_sink: List[LMResponse]):
        import minisweagent
        from minisweagent.models.utils.actions_toolcall import (
            BASH_TOOL, format_toolcall_observation_messages,
            parse_toolcall_actions,
        )
        from minisweagent.models.utils.openai_multimodal import (
            expand_multimodal_content,
        )
        from minisweagent.exceptions import FormatError
        self.client = client
        self.usage_sink = usage_sink
        self.BASH_TOOL = BASH_TOOL
        self._parse = parse_toolcall_actions
        self._format_obs = format_toolcall_observation_messages
        self._expand = expand_multimodal_content
        self.FormatError = FormatError
        # consistent with litellm_model's config defaults
        self.format_error_template = "{{ error }}"
        self.observation_template = (
            "{% if output.exception_info %}<exception>{{output.exception_info}}"
            "</exception>\n{% endif %}"
            "<returncode>{{output.returncode}}</returncode>\n"
            "<output>\n{{output.output}}</output>")
        self.multimodal_regex = ""
        self.config = self  # Protocol's config: Any

    def _prepare(self, messages):
        return [{k: v for k, v in m.items() if k != "extra"}
                for m in messages]

    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        resp = self.client.complete(
            self._prepare(messages), tools=[self.BASH_TOOL], temperature=0.0,
            max_tokens=self.client.max_output_tokens)
        self.usage_sink.append(resp)
        cost = self.client.cost_for(resp.prompt_tokens, resp.completion_tokens)
        if resp.error:
            raise self.FormatError({
                "role": "user",
                "content": str(resp.error),
                "extra": {"interrupt_type": "FormatError", "cost": cost},
            })
        tool_calls = [_ToolCall(c) for c in (resp.tool_calls or [])]
        try:
            actions = self._parse(
                tool_calls, format_error_template=self.format_error_template,
                template_kwargs={})
        except self.FormatError as e:
            e.messages[0]["extra"].update({"cost": cost})
            raise
        # native tool_calls structure (for the downstream trace record and message history)
        native = [{"id": c.id, "type": "function",
                   "function": {"name": c.function.name,
                                "arguments": c.function.arguments}}
                  for c in tool_calls]
        message = {"role": "assistant", "content": resp.raw_text or None,
                   "tool_calls": native}
        message["extra"] = {
            "actions": actions,
            "response": {"raw_text": resp.raw_text, "tool_calls": native},
            "cost": cost,
            "timestamp": time.time(),
        }
        return message

    def format_message(self, **kw) -> Dict[str, Any]:
        return self._expand(kw, pattern=self.multimodal_regex)

    def format_observation_messages(self, message, outputs,
                                    template_vars=None) -> List[Dict[str, Any]]:
        actions = (message.get("extra") or {}).get("actions", [])
        return self._format_obs(
            actions=actions, outputs=outputs,
            observation_template=self.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.multimodal_regex)

    def get_template_vars(self, **kw) -> Dict[str, Any]:
        return {}

    def serialize(self) -> Dict[str, Any]:
        return {"model": self.client.model}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
@dataclass
class SwebenchSession(RestoredSession):
    agent: Any = None
    env: Any = None
    model_adapter: Optional[SwebenchModelAdapter] = None
    workspace_dir: Optional[Path] = None
    span: Optional[SpanRecord] = None
    trace: Optional[TraceRun] = None
    injected: Any = None
    usage: List[LMResponse] = field(default_factory=list)
    downstream_trace: Optional[Dict[str, Any]] = None
    task: Optional[Dict[str, Any]] = None
    exit_status: str = ""
    completed: bool = False
    restored_env_hash_raw: str = ""


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class SwebenchRuntime(AgentRuntime):
    name = "swebench-live"

    def __init__(self, manifest=None, work_root: Optional[str] = None):
        self.manifest = manifest
        self.work_root = work_root or SWE_WORK_ROOT

    # -- 14.4 step 3 --------------------------------------------------------
    def restore(self, trace: TraceRun, span: SpanRecord,
                baseline_client: Any = None) -> SwebenchSession:
        _minisweagent()  # backend hard check: raise directly if unavailable
        import minisweagent
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.local import LocalEnvironment

        task = {}
        if self.manifest is not None:
            mt = self.manifest.get(trace.run_meta.task_id)
            if mt is not None:
                task = dict(mt.evaluator)
        if not task:
            raise RuntimeError("manifest missing task info: %s"
                               % trace.run_meta.task_id)
        # workspace: base_commit checkout + delta replay + hash-chain verification
        ws = ensure_workspace(task, self.work_root)
        reset_workspace(ws)
        ref = span.state_snapshot.environment_snapshot_ref
        snap = trace.artifacts["state_snapshots"][ref]
        final_hash, notes = restore_deltas(
            ws, trace.artifacts["env_delta_log"],
            snap["delta_log_index"])
        # session: msg:j (tokens -> real paths)
        msgs = [dict(m) for m in trace.artifacts["messages"]["msg:%d"
                                                           % span.invocation_index]]
        for m in msgs:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace(
                    WORKSPACE_TOKEN, str(ws) + "/")
                m["content"] = m["content"].replace(
                    TESTBED_TOKEN, str(ws))
        usage: List[LMResponse] = []  # the adapter and session share one accounting list
        adapter = SwebenchModelAdapter(baseline_client, usage)
        env = LocalEnvironment(cwd=str(ws), timeout=30)
        agent = DefaultAgent(
            adapter, env, system_template="(seeded)",
            instance_template="", step_limit=50, cost_limit=5.0,
            max_consecutive_format_errors=3)
        agent.messages = msgs
        agent.n_calls = span.invocation_index
        agent.cost = 0.0
        agent.extra_template_vars = {}
        return SwebenchSession(
            runtime=self, agent=agent, env=env, model_adapter=adapter,
            workspace_dir=ws, span=span, trace=trace, injected=None,
            baseline_client=baseline_client,
            usage=usage,
            snapshot_hash="env_delta:" + final_hash,
            # raw-to-raw gate: the actual chain hash after replay vs the collection side's recorded value -- the chain invariant guarantees equality (entry idx's
            # hash_before = the accumulation after applying the first idx deltas); a mismatch means the delta replay drifted from the collection side's semantics, exposed
            # honestly
            restored_env_hash_raw=final_hash,
            task=task,
            downstream_trace={
                "meta": {"task_id": trace.run_meta.task_id,
                         "run_id": trace.run_meta.run_id,
                         "baseline_model": trace.run_meta.baseline_model,
                         "continuation_model": (
                             baseline_client.model if baseline_client else "")},
                "spans": [], "terminal": {},
            },
        )

    # -- 14.4 step 4 --------------------------------------------------------
    def resolve_request(self, trace: TraceRun, span: SpanRecord,
                        session: Optional[SwebenchSession] = None
                        ) -> ResolvedRequest:
        msgs = [dict(m) for m in trace.artifacts["messages"]["msg:%d"
                                                           % span.invocation_index]]
        if session is not None and session.workspace_dir is not None:
            for m in msgs:
                if isinstance(m.get("content"), str):
                    m["content"] = m["content"].replace(
                        WORKSPACE_TOKEN, str(session.workspace_dir) + "/")
                    m["content"] = m["content"].replace(
                        TESTBED_TOKEN, str(session.workspace_dir))
        tools = trace.artifacts["recorded_tools"].get(
            "tools:%d" % span.invocation_index) or None
        return ResolvedRequest(
            messages=msgs, tools=tools, response_format=None,
            temperature=0.0, max_output_tokens=8192,
            meta_hint=span.meta_hint, tool_choice=None,
        )

    # -- 14.4 step 5 --------------------------------------------------------
    def parse(self, span: SpanRecord, raw: str) -> ParseResult:
        """Text-fallback parsing: a JSON tool_call list -> the contract; everything else fails (the real agent raises FormatError on non-tool-call output; mapped
        honestly)."""
        try:
            parsed = json.loads(raw)
        except ValueError:
            return ParseResult(ok=False, parsed=None,
                               error="expected bash tool call JSON, got %r"
                                     % raw[:120])
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            return ParseResult(ok=True, parsed=parsed, error="")
        return ParseResult(ok=False, parsed=None,
                           error="no tool_calls in text output")

    def schema_check(self, session: SwebenchSession, span: SpanRecord,
                     parsed: Any) -> bool:
        if not isinstance(parsed, dict) or not parsed.get("tool_calls"):
            return False
        for c in parsed["tool_calls"]:
            if str(c.get("name", "")) != "bash":
                return False
            args = c.get("arguments")
            if not isinstance(args, dict) or not isinstance(
                    args.get("command"), str):
                return False
        return True

    def verify_local(self, session: SwebenchSession, span: SpanRecord,
                     parsed: Any) -> VerifierVerdict:
        """Contract verification: check only the name/command shape, NEVER execute bash (an arbitrary-code red line)."""
        ok = self.schema_check(session, span, parsed)
        return VerifierVerdict(
            triggered=True, detected_failure=not ok, passed=ok,
            executed=["contract_check", "schema_check"],
            errors=[] if ok else ["bash command contract violated"])

    # -- 14.4 step 6 --------------------------------------------------------
    def inject(self, session: SwebenchSession, span: SpanRecord,
               parsed: Any) -> bool:
        from ..appworld_adapter import _native_tool_calls, _synth_call_id
        if isinstance(parsed, dict) and parsed.get("__span_failure__") is not None:
            # SLM output rejected (fallback=none): go through the FormatError shape so the agent recovers on its own
            msg = {"role": "user", "content": str(parsed["__span_failure__"]),
                   "extra": {"interrupt_type": "FormatError"}}
        elif isinstance(parsed, dict) and parsed.get("tool_calls"):
            calls = parsed["tool_calls"]
            # the synthesized id must appear on BOTH the native tool_calls and the actions' tool_call_id:
            # format_observation_messages decides role=tool and pairing by the action's tool_call_id; without it the
            # observation message becomes a user message -> next round vLLM 400s (assistant tool_calls with no paired tool
            # message)
            synth = [{"id": _synth_call_id("inj", i), "name": c["name"],
                      "arguments": c.get("arguments") or {}}
                     for i, c in enumerate(calls)]
            native = _native_tool_calls(synth, id_prefix="inj")
            actions = []
            for i, c in enumerate(calls):
                args = c.get("arguments") or {}
                cmd = str(args.get("command", ""))
                if session.workspace_dir is not None:
                    cmd = cmd.replace(WORKSPACE_TOKEN,
                                      str(session.workspace_dir) + "/")
                    cmd = cmd.replace(TESTBED_TOKEN,
                                      str(session.workspace_dir))
                actions.append({"command": cmd,
                                "tool_call_id": synth[i]["id"]})
            msg = {"role": "assistant", "content": None, "tool_calls": native,
                   "extra": {"actions": actions}}
        else:
            msg = {"role": "user", "content": "no usable output injected",
                   "extra": {"interrupt_type": "FormatError"}}
        session.agent.messages.append(msg)
        session.injected = msg
        return True

    # -- 14.4 steps 7-8 -----------------------------------------------------
    def _trace_step(self, session: SwebenchSession, entry: Dict[str, Any]):
        if session.downstream_trace is not None:
            session.downstream_trace["spans"].append(entry)

    def _trace_round(self, session: SwebenchSession, i: int, before: int):
        """Record the messages produced by one step(): model messages -> llm.call, observation messages -> tool.execution (ok judged by returncode, same basis as the collection
        side)."""
        step = i - session.span.invocation_index
        for m in session.agent.messages[before:]:
            role = m.get("role")
            if role == "assistant":
                tcs = m.get("tool_calls") or []
                # LLM thinking: persisted when the message dict carries reasoning_content/reasoning
                # ("" when not thinking); thinking kept in full, not truncated (same basis as
                # appworld)
                reasoning = str(m.get("reasoning_content")
                                or m.get("reasoning") or "")
                if tcs:
                    output = {"tool_calls": [
                        {"name": tc.get("function", {}).get("name", ""),
                         "arguments": (json.loads(
                             tc.get("function", {}).get("arguments", "{}")
                             or "{}") if isinstance(
                                 tc.get("function", {}).get("arguments"), str)
                             else tc.get("function", {}).get("arguments", {}))}
                        for tc in tcs],
                        "reasoning": reasoning}
                else:
                    output = {"text": str(m.get("content", ""))
                              [:TRACE_RESULT_MAX_LEN],
                              "reasoning": reasoning}
                self._trace_step(session, {
                    "span_type": "llm.call", "step": step,
                    "source": "baseline", "output": output})
            elif role == "tool":
                extra = m.get("extra") or {}
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": step,
                    "source": "baseline", "tool": "bash",
                    "arguments": {"command": ""},
                    "ok": (extra.get("returncode") == 0),
                    "result": str(m.get("content", ""))
                    [:TRACE_RESULT_MAX_LEN]})

    def run_to_node(self, session: SwebenchSession, span: SpanRecord,
                    client: Any, max_steps: int, timeout_s: float,
                    budget_usd: float) -> Tuple[DownstreamOutcome,
                                                List[LMResponse]]:
        outcome = DownstreamOutcome()
        try:
            self._consume_injected(session)
            outcome.next_node_reached = True
            outcome.next_tool_executed = True
            outcome.downstream_execution_valid = True
        except Exception as exc:
            outcome.unhandled_exception = True
            session.downstream_trace["terminal"] = {
                "status": "exception: %s" % str(exc)[:200]}
        return outcome, []

    def _consume_injected(self, session: SwebenchSession):
        """Execute the injected span's actions (the real agent's execute_actions path)."""
        import minisweagent
        from minisweagent.exceptions import InterruptAgentFlow
        injected = session.injected
        if not isinstance(injected, dict) or not (injected.get("extra") or {}).get("actions"):
            return
        actions = injected["extra"]["actions"]
        self._trace_step(session, {
            "span_type": "llm.call", "step": 0, "source": "slm",
            "output": {"tool_calls": [
                {"name": "bash", "arguments": {"command": a.get("command", "")}}
                for a in actions]}})
        try:
            outputs = session.agent.execute_actions(injected)
        except InterruptAgentFlow as e:
            session.agent.add_messages(*e.messages)
            if (e.messages and
                    e.messages[-1].get("extra", {}).get("exit_status")):
                session.exit_status = e.messages[-1]["extra"]["exit_status"]
        else:
            outputs = outputs or []
        for i, out in enumerate(outputs):
            ok = (out.get("extra", {}) or {}).get("returncode") == 0
            cmd = actions[i].get("command", "") if i < len(actions) else ""
            self._trace_step(session, {
                "span_type": "tool.execution", "step": 0, "source": "slm",
                "tool": "bash", "arguments": {"command": cmd}, "ok": bool(ok),
                "result": str(out.get("content", ""))[:TRACE_RESULT_MAX_LEN]})
        if session.agent.messages and session.agent.messages[-1].get("role") == "exit":
            session.completed = True
            session.exit_status = (session.agent.messages[-1].get("extra") or {}) \
                .get("exit_status", "")

    def run_to_completion(self, session: SwebenchSession, span: SpanRecord,
                          client: Any, max_steps: int, timeout_s: float,
                          budget_usd: float) -> Tuple[FinalOutcome,
                                                      List[LMResponse]]:
        final = FinalOutcome()
        import minisweagent
        from minisweagent.exceptions import InterruptAgentFlow
        deadline = time.perf_counter() + timeout_s
        steps_left = max(1, max_steps)
        i = span.invocation_index
        try:
            while not session.completed and steps_left > 0:
                self.budget_guard(steps_left, deadline)
                before = len(session.agent.messages)
                try:
                    session.agent.step()
                    session.agent.n_consecutive_format_errors = 0
                except session.model_adapter.FormatError as e:
                    session.agent.cost += (e.messages[0].get("extra") or {}) \
                        .get("cost", 0.0)
                    session.agent.n_consecutive_format_errors += 1
                    if (0 < 3 <=
                            session.agent.n_consecutive_format_errors):
                        session.agent.add_messages(
                            *e.messages,
                            {"role": "exit",
                             "content": "RepeatedFormatError",
                             "extra": {"exit_status": "RepeatedFormatError",
                                       "submission": ""}})
                    else:
                        session.agent.add_messages(*e.messages)
                except InterruptAgentFlow as e:
                    session.agent.add_messages(*e.messages)
                    for m in e.messages:
                        st = (m.get("extra") or {}).get("exit_status")
                        if st:
                            session.exit_status = st
                # record this round's llm.call / tool.execution (structured downstream trace)
                self._trace_round(session, i, before)
                i += 1
                steps_left -= 1
                last = session.agent.messages[-1] if session.agent.messages else {}
                if last.get("role") == "exit":
                    session.completed = True
                    session.exit_status = (last.get("extra") or {}) \
                        .get("exit_status", "")
        except StopReplay as stop:
            final.timeout = stop.flag == "timeout"
            final.budget_exhausted = stop.flag == "budget_exhausted"
            session.downstream_trace["terminal"] = {
                "status": "stopped: %s" % stop.flag}
            return final, list(session.usage)

        if session.completed:
            session.downstream_trace["terminal"] = {
                "status": "completed", "exit_status": session.exit_status}
            if not final.timeout:
                try:
                    ev = run_pytest_eval(session.workspace_dir, session.task)
                    final.goal_state_passed = bool(ev["success"])
                    final.final_task_success = bool(ev["success"])
                    final.evaluator_details = [
                        "FTP_ok=%s PTP_ok=%s tests=%d" % (
                            ev["FAIL_TO_PASS_ok"], ev["PASS_TO_PASS_ok"],
                            ev["total_tests"])]
                    for t, ok in ev.get("details", {}).items():
                        if not ok:
                            final.evaluator_details.append(
                                "%s: FAIL(%s)" % (
                                    t, ev.get("tails", {}).get(t, "")[-160:]))
                except Exception as exc:
                    final.evaluator_details.append(
                        "evaluation error: %s: %s"
                        % (type(exc).__name__, str(exc)[:200]))
        else:
            session.downstream_trace["terminal"] = {
                "status": "stopped: budget exhausted"}
            final.budget_exhausted = True
        return final, list(session.usage)

    def rollback(self, session: SwebenchSession) -> bool:
        # pre-commit verifier only: no state committed before injection
        return True
