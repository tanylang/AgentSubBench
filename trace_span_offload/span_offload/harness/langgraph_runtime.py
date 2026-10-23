"""LoCoMo + LangGraph ReAct + LangMem live replay runtime.

Hard rule (same as the AppWorld/SWE/locomo runtimes): the downstream continuation must be driven by the REAL
LangGraph source -- create_react_agent + langmem tools + the collection side's own ChatOpenAI binding and
invoke config (recursion_limit=12); report backend unavailability honestly, never degrade to an approximation
loop.

Key design: - restore = empty-store start + memory_delta_log differential
replay (ingestion deltas included in the chain) + per-entry hash verification;
rebuild the real InMemoryStore (deterministic hash embedder) - injection:
tool_call_list -> AIMessage (synthesized-id native tool_calls) + real langmem
tool execution + ToolMessage backfill, the current question continues via the
real graph; freeform -> the injected text is that question's answer, bookkeep
and advance q_idx - each question = prior Q/A history
(compact_history_messages) + the new question, a single agent.invoke
(recursion_limit=12, verbatim the same as the collection side) - verify_local:
tool span = schema shape; answer span = the official F1 (commit gate =
non-empty text; the 0.5 threshold belongs to the final evaluator) - final
evaluation = the official eval_question_answering (avg F1 >= 0.5; this arm has
no read set, recall is always None) - C0 frozen: downstream injects the
trace-recorded output, only really executes the memory tools (store is local),
zero LM calls throughout
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..llm_client import LMResponse
from langchain_core.callbacks import BaseCallbackHandler as _BaseCallbackHandler
from ..runtime import (
    AgentRuntime, ParseResult, ResolvedRequest, RestoredSession, StopReplay,
)
from ..schemas import (
    DownstreamOutcome, FinalOutcome, SpanRecord, TraceRun,
)
from ..timing_util import monotonic_ms
from ..verifiers import VerifierVerdict
from .langgraph_env import (
    SYSTEM_PROMPT, compact_history_messages, detect_qa_template,
    question_prompt, question_prompt_core, rebuild_store, replay_memory_log,
)
from .locomo_env import (
    PASS_THRESHOLD, evaluate_answers, load_sample,
)

RECURSION_LIMIT = 12
QA_F1_COMMIT_GATE = 0.0
TRACE_RESULT_MAX_LEN = 2000
AGENT_ID = "LoCoMo-LangGraph-ReAct-LangMem"
LANGGRAPH_TOOL_NAMES = ("manage_memory", "search_memory")

#: reasoning_content placeholder for the injected assistant message (the
#: DeepSeek thinking endpoint requires non-empty; harmless on non-thinking endpoints)
RC_PLACEHOLDER = "Determining the next memory operation."


# ---------------------------------------------------------------------------
# replay-side model binding: reasoning capture on the same basis as the collection side's TracedReasoningChatOpenAI + timing/trajectory recording (the collection side's collector duty lands on the session on the replay
# side)
# ---------------------------------------------------------------------------
def _make_replay_model(session: "LangGraphSession"):
    from langchain_openai import ChatOpenAI

    bc = session.baseline_client

    class ReplayReasoningChatOpenAI(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            if isinstance(response, dict):
                raw_response = response
            elif hasattr(response, "model_dump"):
                raw_response = response.model_dump(mode="json")
            else:
                raw_response = {}
            result = super()._create_chat_result(response, generation_info)
            for generation, choice in zip(
                    result.generations, raw_response.get("choices") or []):
                raw_message = choice.get("message") or {}
                for key in ("reasoning", "reasoning_content",
                            "output_reasoning"):
                    reasoning = raw_message.get(key)
                    if reasoning not in (None, ""):
                        generation.message.additional_kwargs["reasoning"] = reasoning
                        generation.message.additional_kwargs[
                            "reasoning_source_field"] = key
                        break
            return result

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            start_ms = monotonic_ms()
            try:
                result = super()._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as error:
                elapsed = monotonic_ms() - start_ms
                session.runtime._trace_llm(session,
                    messages, None, elapsed, error=str(error)[:200])
                raise
            elapsed = monotonic_ms() - start_ms
            session.runtime._trace_llm(session, messages, result, elapsed)
            return result

    os.environ.setdefault("OPENAI_API_KEY", bc.api_key or "EMPTY")
    return ReplayReasoningChatOpenAI(
        model=bc.model,
        base_url=bc.base_url,
        api_key=bc.api_key or "EMPTY",
        temperature=0,
        timeout=300,
        max_retries=2,
    )


class _ReplayToolHandler(_BaseCallbackHandler):
    """LangGraph callback: real tool execution -> downstream_trace (same duty as the collection side's ToolTraceHandler,
    different sink). Inherits BaseCallbackHandler to satisfy the langgraph callback protocol (raise_error/ignore_chain etc.
    attributes)."""

    def __init__(self, session: "LangGraphSession"):
        super().__init__()
        self.session = session

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None,
                      **kwargs):
        self.session._pending_tool[str(run_id)] = {
            "name": serialized.get("name") or "unknown_tool",
            "arguments": inputs if inputs is not None else input_str,
            "start_ms": monotonic_ms(),
        }

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.session.runtime._end_tool(self.session, str(run_id), output=output)

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.session.runtime._end_tool(self.session, str(run_id), error=error)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
@dataclass
class LangGraphSession(RestoredSession):
    sample: Any = None
    records: List[Dict[str, Any]] = field(default_factory=list)
    store: Any = None
    span: Optional[SpanRecord] = None
    trace: Optional[TraceRun] = None
    baseline_client: Any = None
    agent: Any = None                   # the real create_react_agent (lazy-built)
    tools: List[Any] = field(default_factory=list)
    namespace: List[str] = field(default_factory=list)
    model: Any = None
    handler: Any = None
    q_idx: int = 0
    num_questions: int = 0
    qa_template: str = "full"          # "full" (production) / "core" (smoke test) basis
    phase: str = "qa"                  # the phase the target span sits in (ingestion/qa)
    qa_index: int = 0                  # this unit's global question number
    ingestion_remaining: List[Dict[str, Any]] = field(default_factory=list)
    question_texts: List[str] = field(default_factory=list)
    answers: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    question_messages: List[Any] = field(default_factory=list)  # LangChain messages
    question_done: bool = True
    injected: Any = None
    usage: List[LMResponse] = field(default_factory=list)
    downstream_trace: Dict[str, Any] = field(default_factory=dict)
    restored_env_hash_raw: str = ""
    prev_ts_end: float = 0.0
    _pending_tool: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _tool_ep: Any = None                # langgraph.func.entrypoint (out-of-graph tool execution)
    # C0 frozen state
    frozen_downstream: bool = False
    frozen_plan: Optional[List[SpanRecord]] = None
    frozen_index: int = 0
    frozen_completed: bool = False
    c0_fidelity: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class LangGraphRuntime(AgentRuntime):
    name = "langgraph-live"

    def __init__(self, manifest=None):
        self.manifest = manifest

    # -- backend hard check ------------------------------------------------
    def _backend_check(self):
        try:
            from langgraph.prebuilt import create_react_agent  # noqa: F401
            from langgraph.store.memory import InMemoryStore  # noqa: F401
            from langmem import (  # noqa: F401
                create_manage_memory_tool, create_search_memory_tool,
            )
        except ImportError as exc:
            raise RuntimeError(
                "backend unavailable: langgraph/langmem not importable (%s); "
                "the downstream continuation must run the REAL LangGraph "
                "agent source" % exc)

    def _ensure_tools(self, session: LangGraphSession):
        """The real langmem tools (bound to session.store); the frozen path makes zero LM calls and builds no model/graph, but tool execution must still go through the real
        source."""
        if session.tools:
            return session.tools
        self._backend_check()
        from langmem import (
            create_manage_memory_tool, create_search_memory_tool,
        )
        namespace = tuple(session.namespace)
        session.tools = [
            create_manage_memory_tool(namespace=namespace),
            create_search_memory_tool(namespace=namespace),
        ]
        return session.tools

    def _invoke_tool(self, session: LangGraphSession, name: str, args: Dict):
        """Execute langmem tools outside the graph: langgraph.func.entrypoint(store=...) provides the Pregel runtime context (the langmem official docs pattern; zero LM
        calls)."""
        if session._tool_ep is None:
            from langgraph.func import entrypoint

            @entrypoint(store=session.store)
            def _ep(payload: Dict[str, Any]) -> Any:
                tool = next(t for t in session.tools
                            if getattr(t, "name", "") == payload["name"])
                return tool.invoke(payload["args"])

            session._tool_ep = _ep
        return session._tool_ep.invoke({"name": name, "args": args})

    def _make_agent(self, session: LangGraphSession):
        if session.agent is not None:
            return session.agent
        self._backend_check()
        from langgraph.prebuilt import create_react_agent
        self._ensure_tools(session)
        session.model = _make_replay_model(session)
        session.handler = _ReplayToolHandler(session)
        # verbatim the same as the collection side: create_react_agent(model, tools, store, prompt)
        session.agent = create_react_agent(
            model=session.model, tools=session.tools, store=session.store,
            prompt=SYSTEM_PROMPT)
        return session.agent

    # -- downstream trace + timing (same basis as appworld's _trace_step) -
    def _trace_step(self, session: LangGraphSession, entry: Dict[str, Any],
                    elapsed_ms: float = 0.0):
        now = monotonic_ms()
        entry["elapsed_ms"] = round(elapsed_ms, 3)
        entry["ts_end_ms"] = now
        entry["ts_start_ms"] = now - round(elapsed_ms, 3)
        entry["duration_ms"] = round(elapsed_ms, 3)
        entry["gap_from_prev_ms"] = round(
            max(0.0, entry["ts_start_ms"] - session.prev_ts_end), 3)
        session.prev_ts_end = now
        session.downstream_trace["spans"].append(entry)

    def _trace_llm(self, session: LangGraphSession, messages, result,
                   elapsed_ms: float, error: str = ""):
        output_message = None if result is None else result.generations[0].message
        output_text = "" if output_message is None else str(
            output_message.content or "")
        reasoning = ""
        tool_calls: List[Dict[str, Any]] = []
        usage_in = usage_out = 0
        if output_message is not None:
            reasoning = str(output_message.additional_kwargs.get("reasoning")
                            or "")
            tool_calls = [{"name": str(tc.get("name", "")),
                           "arguments": (json.loads(tc["args"])
                                         if isinstance(tc.get("args"), str)
                                         else (tc.get("args") or {}))}
                          for tc in (output_message.tool_calls or [])]
            usage = output_message.usage_metadata or {}
            usage_in = int(usage.get("input_tokens") or 0)
            usage_out = int(usage.get("output_tokens") or 0)
        session.usage.append(LMResponse(
            raw_text=output_text, prompt_tokens=usage_in,
            completion_tokens=usage_out))
        if tool_calls:
            self._trace_step(session, {
                "span_type": "llm.call", "step": session.q_idx,
                "source": "baseline",
                "output": {"tool_calls": tool_calls,
                           "reasoning": reasoning}})
        else:
            self._trace_step(session, {
                "span_type": "llm.call", "step": session.q_idx,
                "source": "baseline",
                "output": {"text": output_text[:TRACE_RESULT_MAX_LEN],
                           "reasoning": reasoning}})
        if error:
            session.downstream_trace["spans"][-1]["error"] = error

    def _end_tool(self, session: LangGraphSession, run_id: str,
                  output=None, error=None):
        pending = session._pending_tool.pop(run_id, None)
        if pending is None:
            return
        elapsed = monotonic_ms() - pending["start_ms"]
        if hasattr(output, "content"):
            result_value = output.content
        else:
            result_value = output
        self._trace_step(session, {
            "span_type": "tool.execution", "step": session.q_idx,
            "source": "baseline", "tool": pending["name"],
            "arguments": pending["arguments"] or {},
            "ok": error is None,
            "result": ("" if result_value is None
                       else (result_value if isinstance(result_value, str)
                             else json.dumps(result_value, ensure_ascii=False,
                                             default=str))
                       )[:TRACE_RESULT_MAX_LEN],
        }, elapsed_ms=elapsed)

    # -- message reconstruction (inverse of the collection side's trace_message) --
    def _to_lc_messages(self, recorded: List[Dict[str, Any]]) -> List[Any]:
        from langchain_core.messages import (
            AIMessage, HumanMessage, ToolMessage,
        )
        out: List[Any] = []
        for m in recorded:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                out.append(HumanMessage(content=content))
            elif role == "assistant":
                tcs = []
                for tc in m.get("tool_calls") or []:
                    args = tc.get("args") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    # langchain 0.3 validation: tool_calls[].args must be a dict
                    tcs.append({
                        "name": str(tc.get("name", "")),
                        "args": args if isinstance(args, dict) else {},
                        "id": str(tc.get("id", "")),
                    })
                if tcs:
                    # passing None explicitly is rejected by langchain validation (must be a list); omit the arg when there are no tool_calls (AI answers in the prior
                    # history)
                    out.append(AIMessage(content=content or "",
                                         tool_calls=tcs))
                else:
                    out.append(AIMessage(content=content or ""))
            elif role == "tool":
                out.append(ToolMessage(
                    content=content,
                    tool_call_id=str(m.get("tool_call_id", ""))))
        return out

    # -- 14.4 step 3 --------------------------------------------------------
    def restore(self, trace: TraceRun, span: SpanRecord,
                baseline_client: Any = None) -> LangGraphSession:
        agent_name = trace.run_meta.agent_id
        if agent_name != AGENT_ID:
            raise RuntimeError(
                "backend unavailable: agent %r not supported by langgraph "
                "runtime (LoCoMo-LangGraph-ReAct-LangMem only)" % agent_name)
        self._backend_check()
        # the sample id differs from the unit task_id (task_id = locomo:<sample>:q<qi>)
        sample_id = str(trace.artifacts["langgraph"].get("sample_id")
                        or trace.run_meta.task_id)
        sample = load_sample(sample_id)
        snap = trace.artifacts["state_snapshots"][
            span.state_snapshot.environment_snapshot_ref]
        step = int(snap.get("step", 0) or 0)
        records, final_hash, notes = replay_memory_log(
            sample_id, trace.artifacts["memory_delta_log"], step)
        recorded_hash = str(span.state_snapshot.application_state_hash or "")
        if recorded_hash and final_hash != recorded_hash:
            raise RuntimeError(
                "langgraph restore hash mismatch for span %d: recorded %s != "
                "rebuilt %s (%s)" % (
                    span.invocation_index, recorded_hash[:12],
                    final_hash[:12], "; ".join(notes)))
        store = rebuild_store(records, ["locomo", sample_id, "memories"])

        agent_snap = trace.artifacts["state_snapshots"].get(
            "agent:%d" % span.invocation_index) or {}
        q_idx = int(agent_snap.get("q_idx", 0) or 0)
        num_q = int(agent_snap.get("num_questions", 0) or 1)
        phase = str(agent_snap.get("locomo_phase", "") or "qa")
        lg = trace.artifacts.get("langgraph") or {}
        qa_index = int(lg.get("qa_index", 0) or 0)
        qa_template = str(lg.get("qa_template", "full") or "full")
        question_text = str(lg.get("question_text", "") or "")
        # unit semantics: one unit trace = this QA (each question independent; no prior history/answer seed)
        question_texts = [question_text]
        answers: List[Dict[str, Any]] = []
        history: List[Dict[str, Any]] = []
        if q_idx > 0 and lg.get("answers"):
            # in a single-question trace the target span sits after an answered question (a theoretical edge): seed that question's recorded answer for the identity
            # comparison
            for i, a in enumerate(lg["answers"][:q_idx]):
                answers.append({
                    "idx": int(a.get("question_index", i)),
                    "question": a.get("question", ""),
                    "pred": a.get("predicted_answer", "")})
                history.append({
                    "question_index": int(a.get("question_index", i)),
                    "question": a.get("question", ""),
                    "predicted_answer": a.get("predicted_answer", "")})

        # the system prompt goes through the real agent source (SYSTEM_PROMPT
        # authoritative import); the recorded value is only a cross-check snapshot
        recorded_prompt = trace.artifacts["system_prompts"].get(
            "sys:%d" % span.invocation_index) or ""
        prompt_drift = bool(recorded_prompt and recorded_prompt != SYSTEM_PROMPT)

        # the current question's message list (inverse transform of msg:j; contains prior history + question + loop
        # messages); QA template basis: the unit artifact was already recorded per this QA; if the target span's
        # messages contain a question message, use it as authoritative (ingestion spans' messages only have turn
        # text)
        recorded_msgs = [dict(m) for m in trace.artifacts["messages"][
            "msg:%d" % span.invocation_index]]
        question_messages = self._to_lc_messages(recorded_msgs)
        if phase == "qa":
            # QA spans' messages contain the question text; use them as authoritative to detect the template basis
            for m in recorded_msgs:
                if m.get("role") == "user":
                    qa_template = detect_qa_template(
                        str(m.get("content", "") or ""))
                    break

        # ingestion remaining turns (turns still to ingest after the target span; [] in the QA phase)
        ingestion_remaining: List[Dict[str, Any]] = []
        if phase == "ingestion":
            dia_id = str(agent_snap.get("ingestion_dia_id", "") or "")
            if dia_id:
                turns = list(getattr(sample, "turns", []) or [])
                pos = next((i for i, t in enumerate(turns)
                            if t.dia_id == dia_id), None)
                if pos is not None:
                    ingestion_remaining = [
                        {"dia_id": t.dia_id, "session_key": t.session_key,
                         "session_date": t.session_date, "speaker": t.speaker,
                         "text": t.text}
                        for t in turns[pos + 1:]]

        return LangGraphSession(
            runtime=self, sample=sample, records=records, store=store,
            namespace=["locomo", sample_id, "memories"],
            span=span, trace=trace, baseline_client=baseline_client,
            q_idx=q_idx, num_questions=num_q or 1,
            qa_template=qa_template, phase=phase, qa_index=qa_index,
            ingestion_remaining=ingestion_remaining,
            question_texts=question_texts, answers=answers, history=history,
            question_messages=question_messages, question_done=True,
            injected=None,
            snapshot_hash="memory:" + final_hash,
            restored_env_hash_raw=final_hash,
            prev_ts_end=monotonic_ms(),
            downstream_trace={
                "meta": {"task_id": trace.run_meta.task_id,
                         "run_id": trace.run_meta.run_id,
                         "baseline_model": trace.run_meta.baseline_model,
                         "continuation_model": (
                             baseline_client.model if baseline_client else ""),
                         "system_prompt_source": "agent_source",
                         "system_prompt_drift": prompt_drift},
                "spans": [], "terminal": {},
            },
        )

    # -- 14.4 step 4 --------------------------------------------------------
    def resolve_request(self, trace: TraceRun, span: SpanRecord,
                        session: Optional[RestoredSession] = None
                        ) -> ResolvedRequest:
        msgs = [dict(m) for m in trace.artifacts["messages"][
            "msg:%d" % span.invocation_index]]
        # the recorded messages' tool_calls use the collection side's jsonable shape ({name, args, id, type: "tool_call"})
        # -- requests to an OpenAI-compatible endpoint must be in the function-wrapped format, otherwise vLLM validation
        # 400s
        for m in msgs:
            tcs = m.get("tool_calls")
            if not tcs:
                continue
            native = []
            for tc in tcs:
                args = tc.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                native.append({
                    "id": str(tc.get("id", "") or ""),
                    "type": "function",
                    "function": {
                        "name": str(tc.get("name", "") or ""),
                        "arguments": json.dumps(args, ensure_ascii=False),
                    }})
            m["tool_calls"] = native
        tools = trace.artifacts["recorded_tools"].get(
            "tools:%d" % span.invocation_index) or None
        return ResolvedRequest(
            messages=msgs, tools=tools, response_format=None,
            temperature=0.0, max_output_tokens=8192,
            meta_hint=span.meta_hint, tool_choice=None,
        )

    # -- 14.4 step 5 --------------------------------------------------------
    def parse(self, span: SpanRecord, raw: str) -> ParseResult:
        try:
            parsed = json.loads(raw)
        except ValueError:
            return ParseResult(ok=True, parsed={"text": raw}, error="")
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            return ParseResult(ok=True, parsed=parsed, error="")
        if isinstance(parsed, dict) and parsed.get("text") is not None:
            return ParseResult(ok=True, parsed=parsed, error="")
        return ParseResult(ok=True, parsed={"text": raw}, error="")

    def schema_check(self, session: LangGraphSession, span: SpanRecord,
                     parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        if parsed.get("tool_calls"):
            if not isinstance(parsed["tool_calls"], list) or not parsed["tool_calls"]:
                return False
            for c in parsed["tool_calls"]:
                name = str(c.get("name", ""))
                if name not in LANGGRAPH_TOOL_NAMES:
                    return False
                if not isinstance(c.get("arguments"), dict):
                    return False
                args = c["arguments"]
                if name == "manage_memory" and "action" not in args:
                    return False
                if name == "search_memory" and "query" not in args:
                    return False
            return True
        text = parsed.get("text")
        return isinstance(text, str) and bool(text.strip())

    def verify_local(self, session: LangGraphSession, span: SpanRecord,
                     parsed: Any) -> VerifierVerdict:
        if span.output_contract_runtime.contract_type == "tool_call_list":
            ok = self.schema_check(session, span, parsed)
            return VerifierVerdict(
                triggered=True, detected_failure=not ok, passed=ok,
                executed=["contract_check", "schema_check"],
                errors=[] if ok else ["memory tool contract violated"])
        text = str((parsed or {}).get("text", "") or "")
        if not text.strip():
            return VerifierVerdict(
                triggered=True, detected_failure=True, passed=False,
                executed=["locomo_qa_f1"], errors=["empty answer text"])
        # ingestion ACK: no gold, only a structural gate
        sc = span.output_contract_runtime.semantic_constraints or {}
        if sc.get("locomo_phase") == "ingestion":
            return VerifierVerdict(
                triggered=True, detected_failure=False, passed=True,
                executed=["contract_check"], errors=[])
        f1 = None
        try:
            ev = evaluate_answers(
                session.sample,
                [{"idx": session.q_idx,
                  "question": (session.question_texts[session.q_idx]
                               if session.q_idx < len(session.question_texts)
                               else ""),
                  "pred": text}],
                set())
            pq = (ev.get("per_question") or [{}])[0]
            f1 = pq.get("f1")
        except Exception as exc:
            return VerifierVerdict(
                triggered=True, detected_failure=False, passed=True,
                executed=["locomo_qa_f1"],
                errors=["f1 unavailable: %s" % str(exc)[:120]])
        passed = (f1 is None) or (f1 >= QA_F1_COMMIT_GATE)
        return VerifierVerdict(
            triggered=True, detected_failure=not passed, passed=bool(passed),
            executed=["locomo_qa_f1=%.4f" % (f1 if f1 is not None else -1.0)],
            errors=[] if passed else ["answer f1 below commit gate"])

    # -- 14.4 step 6 --------------------------------------------------------
    def _record_answer(self, session: LangGraphSession, pred: str):
        session.answers.append({
            "idx": session.q_idx,
            "question": (session.question_texts[session.q_idx]
                         if session.q_idx < len(session.question_texts) else ""),
            "pred": pred})
        session.history.append({
            "question_index": session.q_idx,
            "question": (session.question_texts[session.q_idx]
                         if session.q_idx < len(session.question_texts) else ""),
            "predicted_answer": pred})
        session.q_idx += 1
        session.question_done = True
        session.question_messages = []

    def inject(self, session: LangGraphSession, span: SpanRecord,
               parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            session.injected = parsed
            return False
        if parsed.get("__span_failure__") is not None:
            self._record_answer(
                session, str(parsed["__span_failure__"])[:TRACE_RESULT_MAX_LEN])
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"text": str(parsed["__span_failure__"])
                           [:TRACE_RESULT_MAX_LEN]}})
            session.injected = parsed
            return True

        if parsed.get("tool_calls"):
            from langchain_core.messages import AIMessage, ToolMessage
            self._ensure_tools(session)
            calls = parsed["tool_calls"]
            # langchain 0.3 validation: tool_calls[].args must be a dict
            synth = [{"id": "call_inject_%d" % i, "name": c["name"],
                      "args": (c.get("arguments") or {})
                      if isinstance(c.get("arguments"), dict)
                      else {}}
                     for i, c in enumerate(calls)]
            session.question_messages.append(AIMessage(
                content="", tool_calls=synth))
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"tool_calls": [
                    {"name": c.get("name", ""),
                     "arguments": c.get("arguments")} for c in calls]}})
            # frozen path: if the target span's calls were never executed in the records (recursion cutoff), the walk-through mirrors the recorded behavior -- skip
            # execution
            frozen_recorded = []
            if getattr(session, "frozen_downstream", False):
                frozen_recorded = session.trace.artifacts[
                    "span_tool_results"].get(
                    "span:%d" % span.invocation_index, [])
            if getattr(session, "frozen_downstream", False) and \
                    not frozen_recorded:
                session.c0_fidelity.append({
                    "invocation_index": span.invocation_index,
                    "node_pattern": span.span_descriptor.node_pattern,
                    "tool_match": None,
                    "note": "target span has tool_calls but no recorded "
                            "executions (recursion cutoff); skipped"})
                session.question_done = True
                session.injected = parsed
                return True
            # really execute the langmem tools (store-local operations)
            executed = []
            for i, tc in enumerate(calls):
                name = str(tc.get("name", ""))
                args = tc.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}
                start_ms = monotonic_ms()
                try:
                    if getattr(session, "frozen_downstream", False):
                        for r in frozen_recorded:
                            if r.get("tool") == name:
                                args = r.get("arguments") or args
                                break
                    result = self._invoke_tool(session, name, args)
                    ok = True
                except Exception as exc:
                    result = json.dumps(
                        {"error": "%s: %s" % (
                            type(exc).__name__, str(exc)[:200])},
                        ensure_ascii=False)
                    ok = False
                elapsed = monotonic_ms() - start_ms
                result_value = (result.content if hasattr(result, "content")
                                else result)
                session.question_messages.append(ToolMessage(
                    content=result_value,
                    tool_call_id=synth[i]["id"]))
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": 0, "source": "slm",
                    "tool": name, "arguments": args, "ok": bool(ok),
                    "result": ("" if result_value is None
                               else (result_value
                                     if isinstance(result_value, str)
                                     else json.dumps(result_value,
                                                     ensure_ascii=False,
                                                     default=str))
                               )[:TRACE_RESULT_MAX_LEN],
                }, elapsed_ms=elapsed)
                executed.append({"tool": name, "ok": bool(ok),
                                 "result": ("" if result_value is None
                                            else (result_value
                                                  if isinstance(result_value, str)
                                                  else json.dumps(result_value,
                                                                  ensure_ascii=False,
                                                                  default=str))
                                            )[:TRACE_RESULT_MAX_LEN]})
            if getattr(session, "frozen_downstream", False):
                session.c0_fidelity.append(self._compare_frozen(
                    span, executed, frozen_recorded))
            session.question_done = False
            session.injected = session.question_messages[-1]
            return True

        # free text: QA phase = the injected text is the answer; ingestion
        # phase = ACK, appended to the current turn's messages, no answer
        # bookkeeping
        if session.phase == "ingestion":
            from langchain_core.messages import AIMessage
            session.question_messages.append(
                AIMessage(content=str(parsed.get("text", ""))))
            session.question_done = True
        else:
            self._record_answer(session, str(parsed.get("text", "")))
        self._trace_step(session, {
            "span_type": "llm.call", "step": 0, "source": "slm",
            "output": {"text": str(parsed.get("text", ""))
                       [:TRACE_RESULT_MAX_LEN]}})
        session.injected = parsed
        return True

    # -- real-graph continuation -------------------------------------------
    def _invoke_current(self, session: LangGraphSession, deadline: float):
        """The current question's graph call (verbatim like the collection side: recursion_limit=12 + handler)."""
        agent = self._make_agent(session)
        self.budget_guard(1, deadline)
        return agent.invoke(
            {"messages": list(session.question_messages)},
            config={"recursion_limit": RECURSION_LIMIT,
                    "callbacks": [session.handler]},
        )

    # -- 14.4 steps 7-8 -----------------------------------------------------
    def run_to_node(self, session: LangGraphSession, span: SpanRecord,
                    client: Any, max_steps: int, timeout_s: float,
                    budget_usd: float) -> Tuple[DownstreamOutcome,
                                                List[LMResponse]]:
        outcome = DownstreamOutcome()
        deadline = monotonic_ms() + timeout_s * 1000
        try:
            if getattr(session, "frozen_downstream", False):
                completed = self._continue_frozen(
                    session, stop_after_span=True)
                outcome.next_node_reached = True
                outcome.next_tool_executed = True
                outcome.downstream_execution_valid = True
                outcome.downstream_checkpoint_reached = completed
                return outcome, []
            if session.question_done:
                outcome.next_node_reached = True
                outcome.next_tool_executed = False
                outcome.downstream_execution_valid = True
                outcome.downstream_checkpoint_reached = True
                return outcome, []
            result = self._invoke_current(session, deadline)
            final_message = result["messages"][-1]
            predicted = str(final_message.content).strip()
            if session.phase == "ingestion":
                # the current ingestion turn is complete (ACK); no answer bookkeeping
                session.question_done = True
            else:
                self._record_answer(session, predicted)
            outcome.next_node_reached = True
            outcome.next_tool_executed = True
            outcome.downstream_execution_valid = True
            outcome.downstream_checkpoint_reached = True
        except StopReplay as stop:
            session.downstream_trace["terminal"] = {
                "status": "stopped: %s" % stop.flag}
            outcome.downstream_checkpoint_reached = False
        except Exception as exc:
            outcome.unhandled_exception = True
            outcome.downstream_execution_valid = False
            session.downstream_trace["terminal"] = {
                "status": "exception: %s" % str(exc)[:200]}
        return outcome, []

    def run_to_completion(self, session: LangGraphSession, span: SpanRecord,
                          client: Any, max_steps: int, timeout_s: float,
                          budget_usd: float) -> Tuple[FinalOutcome,
                                                      List[LMResponse]]:
        final = FinalOutcome()
        deadline = monotonic_ms() + timeout_s * 1000
        try:
            if getattr(session, "frozen_downstream", False):
                self._continue_frozen(session, stop_after_span=False)
                if not session.answers:
                    # the unit has no answer span (recursion cutoff): backfill the recorded pred; evaluating with n=1 is the same basis as the collection side (its pred is the empty
                    # string)
                    lg = session.trace.artifacts.get("langgraph") or {}
                    rec_ans = (lg.get("answers") or [{}])[0]
                    self._record_answer(session, str(
                        rec_ans.get("predicted_answer", "") or ""))
            else:
                if not session.question_done:
                    self.budget_guard(1, deadline)
                    result = self._invoke_current(session, deadline)
                    if session.phase == "ingestion":
                        session.question_done = True
                    else:
                        predicted = str(result["messages"][-1].content).strip()
                        self._record_answer(session, predicted)
                if session.phase == "ingestion":
                    # ingest the remaining turns (the collection side invokes agent per turn), then this QA
                    for turn in session.ingestion_remaining:
                        self.budget_guard(
                            len(session.ingestion_remaining), deadline)
                        self._invoke_ingestion_turn(session, turn, deadline)
                    session.ingestion_remaining = []
                while session.q_idx < session.num_questions:
                    self.budget_guard(session.num_questions - session.q_idx,
                                      deadline)
                    self._invoke_next_question(session, deadline)
            # single-question determination: when the answer is verbatim the same as the collection side's record, reuse the recorded determination directly (a full
            # replay = same answer same determination, avoiding second-judge basis drift); only new answers (paid arms) run the verifier (same basis as
            # validate_locomot).
            from .langgraph_env import verify_qa_answer
            lg = session.trace.artifacts.get("langgraph") or {}
            col_row = lg.get("collector_row") or {}
            row = (session.answers or [{}])[0]
            qtext = str(row.get("question", "") or "")
            pred_text = str(row.get("pred", "") or "")
            # answer verbatim the same as the collection side's record -> reuse the collection side's determination directly. Basis:
            # per_question.pred = stripped then truncated (<=800 chars, truncation happens after the strip); compare aligned to the determination
            # row's length
            col_pred = str(col_row.get("pred", "") or "").strip()
            if (col_row.get("pass") and col_pred
                    and col_pred == pred_text.strip()[:len(col_pred)]):
                ev_row = {"pass": str(col_row.get("pass", "")),
                          "score": col_row.get("score", 0.0),
                          "source": "collector",
                          "f1": col_row.get("f1", 0.0),
                          "num_f1": col_row.get("num_f1", 0.0),
                          "contains": col_row.get("contains"),
                          "rouge_l": col_row.get("rouge_l", 0.0),
                          "verdict": col_row.get("verdict", ""),
                          "reason": str(col_row.get("reason", ""))[:200]}
            else:
                gold = None
                category = 0
                for q in getattr(session.sample, "qas", []) or []:
                    if q.question == qtext:
                        gold = q.answer
                        category = q.category
                        break
                if gold is None:
                    ev_row = {"pass": "fail", "score": 0.0, "source": "rule",
                              "f1": 0.0, "num_f1": 0.0, "contains": None,
                              "rouge_l": 0.0, "verdict": "incorrect",
                              "reason": "gold missing"}
                else:
                    ev_row = verify_qa_answer(
                        qtext, gold, category, pred_text,
                        use_llm=True)
            passed = ev_row["pass"] == "success"
            final.goal_state_passed = passed
            final.final_task_success = passed
            final.collateral_damage = False
            final.evaluator_details = [
                "pass=%s score=%.3f source=%s f1=%.4f num_f1=%.4f "
                "contains=%s rouge_l=%.4f" % (
                    ev_row["pass"], ev_row["score"], ev_row["source"],
                    ev_row.get("f1", 0.0), ev_row.get("num_f1", 0.0),
                    ev_row.get("contains"), ev_row.get("rouge_l", 0.0)),
                "verdict=%s reason=%s" % (
                    ev_row.get("verdict", ""),
                    str(ev_row.get("reason", ""))[:200]),
            ]
            session.downstream_trace["terminal"] = {
                "status": "completed", "pass": ev_row["pass"],
                "score": ev_row["score"], "source": ev_row["source"],
                "backend_info": ["backend=langgraph agent=%s" % AGENT_ID]}
        except StopReplay as stop:
            final.timeout = stop.flag == "timeout"
            final.budget_exhausted = stop.flag == "budget_exhausted"
            session.downstream_trace["terminal"] = {
                "status": "stopped: %s" % stop.flag}
            return final, list(session.usage)
        except Exception as exc:
            final.evaluator_details.append(
                "evaluation error: %s: %s"
                % (type(exc).__name__, str(exc)[:200]))
            session.downstream_trace["terminal"] = {
                "status": "exception: %s" % str(exc)[:200]}
        return final, list(session.usage)

    def _invoke_next_question(self, session: LangGraphSession,
                              deadline: float):
        """Remaining questions: prior history + the new question (verbatim like 10_run_locomo_sample)."""
        from langchain_core.messages import HumanMessage
        self.budget_guard(1, deadline)
        qi = session.q_idx
        qtext = (session.question_texts[qi]
                 if qi < len(session.question_texts) else "")
        messages = compact_history_messages(session.history)
        if session.qa_template == "core":
            prompt_text = question_prompt_core(qtext)
        else:
            prompt_text = question_prompt(qi, qtext)
        messages.append(HumanMessage(content=prompt_text))
        session.question_messages = messages
        agent = self._make_agent(session)
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": RECURSION_LIMIT,
                    "callbacks": [session.handler]},
        )
        predicted = str(result["messages"][-1].content).strip()
        self._record_answer(session, predicted)

    def _invoke_ingestion_turn(self, session: LangGraphSession,
                               turn: Dict[str, Any], deadline: float):
        """Ingest one turn (same basis as the collection side's per-turn invoke; ACK with no answer bookkeeping)."""
        from langchain_core.messages import HumanMessage
        from .langgraph_env import ingestion_turn_prompt
        self.budget_guard(1, deadline)
        text = ingestion_turn_prompt(
            turn.get("session_date", ""), turn.get("dia_id", ""),
            turn.get("speaker", ""), turn.get("text", ""))
        session.question_messages = [HumanMessage(content=text)]
        agent = self._make_agent(session)
        agent.invoke(
            {"messages": session.question_messages},
            config={"recursion_limit": RECURSION_LIMIT,
                    "callbacks": [session.handler]},
        )

    # -- C0 frozen walk-through ---------------------------------------------
    def _continue_frozen(self, session: LangGraphSession,
                         stop_after_span: bool) -> bool:
        span = session.span
        trace = session.trace
        assert span is not None and trace is not None
        if session.frozen_plan is None:
            session.frozen_plan = sorted(
                (sp for sp in trace.spans
                 if sp.invocation_index > span.invocation_index),
                key=lambda sp: sp.invocation_index)
            session.frozen_index = 0
        if session.frozen_completed:
            return True
        if stop_after_span:
            return False
        self._ensure_tools(session)
        while session.frozen_index < len(session.frozen_plan):
            rec = session.frozen_plan[session.frozen_index]
            session.frozen_index += 1
            recorded_results = trace.artifacts["span_tool_results"].get(
                "span:%d" % rec.invocation_index, [])
            parsed = rec.baseline_response.parsed or {}
            if (rec.output_contract_runtime.contract_type == "tool_call_list"
                    and parsed.get("tool_calls")):
                if not recorded_results:
                    # the records show this span's calls were never executed (recursion_limit cutoff): the walk-through mirrors the recorded behavior and does not execute unexecuted
                    # calls
                    session.c0_fidelity.append({
                        "invocation_index": rec.invocation_index,
                        "node_pattern": rec.span_descriptor.node_pattern,
                        "tool_match": None,
                        "note": "recorded span has tool_calls but no "
                                "executions (recursion cutoff); skipped"})
                    continue
                calls = parsed["tool_calls"]
                executed: List[Dict[str, Any]] = []
                for tc in calls:
                    name = str(tc.get("name", ""))
                    args = tc.get("arguments") or {}
                    if not isinstance(args, dict):
                        args = {}
                    for r in recorded_results:
                        if r.get("tool") == name:
                            args = r.get("arguments") or args
                            break
                    start_ms = monotonic_ms()
                    try:
                        result = self._invoke_tool(session, name, args)
                        ok = True
                    except Exception as exc:
                        result = json.dumps(
                            {"error": "%s: %s" % (
                                type(exc).__name__, str(exc)[:200])},
                            ensure_ascii=False)
                        ok = False
                    elapsed = monotonic_ms() - start_ms
                    result_value = (result.content
                                    if hasattr(result, "content") else result)
                    executed.append({
                        "tool": name, "ok": bool(ok),
                        "result": ("" if result_value is None
                                   else (result_value
                                         if isinstance(result_value, str)
                                         else json.dumps(result_value,
                                                         ensure_ascii=False,
                                                         default=str))
                                   )[:TRACE_RESULT_MAX_LEN]})
                    self._trace_step(session, {
                        "span_type": "tool.execution",
                        "step": rec.invocation_index, "source": "frozen",
                        "tool": name, "arguments": args, "ok": bool(ok),
                        "result": executed[-1]["result"],
                    }, elapsed_ms=elapsed)
                session.c0_fidelity.append(self._compare_frozen(
                    rec, executed, recorded_results))
            else:
                text = str(parsed.get("text", "")) if isinstance(
                    parsed, dict) else ""
                sc = rec.output_contract_runtime.semantic_constraints or {}
                is_ingestion = sc.get("locomo_phase") == "ingestion"
                if is_ingestion:
                    # ingestion ACK: no answer semantics, the walk-through continues
                    session.c0_fidelity.append({
                        "invocation_index": rec.invocation_index,
                        "node_pattern": rec.span_descriptor.node_pattern,
                        "tool_match": None,
                        "note": "recorded ingestion ACK; no bookkeeping"})
                else:
                    session.c0_fidelity.append({
                        "invocation_index": rec.invocation_index,
                        "node_pattern": rec.span_descriptor.node_pattern,
                        "tool_match": None,
                        "note": "recorded answer span; pred bookkept"})
                self._trace_step(session, {
                    "span_type": "llm.call", "step": rec.invocation_index,
                    "source": "frozen",
                    "output": {"text": text[:TRACE_RESULT_MAX_LEN]}})
                if not is_ingestion:
                    self._record_answer(session, text)
        session.frozen_completed = True
        return session.q_idx >= session.num_questions

    def _compare_frozen(self, rec: SpanRecord,
                        executed: List[Dict[str, Any]],
                        recorded: List[Dict[str, Any]]) -> Dict[str, Any]:
        from ..local_equivalence import compare_values, _normalize_tool_name
        exec_names = [_normalize_tool_name(e.get("tool")) for e in executed]
        rec_names = [_normalize_tool_name(r.get("tool")) for r in recorded]
        names_match = sorted(exec_names) == sorted(rec_names)
        ok_match = all(e.get("ok") for e in executed) == \
            all(r.get("ok") for r in recorded)
        value_details = []
        for i, e in enumerate(executed):
            if i < len(recorded):
                value_details.append(
                    compare_values(e.get("result"),
                                   recorded[i].get("result")))
        value_match = all(v for v in value_details) if value_details else None
        return {
            "invocation_index": rec.invocation_index,
            "node_pattern": rec.span_descriptor.node_pattern,
            "tool_match": bool(names_match and ok_match),
            "value_match": bool(value_match) if value_match is not None
            else None,
            "note": "" if names_match else
            "exec=%s vs recorded=%s" % (exec_names, rec_names),
        }

    def rollback(self, session: LangGraphSession) -> bool:
        # pre-commit verifier only: no state committed before injection (the store is local and rebuildable)
        return True
