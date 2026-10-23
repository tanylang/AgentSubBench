"""LocoMo live replay runtime (simplified continuation backend + C0 frozen walk-through).

Hard rule (same as the AppWorld/SWE runtimes): the downstream continuation must be driven by the REAL simplified LanguageModel source
(appworld code.simplified, the same code as the collection side); report backend unavailability honestly, never degrade to an approximation
loop.

Key design: - restore = corpus load + span_log hash-chain rebuild (only replay tool
records with step < target; mark_read is a pure function of the retrieved results, so the
rebuild is bit-exact) + final hash checked against the span's recorded value
(restored_env_hash_raw feeds the engine's raw-to-raw gate) - injection semantics:
tool_call_list -> splice an assistant message (tool_calls, synthesized ids) + immediately
really execute each call and backfill role=tool messages, the current question continues;
freeform_text -> the injected text IS the question's answer, bookkeep it and advance q_idx
(that question ends) - verify_local: tool span = schema shape check; answer span = the
official F1 (commit gate = non-empty text; F1 recorded into executed/details -- the 0.5
threshold belongs to the final evaluator layer, avoiding identity/C0 arm distortion on
fail traces) - final evaluation = the official eval_question_answering (avg F1 >= 0.5) +
retrieval recall - C0 frozen: downstream injects the trace-recorded output, only really
executes tools (read-only), zero LM calls throughout -- it answers replay fidelity, not
model behavior
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..appworld_adapter import _native_tool_calls, _synth_call_id
from ..llm_client import LMResponse
from ..runtime import (
    AgentRuntime, ParseResult, ResolvedRequest, RestoredSession, StopReplay,
)
from ..schemas import (
    DownstreamOutcome, FinalOutcome, SpanRecord, TraceRun,
)
from ..verifiers import VerifierVerdict
from .locomo_env import (
    LOC_TOOL_NAMES, OAI_TOOL_SCHEMAS, PASS_THRESHOLD, QUESTION_RE,
    build_question_prompt, build_system_prompt, evaluate_answers,
    execute_tool, load_sample, new_memory, replay_span_log, run_question_loop,
)

REPO_ROOT = Path(__file__).parent.parent.parent
#: appworld experiments/code (where the collection side's simplified LanguageModel lives)
_EXP_CODE = str(REPO_ROOT / "benchmarks" / "appworld" / "experiments" / "code")
#: the answer span's F1 commit gate (0.0 = structural gate: non-empty text commits; strict basis can set 0.5)
QA_F1_COMMIT_GATE = 0.0
TRACE_RESULT_MAX_LEN = 2000

#: reasoning_content placeholder for the injected assistant message (the
#: DeepSeek thinking endpoint requires non-empty; harmless on non-thinking endpoints -- only the baseline continuation path uses it)
RC_PLACEHOLDER = "Determining the next retrieval call."

#: whether the LanguageModel construction and raw-caller hardening are
#: already installed (same basis as the collection driver's _install_hooks; idempotent, installed once per process)
_LM_HOOKED = False

# : system prompt builder dispatched per agent identity (agent prompt build :
# logic differs per framework -- hard rule: v1 implements only the : simplified
# backend, other identities error out honestly). : extension guide (verified on
# the collection side): : - LocoMo-Legacy / LocoMo-OpenAI-Agents / LocoMo-AutoGen
# / : LocoMo-Smolagents: text same as build_system_prompt, injection : mechanisms
# differ (base_messages / instructions= / system_message= : / template override) :
# - LocoMo-Tau2: text = build_system_prompt + append : "Answer the current
# question using the tools if needed. Stop when : you have the answer."
# (domain_policy), and the loop is a : get_init_state + generate_next_message
# state machine
AGENT_PROMPT_BUILDERS = {
    "LocoMo-Simplified": build_system_prompt,
}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
@dataclass
class LocomoSession(RestoredSession):
    sample: Any = None
    memory: Any = None
    span: Optional[SpanRecord] = None
    trace: Optional[TraceRun] = None
    baseline_client: Any = None
    lm: Any = None                     # the real simplified LanguageModel (lazy-built)
    base_messages: List[Dict[str, Any]] = field(default_factory=list)
    q_idx: int = 0
    num_questions: int = 0
    max_tool_steps: int = 10
    question_text: str = ""            # the current question's raw text (for gold alignment)
    question_texts: List[str] = field(default_factory=list)  # all question texts
    messages: List[Dict[str, Any]] = field(default_factory=list)
    answers: List[Dict[str, Any]] = field(default_factory=list)
    question_done: bool = True
    injected: Any = None
    usage: List[LMResponse] = field(default_factory=list)
    downstream_trace: Dict[str, Any] = field(default_factory=dict)
    restored_env_hash_raw: str = ""
    # C0 frozen state (the engine sets frozen_downstream=True for arm=="C0")
    frozen_downstream: bool = False
    frozen_plan: Optional[List[SpanRecord]] = None
    frozen_index: int = 0
    frozen_completed: bool = False
    c0_fidelity: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class LocomoRuntime(AgentRuntime):
    name = "locomo-live"

    def __init__(self, manifest=None):
        self.manifest = manifest

    # -- backend hard check + LanguageModel construction ------------------
    def _make_lm(self, session: LocomoSession):
        """The real simplified LanguageModel (lazy-built; the appworld-root
        guard must run before construction -- language_model imports appworld at module level and instantiates a joblib cache)."""
        if session.lm is not None:
            return session.lm
        if not os.path.isdir(_EXP_CODE):
            raise RuntimeError(
                "backend unavailable: appworld simplified source not found at "
                "%s; the downstream continuation must run the REAL agent "
                "source" % _EXP_CODE)
        from .appworld_env import _ensure_appworld_root
        _ensure_appworld_root()
        for p in (_EXP_CODE,):
            if p not in sys.path:
                sys.path.insert(0, p)
        from appworld_agents.code.simplified import language_model as _lm_mod
        from appworld_agents.code.simplified.language_model import (
            LanguageModel as _LM,
        )
        # raw-caller timeout hardening (same basis as the collection driver's _install_hooks; without the trace stash -- that is collection-only): timeout=600,
        # max_retries=3
        global _LM_HOOKED
        if not _LM_HOOKED:
            _orig_grlc = _lm_mod.get_raw_lm_caller

            def _patched_grlc(api_type, client_name, api_key=None,
                              base_url=None):
                f = _orig_grlc(api_type, client_name, api_key, base_url)
                _c = getattr(f, "__self__", None)
                if _c is not None and hasattr(_c, "timeout"):
                    _c.timeout = 600.0
                    _c.max_retries = 3
                return f

            _lm_mod.get_raw_lm_caller = _patched_grlc
            _LM_HOOKED = True
        bc = session.baseline_client
        os.environ.setdefault("MODEL_SERVER_URL",
                              bc.base_url or "http://localhost:8000")
        os.environ.setdefault("OPENAI_API_KEY", bc.api_key or "placeholder")
        # constructor args match the collection side's launch item by item (max_retries=100 is the constructor's basis)
        session.lm = _LM(
            name=bc.model,
            client_name=os.environ.get("SPAN_OFFLOAD_CONT_LM_CLIENT", "openai"),
            api_type="chat_completions",
            base_url=bc.base_url,
            api_key=bc.api_key or "placeholder",
            temperature=bc.temperature,
            max_tokens=bc.max_output_tokens,
            seed=100,
            tool_choice="auto",
            extra_body=getattr(bc, "extra_body", None) or {},
            use_cache=False,
            max_retries=100,
            retry_after_n_seconds=15,
            cost_per_token={"input_cache_miss": 0, "input_cache_hit": 0,
                            "input_cache_write": 0, "output": 0},
        )
        return session.lm

    # -- 14.4 step 3 --------------------------------------------------------
    def restore(self, trace: TraceRun, span: SpanRecord,
                baseline_client: Any = None) -> LocomoSession:
        # hard identity gate: the continuation must be driven by this agent's real source; v1 supports only the simplified backend, other identities error out honestly (never degrade to an approximation
        # loop)
        agent_name = trace.run_meta.agent_id
        prompt_builder = AGENT_PROMPT_BUILDERS.get(agent_name)
        if prompt_builder is None:
            raise RuntimeError(
                "backend unavailable: agent %r not supported by locomo "
                "runtime v1 (simplified only); the downstream continuation "
                "must run the REAL per-agent source" % agent_name)
        sample = load_sample(trace.run_meta.task_id)
        span_log = trace.artifacts["span_log"]
        snap = trace.artifacts["state_snapshots"][
            span.state_snapshot.environment_snapshot_ref]
        step = int(snap.get("step", 0) or 0)
        memory, final_hash, notes = replay_span_log(sample, span_log, step)
        recorded_hash = str(span.state_snapshot.application_state_hash or "")
        if recorded_hash and final_hash != recorded_hash:
            raise RuntimeError(
                "locomo restore hash mismatch for span %d: recorded %s != "
                "rebuilt %s (%s)" % (
                    span.invocation_index, recorded_hash[:12],
                    final_hash[:12], "; ".join(notes)))

        agent_snap = trace.artifacts["state_snapshots"].get(
            "agent:%d" % span.invocation_index) or {}
        q_idx = int(agent_snap.get("q_idx", 0) or 0)
        num_q = int(agent_snap.get("num_questions", 0) or 0)
        loco = trace.artifacts.get("locomo") or {}
        recorded_answers = loco.get("answers") or []
        max_tool_steps = int(loco.get("max_tool_steps", 0) or 10)

        # already-answered questions keep the trace-recorded answers (never re-sample questions on replay); the current question's text is parsed from msg:j's first user message (for gold
        # alignment)
        question_texts = [str(a.get("question", ""))
                          for a in recorded_answers]
        answers = [
            {"idx": int(a.get("idx", i)), "question": a.get("question", ""),
             "pred": a.get("pred", "")}
            for i, a in enumerate(recorded_answers[:q_idx])]
        msgs = [dict(m) for m in trace.artifacts["messages"][
            "msg:%d" % span.invocation_index]]
        question_text = ""
        for m in msgs:
            if m.get("role") != "user":
                continue
            mm = QUESTION_RE.search(str(m.get("content", "") or ""))
            if mm:
                question_text = str(m.get("content", "") or "")[
                    mm.end():].strip()
            break
        if not question_text and 0 <= q_idx < len(question_texts):
            question_text = question_texts[q_idx]
        # system prompt: go through this agent's real build path (the only source of the agent's internal prompt); the trace-recorded sys:j is only a cross-check snapshot -- drift is recorded
        # honestly
        recorded_prompt = trace.artifacts["system_prompts"].get(
            "sys:%d" % span.invocation_index) or ""
        system_prompt = prompt_builder(sample)
        prompt_drift = ""
        if recorded_prompt and system_prompt != recorded_prompt:
            prompt_drift = ("system_prompt_drift: rebuilt != recorded "
                            "(template evolved since collection?)")

        return LocomoSession(
            runtime=self, sample=sample, memory=memory,
            span=span, trace=trace, baseline_client=baseline_client,
            base_messages=[{"role": "system", "content": system_prompt}],
            q_idx=q_idx, num_questions=num_q or len(recorded_answers),
            max_tool_steps=max_tool_steps,
            question_text=question_text, question_texts=question_texts,
            messages=[], answers=answers,
            question_done=True, injected=None,
            snapshot_hash="readset:" + final_hash,
            restored_env_hash_raw=final_hash,
            downstream_trace={
                "meta": {"task_id": trace.run_meta.task_id,
                         "run_id": trace.run_meta.run_id,
                         "baseline_model": trace.run_meta.baseline_model,
                         "continuation_model": (
                             baseline_client.model if baseline_client else ""),
                         "system_prompt_source": "rebuilt_from_agent_template",
                         "system_prompt_drift": bool(prompt_drift),
                         "system_prompt_drift_note": prompt_drift},
                "spans": [], "terminal": {},
            },
        )

    # -- 14.4 step 4 --------------------------------------------------------
    def resolve_request(self, trace: TraceRun, span: SpanRecord,
                        session: Optional[RestoredSession] = None
                        ) -> ResolvedRequest:
        msgs = [dict(m) for m in trace.artifacts["messages"][
            "msg:%d" % span.invocation_index]]
        tools = trace.artifacts["recorded_tools"].get(
            "tools:%d" % span.invocation_index) or list(OAI_TOOL_SCHEMAS)
        return ResolvedRequest(
            messages=msgs, tools=tools, response_format=None,
            temperature=0.0, max_output_tokens=8192,
            meta_hint=span.meta_hint, tool_choice=None,
        )

    # -- 14.4 step 5 --------------------------------------------------------
    def parse(self, span: SpanRecord, raw: str) -> ParseResult:
        """Text-fallback parsing: a JSON tool_call list -> the contract; anything else is treated as a free-text answer (free text without tool_calls is a legal
        output)."""
        try:
            parsed = json.loads(raw)
        except ValueError:
            return ParseResult(ok=True, parsed={"text": raw}, error="")
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            return ParseResult(ok=True, parsed=parsed, error="")
        if isinstance(parsed, dict) and parsed.get("text") is not None:
            return ParseResult(ok=True, parsed=parsed, error="")
        return ParseResult(ok=True, parsed={"text": raw}, error="")

    def schema_check(self, session: LocomoSession, span: SpanRecord,
                     parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        if parsed.get("tool_calls"):
            if not isinstance(parsed["tool_calls"], list) or not parsed["tool_calls"]:
                return False
            for c in parsed["tool_calls"]:
                name = str(c.get("name", ""))
                if name not in LOC_TOOL_NAMES:
                    return False
                if not isinstance(c.get("arguments"), dict):
                    return False
            return True
        text = parsed.get("text")
        return isinstance(text, str) and bool(text.strip())

    def verify_local(self, session: LocomoSession, span: SpanRecord,
                     parsed: Any) -> VerifierVerdict:
        """Local verification: tool span = shape check; answer span = the official F1 (commit gate = non-empty text, F1 recorded into
        executed/details)."""
        if span.output_contract_runtime.contract_type == "tool_call_list":
            ok = self.schema_check(session, span, parsed)
            return VerifierVerdict(
                triggered=True, detected_failure=not ok, passed=ok,
                executed=["contract_check", "schema_check"],
                errors=[] if ok else ["retrieval tool contract violated"])
        # answer span: the official F1 criterion
        text = str((parsed or {}).get("text", "") or "")
        if not text.strip():
            return VerifierVerdict(
                triggered=True, detected_failure=True, passed=False,
                executed=["locomo_qa_f1"],
                errors=["empty answer text"])
        f1 = None
        try:
            ev = evaluate_answers(
                session.sample,
                [{"idx": session.q_idx, "question": session.question_text,
                  "pred": text}],
                session.memory.read_dia_ids)
            pq = (ev.get("per_question") or [{}])[0]
            f1 = pq.get("f1")
        except Exception as exc:
            return VerifierVerdict(
                triggered=True, detected_failure=False, passed=True,
                executed=["locomo_qa_f1"],
                errors=["f1 unavailable: %s" % str(exc)[:120]])
        passed = (f1 is None) or (f1 >= QA_F1_COMMIT_GATE)
        errs = [] if passed else ["answer f1 below commit gate"]
        return VerifierVerdict(
            triggered=True, detected_failure=not passed, passed=bool(passed),
            executed=["locomo_qa_f1=%.4f" % (f1 if f1 is not None else -1.0)],
            errors=errs)

    # -- 14.4 step 6 --------------------------------------------------------
    def _trace_step(self, session: LocomoSession, entry: Dict[str, Any]):
        if session.downstream_trace is not None:
            session.downstream_trace["spans"].append(entry)

    def _record_answer(self, session: LocomoSession, pred: str):
        session.answers.append({
            "idx": session.q_idx, "question": session.question_text,
            "pred": pred})
        session.q_idx += 1
        session.question_done = True
        session.messages = []
        # advance the current question's text (gold alignment; the recorded answer list = the collection side's sampled subset)
        if session.q_idx < len(session.question_texts):
            session.question_text = session.question_texts[session.q_idx]
        else:
            session.question_text = ""

    def inject(self, session: LocomoSession, span: SpanRecord,
               parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            session.injected = parsed
            return False
        if parsed.get("__span_failure__") is not None:
            # SLM output rejected (fallback=none): record the failure text as this question's answer and advance
            self._record_answer(
                session, str(parsed["__span_failure__"])[:TRACE_RESULT_MAX_LEN])
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"text": str(parsed["__span_failure__"])
                           [:TRACE_RESULT_MAX_LEN]}})
            session.injected = parsed
            return True

        if parsed.get("tool_calls"):
            calls = parsed["tool_calls"]
            synth = [{"id": _synth_call_id("inj", i), "name": c["name"],
                      "arguments": c.get("arguments") or {}}
                     for i, c in enumerate(calls)]
            native = _native_tool_calls(synth, id_prefix="inj")
            assistant = {"role": "assistant", "content": None,
                         "tool_calls": native,
                         "reasoning_content": RC_PLACEHOLDER}
            # session seed: msg:j's full preceding context (minus system) + the injected assistant -- the target span may sit anywhere within a question, so earlier turns must be
            # kept
            msg_j = [dict(m) for m in session.trace.artifacts["messages"][
                "msg:%d" % span.invocation_index]]
            session.messages = [
                m for m in msg_j if m.get("role") != "system"]
            session.messages.append(assistant)
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"tool_calls": [
                    {"name": c.get("name", ""),
                     "arguments": c.get("arguments")} for c in calls]}})
            executed = []
            for i, tc in enumerate(calls):
                name = str(tc.get("name", ""))
                args = tc.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}
                # C0 frozen: the target span replays with the recorded execution args (the framework layer may pad defaults)
                if getattr(session, "frozen_downstream", False):
                    recorded_results = session.trace.artifacts[
                        "span_tool_results"].get(
                        "span:%d" % span.invocation_index, [])
                    for r in recorded_results:
                        if r.get("tool") == name:
                            args = r.get("arguments") or args
                            break
                result = execute_tool(
                    name, args, session.sample, session.memory)
                ok = not result.startswith('{"error"')
                executed.append({"tool": name, "ok": bool(ok),
                                 "result": str(result)
                                 [:TRACE_RESULT_MAX_LEN]})
                session.messages.append({
                    "role": "tool", "tool_call_id": synth[i]["id"],
                    "name": name, "content": result})
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": 0, "source": "slm",
                    "tool": name, "arguments": args, "ok": bool(ok),
                    "result": str(result)[:TRACE_RESULT_MAX_LEN]})
            if getattr(session, "frozen_downstream", False):
                recorded_results = session.trace.artifacts[
                    "span_tool_results"].get(
                    "span:%d" % span.invocation_index, [])
                session.c0_fidelity.append(self._compare_frozen(
                    span, executed, recorded_results))
            session.question_done = False
            session.injected = assistant
            return True

        # free text: the injected text is the answer, this question ends
        self._record_answer(session, str(parsed.get("text", "")))
        self._trace_step(session, {
            "span_type": "llm.call", "step": 0, "source": "slm",
            "output": {"text": str(parsed.get("text", ""))
                       [:TRACE_RESULT_MAX_LEN]}})
        session.injected = parsed
        return True

    # -- baseline continuation loop ----------------------------------------
    class _ReplayCollector:
        """A recording shim for run_question_loop: write the real loop's
        tool executions into downstream_trace (structure same as the collection side's log_tool basis)."""

        def __init__(self, runtime: "LocomoRuntime",
                     session: "LocomoSession"):
            self.runtime = runtime
            self.session = session

        def log_tool(self, tool_name, arguments, result, duration_ms,
                     success):
            self.runtime._trace_step(self.session, {
                "span_type": "tool.execution", "step": self.session.q_idx,
                "source": "baseline", "tool": tool_name,
                "arguments": arguments or {}, "ok": bool(success),
                "result": str(result)[:TRACE_RESULT_MAX_LEN]})

    def _run_fresh_question(self, session: LocomoSession,
                            deadline: float) -> str:
        """Whole-question loop: go through the collection side's real locomot_common.run_question_loop (the authoritative implementation of
        the agent's internal logic); step_fn only supplies the real LanguageModel + usage/trajectory capture. Returns the question's
        answer."""
        lm = self._make_lm(session)
        shim = self._ReplayCollector(self, session)

        def step_fn(messages):
            self.budget_guard(session.max_tool_steps, deadline)
            resp = lm.generate(session.base_messages + messages,
                               tools=OAI_TOOL_SCHEMAS)
            content = str(resp.get("content") or "")
            tcs = list(resp.get("tool_calls") or [])
            toks = resp.get("standardized_usage")
            p = c = 0
            try:
                t = getattr(toks, "tokens", None)
                p = int(getattr(t, "input_cache_miss", 0) or 0) \
                    + int(getattr(t, "input_cache_hit", 0) or 0)
                c = int(getattr(t, "output", 0) or 0)
            except Exception:
                pass
            session.usage.append(LMResponse(
                raw_text=content, prompt_tokens=p, completion_tokens=c))
            norm = []
            for tc in tcs:
                if "function" in tc:
                    fn = tc["function"]
                    norm.append({"name": fn.get("name", ""),
                                 "arguments": fn.get("arguments", {})})
                else:
                    norm.append({"name": tc.get("name", ""),
                                 "arguments": tc.get("arguments") or {}})
            # LLM thinking: the message dict's reasoning_content (returned in thinking mode; "" when not thinking); small models do
            # not take this path (the baseline continuation is always 27B); thinking kept in full, not truncated (same basis as
            # appworld)
            reasoning = str(resp.get("reasoning_content") or "")
            if norm:
                self._trace_step(session, {
                    "span_type": "llm.call", "step": session.q_idx,
                    "source": "baseline", "output": {
                        "tool_calls": norm,
                        "reasoning": reasoning}})
            else:
                self._trace_step(session, {
                    "span_type": "llm.call", "step": session.q_idx,
                    "source": "baseline", "output": {
                        "text": content[:TRACE_RESULT_MAX_LEN],
                        "reasoning": reasoning}})
            return content, tcs

        prompt = build_question_prompt(
            session.q_idx, session.num_questions, session.question_text)
        answer, _steps = run_question_loop(
            session.memory, shim, prompt, step_fn, session.max_tool_steps)
        return str(answer or "")

    def _step_question(self, session: LocomoSession,
                       deadline: float) -> Optional[str]:
        """One step of mid-question resume (continuing from the seeded message list after injection -- a scenario the collection side does not have; the loop body corresponds
        line by line with locomot_common.run_question_loop, the only difference being continuation instead of a fresh loop). A non-None return means the question has been
        answered."""
        lm = self._make_lm(session)
        if not session.messages:
            session.messages = [{"role": "user", "content": build_question_prompt(
                session.q_idx, session.num_questions, session.question_text)}]
        self.budget_guard(session.max_tool_steps, deadline)
        try:
            resp = lm.generate(session.base_messages + session.messages,
                               tools=OAI_TOOL_SCHEMAS)
        except StopReplay:
            raise
        except Exception as exc:
            # LM call failed: record the failure text as this question's answer (advance honestly, no infinite retry)
            self._trace_step(session, {
                "span_type": "llm.call", "step": len(session.messages),
                "source": "baseline", "output": {
                    "text": "LM error: %s" % str(exc)[:120]}})
            self._record_answer(
                session, "LM error: %s" % str(exc)[:TRACE_RESULT_MAX_LEN])
            return session.answers[-1]["pred"]
        content = str(resp.get("content") or "")
        tcs = list(resp.get("tool_calls") or [])
        toks = resp.get("standardized_usage")
        p = c = 0
        try:
            t = getattr(toks, "tokens", None)
            p = int(getattr(t, "input_cache_miss", 0) or 0) \
                + int(getattr(t, "input_cache_hit", 0) or 0)
            c = int(getattr(t, "output", 0) or 0)
        except Exception:
            pass
        session.usage.append(LMResponse(
            raw_text=content, prompt_tokens=p, completion_tokens=c))
        norm_calls = []
        for tc in tcs:
            if "function" in tc:
                fn = tc["function"]
                norm_calls.append({
                    "id": tc.get("id", ""), "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", {})})
            else:
                norm_calls.append(dict(tc))
        # LLM thinking: the message dict's reasoning_content (returned in
        # thinking mode; "" when not thinking); the LM-error branch has no
        # output and is not recorded; thinking kept in full, not truncated
        # (same basis as appworld)
        reasoning = str(resp.get("reasoning_content") or "")
        if not norm_calls:
            self._trace_step(session, {
                "span_type": "llm.call", "step": len(session.messages),
                "source": "baseline", "output": {
                    "text": content[:TRACE_RESULT_MAX_LEN],
                    "reasoning": reasoning}})
            self._record_answer(session, content)
            return content
        self._trace_step(session, {
            "span_type": "llm.call", "step": len(session.messages),
            "source": "baseline", "output": {"tool_calls": [
                {"name": tc.get("name", ""), "arguments": tc.get("arguments")}
                for tc in norm_calls],
                "reasoning": reasoning}})
        session.messages.append({"role": "assistant", "content": "",
                                 "tool_calls": tcs})
        for tc in norm_calls:
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            result = execute_tool(str(tc.get("name", "")), args,
                                  session.sample, session.memory)
            ok = not result.startswith('{"error"')
            session.messages.append({
                "role": "tool", "tool_call_id": tc.get("id", ""),
                "name": tc.get("name", ""), "content": result})
            self._trace_step(session, {
                "span_type": "tool.execution",
                "step": len(session.messages), "source": "baseline",
                "tool": tc.get("name", ""), "arguments": args, "ok": bool(ok),
                "result": str(result)[:TRACE_RESULT_MAX_LEN]})
        return None

    # -- 14.4 steps 7-8 -----------------------------------------------------
    def run_to_node(self, session: LocomoSession, span: SpanRecord,
                    client: Any, max_steps: int, timeout_s: float,
                    budget_usd: float) -> Tuple[DownstreamOutcome,
                                                List[LMResponse]]:
        outcome = DownstreamOutcome()
        deadline = time.perf_counter() + timeout_s
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
                # the answer span was already bookkept in inject; no downstream execution
                outcome.next_node_reached = True
                outcome.next_tool_executed = False
                outcome.downstream_execution_valid = True
                outcome.downstream_checkpoint_reached = True
                return outcome, []
            # after a tool span's injection: the baseline continues the current question to the answer
            step_budget = max(1, session.max_tool_steps)
            answer = None
            while answer is None and step_budget > 0:
                self.budget_guard(step_budget, deadline)
                answer = self._step_question(session, deadline)
                step_budget -= 1
            if answer is None:
                # over budget: same basis as run_question_loop (empty last output -> empty answer)
                self._record_answer(session, "")
                outcome.downstream_checkpoint_reached = False
            else:
                outcome.downstream_checkpoint_reached = True
            outcome.next_node_reached = True
            outcome.next_tool_executed = True
            outcome.downstream_execution_valid = True
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

    def run_to_completion(self, session: LocomoSession, span: SpanRecord,
                          client: Any, max_steps: int, timeout_s: float,
                          budget_usd: float) -> Tuple[FinalOutcome,
                                                      List[LMResponse]]:
        final = FinalOutcome()
        deadline = time.perf_counter() + timeout_s
        try:
            if getattr(session, "frozen_downstream", False):
                self._continue_frozen(session, stop_after_span=False)
            else:
                # finish the current unanswered question first, then continue the remaining questions one by one
                if not session.question_done:
                    step_budget = max(1, session.max_tool_steps)
                    while not session.question_done and step_budget > 0:
                        self.budget_guard(step_budget, deadline)
                        self._step_question(session, deadline)
                        step_budget -= 1
                    if not session.question_done:
                        # over budget: same basis as run_question_loop (empty last output -> "")
                        self._record_answer(session, "")
                while session.q_idx < session.num_questions:
                    self.budget_guard(session.num_questions - session.q_idx,
                                      deadline)
                    # the whole-question loop goes through the collection side's real run_question_loop
                    answer = self._run_fresh_question(session, deadline)
                    self._record_answer(session, answer)
            ev = evaluate_answers(
                session.sample, session.answers,
                session.memory.read_dia_ids)
            final.goal_state_passed = bool(ev["success"])
            final.final_task_success = bool(ev["success"])
            final.collateral_damage = False
            final.evaluator_details = [
                "f1=%.4f threshold=%.1f n=%d recall=%s" % (
                    ev["f1"], ev["threshold"], ev["num_questions"],
                    ev["overall_recall"]),
                "category_avg_f1=%s" % ev["category_avg_f1"],
            ]
            for q in ev["per_question"]:
                if q.get("f1") is not None and q["f1"] < PASS_THRESHOLD:
                    final.evaluator_details.append(
                        "q%d(cat%s) f1=%.3f pred=%r gold=%r" % (
                            q.get("idx"), q.get("category"), q["f1"],
                            q.get("pred"), q.get("gold")))
            session.downstream_trace["terminal"] = {
                "status": "completed", "f1": ev["f1"],
                "success": ev["success"]}
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

    # -- C0 frozen walk-through ---------------------------------------------
    def _continue_frozen(self, session: LocomoSession,
                         stop_after_span: bool) -> bool:
        """Downstream injects the trace-recorded output, only really executes tools (read-only), and compares execution results with the recorded observations per span; answer spans bookkeep the recorded pred and advance the question. Zero LM
        calls."""
        span = session.span
        trace = session.trace
        assert span is not None and trace is not None
        if session.frozen_plan is None:
            session.frozen_plan = sorted(
                (sp for sp in trace.spans
                 if sp.invocation_index > span.invocation_index),
                key=lambda sp: sp.invocation_index)
            session.frozen_index = 0
        plan = session.frozen_plan
        if session.frozen_completed:
            return True
        if stop_after_span:
            # the target span's tools were already really executed and compared in inject;
            # run_to_node consumes no downstream span, the walk-through is finished by run_to_completion
            return False
        while session.frozen_index < len(plan):
            rec = plan[session.frozen_index]
            session.frozen_index += 1
            recorded_results = trace.artifacts["span_tool_results"].get(
                "span:%d" % rec.invocation_index, [])
            parsed = rec.baseline_response.parsed or {}
            if (rec.output_contract_runtime.contract_type == "tool_call_list"
                    and parsed.get("tool_calls")):
                calls = parsed["tool_calls"]
                executed: List[Dict[str, Any]] = []
                for tc in calls:
                    name = str(tc.get("name", ""))
                    args = tc.get("arguments") or {}
                    if not isinstance(args, dict):
                        args = {}
                    # the frozen walk-through replays with the recorded execution args (same basis as _with_recorded_args): the framework layer may pad defaults, replaying by the model output would
                    # distort
                    for r in recorded_results:
                        if r.get("tool") == name:
                            args = r.get("arguments") or args
                            break
                    result = execute_tool(
                        name, args, session.sample, session.memory)
                    ok = not result.startswith('{"error"')
                    executed.append({"tool": name, "ok": bool(ok),
                                     "result": str(result)
                                     [:TRACE_RESULT_MAX_LEN]})
                    self._trace_step(session, {
                        "span_type": "tool.execution",
                        "step": rec.invocation_index, "source": "frozen",
                        "tool": name, "arguments": args, "ok": bool(ok),
                        "result": str(result)[:TRACE_RESULT_MAX_LEN]})
                session.c0_fidelity.append(self._compare_frozen(
                    rec, executed, recorded_results))
            else:
                # answer span: bookkeep the recorded pred, advance the question (no state effect, the walk-through continues)
                text = str(parsed.get("text", "")) if isinstance(
                    parsed, dict) else ""
                session.c0_fidelity.append({
                    "invocation_index": rec.invocation_index,
                    "node_pattern": rec.span_descriptor.node_pattern,
                    "tool_match": None,
                    "note": "recorded answer span; pred bookkept"})
                self._trace_step(session, {
                    "span_type": "llm.call", "step": rec.invocation_index,
                    "source": "frozen",
                    "output": {"text": text[:TRACE_RESULT_MAX_LEN]}})
                self._record_answer(session, text)
        session.frozen_completed = True
        return session.q_idx >= session.num_questions

    def _compare_frozen(self, rec: SpanRecord,
                        executed: List[Dict[str, Any]],
                        recorded: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Two-layer comparison: tool_match (hard: names/counts/ok flags
        match) + value_match (soft: result text semantically consistent)."""
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

    def rollback(self, session: LocomoSession) -> bool:
        # pre-commit verifier only: no state committed before injection (read-only baseline)
        return True
