"""Trace-local replay runtime: the LOCAL layer of the replay protocol
(spec 14.4 steps 3-5) for traces without environment checkpoints.

What it does (honestly):
  - feeds the candidate SLM the exact recorded prompt (same request the
    baseline model saw),
  - parses the SLM output with the original parser,
  - checks the recorded output contract (tool-call list schema, known tool
    names, JSON arguments / answer schema),
  - reports the local verifier verdict.

What it does NOT do (and never fakes):
  - downstream continuation and the final evaluator.  Those require the
    framework's full state and a live/checkpointed AppWorld environment, which
    the current trace format does not carry.  Records are stamped
    replay_meta["outcome_source"] = "trace_local" and are EXCLUDED from the
    strict-success profiles O_m^strict (14.6); they feed the separate
    local-only profile instead (parse/schema/contract rates per span class).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .appworld_adapter import parse_llm_output, parse_messages, ANSWER_SCHEMA, TOOL_CALL_SCHEMA
from .llm_client import LMResponse
from .runtime import (
    AgentRuntime, ParseResult, ResolvedRequest, RestoredSession, SpanContext,
)
from .schemas import (
    DownstreamOutcome, FinalOutcome, SpanRecord, TraceRun,
)
from .verifiers import VerifierVerdict, validate_schema


@dataclass
class TraceLocalSession(RestoredSession):
    trace: Optional[TraceRun] = None
    span: Optional[SpanRecord] = None
    injected: Optional[Any] = None


class TraceLocalRuntime(AgentRuntime):
    """Local-only replay backend for checkpoint-less traces (e.g. the current
    AppWorld exporter format)."""

    name = "trace-local"
    outcome_source = "trace_local"

    def restore(self, trace: TraceRun, span: SpanRecord,
                baseline_client: Any = None) -> TraceLocalSession:
        return TraceLocalSession(
            runtime=self, trace=trace, span=span,
            snapshot_hash=span.state_snapshot.application_state_hash,
            next_invocation_index=span.invocation_index,
        )

    def resolve_request(self, trace: TraceRun, span: SpanRecord,
                        session: Any = None) -> ResolvedRequest:
        messages = (trace.artifacts.get("messages") or {}).get(span.request.messages_ref, [])
        return ResolvedRequest(
            messages=list(messages),
            tools=None,
            response_format=None,
            temperature=span.request.temperature,
            max_output_tokens=span.request.max_output_tokens,
            meta_hint=span.meta_hint,
        )

    def parse(self, span: SpanRecord, raw: str) -> ParseResult:
        parsed, kind = parse_llm_output(raw)
        expected = span.output_contract_runtime.contract_type
        if kind == "freeform_text" and expected != "freeform_text":
            return ParseResult(ok=False, parsed=None,
                               error="expected %s output, got freeform text" % expected)
        if kind != "freeform_text" and expected == "freeform_text":
            return ParseResult(ok=False, parsed=None,
                               error="expected freeform text, got structured output")
        return ParseResult(ok=True, parsed=parsed, error="")

    def schema_check(self, session: TraceLocalSession, span: SpanRecord,
                     parsed: Any) -> bool:
        contract = span.output_contract_runtime.contract_type
        if contract == "tool_call_list":
            return not validate_schema(parsed, TOOL_CALL_SCHEMA)
        if contract == "answer_json":
            return not validate_schema(parsed, ANSWER_SCHEMA)
        return True

    def verify_local(self, session: TraceLocalSession, span: SpanRecord,
                     parsed: Any) -> VerifierVerdict:
        verdict = VerifierVerdict()
        contract = span.output_contract_runtime.contract_type
        if contract == "tool_call_list":
            allowed = {t.replace("__", ".") for t in span.output_contract_runtime.allowed_tools}
            ok = True
            errors: List[str] = []
            for call in (parsed or {}).get("tool_calls", []):
                name = str(call.get("name", "")).replace("__", ".")
                if name not in allowed:
                    ok = False
                    errors.append("tool %r not in the trace's known tool set" % name)
                if not isinstance(call.get("arguments"), dict):
                    ok = False
                    errors.append("tool %r arguments not a JSON object" % name)
            verdict = VerifierVerdict(triggered=True, detected_failure=not ok,
                                      passed=ok, executed=["contract_check"], errors=errors)
        if contract == "answer_json":
            ok = "status" in (parsed or {})
            verdict = VerifierVerdict(triggered=True, detected_failure=not ok,
                                      passed=ok, executed=["contract_check"],
                                      errors=[] if ok else ["answer_json missing 'status'"])
        return verdict

    def inject(self, session: TraceLocalSession, span: SpanRecord, parsed: Any) -> bool:
        session.injected = parsed
        return True

    def run_to_node(self, session: TraceLocalSession, span: SpanRecord,
                    client: Any, max_steps: int, timeout_s: float,
                    budget_usd: float) -> Tuple[DownstreamOutcome, List[LMResponse]]:
        # continuation requires a live/checkpointed environment -- not measured
        return DownstreamOutcome(), []

    def run_to_completion(self, session: TraceLocalSession, span: SpanRecord,
                          client: Any, max_steps: int, timeout_s: float,
                          budget_usd: float) -> Tuple[FinalOutcome, List[LMResponse]]:
        final = FinalOutcome()
        final.evaluator_details.append(
            "not_measured: trace_local mode (no environment checkpoint; "
            "downstream/final layers require a live AppWorld runtime)")
        return final, []

    def rollback(self, session: TraceLocalSession) -> bool:
        return True


class RecordedEchoSLM:
    """OFFLINE SANITY ONLY: an 'SLM' that echoes each span's recorded baseline
    output (keyed by meta_hint).  An identity SLM must pass every local check;
    use it to smoke-test the trace-local measurement machinery before pointing
    it at a real SLM endpoint."""

    def __init__(self, outputs: Dict[str, str], model: str = "echo-slm",
                 price_per_1k_input: float = 0.05, price_per_1k_output: float = 0.3):
        self.outputs = outputs
        self.model = model
        self.price_per_1k_input = price_per_1k_input
        self.price_per_1k_output = price_per_1k_output

    @property
    def name(self) -> str:
        return self.model

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.price_per_1k_input + \
               (completion_tokens / 1000.0) * self.price_per_1k_output

    def complete(self, messages: List[Dict[str, Any]], tools: Any = None,
                 response_format: Any = None, temperature: Any = None,
                 max_tokens: Any = None, tool_choice: Any = None,
                 ctx: Any = None) -> LMResponse:
        hint = getattr(ctx, "meta_hint", "") if ctx is not None else ""
        raw = self.outputs.get(hint)
        if raw is None:
            return LMResponse(error="echo-slm: no recorded output for hint %r" % hint)
        return LMResponse(
            raw_text=raw,
            prompt_tokens=max(1, sum(len(m.get("content", "")) // 4 for m in messages)),
            completion_tokens=max(1, len(raw) // 4),
            latency_ms=15.0,
            finish_reason="stop",
        )


def build_echo_outputs(traces: List[TraceRun]) -> Dict[str, str]:
    """meta_hint -> recorded baseline output, for RecordedEchoSLM."""
    outputs: Dict[str, str] = {}
    for trace in traces:
        for span in trace.spans:
            outputs[span.meta_hint] = span.baseline_response.raw
    return outputs
