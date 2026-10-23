"""Agent runtime interface for Stage 10 (spec 14.4).

The replay protocol is runtime-agnostic; everything environment-specific
(state restore, original agent parser, local verifier, workflow continuation)
lives behind AgentRuntime.  The bundled MockAppWorldRuntime implements it for
the pilot harness; a real AppWorld integration provides the same interface
over the frozen representative agent + AppWorld executor.

Protocol steps (14.4), mapped to runtime calls:
    1-2  trace from the frozen agent + baseline LLM           (pre-recorded)
    3    restore identical pre-span state                     restore()
    4    same prompt / tool schema / output contract -> SLM    resolve_request()
    5    parse with original parser + local verifier           parse() / verify_local()
    6    inject parsed output into the workflow                inject()
    7    all later LM calls use baseline LLM                   (driver default)
    8    continue to the final evaluator                       run_to_node() / run_to_completion()
    9    record all outcome layers                             (engine side)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import LMResponse
from .schemas import (
    DownstreamOutcome, FinalOutcome, SpanRecord, TraceRun,
)
from .verifiers import VerifierVerdict


class StopReplay(Exception):
    """Raised by drivers when a step/time/cost budget is exhausted."""

    def __init__(self, flag: str, detail: str = ""):
        super().__init__("%s: %s" % (flag, detail))
        self.flag = flag  # "budget_exhausted" | "timeout"
        self.detail = detail


@dataclass
class ResolvedRequest:
    """The exact LM request for span j, resolved from trace artifacts."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tools: Optional[List[Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    temperature: float = 0.0
    max_output_tokens: int = 8192
    meta_hint: str = ""
    #: native tool_choice ("required" / "auto" / None): legacy and smolagents' native
    #: protocol forces tool_choice="required"; other agents keep None
    tool_choice: Optional[str] = None


@dataclass
class ParseResult:
    ok: bool = False
    parsed: Any = None
    error: str = ""


@dataclass
class InjectedResponse(LMResponse):
    """Driver response for the intervened span: the parsed output is provided
    directly, no LM call happens for span j."""
    parsed: Any = None


@dataclass
class RestoredSession:
    """Runtime-specific handle to the restored pre-span state."""
    runtime: Any = None
    handle: Any = None
    snapshot_hash: str = ""
    next_invocation_index: int = -1


@dataclass
class SpanContext:
    """Everything a driver/policy needs about the span being invoked."""
    invocation_index: int = -1
    node_pattern: str = "other"
    control_role: str = "worker"
    contract_type: str = "json_schema"
    span_subtask_type: str = ""
    descriptor: Any = None        # SpanDescriptor
    verifier_runtime: Any = None  # VerifierRuntime
    meta_hint: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 8192
    allowed_tools: List[str] = field(default_factory=list)


class AgentRuntime(ABC):
    """Pluggable execution backend for single-span counterfactual replay."""

    name = "abstract"

    @abstractmethod
    def restore(self, trace: TraceRun, span: SpanRecord,
                baseline_client: Any = None) -> RestoredSession:
        """Restore agent/environment state to exactly the span-before snapshot
        (spec 14.4 step 3).  Must be deterministic and side-effect free so every
        SLM candidate starts from the identical snapshot (14.3)."""

    @abstractmethod
    def resolve_request(self, trace: TraceRun, span: SpanRecord,
                        session: Optional[RestoredSession] = None) -> ResolvedRequest:
        """Resolve the recorded request refs into the concrete prompt payload
        (spec 14.4 step 4: same prompt, tool schema, output contract).

        `session` (the product of restore()) is always available in the replay engine before
        this method; the runtime can use it for session-level rebuilds (e.g. an OpenAI-SDK trace
        does not record the conversation and needs the world context to re-render the real system prompt)."""

    @abstractmethod
    def parse(self, span: SpanRecord, raw: str) -> ParseResult:
        """Original agent parser on SLM output (step 5)."""

    @abstractmethod
    def schema_check(self, session: "RestoredSession", span: SpanRecord,
                     parsed: Any) -> bool:
        """Contract schema validity of the parsed output."""

    @abstractmethod
    def verify_local(self, session: RestoredSession, span: SpanRecord,
                     parsed: Any) -> VerifierVerdict:
        """Local verifier(s) per span.verifier_runtime (step 5)."""

    @abstractmethod
    def inject(self, session: RestoredSession, span: SpanRecord, parsed: Any) -> bool:
        """Inject the parsed SLM output into the workflow at span j (step 6)."""

    @abstractmethod
    def run_to_node(self, session: RestoredSession, span: SpanRecord,
                    client: Any, max_steps: int,
                    timeout_s: float, budget_usd: float) -> tuple:
        """Execute the workflow (baseline LLM after span j) until
        continuation.resume_node_id is reached or a budget trips.
        Returns (DownstreamOutcome, List[LMResponse])."""

    @abstractmethod
    def run_to_completion(self, session: RestoredSession, span: SpanRecord,
                          client: Any, max_steps: int,
                          timeout_s: float, budget_usd: float) -> tuple:
        """Continue from the current session state to the final evaluator.
        Returns (FinalOutcome, List[LMResponse])."""

    @abstractmethod
    def rollback(self, session: RestoredSession) -> bool:
        """Roll back external state effects if the verifier rejects a
        post-commit span.  Pre-commit spans have nothing to roll back and must
        report True with a 'nothing committed' note."""

    def budget_guard(self, steps_left: int, deadline: float) -> None:
        """Shared budget check; drivers call this before each LM/tool step."""
        import time as _time
        if steps_left <= 0:
            raise StopReplay("budget_exhausted", "step budget exhausted")
        if _time.perf_counter() > deadline:
            raise StopReplay("timeout", "wall-clock timeout")
