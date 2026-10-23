"""Core data schemas for Stage 10 (spec 14.1-14.6).

All benchmark data is stored as structured categorical / multi-label metadata
(spec 14.2.6).  One-hot / multi-hot / ordinal encoding happens only inside the
statistical analysis module; raw canonical descriptors are never replaced by
dense embeddings.

Three-layer conditioning (spec 14.2.1):
    phi_s = SpanDescriptor(s)      span-intrinsic, obtainable before replay
    z_t   = TaskCondition(t)       frozen task manifest, NOT inferred from outcomes
    a_H   = AgentCondition(H)      representative agent / architecture context

Replay outcomes (spec 14.2.5) are labels and MUST NOT enter any descriptor:
SLM local success, downstream success, final-task success, collateral damage,
fallback success, measured cost/latency saving.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

TRACE_SCHEMA_VERSION = "stage10-native-trace/v1"
MANIFEST_SCHEMA_VERSION = "stage10-task-manifest/v1"

# ---------------------------------------------------------------------------
# Canonical vocabularies (spec 14.2.2).  Extraction warns on unknown values but
# never hard-fails, so extended taxonomies can be introduced incrementally.
# ---------------------------------------------------------------------------

# A. node_pattern -- what the invocation does (spec 14.2.2.A)
NODE_PATTERNS = {
    "planning",
    "tool_selection",
    "tool_argument_generation",
    "observation_interpretation",
    "failure_diagnosis",
    "replanning",
    "final_synthesis",
    # additional span types named in the spec 14.2 span-type list
    "intent_interpretation",
    "policy_checking",
    "other",
}

# B. control_role -- where in the control structure the node sits (14.2.2.B)
CONTROL_ROLES = {
    "global_planner", "local_executor", "router", "verifier",
    "repair_controller", "supervisor", "worker", "terminal",
}

# D. mechanism_tags (14.2.2.D) -- fixed four groups in v0.3
INTERFACES = {"api_tool", "browser", "gui", "shell", "file_system", "database"}
CONTEXT_SOURCES = {
    "conversation_history", "tool_schema", "previous_tool_results",
    "memory", "retrieved_docs", "policy_doc",
}
OUTPUT_CONTRACTS = {
    "json_schema", "tool_name", "tool_arguments", "structured_plan",
    "boolean_decision", "freeform_text",
}
STATE_EFFECTS = {
    "read_only", "external_state_mutation", "memory_write",
    "artifact_generation", "permission_sensitive",
}

# E. verifier_type -- verifier capability, NOT the replay outcome (14.2.2.E)
VERIFIER_TYPES = {
    "schema_check", "tool_exec_success", "policy_check",
    "goal_state_match", "db_state_match", "none",
}

# F. graph_context (14.2.2.F)
CONTROL_PHASES = {"initial", "execution", "verification", "recovery", "terminal"}

# task condition vocabularies (14.2.3)
HORIZONS = {"short", "medium", "long"}
BRANCHING = {"none", "conditional", "loop"}
COLLATERAL_RISKS = {"low", "medium", "high"}
TASK_STATE_EFFECTS = {"read_only", "state_mutating"}
EVALUATOR_TYPES = {"goal_state_match", "db_state_match", "policy_check", "answer_match"}

VERIFIER_LOCATIONS = {"pre_commit", "post_commit"}

# Fields that are replay OUTCOMES / labels and are forbidden inside descriptors
# (spec 14.2.5 -- candidate model identity and replay-only outcomes).
FORBIDDEN_DESCRIPTOR_FIELDS = {
    "slm_local_success", "slm_downstream_success", "slm_final_task_success",
    "collateral_damage", "fallback_success", "measured_cost", "measured_latency",
    "candidate_model", "model_id", "replay_outcome", "strict_success",
    "local_verifier_passed", "final_task_success",
}

# Suggested checklist to revisit when the descriptor completeness test (14.7.5)
# shows large conditioned-vs-marginal gaps.
MISSING_DESCRIPTOR_CHECKLIST = [
    "cross_app_dependency",
    "context_size",
    "context_source_complexity",
    "secondary_patterns",
    "mechanism_tags_depth",
    "failure_recovery_context",
    "output_contract_strength",
    "state_risk_granularity",
]


# ---------------------------------------------------------------------------
# Generic dataclass <-> JSON serialization
# ---------------------------------------------------------------------------

def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    return obj


def _convert(value: Any, hint: Any, path: str) -> Any:
    origin = get_origin(hint)
    if origin in (list, tuple, List, getattr(__import__("typing"), "List", None)):
        args = get_args(hint)
        inner = args[0] if args else None
        if inner is not None and dataclasses.is_dataclass(inner):
            return [from_dict(inner, v) for v in value]
        return list(value)
    if origin is dict or origin is getattr(__import__("typing"), "Dict", None):
        return dict(value)
    if origin is Union or origin is getattr(__import__("typing"), "Union", None):
        for arg in get_args(hint):
            if arg is type(None):
                continue
            if dataclasses.is_dataclass(arg):
                return from_dict(arg, value)
            return value
    if dataclasses.is_dataclass(hint):
        return from_dict(hint, value)
    return value


def from_dict(cls: Any, data: Any) -> Any:
    if data is None:
        return None
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        name = f.name
        hint = hints.get(name)
        if name not in data or data[name] is None:
            if name in data and data[name] is None:
                kwargs[name] = None
            elif f.default is not dataclasses.MISSING:
                kwargs[name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
                kwargs[name] = f.default_factory()  # type: ignore[attr-defined]
            else:
                kwargs[name] = None
            continue
        kwargs[name] = _convert(data[name], hint, name)
    return cls(**kwargs)


def canonical(obj: Any) -> Any:
    """Sort all lists in-place so dict equality / hashing is order-insensitive."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = canonical(v)
        return obj
    if isinstance(obj, list):
        try:
            return sorted(canonical(x) for x in obj)
        except TypeError:
            return [canonical(x) for x in obj]
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(canonical(to_dict(obj)), sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Span descriptor (14.2.2) -- intrinsic, pre-replay, explainable
# ---------------------------------------------------------------------------

@dataclass
class MechanismTags:
    interface: List[str] = field(default_factory=list)
    context_source: List[str] = field(default_factory=list)
    output_contract: List[str] = field(default_factory=list)
    state_effect: List[str] = field(default_factory=list)


@dataclass
class GraphContext:
    inside_loop: bool = False
    control_phase: str = "execution"
    predecessor_class: str = ""
    after_failure: bool = False
    after_verification_failure: bool = False
    loop_iteration_bucket: str = "1"


@dataclass
class SpanDescriptor:
    node_pattern: str = "other"
    control_role: str = "worker"
    span_subtask_type: str = ""
    mechanism_tags: MechanismTags = field(default_factory=MechanismTags)
    verifier_type: List[str] = field(default_factory=lambda: ["none"])
    graph_context: GraphContext = field(default_factory=GraphContext)


@dataclass
class TaskCondition:
    task_type: str = ""
    involved_apps: List[str] = field(default_factory=list)
    cross_app: bool = False
    horizon: str = "short"
    task_state_effect: str = "read_only"
    branching: str = "none"
    evaluator_type: str = "goal_state_match"
    collateral_risk: str = "low"


@dataclass
class AgentCondition:
    representative_id: str = ""
    source_cluster_id: str = ""
    architecture_annotation: str = ""


# ---------------------------------------------------------------------------
# Replay input (14.3).  Execution payload and analysis metadata are kept apart:
#   - lm_request / state_snapshot / output_contract_runtime / continuation are
#     the actual replay execution inputs (references into the trace artifacts)
#   - span_descriptor / task_condition / agent_condition are for statistics and
#     conditioned analyses only.
# ---------------------------------------------------------------------------

@dataclass
class LMRequest:
    system_prompt_ref: str = ""
    messages_ref: str = ""
    previous_tool_results_ref: str = ""
    available_tools_ref: str = ""
    tool_schemas_ref: str = ""
    response_format_ref: str = ""
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None


@dataclass
class StateSnapshot:
    agent_state_ref: str = ""
    environment_snapshot_ref: str = ""
    conversation_history_ref: str = ""
    plan_state_ref: str = ""
    application_state_hash: str = ""


@dataclass
class OutputContractRuntime:
    contract_type: str = "json_schema"
    schema_ref: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    semantic_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifierRuntime:
    verifier_type: List[str] = field(default_factory=lambda: ["none"])
    verifier_location: str = "pre_commit"
    pre_commit: bool = True
    rollback_available: bool = False


@dataclass
class Continuation:
    resume_node_id: str = ""
    later_lm_model: str = "baseline_llm"
    evaluator_ref: str = ""
    max_remaining_steps: int = 64


@dataclass
class ReplayInput:
    lm_request: LMRequest = field(default_factory=LMRequest)
    state_snapshot: StateSnapshot = field(default_factory=StateSnapshot)
    span_descriptor: SpanDescriptor = field(default_factory=SpanDescriptor)
    task_condition: TaskCondition = field(default_factory=TaskCondition)
    agent_condition: AgentCondition = field(default_factory=AgentCondition)
    output_contract_runtime: OutputContractRuntime = field(default_factory=OutputContractRuntime)
    verifier_runtime: VerifierRuntime = field(default_factory=VerifierRuntime)
    continuation: Continuation = field(default_factory=Continuation)


# ---------------------------------------------------------------------------
# Native trace format (a successful baseline run r of agent H on task t)
# ---------------------------------------------------------------------------

@dataclass
class BaselineResponse:
    raw: str = ""
    parsed: Any = None
    parse_success: bool = False
    parse_error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0


@dataclass
class ToolExecution:
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    result: Any = None
    error: str = ""
    node_id: str = ""


@dataclass
class SpanRecord:
    invocation_index: int = 0
    #: The native span_id of this llm.call in the original OTel trace (16 hex digits);
    #: used to map the synthetic span_id back to the original trace's span tree
    native_span_id: str = ""
    node_id: str = ""
    span_descriptor: SpanDescriptor = field(default_factory=SpanDescriptor)
    request: LMRequest = field(default_factory=LMRequest)
    state_snapshot: StateSnapshot = field(default_factory=StateSnapshot)
    output_contract_runtime: OutputContractRuntime = field(default_factory=OutputContractRuntime)
    verifier_runtime: VerifierRuntime = field(default_factory=VerifierRuntime)
    continuation: Continuation = field(default_factory=Continuation)
    baseline_response: BaselineResponse = field(default_factory=BaselineResponse)
    tool_execution: Optional[ToolExecution] = None
    meta_hint: str = ""


@dataclass
class RunMeta:
    agent_id: str = ""
    task_id: str = ""
    run_id: str = ""
    baseline_model: str = ""
    created_at: str = ""
    final_evaluator_passed: bool = False
    goal_state_passed: bool = False
    collateral_damage: bool = False
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    attempts: int = 1
    notes: str = ""


@dataclass
class TraceRun:
    schema_version: str = TRACE_SCHEMA_VERSION
    run_meta: RunMeta = field(default_factory=RunMeta)
    task_condition: TaskCondition = field(default_factory=TaskCondition)
    agent_condition: AgentCondition = field(default_factory=AgentCondition)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    node_ids: List[str] = field(default_factory=list)
    spans: List[SpanRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Extracted span instance -- the basic measurement unit s = (H, t, r, j)
# (spec 14.1)
# ---------------------------------------------------------------------------

@dataclass
class SpanInstance:
    span_id: str = ""
    #: The native span_id of this llm.call in the original OTel trace
    native_span_id: str = ""
    trace_path: str = ""
    agent_id: str = ""
    task_id: str = ""
    run_id: str = ""
    invocation_index: int = -1
    node_id: str = ""
    span_descriptor: SpanDescriptor = field(default_factory=SpanDescriptor)
    task_condition: TaskCondition = field(default_factory=TaskCondition)
    agent_condition: AgentCondition = field(default_factory=AgentCondition)
    replay_input: ReplayInput = field(default_factory=ReplayInput)
    stratification_unit: Dict[str, Any] = field(default_factory=dict)
    unit_key: str = ""
    experiment_split: str = "pilot"
    eligible: bool = False
    ineligibility_reasons: List[str] = field(default_factory=list)
    baseline_failed: bool = False
    baseline_span_cost: float = 0.0
    baseline_span_latency_ms: float = 0.0
    baseline_tokens: int = 0
    meta_hint: str = ""


# ---------------------------------------------------------------------------
# Replay outcomes (14.5) -- labels, never descriptors
# ---------------------------------------------------------------------------

@dataclass
class LocalOutcome:
    parse_success: bool = False
    schema_valid: bool = False
    local_output_valid: bool = False
    local_verifier_passed: bool = False
    parse_error: str = ""
    verifier_errors: List[str] = field(default_factory=list)
    #: Local equivalence verifier result: the consistency verdict of SLM output vs baseline
    #: output (tool name/argument/execution-result comparison, or text semantic detection;
    #: modeled on AppWorld TestTracker's requirement list + passed summary, see span_offload/local_equivalence.py)
    equivalence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DownstreamOutcome:
    next_node_reached: bool = False
    next_tool_executed: bool = False
    downstream_execution_valid: bool = False
    downstream_checkpoint_reached: bool = False
    unhandled_exception: bool = False


@dataclass
class FinalOutcome:
    final_task_success: bool = False
    goal_state_passed: bool = False
    collateral_damage: bool = False
    timeout: bool = False
    budget_exhausted: bool = False
    evaluator_details: List[str] = field(default_factory=list)


@dataclass
class FallbackOutcome:
    verifier_triggered: bool = False
    verifier_detected_failure: bool = False
    fallback_triggered: bool = False
    rollback_succeeded: bool = False
    fallback_recovered: bool = False
    fallback_model: str = ""


@dataclass
class ResourceOutcome:
    baseline_span_cost: float = 0.0
    slm_cost: float = 0.0
    #: Target span generation cost of the B arm (baseline-LLM resampling); T_span's
    #: NetSaving is baselined on B's live resampling cost (not the trace's recorded historical cost)
    baseline_target_cost: float = 0.0
    verifier_cost: float = 0.0
    fallback_cost: float = 0.0
    total_counterfactual_cost: float = 0.0
    cost_saving: float = 0.0
    baseline_span_latency: float = 0.0
    total_counterfactual_latency: float = 0.0
    latency_saving: float = 0.0
    #: [legacy field, disabled for analysis] baseline_tokens (collection-side convention) −
    #: target generation tokens (live serving convention) — two incomparable yardsticks; it
    #: covers only the target layer with no downstream compensation and once misled the utility
    #: conclusions. NetSaving is always computed from the itemized fields below at the analysis layer via per-span B vs T pairing: total = target + downstream (prompt+completion)
    token_delta: int = 0
    #: Itemized tokens and context load (unified cost convention, plan §9.3/§15): the target
    #: span and the downstream record prompt/completion separately; context_load = the
    #: cumulative prompt-token area from the target span to termination (0 for identity/C0 target items)
    target_prompt_tokens: int = 0
    target_completion_tokens: int = 0
    downstream_prompt_tokens: int = 0
    downstream_completion_tokens: int = 0
    context_load: int = 0
    #: Target-layer generation time (ms): T_span=SLM call, B=baseline live resampling
    #: (both take slm_resp.latency_ms, single successful attempt convention); I/C0/identity
    #: =0.0 (no target-layer model generation). Old data missing the key backfills 0.0 via from_dict
    target_generation_ms: float = 0.0


@dataclass
class ReplayRecord:
    span_id: str = ""
    #: The native span_id of this llm.call in the original OTel trace (for looking back up the original span tree)
    native_span_id: str = ""
    agent_id: str = ""
    task_id: str = ""
    run_id: str = ""
    invocation_index: int = -1
    model: str = ""
    span_descriptor: SpanDescriptor = field(default_factory=SpanDescriptor)
    task_condition: TaskCondition = field(default_factory=TaskCondition)
    agent_condition: AgentCondition = field(default_factory=AgentCondition)
    stratification_unit: Dict[str, Any] = field(default_factory=dict)
    unit_key: str = ""
    local_outcome: LocalOutcome = field(default_factory=LocalOutcome)
    downstream_outcome: DownstreamOutcome = field(default_factory=DownstreamOutcome)
    final_outcome: FinalOutcome = field(default_factory=FinalOutcome)
    fallback: FallbackOutcome = field(default_factory=FallbackOutcome)
    resource: ResourceOutcome = field(default_factory=ResourceOutcome)
    strict_success: bool = False
    replay_meta: Dict[str, Any] = field(default_factory=dict)
    #: SLM output vs baseline comparison material (offload consistency analysis):
    #:   baseline_tool_calls / slm_tool_calls / injected_tool_calls are all normalized to
    #:   [{"name": ..., "arguments": ...}]; slm_raw_text is the SLM's raw output (truncated);
    #:   slm_error is the SLM call error (if any); reasoning is the target span call's large-model
    #:   thinking (a separate output part, "" when the SLM has no thinking); fallback_reasoning
    #:   is the live_baseline fallback call's thinking (key present only on the successful fallback path)
    slm_output: Dict[str, Any] = field(default_factory=dict)
    #: The full structured trace of injection + continuation by the baseline LLM, aligned with the original trace's structure:
    #:   {
    #:     "meta": {task/run/offloaded span info/model/fallback},
    #:     "prefix": {"messages": [...the conversation window before the span (upstream info)...],
    #:                "previous_tool_results": [...]},
    #:     "spans": [span list, each corresponding to one span of the original trace:
    #:        {"span_type": "llm.call"|"tool.execution", "step": n,
    #:         "source": "slm"|"baseline",
    #:         "output": {tool_calls/text}(llm.call; a source=="baseline"
    #:                   live call also carries a reasoning key, "" when no thinking,
    #:                   slm spans do not carry it — small-model output is all in content),
    #:         "tool"/"arguments"/"ok"/"result"(tool.execution), ...],
    #:     "terminal": {"status": ..., "world_completed": ...}
    #:   }
    #: Accumulated at runtime by each continuation backend (simplified/autogen/fallback)
    downstream_trace: Dict[str, Any] = field(default_factory=dict)


def derive_strict_success(final: FinalOutcome) -> bool:
    """S^strict = I[final_task_success=1 AND collateral_damage=0] (spec 14.5)."""
    return bool(final.final_task_success) and not bool(final.collateral_damage)


# ---------------------------------------------------------------------------
# Task manifest (14.2.3) -- frozen, the sole source of TaskCondition
# ---------------------------------------------------------------------------

@dataclass
class ManifestTask:
    task_id: str = ""
    statement: str = ""
    task_type: str = ""
    involved_apps: List[str] = field(default_factory=list)
    cross_app: bool = False
    horizon: str = "short"
    task_state_effect: str = "read_only"
    branching: str = "none"
    evaluator_type: str = "goal_state_match"
    collateral_risk: str = "low"
    split: str = "pilot"
    world_variant: str = "default"
    evaluator: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskManifest:
    manifest_version: str = MANIFEST_SCHEMA_VERSION
    tasks: List[ManifestTask] = field(default_factory=list)

    def get(self, task_id: str) -> Optional[ManifestTask]:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None


# ---------------------------------------------------------------------------
# Minimal stratification unit (14.2.7): u = (node_pattern, control_role,
# state_effect, verifier_type)
# ---------------------------------------------------------------------------

def stratification_unit(descriptor: SpanDescriptor) -> Dict[str, Any]:
    return {
        "node_pattern": descriptor.node_pattern,
        "control_role": descriptor.control_role,
        "state_effect": sorted(descriptor.mechanism_tags.state_effect),
        "verifier_type": sorted(descriptor.verifier_type),
    }


def unit_key(descriptor: SpanDescriptor) -> str:
    return json.dumps(stratification_unit(descriptor), sort_keys=True, ensure_ascii=False)


def serialize_descriptor(descriptor: SpanDescriptor) -> str:
    """Canonical, order-insensitive key of the full intrinsic descriptor."""
    return canonical_json(descriptor)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_descriptor_values(descriptor: SpanDescriptor) -> List[str]:
    """Soft validation against the canonical vocabularies; warnings only."""
    warnings: List[str] = []
    if descriptor.node_pattern not in NODE_PATTERNS:
        warnings.append("unknown node_pattern: %s" % descriptor.node_pattern)
    if descriptor.control_role not in CONTROL_ROLES:
        warnings.append("unknown control_role: %s" % descriptor.control_role)
    for tag in descriptor.mechanism_tags.interface:
        if tag not in INTERFACES:
            warnings.append("unknown interface tag: %s" % tag)
    for tag in descriptor.mechanism_tags.context_source:
        if tag not in CONTEXT_SOURCES:
            warnings.append("unknown context_source tag: %s" % tag)
    for tag in descriptor.mechanism_tags.output_contract:
        if tag not in OUTPUT_CONTRACTS:
            warnings.append("unknown output_contract tag: %s" % tag)
    for tag in descriptor.mechanism_tags.state_effect:
        if tag not in STATE_EFFECTS:
            warnings.append("unknown state_effect tag: %s" % tag)
    for tag in descriptor.verifier_type:
        if tag not in VERIFIER_TYPES:
            warnings.append("unknown verifier_type tag: %s" % tag)
    if descriptor.graph_context.control_phase not in CONTROL_PHASES:
        warnings.append("unknown control_phase: %s" % descriptor.graph_context.control_phase)
    return warnings


def assert_no_label_leakage(descriptor: SpanDescriptor) -> List[str]:
    """Guard: descriptor must not carry outcome / model fields (14.2.5)."""
    d = to_dict(descriptor)
    keys = set(d.keys())
    for f in dataclasses.fields(SpanDescriptor):
        sub = d.get(f.name)
        if isinstance(sub, dict):
            keys |= set(sub.keys())
    leaks = sorted(keys & FORBIDDEN_DESCRIPTOR_FIELDS)
    return leaks


def state_hash(state: Any) -> str:
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()
