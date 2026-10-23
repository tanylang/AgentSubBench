"""Span extraction from native traces (spec 14.1-14.2).

For every LM invocation in a baseline run, build the basic measurement unit
s = (H, t, r, j) with:
  - the three-layer conditioning (span descriptor / task condition / agent
    condition, 14.2.1)
  - the full replay input I_s (14.3)
  - an eligibility verdict against the six conditions of 14.1

Baseline-failed runs are kept in the census (failure analysis) but flagged so
they never enter the conditional offloadability denominator (14.1).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .schemas import (
    SpanInstance, TraceRun, TaskManifest, SpanRecord, ReplayInput,
    stratification_unit, unit_key, assert_no_label_leakage, to_dict,
)


def _span_id(trace: TraceRun, span: SpanRecord) -> str:
    return "%s__%s__%s__%d" % (
        trace.run_meta.agent_id, trace.run_meta.task_id,
        trace.run_meta.run_id, span.invocation_index,
    )


def check_eligibility(trace: TraceRun, span: SpanRecord) -> List[str]:
    """The six eligibility conditions of 14.1.  Empty list == eligible."""
    reasons: List[str] = []
    meta = trace.run_meta

    # 5. baseline run passed the AppWorld final evaluator
    if not meta.final_evaluator_passed:
        reasons.append("baseline_run_failed_final_evaluator")
    if not meta.goal_state_passed:
        reasons.append("baseline_goal_state_failed")

    # 4. baseline span parseable and resumable
    if not span.baseline_response.parse_success:
        reasons.append("baseline_span_not_parseable")
    if span.baseline_response.parsed is None:
        reasons.append("baseline_span_not_resumable")

    # 6. source node -> normalized logical span mapping valid
    if not span.node_id or span.node_id not in (trace.node_ids or []):
        reasons.append("source_node_mapping_invalid")

    # 1. pre-span agent/environment state recoverable
    snap = span.state_snapshot
    if not (snap.agent_state_ref and snap.environment_snapshot_ref and snap.application_state_hash):
        reasons.append("state_not_recoverable")
    else:
        states = trace.artifacts.get("state_snapshots") or {}
        if snap.agent_state_ref not in states or snap.environment_snapshot_ref not in states:
            reasons.append("state_snapshot_ref_unresolvable")
        else:
            env = states[snap.environment_snapshot_ref]
            if isinstance(env, dict) and env.get("checkpointed") is False:
                # exporter did not checkpoint the live environment (e.g. no
                # AppWorld simulator snapshot); downstream continuation cannot
                # be re-executed against a real environment
                reasons.append("environment_state_not_checkpointed")
            elif isinstance(env, dict) \
                    and env.get("application_state_hash_is_placeholder"):
                # the collector exported an empty check_point shell: the hash is the EMPTY_SHA256
                # placeholder with no env_changes deltas -> on restore the world stays at the initial
                # state, the real pre-span state is unverifiable (the oai_qwen batch measured
                # all-placeholder). Per plan §4.4 such spans stay out of the offloadability denominator
                reasons.append("environment_state_not_checkpointed: "
                               "placeholder hash (env deltas not exported)")
            agent_state = states[snap.agent_state_ref]
            if isinstance(agent_state, dict) and \
                    agent_state.get("has_full_conversation") is False:
                # only the tail window was recorded; the agent's full state
                # (framework memory) cannot be restored for continuation
                reasons.append("agent_state_not_checkpointed: conversation tail window only")

    # 2. LM request, tool schemas and output contract complete
    messages = trace.artifacts.get("messages") or {}
    if not span.request.messages_ref or span.request.messages_ref not in messages or not messages[span.request.messages_ref]:
        reasons.append("lm_request_incomplete")
    else:
        recorded = messages[span.request.messages_ref]
        has_conversation = any(isinstance(m, dict) and m.get("role") in ("user", "assistant")
                               for m in recorded)
        if not has_conversation:
            # e.g. the AutoGen exporter records only a placeholder system
            # message -- the exact LM request is not reconstructable
            reasons.append("lm_request_incomplete: no conversation messages "
                           "(placeholder system-only request)")
    if span.request.system_prompt_ref and span.request.system_prompt_ref not in (trace.artifacts.get("system_prompts") or {}):
        reasons.append("system_prompt_incomplete")
    if span.request.tool_schemas_ref and span.request.tool_schemas_ref not in (trace.artifacts.get("tool_schemas") or {}):
        reasons.append("tool_schemas_incomplete")
    if span.output_contract_runtime.contract_type == "json_schema" and span.output_contract_runtime.schema_ref \
            and span.output_contract_runtime.schema_ref not in (trace.artifacts.get("schemas") or {}):
        reasons.append("output_contract_incomplete")

    # 3. post-span workflow continuation point locatable
    cont = span.continuation
    if not cont.resume_node_id:
        reasons.append("continuation_point_missing")
    elif cont.resume_node_id not in (trace.node_ids or []) and cont.resume_node_id != "final_evaluator":
        reasons.append("continuation_point_not_locatable")

    return reasons


def _build_replay_input(trace: TraceRun, span: SpanRecord) -> ReplayInput:
    return ReplayInput(
        lm_request=span.request,
        state_snapshot=span.state_snapshot,
        span_descriptor=span.span_descriptor,
        task_condition=trace.task_condition,
        agent_condition=trace.agent_condition,
        output_contract_runtime=span.output_contract_runtime,
        verifier_runtime=span.verifier_runtime,
        continuation=span.continuation,
    )


def extract_span(trace: TraceRun, span: SpanRecord, trace_path: str,
                 manifest: TaskManifest) -> SpanInstance:
    task = manifest.get(trace.run_meta.task_id)
    split = task.split if task else "pilot"
    baseline_failed = (not trace.run_meta.final_evaluator_passed) or (not trace.run_meta.goal_state_passed)
    reasons = check_eligibility(trace, span)

    descriptor = span.span_descriptor
    leaks = assert_no_label_leakage(descriptor)
    if leaks:
        # hard guard: descriptor carrying label fields would poison every
        # downstream conditional estimate
        reasons.append("label_leakage_in_descriptor:" + ",".join(leaks))

    unit = stratification_unit(descriptor)
    resp = span.baseline_response
    return SpanInstance(
        span_id=_span_id(trace, span),
        native_span_id=span.native_span_id,
        trace_path=trace_path,
        agent_id=trace.run_meta.agent_id,
        task_id=trace.run_meta.task_id,
        run_id=trace.run_meta.run_id,
        invocation_index=span.invocation_index,
        node_id=span.node_id,
        span_descriptor=descriptor,
        task_condition=trace.task_condition,
        agent_condition=trace.agent_condition,
        replay_input=_build_replay_input(trace, span),
        stratification_unit=unit,
        unit_key=unit_key(descriptor),
        experiment_split=split,
        eligible=not reasons,
        ineligibility_reasons=reasons,
        baseline_failed=baseline_failed,
        baseline_span_cost=resp.cost,
        baseline_span_latency_ms=resp.latency_ms,
        baseline_tokens=resp.prompt_tokens + resp.completion_tokens,
        meta_hint=span.meta_hint,
    )


def extract_spans_from_trace(trace: TraceRun, trace_path: str,
                             manifest: TaskManifest) -> List[SpanInstance]:
    return [extract_span(trace, span, trace_path, manifest) for span in trace.spans]


def extract_all(traces: List[Tuple[TraceRun, str]], manifest: TaskManifest) -> List[SpanInstance]:
    out: List[SpanInstance] = []
    for trace, path in traces:
        out.extend(extract_spans_from_trace(trace, path, manifest))
    return out


def census(instances: List[SpanInstance]) -> Dict:
    """Aggregate extraction statistics for the span census section."""
    total = len(instances)
    eligible = [s for s in instances if s.eligible]
    ineligible = [s for s in instances if not s.eligible]
    baseline_failed = [s for s in instances if s.baseline_failed]

    reason_counts: Dict[str, int] = {}
    for s in ineligible:
        for r in s.ineligibility_reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    node_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    effect_counts: Dict[str, int] = {}
    verifier_counts: Dict[str, int] = {}
    for s in instances:
        d = s.span_descriptor
        node_counts[d.node_pattern] = node_counts.get(d.node_pattern, 0) + 1
        role_counts[d.control_role] = role_counts.get(d.control_role, 0) + 1
        for tag in d.mechanism_tags.state_effect:
            effect_counts[tag] = effect_counts.get(tag, 0) + 1
        for tag in d.verifier_type:
            verifier_counts[tag] = verifier_counts.get(tag, 0) + 1

    return {
        "total_spans": total,
        "eligible_spans": len(eligible),
        "ineligible_spans": len(ineligible),
        "baseline_failed_spans": len(baseline_failed),
        "ineligibility_reason_counts": reason_counts,
        "node_pattern_counts": node_counts,
        "control_role_counts": role_counts,
        "state_effect_counts": effect_counts,
        "verifier_type_counts": verifier_counts,
    }


def save_spans(instances: List[SpanInstance], path: str) -> str:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(__import__("json").dumps(to_dict(inst), ensure_ascii=False) + "\n")
    return path


def load_spans(path: str) -> List[dict]:
    import json
    spans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans
