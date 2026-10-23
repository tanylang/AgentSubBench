"""LocoMo trace adapter: trace JSON collected by trace_record -> canonical TraceRun
(same structure as the AppWorld/SWE adapters, reusing their splice/verifier components).

Trace structure and key semantics (trace_record project, adapter/locomot.py +
scripts/locomot_common.py):
  - spans include invoke_agent (root) / llm.call (CLIENT) / tool.execution (INTERNAL),
    strictly increasing by attributes["agent.step"]; the tools triggered by llm j =
    the tool spans with step in (llm[j].step, llm[j+1].step)
  - purely read-only baseline: check_point.env_changes is always an empty dict; the only
    mutable state = the memory read set (read_dia_ids), application_state_hash = the sha256
    chained cumulative hash **before** this span starts; the chain-start e3b0c442... is a legal empty state (not a placeholder!)
  - llm.call's check_point.conversation = {has_full_conversation,
    messages (including system + the "Question i/n:" user message at the start of each
    question), output_text, tool_calls[{id,name,arguments (JSON string)}]} — the
    assistant's tool_calls structure is spliced back by conv.tool_calls' real call ids
  - an independent message list per question (a question-boundary span's raw list resets to [system,user]); a freeform span
    (no tool_calls) output is that question's answer (legal, not a FormatError)
  - tool span key name: gen_ai.tool.name (note: not swebench's call.name)
  - top-level locomo = {sample_id, answers:[{idx,question,pred,...}], read_ids,
    num_questions, max_tool_steps}; pass=="success" is the official F1 convention
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .appworld_adapter import (  # reuse: splice and verifier components, zero changes
    _link_tool_messages, _native_tool_calls, span_verifier,
)
from .harness.locomo_env import (
    LOC_TOOL_NAMES, OAI_TOOL_SCHEMAS, QUESTION_RE,
)
from .schemas import (
    AgentCondition, BaselineResponse, Continuation, GraphContext, LMRequest,
    MechanismTags, RunMeta, SpanDescriptor, SpanRecord, StateSnapshot,
    TaskCondition, ToolExecution, TraceRun, OutputContractRuntime,
)

ADAPTER_NAME = "locomo"
AGENT_ID = "LocoMo-Simplified"

#: Output contract schema: tool_call_list (4 read-only retrieval tools)
LOC_CALL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["tool_calls"],
    "properties": {"tool_calls": {"type": "array", "minItems": 1}},
}

#: Collection-side per-question tool call cap (locomot_common.MAX_TOOL_STEPS default)
DEFAULT_MAX_TOOL_STEPS = 10


def _parse_baseline_output(conv: Dict[str, Any]
                           ) -> Tuple[Any, str, bool]:
    """Restore the baseline output into an injectable parsed form.
    - has tool_calls -> {"tool_calls": [{name, arguments}]} (tool_call_list)
    - no tool_calls -> {"text": output_text} with parse_success=True
      (freeform text is that question's answer, a legal output — unlike swebench's FormatError semantics)
    """
    calls = conv.get("tool_calls") or []
    if calls:
        parsed = {"tool_calls": [
            {"name": str(c.get("name", "")),
             "arguments": json.loads(c["arguments"])
             if isinstance(c.get("arguments"), str) else (c.get("arguments") or {})}
            for c in calls]}
        return parsed, "tool_call_list", True
    return ({"text": str(conv.get("output_text", "") or "")},
            "freeform_text", True)


def _classify_locomo_span(has_calls: bool, first_of_question: bool,
                          prev_tool_failed: bool
                          ) -> Tuple[SpanDescriptor, str]:
    """LocoMo span classification: retrieval tool calls vs final answer. Returns (descriptor, phase)."""
    if not has_calls:
        d = SpanDescriptor(
            node_pattern="final_synthesis", control_role="terminal",
            span_subtask_type="locomo_answer",
            mechanism_tags=MechanismTags(
                interface=["api_tool"],
                context_source=["conversation_history", "memory"],
                output_contract=["freeform_text"],
                state_effect=["read_only"],
            ),
            verifier_type=["goal_state_match"],
            graph_context=GraphContext(control_phase="terminal",
                                       after_failure=False,
                                       inside_loop=False),
        )
        return d, "terminal"
    phase = "initial" if first_of_question else "execution"
    d = SpanDescriptor(
        node_pattern="tool_argument_generation", control_role="local_executor",
        span_subtask_type="locomo_retrieval",
        mechanism_tags=MechanismTags(
            interface=["api_tool"],
            context_source=["conversation_history", "tool_schema",
                            "previous_tool_results", "memory"],
            output_contract=["tool_name", "tool_arguments", "json_schema"],
            state_effect=["read_only"],
        ),
        verifier_type=["schema_check"],
        graph_context=GraphContext(control_phase=phase,
                                   after_failure=prev_tool_failed,
                                   inside_loop=True),
    )
    return d, phase


class LocomoTraceAdapter:
    """trace_record's LocoMo trace JSON -> canonical TraceRun."""

    name = ADAPTER_NAME

    def __init__(self, manifest=None):
        self.manifest = manifest

    # ------------------------------------------------------------------
    def load(self, path: str) -> TraceRun:
        path = str(path).replace("\\", "/")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loco = data.get("locomo") or {}
        sample_id = loco.get("sample_id", "unknown")
        agent_name = data.get("agent_name", AGENT_ID)
        num_questions = int(loco.get("num_questions", 0) or 0)
        max_tool_steps = int(loco.get("max_tool_steps", 0)
                             or DEFAULT_MAX_TOOL_STEPS)

        # Sort spans by step (root span step=None sorts first)
        spans = sorted(
            (s for s in data.get("spans", []) if s.get("type") != "invoke_agent"),
            key=lambda s: int(s.get("attributes", {}).get("agent.step", 0) or 0))
        root = next((s for s in data.get("spans", [])
                     if s.get("type") == "invoke_agent"), {})
        root_attrs = root.get("attributes", {})
        llm_steps = [int(s["attributes"]["agent.step"]) for s in spans
                     if s.get("type") == "llm.call"]
        tool_by_step = {int(s["attributes"]["agent.step"]): s for s in spans
                        if s.get("type") == "tool.execution"}
        llm_spans = [s for s in spans if s.get("type") == "llm.call"]

        # span_log: one entry per non-root span (tools included), in step order — the runtime replay basis
        span_log: List[Dict[str, Any]] = []
        for s in spans:
            cp = s.get("check_point") or {}
            entry: Dict[str, Any] = {
                "step": int(s["attributes"].get("agent.step", 0) or 0),
                "type": s.get("type"),
                "span_id": s.get("span_id"),
                "hash_before": str(cp.get("application_state_hash", "") or ""),
                "env_changes": cp.get("env_changes") or {},
            }
            if s.get("type") == "tool.execution":
                ta = s.get("attributes", {})
                raw_args = ta.get("gen_ai.tool.call.arguments", "{}") or "{}"
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except ValueError:
                        args = {}
                else:
                    args = raw_args
                entry["tool"] = {
                    "name": str(ta.get("gen_ai.tool.name", "")),
                    "arguments": args,
                    "ok": str(ta.get("tool.success", "")).lower() == "true",
                    "result": ta.get("gen_ai.tool.call.result", ""),
                }
            span_log.append(entry)

        # The conversation of each llm j: splice the native tool_calls structure back in
        convs = []
        for s in llm_spans:
            cp = s.get("check_point") or {}
            conv = cp.get("conversation") or {}
            convs.append({
                "messages": conv.get("messages", []),
                "output_text": conv.get("output_text", ""),
                "tool_calls": conv.get("tool_calls", []),
            })

        artifacts: Dict[str, Any] = {
            "messages": {}, "system_prompts": {}, "recorded_tools": {},
            "tool_schemas": {}, "schemas": {}, "state_snapshots": {},
            "span_tool_results": {}, "previous_tool_results": {},
            "span_log": span_log,
            "locomo": loco,
        }
        node_ids: List[str] = []

        records: List[SpanRecord] = []
        prev_tool_failed = False
        prev_spliced: Optional[List[Dict[str, Any]]] = None
        carry_q_idx: Optional[int] = None
        carry_num_q: int = num_questions
        for j, s in enumerate(llm_spans):
            attrs = s.get("attributes", {})
            conv = convs[j]
            step = int(attrs.get("agent.step", 0) or 0)
            nxt = llm_steps[j + 1] if j + 1 < len(llm_steps) else None
            attached = [tool_by_step[st] for st in sorted(tool_by_step)
                        if st > step and (nxt is None or st < nxt)]

            # --- conversation splice: repair the last turn's assistant/tool_call_id ---
            # question-boundary spans (raw reset to [system,user]) naturally fall into the degraded branch
            raw_msgs = [dict(m) for m in conv["messages"]]
            calls = conv["tool_calls"]
            degraded = False
            if j == 0:
                spliced = raw_msgs
            else:
                base_len = len(convs[j - 1]["messages"])
                tail = raw_msgs[base_len:]
                prev_calls = convs[j - 1]["tool_calls"]
                if (prev_calls and len(tail) == 1 + len(prev_calls)
                        and tail[0].get("role") == "assistant"):
                    native = _native_tool_calls(prev_calls, id_prefix="loc")
                    assistant = {"role": "assistant", "content": None,
                                 "tool_calls": native}
                    fixed = [assistant] + [dict(m) for m in tail[1:]]
                    _link_tool_messages(fixed, 1, native)
                    spliced = list(prev_spliced) + fixed
                else:
                    spliced = raw_msgs
                    degraded = True
            prev_spliced = spliced

            # --- question boundary: the first user message's "Question i/n:" ---
            q_idx = carry_q_idx
            num_q = carry_num_q
            for m in spliced:
                if m.get("role") != "user":
                    continue
                mm = QUESTION_RE.search(str(m.get("content", "") or ""))
                if mm:
                    q_idx = int(mm.group(1)) - 1
                    num_q = int(mm.group(2))
                    carry_q_idx = q_idx
                    carry_num_q = num_q
                    break

            parsed, contract_kind, parse_success = _parse_baseline_output(
                {"output_text": conv["output_text"], "tool_calls": calls})

            has_calls = bool(calls)
            first_of_question = (j == 0) or (prev_q_idx != q_idx)
            descriptor, phase = _classify_locomo_span(
                has_calls, bool(first_of_question), prev_tool_failed)
            prev_q_idx = q_idx

            # Attached tools: record results (for the equivalence check) and update prev_tool_failed
            tool_exec: Optional[ToolExecution] = None
            span_tool_results: List[Dict[str, Any]] = []
            for k, t in enumerate(attached):
                ta = t.get("attributes", {})
                ok = str(ta.get("tool.success", "")).lower() == "true"
                prev_tool_failed = prev_tool_failed or not ok
                raw_args = ta.get("gen_ai.tool.call.arguments", "{}") or "{}"
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except ValueError:
                        args = {}
                else:
                    args = raw_args
                name = str(ta.get("gen_ai.tool.name", ""))
                span_tool_results.append({
                    "tool": name, "arguments": args, "ok": ok,
                    "result": ta.get("gen_ai.tool.call.result", ""),
                })
                if tool_exec is None:
                    tool_exec = ToolExecution(
                        tool_name=name,
                        arguments=args,
                        success=ok,
                        result=span_tool_results[-1]["result"],
                        node_id="tool:%d:%d" % (j, k),
                    )

            node_ids.append("lm:%d" % j)
            for k in range(len(attached)):
                node_ids.append("tool:%d:%d" % (j, k))

            ref = "env:%d" % j
            artifacts["state_snapshots"][ref] = {
                "checkpointed": True,
                # Note: no application_state_hash_is_placeholder — e3b0c442... is
                # locomo's legal chain start for an empty read set, not a placeholder
                "application_state_hash": str(
                    (s.get("check_point") or {}).get(
                        "application_state_hash", "") or ""),
                "step": step,
            }
            artifacts["state_snapshots"]["agent:%d" % j] = {
                "has_full_conversation": True, "step": step,
                "q_idx": q_idx, "num_questions": num_q,
            }
            artifacts["messages"]["msg:%d" % j] = spliced
            artifacts["system_prompts"]["sys:%d" % j] = next(
                (m["content"] for m in spliced if m.get("role") == "system"),
                "")
            recorded_tools = None
            raw_tools = attrs.get("gen_ai.request.tools", "")
            if isinstance(raw_tools, str) and raw_tools.strip():
                try:
                    recorded_tools = json.loads(raw_tools)
                except ValueError:
                    recorded_tools = None
            artifacts["recorded_tools"]["tools:%d" % j] = (
                recorded_tools or list(OAI_TOOL_SCHEMAS))
            artifacts["tool_schemas"]["tools:%d" % j] = {
                s["function"]["name"]: s for s in OAI_TOOL_SCHEMAS}
            artifacts["schemas"]["schema:%d" % j] = (
                LOC_CALL_SCHEMA if contract_kind == "tool_call_list"
                else {"type": "object"})
            artifacts["span_tool_results"]["span:%d" % j] = span_tool_results
            artifacts["previous_tool_results"]["prev:%d" % j] = span_tool_results

            records.append(SpanRecord(
                invocation_index=j,
                native_span_id=str(s.get("span_id", "")),
                node_id="lm:%d" % j,
                span_descriptor=descriptor,
                request=LMRequest(
                    messages_ref="msg:%d" % j,
                    system_prompt_ref="sys:%d" % j,
                    tool_schemas_ref="tools:%d" % j,
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
                state_snapshot=StateSnapshot(
                    agent_state_ref="agent:%d" % j,
                    environment_snapshot_ref=ref,
                    application_state_hash=str(
                        (s.get("check_point") or {}).get(
                            "application_state_hash", "") or ""),
                ),
                output_contract_runtime=OutputContractRuntime(
                    contract_type=contract_kind,
                    schema_ref="schema:%d" % j,
                    allowed_tools=list(LOC_TOOL_NAMES),
                    semantic_constraints={
                        "source_step": step,
                        "n_tool_calls": len(calls),
                        "splice_degraded": degraded,
                        "q_idx": q_idx,
                        "num_questions": num_q,
                    },
                ),
                verifier_runtime=span_verifier(descriptor.verifier_type),
                continuation=Continuation(
                    resume_node_id=("tool:%d:0" % j) if attached else (
                        "lm:%d" % (j + 1) if j + 1 < len(llm_spans)
                        else "final_evaluator"),
                    max_remaining_steps=max(
                        1, (num_q - (q_idx or 0)) * (max_tool_steps + 1)),
                    evaluator_ref="%s#locomo_qa" % sample_id,
                ),
                baseline_response=BaselineResponse(
                    raw=str(conv["output_text"] or ""),
                    parsed=parsed,
                    parse_success=parse_success,
                    prompt_tokens=int(attrs.get("gen_ai.usage.input_tokens", 0)
                                      or 0),
                    completion_tokens=int(
                        attrs.get("gen_ai.usage.output_tokens", 0) or 0),
                ),
                tool_execution=tool_exec,
                meta_hint="%s|q%d|%d|step=%d|%s" % (
                    sample_id, (q_idx or 0) + 1, j, step, phase),
            ))

        passed = (data.get("pass") == "success")
        pass_details = data.get("pass_details") or {}
        run_meta = RunMeta(
            agent_id=agent_name,
            task_id=sample_id,
            run_id=str(data.get("trace_id", "")),
            baseline_model=str(root_attrs.get("model", "") or ""),
            created_at=str(data.get("start_time", "") or ""),
            final_evaluator_passed=passed,
            goal_state_passed=passed,
            collateral_damage=False,
            total_cost=0.0,
            total_latency_ms=float(root_attrs.get("agent.wall_time_ms", 0) or 0),
            notes="f1=%s questions=%d" % (
                str(pass_details.get("f1", "")), num_questions),
        )
        task_condition = TaskCondition()
        if self.manifest is not None:
            mt = self.manifest.get(sample_id)
            if mt is not None:
                task_condition = TaskCondition(
                    task_type=mt.task_type, involved_apps=mt.involved_apps,
                    cross_app=mt.cross_app, horizon=mt.horizon,
                    task_state_effect=mt.task_state_effect,
                    branching=mt.branching,
                    evaluator_type=mt.evaluator_type,
                    collateral_risk=mt.collateral_risk,
                )
        return TraceRun(
            run_meta=run_meta,
            task_condition=task_condition,
            agent_condition=AgentCondition(
                representative_id=agent_name,
                # Derived from the agent identity (e.g. LocoMo-Simplified -> locomo-simplified)
                source_cluster_id="locomo-" + str(agent_name).split("-", 1)[-1].lower(),
                architecture_annotation=(
                    "%s: 4 read-only retrieval tools "
                    "(read-set hash chain)" % agent_name),
            ),
            artifacts=artifacts,
            node_ids=node_ids,
            spans=records,
        )
