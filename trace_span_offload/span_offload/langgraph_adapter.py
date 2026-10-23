"""LoCoMo + LangGraph ReAct + LangMem trace adapter: trace JSON -> canonical
TraceRun. **Offload unit = one unit trace per QA** (user decision 2026-09-21).

Trace structure and key semantics (trace_record langgraph_locomo, locomo_trace_core.py +
10_run_locomo_sample.py):
  - spans include invoke_agent (root) / llm.call (CLIENT) / tool.execution (INTERNAL),
    strictly increasing by attributes["agent.step"]; the trace has two phases: ingestion
    (memory intake, including search_memory queries and manage_memory writes) + qa (Q&A)
  - **offload target = QA-phase spans only** (user decision 2026-09-21); a unit trace =
    that QA's spans. The ingestion deltas are kept in memory_delta_log, restore replays
    the full chain from an empty store -> the QA span's context = full conversation memory
    (complete message list + memory = the state after replaying the full prefix)
  - the only mutable state = the LangMem memory record set; application_state_hash =
    sha256(sorted [{namespace,key,value}]); env_changes: always [] for llm, tool =
    memory deltas; the tool span's check_point is recorded after execute (execute first,
    then check); chain start = empty-store hash
  - conversation (llm spans only): {has_full_conversation, messages (complete, no
    splicing needed), output_text, output_reasoning, tool_calls}
  - top-level pass/pass_details are written by the collection-side finalize; missing
    (None) is treated as baseline_failed (enters the denominator via --relax-baseline-pass)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .appworld_adapter import span_verifier
from .harness.langgraph_env import SYSTEM_PROMPT
from .schemas import (
    AgentCondition, BaselineResponse, Continuation, GraphContext, LMRequest,
    MechanismTags, RunMeta, SpanDescriptor, SpanRecord, StateSnapshot,
    TaskCondition, ToolExecution, TraceRun, OutputContractRuntime,
)

ADAPTER_NAME = "langgraph"
AGENT_ID = "LoCoMo-LangGraph-ReAct-LangMem"
AGENT_DIR = "langgraph"

#: Max recursion depth per QA question (collection-side recursion_limit)
RECURSION_LIMIT = 12

LANGGRAPH_CALL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["tool_calls"],
    "properties": {"tool_calls": {"type": "array", "minItems": 1}},
}

LANGGRAPH_TOOL_NAMES = ("manage_memory", "search_memory")

#: langmem tool schema fallback (when gen_ai.request.tools is missing)
LANGGRAPH_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "manage_memory",
        "description": "Create, update, or delete a memory in a persistent "
                       "knowledge base for future reference.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string",
                       "enum": ["create", "update", "delete"],
                       "description": "The action to perform"},
            "key": {"type": "string", "description": "Memory key"},
            "content": {"type": "string", "description": "Memory content"},
            "category": {"type": "string",
                         "description": "Category like 'fact', 'event', "
                                        "'preference', 'relationship', "
                                        "'plan', or 'dated'"},
        }, "required": ["action", "key", "content", "category"]}}},
    {"type": "function", "function": {
        "name": "search_memory",
        "description": "Search the knowledge base for relevant memories "
                       "using semantic similarity.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
        }, "required": ["query"]}}},
]


def _parse_baseline_output(conv: Dict[str, Any]
                           ) -> Tuple[Any, str, bool]:
    """Restore the baseline output into an injectable parsed form (freeform text is a legal answer/ACK)."""
    calls = conv.get("tool_calls") or []
    if calls:
        parsed = {"tool_calls": [
            {"name": str(c.get("name", "")),
             "arguments": json.loads(c["args"])
             if isinstance(c.get("args"), str) else (c.get("args") or {})}
            for c in calls]}
        return parsed, "tool_call_list", True
    return ({"text": str(conv.get("output_text", "") or "")},
            "freeform_text", True)


def _classify_span(has_calls: bool, phase: str, tool_names: List[str],
                   first_of_question: bool, prev_tool_failed: bool
                   ) -> Tuple[SpanDescriptor, str]:
    """LangGraph span classification: ingestion (memory write/query/ACK) and QA (tools/answer)."""
    if phase == "qa" and not has_calls:
        d = SpanDescriptor(
            node_pattern="final_synthesis", control_role="terminal",
            span_subtask_type="langgraph_qa_answer",
            mechanism_tags=MechanismTags(
                interface=["api_tool"],
                context_source=["conversation_history", "memory",
                                "retrieved_docs"],
                output_contract=["freeform_text"],
                state_effect=["read_only"],
            ),
            verifier_type=["goal_state_match"],
            graph_context=GraphContext(control_phase="terminal",
                                       after_failure=False,
                                       inside_loop=False),
        )
        return d, "terminal"
    if phase == "ingestion" and not has_calls:
        d = SpanDescriptor(
            node_pattern="other", control_role="worker",
            span_subtask_type="langgraph_ingestion_ack",
            mechanism_tags=MechanismTags(
                interface=["api_tool"],
                context_source=["conversation_history", "memory"],
                output_contract=["freeform_text"],
                state_effect=["read_only"],
            ),
            verifier_type=["none"],
            graph_context=GraphContext(control_phase="terminal",
                                       after_failure=False,
                                       inside_loop=False),
        )
        return d, "terminal"
    # Tool span: distinguish the state effect by tool name (manage_memory writes / search read-only)
    state_effect = ("memory_write" if any(
        n == "manage_memory" for n in tool_names) else "read_only")
    phase_hint = "initial" if first_of_question else "execution"
    d = SpanDescriptor(
        node_pattern="tool_argument_generation", control_role="local_executor",
        span_subtask_type="langgraph_memory_operation",
        mechanism_tags=MechanismTags(
            interface=["api_tool"],
            context_source=["conversation_history", "tool_schema",
                            "previous_tool_results", "memory"],
            output_contract=["tool_name", "tool_arguments", "json_schema"],
            state_effect=[state_effect],
        ),
        verifier_type=["schema_check"],
        graph_context=GraphContext(control_phase=phase_hint,
                                   after_failure=prev_tool_failed,
                                   inside_loop=True),
    )
    return d, phase_hint


class LangGraphTraceAdapter:
    """langgraph_locomo trace -> one unit TraceRun per QA.

    load(path) returns the first unit (q0; a single-question smoke trace is the whole
    trace); iter_units(path) yields one per QA; load_unit(path, task_id) fetches on demand.
    """

    name = ADAPTER_NAME

    def __init__(self, manifest=None):
        self.manifest = manifest
        # Cache only the parse result (raw JSON read once); unit TraceRuns are never cached —
        # units carry full message copies, keeping them all during batch extraction would blow
        # up memory (measured: full-batch extraction >50 min unfinished). Callers (runner) cache units on demand.
        self._parse_cache: Dict[str, Dict[str, Any]] = {}
        self._unit_cache: Dict[Tuple[str, str], TraceRun] = {}

    # ------------------------------------------------------------------
    def _parse(self, path: str) -> Dict[str, Any]:
        path = str(path).replace("\\", "/")
        if path in self._parse_cache:
            return self._parse_cache[path]
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agent_name = data.get("agent_name", AGENT_ID)
        root = next((s for s in data.get("spans", [])
                     if s.get("type") == "invoke_agent"), {})
        root_attrs = root.get("attributes", {})
        sample_id = str(root_attrs.get("locomo.sample_id", "") or "")
        if not sample_id:
            sample_id = str(data.get("task", "")).replace(
                "LoCoMo sample ", "")
        recorded_answers = root_attrs.get("locomo.answers") or []
        qa_count = int(root_attrs.get("locomo.qa_count", 0) or 0)
        if not qa_count and not recorded_answers:
            qa_count = 1 if any(
                ((s.get("check_point") or {}).get("locomo") or {}).get("phase")
                == "qa" for s in data.get("spans", [])
                if s.get("type") == "llm.call") else 0

        spans = sorted(
            (s for s in data.get("spans", []) if s.get("type") != "invoke_agent"),
            key=lambda s: int(s.get("attributes", {}).get("agent.step", 0) or 0))

        delta_log: List[Dict[str, Any]] = []
        for s in spans:
            cp = s.get("check_point") or {}
            entry: Dict[str, Any] = {
                "step": int(s["attributes"].get("agent.step", 0) or 0),
                "type": s.get("type"),
                "span_id": s.get("span_id"),
                "hash_before": str(cp.get("application_state_hash", "") or ""),
                "env_changes": cp.get("env_changes") or [],
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
            delta_log.append(entry)

        def _phase_of(s: Dict[str, Any]) -> str:
            cp = s.get("check_point") or {}
            return str((cp.get("locomo") or {}).get("phase")
                       or s.get("attributes", {}).get("locomo.phase") or "")

        ingestion_spans = [s for s in spans if _phase_of(s) == "ingestion"]
        qa_spans = [s for s in spans if _phase_of(s) == "qa"]
        # Group by the global question_index (QA spans missing it go to 0)
        qa_groups: Dict[int, List[Dict[str, Any]]] = {}
        for s in qa_spans:
            cp = s.get("check_point") or {}
            qi = (cp.get("locomo") or {}).get("question_index")
            qi = 0 if qi is None else int(qi)
            qa_groups.setdefault(qi, []).append(s)

        # gold question entries (manifest qa prefix; collection answers all in order)
        gold_questions: List[str] = []
        gold_qa: List[Dict[str, Any]] = []
        if self.manifest is not None:
            mt = self.manifest.get(sample_id)
            if mt is not None:
                gold_qa = list(mt.evaluator.get("qa") or [])
                gold_questions = [str(q.get("question", "")) for q in gold_qa]

        parse = {
            "path": path, "data": data, "root_attrs": root_attrs,
            "sample_id": sample_id, "agent_name": agent_name,
            "recorded_answers": recorded_answers, "qa_count": qa_count,
            "delta_log": delta_log, "ingestion_spans": ingestion_spans,
            "qa_groups": qa_groups, "gold_questions": gold_questions,
            "gold_qa": gold_qa,
        }
        self._parse_cache[path] = parse
        return parse

    # ------------------------------------------------------------------
    def iter_units(self, path: str):
        parse = self._parse(path)
        qa_indexes = sorted(parse["qa_groups"].keys())
        if not qa_indexes:
            qa_indexes = [0]
        for qi in qa_indexes:
            yield "locomo:%s:q%03d" % (parse["sample_id"], qi), \
                self._build_unit(parse, qi)

    def load(self, path: str) -> TraceRun:
        for task_id, tr in self.iter_units(path):
            return tr
        raise ValueError("no units in trace %s" % path)

    def load_unit(self, path: str, task_id: str) -> TraceRun:
        key = (str(path).replace("\\", "/"), task_id)
        unit = self._unit_cache.get(key)
        if unit is not None:
            return unit
        parse = self._parse(key[0])
        unit = self._build_unit(parse, int(task_id.rsplit("q", 1)[-1]))
        self._unit_cache[key] = unit
        return unit

    # ------------------------------------------------------------------
    def _build_unit(self, parse: Dict[str, Any], qi: int) -> TraceRun:
        sample_id = parse["sample_id"]
        data = parse["data"]
        root_attrs = parse["root_attrs"]
        recorded_answers = parse["recorded_answers"]
        qa_count = parse["qa_count"] or len(parse["gold_questions"]) or 1
        gold_questions = parse["gold_questions"]
        delta_log = parse["delta_log"]
        task_id = "locomo:%s:q%03d" % (sample_id, qi)

        # Unit span list = only this QA's spans (user decision 2026-09-21:
        # offload only the QA part). The ingestion deltas remain in memory_delta_log,
        # restore replays the full chain from an empty store as usual -> the QA span's
        # context = full conversation memory, unchanged.
        unit_spans: List[Dict[str, Any]] = list(parse["qa_groups"].get(qi, []))
        unit_spans.sort(key=lambda s: int(
            s["attributes"].get("agent.step", 0) or 0))

        def _phase_of(s: Dict[str, Any]) -> str:
            cp = s.get("check_point") or {}
            return str((cp.get("locomo") or {}).get("phase")
                       or s.get("attributes", {}).get("locomo.phase") or "")

        llm_spans = [s for s in unit_spans if s.get("type") == "llm.call"]
        tool_by_step = {int(s["attributes"]["agent.step"]): s for s in unit_spans
                        if s.get("type") == "tool.execution"}
        llm_steps = [int(s["attributes"]["agent.step"]) for s in llm_spans]

        question_text = (gold_questions[qi]
                         if qi < len(gold_questions) else
                         (recorded_answers[qi].get("question", "")
                          if qi < len(recorded_answers) else ""))
        # QA template convention: identified from any recorded question message of this QA (production full / smoke core)
        qa_template = "full"
        for s in parse["qa_groups"].get(qi, []):
            if s.get("type") != "llm.call":
                continue
            conv = (s.get("check_point") or {}).get("conversation") or {}
            for m in conv.get("messages", []):
                if m.get("role") == "user":
                    from .harness.langgraph_env import detect_qa_template
                    qa_template = detect_qa_template(
                        str(m.get("content", "") or ""))
                    break
            if qa_template:
                break
        # The collection-side full verdict row for this question (reused directly when the
        # replayed answer is verbatim identical, avoiding convention drift from a second judge)
        per_q = ((data.get("pass_details") or {}).get("per_question") or [])
        collector_row = per_q[qi] if qi < len(per_q) else {}

        artifacts: Dict[str, Any] = {
            "messages": {}, "system_prompts": {}, "recorded_tools": {},
            "tool_schemas": {}, "schemas": {}, "state_snapshots": {},
            "span_tool_results": {}, "previous_tool_results": {},
            "memory_delta_log": delta_log,
            "langgraph": {
                "sample_id": sample_id,
                "qa_index": qi,
                "qa_count": qa_count,
                "num_questions": 1,
                "question_text": question_text,
                "qa_template": qa_template,
                "collector_row": collector_row,
                "answers": recorded_answers[qi:qi + 1] if recorded_answers
                else [],
                "model": str(root_attrs.get("model", "") or ""),
            },
        }
        node_ids: List[str] = []

        records: List[SpanRecord] = []
        prev_tool_failed = False
        prev_q_idx: Optional[int] = None
        for j, s in enumerate(llm_spans):
            attrs = s.get("attributes", {})
            cp = s.get("check_point") or {}
            conv = (cp.get("conversation") or {})
            step = int(attrs.get("agent.step", 0) or 0)
            phase = _phase_of(s)
            nxt = llm_steps[j + 1] if j + 1 < len(llm_steps) else None
            attached = [tool_by_step[st] for st in sorted(tool_by_step)
                        if st > step and (nxt is None or st < nxt)]

            # Question index within the unit (QA span = 0; ingestion span also 0, no answer semantics)
            q_idx = 0
            first_of_question = (j == 0) or (
                prev_q_idx is not None and prev_q_idx != q_idx)
            prev_q_idx = q_idx

            parsed, contract_kind, parse_success = _parse_baseline_output(conv)
            has_calls = bool(conv.get("tool_calls") or [])
            tool_names = [str(c.get("name", ""))
                          for c in (conv.get("tool_calls") or [])]
            descriptor, phase_hint = _classify_span(
                has_calls, phase, tool_names, bool(first_of_question),
                prev_tool_failed)

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
                # The empty-store hash is a legal chain start, no placeholder marker
                "application_state_hash": str(
                    cp.get("application_state_hash", "") or ""),
                "step": step,
            }
            agent_snap: Dict[str, Any] = {
                "has_full_conversation": True, "step": step,
                "locomo_phase": phase,
                "q_idx": q_idx, "num_questions": 1, "qa_index": qi,
            }
            # Positioning context for the ingestion round (used for the remaining continuation rounds)
            loco_ctx = cp.get("locomo") or {}
            if phase == "ingestion":
                agent_snap["ingestion_dia_id"] = str(
                    loco_ctx.get("dia_id", "") or "")
                agent_snap["ingestion_session_number"] = loco_ctx.get(
                    "session_number")
                agent_snap["ingestion_turn_index"] = loco_ctx.get("turn_index")
            artifacts["state_snapshots"]["agent:%d" % j] = agent_snap
            artifacts["messages"]["msg:%d" % j] = [
                dict(m) for m in conv.get("messages", [])]
            artifacts["system_prompts"]["sys:%d" % j] = str(
                attrs.get("gen_ai.system", "") or SYSTEM_PROMPT)
            recorded_tools = None
            raw_tools = attrs.get("gen_ai.request.tools", "")
            if isinstance(raw_tools, str) and raw_tools.strip():
                try:
                    recorded_tools = json.loads(raw_tools)
                except ValueError:
                    recorded_tools = None
            artifacts["recorded_tools"]["tools:%d" % j] = (
                recorded_tools or list(LANGGRAPH_TOOL_SCHEMAS))
            artifacts["tool_schemas"]["tools:%d" % j] = {
                s["function"]["name"]: s
                for s in (recorded_tools or LANGGRAPH_TOOL_SCHEMAS)
                if isinstance(s, dict) and "function" in s}
            artifacts["schemas"]["schema:%d" % j] = (
                LANGGRAPH_CALL_SCHEMA if contract_kind == "tool_call_list"
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
                        cp.get("application_state_hash", "") or ""),
                ),
                output_contract_runtime=OutputContractRuntime(
                    contract_type=contract_kind,
                    schema_ref="schema:%d" % j,
                    allowed_tools=list(LANGGRAPH_TOOL_NAMES),
                    semantic_constraints={
                        "source_step": step,
                        "n_tool_calls": len(conv.get("tool_calls") or []),
                        "splice_degraded": False,  # recording is complete, no splicing needed
                        "q_idx": q_idx,
                        "num_questions": 1,
                        "qa_index": qi,
                        "locomo_phase": phase,
                    },
                ),
                verifier_runtime=span_verifier(descriptor.verifier_type),
                continuation=Continuation(
                    resume_node_id=("tool:%d:0" % j) if attached else (
                        "lm:%d" % (j + 1) if j + 1 < len(llm_spans)
                        else "final_evaluator"),
                    max_remaining_steps=max(
                        1, (len(llm_spans) - j) * RECURSION_LIMIT),
                    evaluator_ref="%s#locomo_qa" % task_id,
                ),
                baseline_response=BaselineResponse(
                    raw=str(conv.get("output_text", "") or ""),
                    parsed=parsed,
                    parse_success=parse_success,
                    prompt_tokens=int(attrs.get("gen_ai.usage.input_tokens", 0)
                                      or 0),
                    completion_tokens=int(
                        attrs.get("gen_ai.usage.output_tokens", 0) or 0),
                ),
                tool_execution=tool_exec,
                meta_hint="%s|q%d|%d|step=%d|%s|%s" % (
                    sample_id, qi, j, step, phase, phase_hint),
            ))

        # Unit-level baseline verdict: read the collection-side per-question verdict
        # (pass_details.per_question; convention = collection side computes, this side is read-only).
        # Old formats without per_question fall back to an ad-hoc official F1; empty answers
        # (recursion truncation) are never judged passed; notes marks baseline_answer_empty for the runner to exclude.
        baseline_answer_empty = True
        unit_f1 = None
        pq_pass = ""
        pred = (str(recorded_answers[qi].get("predicted_answer", "") or "")
                if qi < len(recorded_answers) else "")
        baseline_answer_empty = not pred.strip()
        if not baseline_answer_empty and qi < len(per_q):
            pq_pass = str(per_q[qi].get("pass", "") or "")
            unit_f1 = per_q[qi].get("f1")
        elif not baseline_answer_empty and qi < len(parse["gold_qa"]):
            # Fallback (old format): official F1
            try:
                from .harness.locomo_env import _load_scorer
                gold = parse["gold_qa"][qi]
                unit_f1 = float(_load_scorer()([{
                    "answer": str(gold.get("answer", "")),
                    "category": gold.get("category"),
                    "prediction": pred,
                }], eval_key="prediction", metric="f1")[0][0])
            except Exception:
                unit_f1 = None
        if pq_pass:
            passed = (pq_pass == "success")
        else:
            passed = bool(unit_f1 is not None and unit_f1 >= 0.5)
        pass_details = data.get("pass_details") or {}
        run_meta = RunMeta(
            agent_id=parse["agent_name"],
            task_id=task_id,
            run_id=str(data.get("trace_id", "")),
            baseline_model=str(root_attrs.get("model", "") or ""),
            created_at=str(data.get("start_time", "") or ""),
            final_evaluator_passed=passed,
            goal_state_passed=passed,
            collateral_damage=False,
            total_cost=0.0,
            total_latency_ms=float(root_attrs.get("agent.wall_time_ms", 0) or 0),
            notes="qa_pass_source=%s qa_pass=%s qa_f1=%s qa_index=%d/%d "
                  "baseline_answer_empty=%s sample_mode=%s sample_success=%s" % (
                      "collector" if pq_pass else "fallback_f1",
                      pq_pass or ("pass" if passed else "fail"),
                      ("%.4f" % unit_f1) if unit_f1 is not None else "None",
                      qi, max(qa_count - 1, 0),
                      int(baseline_answer_empty),
                      str(pass_details.get("mode", "")),
                      str(pass_details.get("success", ""))),
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
                representative_id=parse["agent_name"],
                source_cluster_id="locomo-langgraph",
                architecture_annotation=(
                    "%s: prebuilt LangGraph ReAct + LangMem semantic "
                    "memory tools (InMemoryStore, deterministic hash "
                    "embedder)" % parse["agent_name"]),
            ),
            artifacts=artifacts,
            node_ids=node_ids,
            spans=records,
        )
