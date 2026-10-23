"""SWE-bench (mini-swe-agent) trace adapter: trace JSON collected by trace_record ->
canonical TraceRun (same structure as the AppWorld adapter, reusing its splice/verifier components).

Trace structure and key semantics (trace_record project, see the plan file):
  - spans include invoke_agent (root) / llm.call (CLIENT) / tool.execution (INTERNAL),
    strictly increasing by attributes["agent.step"]; the tools triggered by llm j = the
    tool spans with step in (llm[j].step, llm[j+1].step)
  - llm.call's check_point.conversation = {has_full_conversation,
    messages (only role/content — the assistant's tool_calls structure and tool_call_id
    are dropped by the collection side and must be spliced back by the real call ids of
    conv.tool_calls), output_text, tool_calls[{id,name,arguments}]}
  - every non-root span's check_point.application_state_hash = the chained cumulative hash
    **before** this span starts; env_changes = the repo file deltas during this span
    {path: full text or null}; the first llm span additionally has env_changes_baseline
    (archive only, not in the hash chain). e3b0c442... is a legal empty state under swebench semantics.
  - top-level pass=="success" is the official pytest convention (all FAIL_TO_PASS pass ∧
    all PASS_TO_PASS pass); swebench = {instance_id, repo, base_commit, FAIL_TO_PASS, PASS_TO_PASS}
  - recorded commands embed the collection-time Windows absolute paths (measured); after I-arm
    injection they would always fail on Linux -> this adapter normalizes them to WORKSPACE_TOKEN,
    and the runtime substitutes the real path
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .appworld_adapter import (  # reuse: splice and verifier components, zero changes
    _link_tool_messages, _native_tool_calls, span_verifier,
)
from .schemas import (
    AgentCondition, BaselineResponse, Continuation, GraphContext, LMRequest,
    MechanismTags, RunMeta, SpanDescriptor, SpanRecord, StateSnapshot,
    TaskCondition, ToolExecution, TraceRun, OutputContractRuntime,
)

ADAPTER_NAME = "swebench"
AGENT_ID = "mini-swe-agent"

#: Verbatim identical to BASH_TOOL in minisweagent/models/utils/actions_toolcall.py:11-27
BASH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

#: Output contract schema: tool_call_list with name=="bash"
BASH_CALL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["tool_calls"],
    "properties": {"tool_calls": {"type": "array", "minItems": 1}},
}

#: Normalization token for the collection workspace path (the runtime substitutes the real workspace path)
WORKSPACE_TOKEN = "<SWE_WORKSPACE>/"

#: Read-only command prefix allowlist (conservative: default external_state_mutation)
_READONLY_PREFIXES = (
    "ls ", "cat ", "head ", "tail ", "grep ", "find ", "wc ", "pwd",
    "git status", "git diff", "git log", "git show", "git branch",
    "cd ", "echo ", "python --version", "sed -n ",
)

#: Windows absolute path shapes (C:/... and C:\\...)
_WIN_PATH_RE = re.compile(
    r'[A-Za-z]:[\\/][^\s"\']*swe_workspaces[^\s"\']*')


def _readonly_command(cmd: str) -> bool:
    c = (cmd or "").strip()
    if not c:
        return True
    return any(c.startswith(p) for p in _READONLY_PREFIXES)


def _rewrite_workspace_paths(text: Any, prefixes: List[str]
                             ) -> Tuple[Any, bool]:
    """Normalize the collection-time workspace absolute paths (Windows shape) to
    WORKSPACE_TOKEN. Returns (rewritten_text, found). None (assistant empty content)
    is kept as-is and must not be str()'d into the string "None"."""
    if text is None:
        return None, False
    s = str(text)
    found = False
    for p in prefixes:
        if p and p in s:
            s = s.replace(p, WORKSPACE_TOKEN)
            found = True
    return s, found


def _parse_baseline_output(conv: Dict[str, Any]
                           ) -> Tuple[Any, str, bool]:
    """Restore the baseline output into an injectable parsed form.
    - has tool_calls -> {"tool_calls": [{name, arguments}]} (tool_call_list contract)
    - text-only -> {"text": output_text} with parse_success=False
      (the real agent raises FormatError on such output; mapped faithfully)
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
            "freeform_text", False)


def _classify_swe_span(has_calls: bool, event_type: str, j: int,
                       prev_tool_failed: bool) -> Tuple[SpanDescriptor, str]:
    """Span classification for the single-bash-tool mini-swe-agent (node_pattern normalized
    to the existing vocabulary). Returns (descriptor, meta_hint suffix)."""
    if not has_calls:
        # text-only output: the real agent raises FormatError -> recovery type
        d = SpanDescriptor(
            node_pattern="other", control_role="worker",
            span_subtask_type="format_error_recovery",
            mechanism_tags=MechanismTags(
                interface=["shell"], output_contract=["freeform_text"],
                state_effect=["read_only"]),
            verifier_type=["none"],
            graph_context=GraphContext(control_phase="execution",
                                       after_failure=True,
                                       inside_loop=True),
        )
        return d, "text_only"
    if event_type == "planning":
        phase = "initial"
        role = "global_planner"
    elif event_type == "verification":
        phase = "verification"
        role = "local_executor"
    elif event_type == "repair" or prev_tool_failed:
        phase = "recovery"
        role = "repair_controller"
    else:
        phase = "initial" if j == 0 else "execution"
        role = "local_executor"
    d = SpanDescriptor(
        node_pattern="tool_argument_generation", control_role=role,
        span_subtask_type="bash_command_generation",
        mechanism_tags=MechanismTags(
            interface=["shell"],
            context_source=["conversation_history", "tool_schema",
                            "previous_tool_results"],
            output_contract=["tool_name", "tool_arguments", "json_schema"],
            state_effect=["read_only"],
        ),
        verifier_type=["schema_check"],
        graph_context=GraphContext(control_phase=phase,
                                   after_failure=(phase == "recovery"),
                                   inside_loop=True),
    )
    return d, "%s" % phase


class SwebenchTraceAdapter:
    """trace_record's SWE trace JSON -> canonical TraceRun."""

    name = ADAPTER_NAME

    def __init__(self, manifest=None):
        self.manifest = manifest

    # ------------------------------------------------------------------
    def load(self, path: str) -> TraceRun:
        path = str(path).replace("\\", "/")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        swe = data.get("swebench") or {}
        instance_id = swe.get("instance_id", "unknown")
        repo = swe.get("repo", "")
        base_commit = swe.get("base_commit", "")
        ftp = swe.get("FAIL_TO_PASS", [])
        ptp = swe.get("PASS_TO_PASS", [])
        agent_name = data.get("agent_name", AGENT_ID)

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
        all_steps = sorted(tool_by_step.keys()) + llm_steps
        llm_spans = [s for s in spans if s.get("type") == "llm.call"]

        # Workspace prefix detection (scan tool commands for Windows swe_workspaces paths)
        prefixes = set()
        for s in spans:
            if s.get("type") != "tool.execution":
                continue
            arg = s["attributes"].get("gen_ai.tool.call.arguments", "") or ""
            for m in _WIN_PATH_RE.finditer(arg):
                p = m.group(0)
                idx = p.find("swe_workspaces")
                prefixes.add(p[:idx] + "swe_workspaces")
        prefixes = sorted(prefixes)

        # env_delta_log: one entry per span (tools included), in step order
        env_delta_log: List[Dict[str, Any]] = []
        for s in spans:
            cp = s.get("check_point") or {}
            env_delta_log.append({
                "step": int(s["attributes"].get("agent.step", 0) or 0),
                "type": s.get("type"),
                "span_id": s.get("span_id"),
                "hash_before": str(cp.get("application_state_hash", "") or ""),
                "env_changes": cp.get("env_changes") or {},
            })

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
            "env_delta_log": env_delta_log,
            "workspace_prefixes": prefixes,
        }
        node_ids: List[str] = []

        records: List[SpanRecord] = []
        prev_tool_failed = False
        prev_spliced: Optional[List[Dict[str, Any]]] = None
        for j, s in enumerate(llm_spans):
            attrs = s.get("attributes", {})
            conv = convs[j]
            step = int(attrs.get("agent.step", 0) or 0)
            nxt = llm_steps[j + 1] if j + 1 < len(llm_steps) else None
            attached = [tool_by_step[st] for st in sorted(tool_by_step)
                        if st > step and (nxt is None or st < nxt)]

            # --- conversation splice: repair the last turn's assistant/tool_call_id ---
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
                    native = _native_tool_calls(prev_calls, id_prefix="swe")
                    assistant = {"role": "assistant", "content": None,
                                 "tool_calls": native}
                    fixed = [assistant] + [dict(m) for m in tail[1:]]
                    _link_tool_messages(fixed, 1, native)
                    spliced = list(prev_spliced) + fixed
                else:
                    spliced = raw_msgs
                    degraded = True
            prev_spliced = spliced

            # --- path normalization (messages + recorded output) ---
            path_rewritten = False
            tok_msgs = []
            for m in spliced:
                mm = dict(m)
                content, found = _rewrite_workspace_paths(
                    mm.get("content", ""), prefixes)
                mm["content"] = content
                path_rewritten = path_rewritten or found
                tok_msgs.append(mm)
            tok_calls = []
            for c in calls:
                cc = dict(c)
                arg, found = _rewrite_workspace_paths(
                    cc.get("arguments", ""), prefixes)
                cc["arguments"] = arg
                path_rewritten = path_rewritten or found
                tok_calls.append(cc)

            parsed, contract_kind, parse_success = _parse_baseline_output(
                {"output_text": conv["output_text"], "tool_calls": tok_calls})

            has_calls = bool(tok_calls)
            descriptor, phase_hint = _classify_swe_span(
                has_calls, str(s.get("event_type") or ""), j,
                prev_tool_failed)

            # Attached tools: record results (for the equivalence check) and update prev_tool_failed
            tool_exec: Optional[ToolExecution] = None
            span_tool_results: List[Dict[str, Any]] = []
            for t in attached:
                ta = t.get("attributes", {})
                ok = str(ta.get("tool.success", "")).lower() == "true"
                prev_tool_failed = prev_tool_failed or not ok
                raw_args = ta.get("gen_ai.tool.call.arguments", "{}") or "{}"
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except ValueError:
                        # The collection-side 5000-char truncation can cut JSON strings in half
                        # (measured: 2 traces); arguments are reference artifacts only, degraded honestly when damaged
                        args = {"command": "<truncated>"}
                else:
                    args = raw_args
                span_tool_results.append({
                    "tool": "bash", "arguments": args, "ok": ok,
                    "result": ta.get("gen_ai.tool.call.result", ""),
                })
                if tool_exec is None:
                    tool_exec = ToolExecution(
                        tool_name="bash",
                        arguments=span_tool_results[-1]["arguments"],
                        success=ok,
                        result=span_tool_results[-1]["result"],
                        node_id="tool:%d:%d" % (j, len(attached)),
                    )

            node_ids.append("lm:%d" % j)
            for k in range(len(attached)):
                node_ids.append("tool:%d:%d" % (j, k))

            ref = "env:%d" % j
            artifacts["state_snapshots"][ref] = {
                "checkpointed": True,
                "delta_log_index": next(
                    i for i, e in enumerate(env_delta_log)
                    if e["step"] == step and e["type"] == "llm.call"),
                "application_state_hash": str(
                    (s.get("check_point") or {}).get(
                        "application_state_hash", "") or ""),
                "step": step,
            }
            artifacts["state_snapshots"]["agent:%d" % j] = {
                "has_full_conversation": True, "step": step,
            }
            artifacts["messages"]["msg:%d" % j] = tok_msgs
            artifacts["system_prompts"]["sys:%d" % j] = next(
                (m["content"] for m in tok_msgs if m.get("role") == "system"),
                "")
            recorded_tools = None
            raw_tools = attrs.get("gen_ai.request.tools", "")
            if isinstance(raw_tools, str) and raw_tools.strip():
                try:
                    recorded_tools = json.loads(raw_tools)
                except ValueError:
                    recorded_tools = None
            artifacts["recorded_tools"]["tools:%d" % j] = (
                recorded_tools or [BASH_TOOL_SCHEMA])
            artifacts["tool_schemas"]["tools:%d" % j] = {
                "bash": BASH_TOOL_SCHEMA}
            artifacts["schemas"]["schema:%d" % j] = (
                BASH_CALL_SCHEMA if contract_kind == "tool_call_list"
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
                    allowed_tools=["bash"],
                    semantic_constraints={
                        "source_event_type": str(s.get("event_type") or ""),
                        "source_step": step,
                        "n_tool_calls": len(tok_calls),
                        "path_rewritten": path_rewritten,
                        "splice_degraded": degraded,
                    },
                ),
                verifier_runtime=span_verifier(descriptor.verifier_type),
                continuation=Continuation(
                    resume_node_id=("tool:%d:0" % j) if attached else (
                        "lm:%d" % (j + 1) if j + 1 < len(llm_spans)
                        else "final_evaluator"),
                    max_remaining_steps=50 - j,
                    evaluator_ref="%s#swebench_pytest" % instance_id,
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
                meta_hint="%s|%s|%d|step=%d|%s" % (
                    instance_id, s.get("event_type") or "", j, step,
                    phase_hint),
            ))

        passed = (data.get("pass") == "success")
        run_meta = RunMeta(
            agent_id=agent_name,
            task_id=instance_id,
            run_id=str(data.get("trace_id", "")),
            baseline_model=str(root_attrs.get("model", "") or ""),
            created_at=str(data.get("start_time", "") or ""),
            final_evaluator_passed=passed,
            goal_state_passed=passed,
            collateral_damage=False,
            total_cost=0.0,
            total_latency_ms=float(root_attrs.get("agent.wall_time_ms", 0) or 0),
            notes="exit=%s pass_details=%s" % (
                str((data.get("agent_extra") or {}).get("exit_status", "")),
                json.dumps(data.get("pass_details") or {},
                           ensure_ascii=False)[:200]),
        )
        task_condition = TaskCondition()
        if self.manifest is not None:
            mt = self.manifest.get(instance_id)
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
                source_cluster_id="swebench-minisweagent",
                architecture_annotation=(
                    "mini-swe-agent 2.4.6 DefaultAgent loop (bash-only)"),
            ),
            artifacts=artifacts,
            node_ids=node_ids,
            spans=records,
        )
