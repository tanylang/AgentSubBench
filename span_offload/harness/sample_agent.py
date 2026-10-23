"""Two representative agent architectures for the Stage 10 pilot harness.

  repA  planner_executor: intent -> plan -> step loop (tool args -> exec ->
        observe -> failure diagnosis) -> goal check -> replan -> final answer
  repB  reactive: intent -> direct tool-call loop -> goal check -> final answer

Both are span-instrumented: every LM invocation goes through lm_invoke(), which
builds a full SpanContext (descriptor + verifier runtime + contract), emits a
deterministic node id, records the span-before state snapshot into the trace
recorder, and is the single point where a replay driver can substitute a
candidate SLM or an injected parsed output (spec 14.4).

The agent state is a phase machine (boot -> plan -> steps -> goal_check ->
[replan -> steps] -> final) with pending-action checkpoints
(pending_observation / pending_failure / pending_policy), so restoring the
span-before state resumes control flow at EXACTLY the span being replayed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..runtime import InjectedResponse, SpanContext
from ..schemas import (
    GraphContext, MechanismTags, SpanDescriptor, VerifierRuntime,
)
from ..verifiers import validate_schema
from .mock_world import MockWorld, TOOL_SPECS, ToolResult

MAX_PLAN_ITERATIONS = 2
MAX_TOOL_CALLS = 14

SYSTEM_PROMPT = (
    "You are the LM core of an autonomous calendar/email assistant agent. "
    "Reply with exactly one JSON object and nothing else, following the "
    "requested schema strictly."
)

CONTRACT_TYPES = {
    "intent": "json_schema",
    "plan": "structured_plan",
    "tool_call": "tool_call",
    "observation": "observation",
    "decision": "decision",
    "failure": "failure_diagnosis",
    "policy": "policy_check",
    "goal_check": "goal_check",
    "final": "final_answer",
}

CONTRACT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "json_schema": {
        "type": "object",
        "required": ["task_summary", "goal", "constraints"],
        "properties": {
            "task_summary": {"type": "string", "minLength": 1},
            "goal": {"type": "string", "minLength": 1},
            "constraints": {"type": "array", "items": {"type": "string"}},
        },
    },
    "structured_plan": {
        "type": "object",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["description"],
                    "properties": {
                        "description": {"type": "string"},
                        "tool_hint": {"type": ["string", "null"]},
                        "guard": {"type": ["object", "null"]},
                    },
                },
            },
        },
    },
    "tool_call": {
        "type": "object",
        "required": ["tool", "arguments"],
        "properties": {
            "tool": {"type": "string"},
            "arguments": {"type": "object"},
        },
    },
    "observation": {
        "type": "object",
        "required": ["summary"],
        "properties": {
            "summary": {"type": "string"},
            "relevant_facts": {"type": "array", "items": {"type": "string"}},
            "decision": {"type": ["string", "null"]},
        },
    },
    "decision": {
        "type": "object",
        "required": ["summary", "decision"],
        "properties": {
            "summary": {"type": "string"},
            "relevant_facts": {"type": "array", "items": {"type": "string"}},
            "decision": {"enum": ["act", "skip"]},
        },
    },
    "failure_diagnosis": {
        "type": "object",
        "required": ["cause", "action"],
        "properties": {
            "cause": {"type": "string"},
            "action": {"enum": ["retry_adjusted", "skip", "abort"]},
            "adjusted_arguments": {"type": ["object", "null"]},
        },
    },
    "policy_check": {
        "type": "object",
        "required": ["decision"],
        "properties": {
            "decision": {"enum": ["allow", "deny"]},
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
    },
    "goal_check": {
        "type": "object",
        "required": ["goal_met"],
        "properties": {
            "goal_met": {"type": "boolean"},
            "evidence": {"type": "string"},
        },
    },
    "final_answer": {
        "type": "object",
        "required": ["final_answer"],
        "properties": {
            "final_answer": {"type": "string", "minLength": 1},
        },
    },
}


def extract_json(raw: str) -> Tuple[bool, Any, str]:
    """The agent's original parser: first '{' .. last '}' as JSON."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return (False, None, "no JSON object found in output")
    try:
        return (True, json.loads(text[start:end + 1]), "")
    except ValueError as exc:
        return (False, None, "JSON parse error: %s" % exc)


@dataclass
class AgentConfig:
    agent_id: str = "repA"
    architecture: str = "planner_executor"  # planner_executor | reactive
    source_cluster_id: str = "planner-executor"
    architecture_annotation: str = "planner-executor with verify/repair loop"


AGENTS: Dict[str, AgentConfig] = {
    "repA": AgentConfig(
        agent_id="repA", architecture="planner_executor",
        source_cluster_id="planner-executor",
        architecture_annotation="planner-executor with verify/repair loop"),
    "repB": AgentConfig(
        agent_id="repB", architecture="reactive",
        source_cluster_id="reactive",
        architecture_annotation="direct prediction-execution (reactive) loop"),
}


@dataclass
class AgentState:
    """Fully serializable agent state (snapshot/restore unit).

    `phase` is the control-flow position; pending_* are checkpoints that let a
    restored agent resume mid-step at exactly the span being replayed."""
    task_id: str = ""
    statement: str = ""
    booted: bool = False
    phase: str = "boot"  # boot | plan | steps | goal_check | replan | final
    intent: Dict[str, Any] = field(default_factory=dict)
    plan: List[Dict[str, Any]] = field(default_factory=list)
    step_index: int = 0
    plan_iteration: int = 0
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    next_invocation_index: int = 0
    node_log: List[Dict[str, Any]] = field(default_factory=list)
    last_decision: Optional[str] = None
    last_observation: Dict[str, Any] = field(default_factory=dict)
    executed_actions: List[str] = field(default_factory=list)
    recent_results: List[Dict[str, Any]] = field(default_factory=list)
    finished: bool = False
    stop_requested: bool = False
    aborted: bool = False
    goal_met: bool = False
    goal_checked: bool = False
    stop_reason: str = ""
    contract_count: Dict[str, int] = field(default_factory=dict)
    last_node: str = ""
    final_answer: str = ""
    final_emitted: bool = False
    pending_observation: Optional[Dict[str, Any]] = None
    pending_failure: Optional[Dict[str, Any]] = None
    pending_policy: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        # DEEP copy: snapshots must be immutable records of the span-before
        # state; live references would be polluted by later mutations.
        import copy
        return copy.deepcopy({
            "task_id": self.task_id, "statement": self.statement,
            "booted": self.booted, "phase": self.phase,
            "intent": self.intent, "plan": self.plan,
            "step_index": self.step_index, "plan_iteration": self.plan_iteration,
            "conversation_history": self.conversation_history,
            "next_invocation_index": self.next_invocation_index,
            "node_log": self.node_log, "last_decision": self.last_decision,
            "last_observation": self.last_observation,
            "executed_actions": self.executed_actions,
            "recent_results": self.recent_results, "finished": self.finished,
            "stop_requested": self.stop_requested, "aborted": self.aborted,
            "goal_met": self.goal_met, "goal_checked": self.goal_checked,
            "stop_reason": self.stop_reason,
            "contract_count": self.contract_count, "last_node": self.last_node,
            "final_answer": self.final_answer, "final_emitted": self.final_emitted,
            "pending_observation": self.pending_observation,
            "pending_failure": self.pending_failure,
            "pending_policy": self.pending_policy,
        })

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentState":
        # deep copy: the caller passes live lists/dicts from trace.artifacts; aliasing them
        # directly would let the agent's in-place appends pollute the artifact in reverse ->
        # the next model replay of the same span would get a span-before state polluted by the previous replay
        import copy
        return cls(**{k: copy.deepcopy(v) for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class SampleAgent:
    """Span-instrumented representative agent (see module docstring)."""

    def __init__(self, cfg: AgentConfig, state: Optional[AgentState] = None):
        self.cfg = cfg
        self.state = state or AgentState()
        self.driver = None
        self.recorder = None
        self.world = None
        self._stop_node: Optional[str] = None

    # -- plumbing -----------------------------------------------------------
    def bind(self, world: MockWorld, driver: Any, recorder: Any = None) -> None:
        self.world = world
        self.driver = driver
        self.recorder = recorder

    def _emit_node(self, node_id: str, **extra: Any) -> None:
        entry: Dict[str, Any] = {"node": node_id, "kind": "node"}
        entry.update(extra)
        self.state.node_log.append(entry)
        if self._stop_node and node_id == self._stop_node:
            self.state.stop_requested = True
            self.state.finished = True
            self.state.stop_reason = "stop_node_reached"

    def _push_history(self, role: str, content: str) -> None:
        self.state.conversation_history.append({"role": role, "content": content})
        if len(self.state.conversation_history) > 40:
            self.state.conversation_history = self.state.conversation_history[-40:]

    def _messages(self, user_content: str) -> List[Dict[str, str]]:
        return ([{"role": "system", "content": SYSTEM_PROMPT}] +
                list(self.state.conversation_history) +
                [{"role": "user", "content": user_content}])

    def _ordinal(self, contract_kind: str) -> int:
        key = "%s:%s" % (self.state.task_id, contract_kind)
        n = self.state.contract_count.get(key, 0)
        self.state.contract_count[key] = n + 1
        return n

    def _hint(self, contract_kind: str) -> str:
        return "%s|%s|%d" % (self.state.task_id, contract_kind, self._ordinal(contract_kind))

    # -- span context builder ----------------------------------------------
    def _span_context(self, node_pattern: str, control_role: str,
                      contract_type: str, subtask: str,
                      interface: List[str], context_source: List[str],
                      output_contract: List[str], state_effect: List[str],
                      verifier_type: List[str], phase: str,
                      predecessor_class: str, inside_loop: bool,
                      after_failure: bool = False,
                      after_verification_failure: bool = False,
                      loop_iteration_bucket: str = "1",
                      hint: str = "", allowed_tools: Optional[List[str]] = None,
                      rollback_available: bool = False) -> SpanContext:
        return SpanContext(
            invocation_index=self.state.next_invocation_index,
            node_pattern=node_pattern,
            control_role=control_role,
            contract_type=contract_type,
            span_subtask_type=subtask,
            descriptor=SpanDescriptor(
                node_pattern=node_pattern,
                control_role=control_role,
                span_subtask_type=subtask,
                mechanism_tags=MechanismTags(
                    interface=interface,
                    context_source=context_source,
                    output_contract=output_contract,
                    state_effect=state_effect,
                ),
                verifier_type=verifier_type,
                graph_context=GraphContext(
                    inside_loop=inside_loop,
                    control_phase=phase,
                    predecessor_class=predecessor_class,
                    after_failure=after_failure,
                    after_verification_failure=after_verification_failure,
                    loop_iteration_bucket=loop_iteration_bucket,
                ),
            ),
            verifier_runtime=VerifierRuntime(
                verifier_type=verifier_type,
                verifier_location="pre_commit",
                pre_commit=True,
                rollback_available=rollback_available,
            ),
            meta_hint=hint,
            allowed_tools=allowed_tools or [],
        )

    def lm_invoke(self, ctx: SpanContext, user_content: str,
                  contract_schema: Optional[Dict[str, Any]] = None,
                  ) -> Optional[Dict[str, Any]]:
        """Single point of LM invocation.  Returns the parsed output dict, or
        None when parsing failed and no fallback rescued it."""
        state_before = self.state.to_dict()  # span-before snapshot (14.3)
        messages = self._messages(user_content)
        resp = self.driver.complete(messages, ctx)
        i = self.state.next_invocation_index
        self.state.next_invocation_index += 1
        self.state.last_node = ctx.node_pattern

        # replay override path: engine already parsed + verified + injected
        if isinstance(resp, InjectedResponse):
            parsed = resp.parsed
            if isinstance(parsed, dict) and "__span_failure__" in parsed:
                if self.recorder:
                    self.recorder.record_span(self, ctx, messages, state_before, None,
                                              "", None, False,
                                              str(parsed.get("__span_failure__", "")),
                                              injected=True)
                self._emit_node("lm:%d" % i)
                self._push_history("assistant", "LM output rejected: %s"
                                   % parsed.get("__span_failure__"))
                return None
            if self.recorder:
                self.recorder.record_span(self, ctx, messages, state_before, parsed,
                                          "", None, True, "", injected=True)
            self._emit_node("lm:%d" % i)
            self._push_history("assistant", json.dumps(parsed, ensure_ascii=False))
            return parsed

        # normal path: raw output -> original parser -> contract schema
        raw = resp.raw_text if resp.ok else ""
        parse_ok, parsed, error = extract_json(raw)
        if not parse_ok:
            error = resp.error or error
            fallback_raw = self._try_fallback(messages, ctx)
            if fallback_raw is not None:
                raw = fallback_raw
                parse_ok, parsed, error = extract_json(raw)
        schema_errors: List[str] = []
        if parse_ok and contract_schema is not None:
            schema_errors = validate_schema(parsed, contract_schema)
            if schema_errors:
                fallback_raw = self._try_fallback(messages, ctx)
                if fallback_raw is not None:
                    raw = fallback_raw
                    parse_ok, parsed, error = extract_json(raw)
                    if parse_ok:
                        schema_errors = validate_schema(parsed, contract_schema)
        ok = parse_ok and not schema_errors
        if self.recorder:
            self.recorder.record_span(self, ctx, messages, state_before, parsed, raw,
                                      resp, ok,
                                      error if not parse_ok else "; ".join(schema_errors),
                                      injected=False)
        self._emit_node("lm:%d" % i)
        self._push_history("assistant", json.dumps(parsed, ensure_ascii=False)
                           if ok else "(unparseable output)")
        return parsed if ok else None

    def _try_fallback(self, messages: List[Dict[str, str]], ctx: SpanContext) -> Optional[str]:
        """Multi-span configs: 'simple schema/verifier fallback' re-asks the
        baseline LLM when the SLM output fails parsing/schema (14.8)."""
        # fallback_enabled gate: previously we only checked whether the driver has a
        # fallback_complete method (it always does) -> the "allowlist" and "allowlist_fallback"
        # configs produced the same results, breaking 14.8's four-config comparison
        if not getattr(self.driver, "fallback_enabled", False):
            return None
        fallback = getattr(self.driver, "fallback_complete", None)
        if fallback is None:
            return None
        resp2 = fallback(messages, ctx)
        return resp2.raw_text if resp2.ok else None

    # -- span emitters ------------------------------------------------------
    def _span_intent(self) -> Optional[Dict[str, Any]]:
        ctx = self._span_context(
            "intent_interpretation", "global_planner", CONTRACT_TYPES["intent"],
            "task_requirement_interpretation",
            interface=[], context_source=["conversation_history"],
            output_contract=["json_schema"], state_effect=["read_only"],
            verifier_type=["schema_check"], phase="initial", predecessor_class="",
            inside_loop=False, hint=self._hint("intent"))
        prompt = ("Task: %s\n\nWorld overview:\n%s\n\nSummarize the task. "
                  "Return JSON {\"task_summary\": str, \"goal\": str, \"constraints\": [str]}."
                  % (self.state.statement, self.world.summary()))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["json_schema"])

    def _span_planning(self) -> Optional[Dict[str, Any]]:
        tools = json.dumps({name: spec.arg_schema for name, spec in TOOL_SPECS.items()}, indent=1)
        ctx = self._span_context(
            "planning", "global_planner", CONTRACT_TYPES["plan"], "task_decomposition",
            interface=[], context_source=["conversation_history", "tool_schema"],
            output_contract=["structured_plan", "json_schema"], state_effect=["read_only"],
            verifier_type=["schema_check"], phase="initial",
            predecessor_class="intent_interpretation", inside_loop=False,
            hint=self._hint("plan"))
        prompt = ("Task summary: %s\nGoal: %s\nConstraints: %s\nAvailable tools and schemas:\n%s\n\n"
                  "Plan the steps. Return JSON {\"steps\": [{\"description\": str, \"tool_hint\": str|null, "
                  "\"guard\": {\"decision\": \"act\"}|null}]}. Use \"guard\" only when a step may only run "
                  "after an observation decided \"act\"."
                  % (self.state.intent.get("task_summary", ""), self.state.intent.get("goal", ""),
                     json.dumps(self.state.intent.get("constraints", [])), tools))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["structured_plan"])

    def _span_tool_args(self, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        hint_tool = step.get("tool_hint")
        tool_spec = TOOL_SPECS.get(hint_tool or "")
        tools_dump = json.dumps({hint_tool: tool_spec.arg_schema} if tool_spec
                                else {name: spec.arg_schema for name, spec in TOOL_SPECS.items()}, indent=1)
        state_effect = [tool_spec.state_effect] if tool_spec else ["read_only"]
        interface = [tool_spec.interface] if tool_spec else ["api_tool"]
        allowed = [hint_tool] if tool_spec else list(TOOL_SPECS.keys()) + ["__done__"]
        ctx = self._span_context(
            "tool_argument_generation", "local_executor", CONTRACT_TYPES["tool_call"],
            self._subtask_for_tool(hint_tool),
            interface=interface,
            context_source=["conversation_history", "tool_schema", "previous_tool_results"],
            output_contract=["tool_name", "tool_arguments", "json_schema"],
            state_effect=state_effect,
            verifier_type=["schema_check", "tool_exec_success"],
            phase="execution",
            predecessor_class=self.state.last_node or "planning",
            inside_loop=True,
            loop_iteration_bucket=str(max(1, self.state.plan_iteration + 1)),
            hint=self._hint("tool_call"),
            allowed_tools=allowed,
            rollback_available=tool_spec is not None and tool_spec.state_effect != "read_only",
        )
        recent = json.dumps(self.state.recent_results[-4:], ensure_ascii=False)
        actions = json.dumps(self.state.executed_actions[-6:], ensure_ascii=False)
        prompt = ("Task: %s\nGoal: %s\nCurrent step: %s\nTool schemas:\n%s\n"
                  "Recent tool results:\n%s\nActions already taken:\n%s\n"
                  "Choose the tool call for this step. Return JSON {\"tool\": str, \"arguments\": {}}. "
                  "If everything is done, return {\"tool\": \"__done__\", \"arguments\": {}}."
                  % (self.state.statement, self.state.intent.get("goal", ""),
                     step.get("description", ""), tools_dump, recent, actions))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["tool_call"])

    def _subtask_for_tool(self, tool_hint: Optional[str]) -> str:
        if tool_hint in ("calendar.add_event", "calendar.remove_event"):
            return "calendar_event_parameter_extraction"
        if tool_hint in ("calendar.list_events", "calendar.find_free_slots"):
            return "calendar_query_formulation"
        if tool_hint == "email.search":
            return "entity_resolution"
        if tool_hint == "email.send":
            return "email_composition"
        if tool_hint == "email.count_unread_urgent":
            return "inbox_aggregation"
        if tool_hint == "docs.read":
            return "document_retrieval"
        return "tool_parameter_extraction"

    def _span_observation(self, result: ToolResult, tool_name: str,
                          decision_required: bool) -> Optional[Dict[str, Any]]:
        contract = CONTRACT_TYPES["decision"] if decision_required else CONTRACT_TYPES["observation"]
        ctx = self._span_context(
            "observation_interpretation", "local_executor", contract,
            "decision_making" if decision_required else "tool_result_interpretation",
            interface=["api_tool"], context_source=["previous_tool_results"],
            output_contract=["json_schema"] + (["boolean_decision"] if decision_required else []),
            state_effect=["read_only"], verifier_type=["none"],
            phase="execution", predecessor_class="tool_argument_generation",
            inside_loop=True,
            loop_iteration_bucket=str(max(1, self.state.plan_iteration + 1)),
            hint=self._hint(contract))
        result_text = ("success: %s" % json.dumps(result.result, ensure_ascii=False)) if result.ok \
            else ("error: %s" % result.error)
        extra = (". Also decide whether the trigger condition is met and continue acting: "
                 "include \"decision\": \"act\"|\"skip\".") if decision_required else ""
        prompt = ("Tool %s returned:\n%s\n\nSummarize the observation%s. Return JSON."
                  % (tool_name, result_text, extra))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS[contract])

    def _span_failure_diagnosis(self, step: Dict[str, Any], call: Optional[Dict[str, Any]],
                                error: str) -> Optional[Dict[str, Any]]:
        ctx = self._span_context(
            "failure_diagnosis", "repair_controller", CONTRACT_TYPES["failure"],
            "failure_diagnosis",
            interface=[], context_source=["previous_tool_results"],
            output_contract=["json_schema"], state_effect=["read_only"],
            verifier_type=["schema_check"], phase="recovery",
            predecessor_class="tool_argument_generation", inside_loop=True,
            after_failure=True,
            loop_iteration_bucket=str(max(1, self.state.plan_iteration + 1)),
            hint=self._hint("failure"))
        prompt = ("Step %r with tool call %s failed: %s\nDiagnose and decide. "
                  "Return JSON {\"cause\": str, \"action\": \"retry_adjusted\"|\"skip\"|\"abort\", "
                  "\"adjusted_arguments\": obj|null}."
                  % (step.get("description", ""), json.dumps(call or {}, ensure_ascii=False), error))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["failure_diagnosis"])

    def _span_policy_check(self, call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        ctx = self._span_context(
            "policy_checking", "verifier", CONTRACT_TYPES["policy"], "policy_checking",
            interface=["file_system"], context_source=["policy_doc", "retrieved_docs"],
            output_contract=["boolean_decision", "json_schema"], state_effect=["read_only"],
            verifier_type=["policy_check"], phase="verification",
            predecessor_class="tool_argument_generation", inside_loop=True,
            loop_iteration_bucket=str(max(1, self.state.plan_iteration + 1)),
            hint=self._hint("policy"))
        policy = self.world.state.get("docs", {}).get("policy", "")
        prompt = ("Policy document:\n%s\n\nProposed action: %s\nDecide whether the action is allowed. "
                  "Return JSON {\"decision\": \"allow\"|\"deny\", \"reasons\": [str]}."
                  % (policy, json.dumps(call, ensure_ascii=False)))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["policy_check"])

    def _span_goal_check(self) -> Optional[Dict[str, Any]]:
        ctx = self._span_context(
            "observation_interpretation", "verifier", CONTRACT_TYPES["goal_check"],
            "goal_state_assessment",
            interface=[], context_source=["previous_tool_results"],
            output_contract=["json_schema", "boolean_decision"], state_effect=["read_only"],
            verifier_type=["goal_state_match"], phase="verification",
            predecessor_class="observation_interpretation", inside_loop=False,
            loop_iteration_bucket=str(max(1, self.state.plan_iteration + 1)),
            hint=self._hint("goal_check"))
        prompt = ("Goal: %s\nWorld overview:\n%s\nActions taken: %s\n\nIs the goal met? "
                  "Return JSON {\"goal_met\": bool, \"evidence\": str}."
                  % (self.state.intent.get("goal", ""), self.world.summary(),
                     json.dumps(self.state.executed_actions, ensure_ascii=False)))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["goal_check"])

    def _span_replanning(self) -> Optional[Dict[str, Any]]:
        ctx = self._span_context(
            "replanning", "repair_controller", CONTRACT_TYPES["plan"], "replanning",
            interface=[], context_source=["conversation_history", "previous_tool_results"],
            output_contract=["structured_plan", "json_schema"], state_effect=["read_only"],
            verifier_type=["schema_check"], phase="recovery",
            predecessor_class="observation_interpretation", inside_loop=False,
            after_verification_failure=True,
            loop_iteration_bucket=str(max(2, self.state.plan_iteration + 1)),
            hint=self._hint("replan"))
        prompt = ("Goal: %s\nActions already taken: %s\nList ONLY the still-needed steps. "
                  "Return JSON {\"steps\": [{\"description\": str, \"tool_hint\": str|null}]}."
                  % (self.state.intent.get("goal", ""),
                     json.dumps(self.state.executed_actions, ensure_ascii=False)))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["structured_plan"])

    def _span_final(self) -> Optional[Dict[str, Any]]:
        ctx = self._span_context(
            "final_synthesis", "terminal", CONTRACT_TYPES["final"], "final_answer_synthesis",
            interface=[], context_source=["conversation_history"],
            output_contract=["json_schema", "freeform_text"],
            state_effect=["read_only", "artifact_generation"], verifier_type=["none"],
            phase="terminal", predecessor_class="observation_interpretation",
            inside_loop=False, hint=self._hint("final"))
        prompt = ("Task: %s\nActions taken: %s\nWorld overview:\n%s\n\nWrite the final answer. "
                  "Return JSON {\"final_answer\": str}."
                  % (self.state.statement, json.dumps(self.state.executed_actions, ensure_ascii=False),
                     self.world.summary()))
        return self.lm_invoke(ctx, prompt, contract_schema=CONTRACT_SCHEMAS["final_answer"])

    # -- execution ----------------------------------------------------------
    def _execute_tool(self, call: Dict[str, Any]) -> Optional[ToolResult]:
        tool = call.get("tool", "")
        args = call.get("arguments") or {}
        if tool == "__done__":
            return None
        if tool == "email.send":
            # policy gate before every permission-sensitive send; the pending
            # checkpoint makes the policy span itself a replayable unit
            if self.state.pending_policy is None:
                self.state.pending_policy = dict(call)
            pol = self._span_policy_check(call)
            self.state.pending_policy = None
            if pol is None or pol.get("decision") != "allow":
                self.state.recent_results.append(
                    {"tool": tool, "ok": False, "error": "policy_denied", "result": None})
                self.state.last_decision = None
                return ToolResult(ok=False, error="policy_denied")
        result = self.world.execute(tool, args)
        i = self.state.next_invocation_index - 1  # span index that chose this call
        self.state.recent_results.append(
            {"tool": tool, "ok": result.ok, "result": result.result, "error": result.error})
        if self.recorder:
            self.recorder.record_tool(self, "tool:%d" % i, tool, args, result)
        self._emit_node("tool:%d" % i, kind="tool", tool=tool, ok=result.ok)
        return result

    def _after_observation(self, obs: Optional[Dict[str, Any]], tool: str) -> None:
        if obs is None:
            return
        self.state.last_observation = obs
        if obs.get("decision"):
            self.state.last_decision = obs.get("decision")
        self.state.executed_actions.append(tool)

    def _after_tool(self, call: Dict[str, Any], step: Dict[str, Any],
                    result: Optional[ToolResult], decision_required: bool) -> None:
        if result is None:
            self.state.step_index += 1
            return
        if not result.ok:
            # set the checkpoint BEFORE checking stop so a resumed agent can
            # continue at the failure-diagnosis span
            self.state.pending_failure = {"step": step, "call": call, "error": result.error}
            if self.state.stop_requested or self.state.finished:
                return
            diag = self._span_failure_diagnosis(step, call, result.error)
            self.state.pending_failure = None
            self._apply_failure(diag, step, call)
            return
        self.state.pending_observation = {
            "tool": call.get("tool", ""), "ok": True, "result": result.result,
            "error": "", "decision_required": decision_required,
        }
        if self.state.stop_requested or self.state.finished:
            return
        obs = self._span_observation(result, call.get("tool", ""), decision_required)
        self.state.pending_observation = None
        self._after_observation(obs, call.get("tool", ""))
        self.state.step_index += 1

    def _after_tool_reactive(self, call: Dict[str, Any],
                             result: Optional[ToolResult], decision_required: bool) -> None:
        if result is None:
            return
        if not result.ok:
            self.state.pending_failure = {"step": {"description": call.get("tool", "")},
                                          "call": call, "error": result.error}
            if self.state.stop_requested or self.state.finished:
                return
            diag = self._span_failure_diagnosis({"description": call.get("tool", "")},
                                                call, result.error)
            self.state.pending_failure = None
            if diag and diag.get("action") == "abort":
                self.state.aborted = True
                self.state.finished = True
                self.state.stop_reason = "abort"
            return
        self.state.pending_observation = {
            "tool": call.get("tool", ""), "ok": True, "result": result.result,
            "error": "", "decision_required": decision_required,
        }
        if self.state.stop_requested or self.state.finished:
            return
        obs = self._span_observation(result, call.get("tool", ""), decision_required)
        self.state.pending_observation = None
        self._after_observation(obs, call.get("tool", ""))

    def _apply_failure(self, diag: Optional[Dict[str, Any]], step: Dict[str, Any],
                       call: Optional[Dict[str, Any]]) -> None:
        if diag is None:
            self.state.step_index += 1
            return
        action = diag.get("action", "skip")
        if action == "abort":
            self.state.aborted = True
            self.state.finished = True
            self.state.stop_reason = "abort"
            return
        if action == "retry_adjusted" and call is not None:
            adjusted = dict(call)
            adjusted["arguments"] = diag.get("adjusted_arguments") or call.get("arguments") or {}
            tool = adjusted.get("tool", "")
            result = self.world.execute(tool, adjusted.get("arguments") or {})
            i = self.state.next_invocation_index - 1
            self.state.recent_results.append(
                {"tool": tool, "ok": result.ok, "result": result.result, "error": result.error})
            if self.recorder:
                self.recorder.record_tool(self, "tool:%d" % i, tool,
                                          adjusted.get("arguments") or {}, result)
            self._emit_node("tool:%d" % i, kind="tool", tool=tool, ok=result.ok)
            if result.ok:
                obs = self._span_observation(result, tool, False)
                self._after_observation(obs, tool)
                self.state.executed_actions.append("%s(retry)" % tool)
                self.state.step_index += 1
                return
        self.state.step_index += 1  # skip

    def _run_steps(self, branching: bool) -> None:
        while self.state.step_index < len(self.state.plan):
            step = self.state.plan[self.state.step_index]
            guard = step.get("guard") or {}
            if guard and self.state.last_decision != guard.get("decision"):
                self.state.step_index += 1
                continue
            # a step flagged "decision" produces a branching observation
            # (output_contract boolean_decision) after its tool executes
            step_decision = bool(step.get("decision"))
            # resume checkpoints inside the step pipeline
            if self.state.pending_policy is not None:
                call = self.state.pending_policy
                result = self._execute_tool(call)
                self._after_tool(call, step, result, step_decision)
                continue
            if self.state.pending_observation is not None:
                po = self.state.pending_observation
                self.state.pending_observation = None
                result = ToolResult(ok=po["ok"], result=po["result"], error=po.get("error", ""))
                obs = self._span_observation(result, po["tool"],
                                             po.get("decision_required", step_decision))
                self._after_observation(obs, po["tool"])
                self.state.step_index += 1
                continue
            if self.state.pending_failure is not None:
                pf = self.state.pending_failure
                self.state.pending_failure = None
                diag = self._span_failure_diagnosis(pf["step"], pf["call"], pf["error"])
                self._apply_failure(diag, pf["step"], pf["call"])
                continue
            call = self._span_tool_args(step)
            if call is None:
                self.state.pending_failure = {"step": step, "call": None,
                                              "error": "LM output unparseable or schema-invalid"}
                diag = self._span_failure_diagnosis(step, None,
                                                    "LM output unparseable or schema-invalid")
                self.state.pending_failure = None
                self._apply_failure(diag, step, None)
                continue
            if call.get("tool") == "__done__":
                self.state.finished = True
                self.state.stop_reason = "done_marker"
                return
            result = self._execute_tool(call)
            self._after_tool(call, step, result, step_decision)

    def _run_reactive(self, branching: bool) -> None:
        calls = 0
        while calls < MAX_TOOL_CALLS:
            if self.state.pending_policy is not None:
                call = self.state.pending_policy
                result = self._execute_tool(call)
                self._after_tool_reactive(call, result,
                                          self._decision_for_tool(call, branching))
                calls += 1
                if self.state.stop_requested or self.state.finished:
                    return
                continue
            if self.state.pending_observation is not None:
                po = self.state.pending_observation
                self.state.pending_observation = None
                result = ToolResult(ok=po["ok"], result=po["result"], error=po.get("error", ""))
                obs = self._span_observation(result, po["tool"], po.get("decision_required", False))
                self._after_observation(obs, po["tool"])
                continue
            if self.state.pending_failure is not None:
                pf = self.state.pending_failure
                self.state.pending_failure = None
                diag = self._span_failure_diagnosis(pf["step"], pf["call"], pf["error"])
                if diag and diag.get("action") == "abort":
                    self.state.aborted = True
                    self.state.finished = True
                    self.state.stop_reason = "abort"
                    return
                continue
            step = {"description": "Choose the next tool call to make progress",
                    "tool_hint": None}
            call = self._span_tool_args(step)
            if call is None:
                self.state.pending_failure = {"step": step, "call": None,
                                              "error": "LM output unparseable or schema-invalid"}
                diag = self._span_failure_diagnosis(step, None,
                                                    "LM output unparseable or schema-invalid")
                self.state.pending_failure = None
                if diag and diag.get("action") == "abort":
                    self.state.aborted = True
                self.state.finished = True
                self.state.stop_reason = "lm_failure"
                return
            if call.get("tool") == "__done__":
                self.state.finished = True
                self.state.stop_reason = "done_marker"
                return
            result = self._execute_tool(call)
            calls += 1
            if self.state.stop_requested or self.state.finished:
                return
            self._after_tool_reactive(call, result,
                                      self._decision_for_tool(call, branching))
        self.state.finished = True
        self.state.stop_reason = "tool_call_budget"

    @staticmethod
    def _decision_for_tool(call: Dict[str, Any], branching: bool) -> bool:
        """Branching observation (boolean_decision contract) only after the
        trigger tool of a conditional task."""
        return bool(branching) and call.get("tool") == "email.count_unread_urgent"

    # -- run / resume -------------------------------------------------------
    def resume(self, world: MockWorld, driver: Any, stop_node: Optional[str] = None,
               recorder: Any = None, branching: bool = False,
               max_plan_iterations: int = MAX_PLAN_ITERATIONS) -> "RunResult":
        """Run/continue the agent until stop_node is reached or the run ends.

        `branching` = the task is conditional (branching != "none"); only the
        trigger-tool observation then uses the boolean_decision contract.

        Idempotent: if the agent already completed (final_emitted), the cached
        result is returned -- this is what makes run_to_completion safe after
        run_to_node reached the final evaluator."""
        self.bind(world, driver, recorder)
        self._stop_node = stop_node
        if self.state.final_emitted:
            return self._completed_result()
        self.state.stop_requested = False
        self.state.finished = False
        if self.state.stop_reason == "stop_node_reached":
            self.state.stop_reason = ""
        while not self.state.finished:
            phase = self.state.phase
            if phase == "boot":
                self.state.booted = True
                intent = self._span_intent()
                if intent is None:
                    self.state.finished = True
                    self.state.stop_reason = "intent_failed"
                    break
                self.state.intent = intent
                self.state.phase = "plan" if self.cfg.architecture == "planner_executor" else "steps"
                if self.state.stop_requested:
                    break
                continue
            if phase == "plan":
                plan = self._span_planning()
                if plan is None:
                    self.state.finished = True
                    self.state.stop_reason = "planning_failed"
                    break
                self.state.plan = plan.get("steps") or []
                self.state.phase = "steps"
                if self.state.stop_requested:
                    break
                continue
            if phase == "steps":
                if self.state.plan_iteration >= max_plan_iterations:
                    self.state.finished = True
                    self.state.stop_reason = "plan_iteration_budget"
                    break
                if self.cfg.architecture == "planner_executor":
                    self._run_steps(branching)
                else:
                    self._run_reactive(branching)
                if self.state.finished and self.state.stop_reason == "done_marker" \
                        and not self.state.goal_checked:
                    # reactive agent declared done: verify the goal before final
                    self.state.finished = False
                    self.state.phase = "goal_check"
                    continue
                if self.state.stop_requested or self.state.finished or self.state.aborted:
                    break
                self.state.phase = "goal_check"
                continue
            if phase == "goal_check":
                check = self._span_goal_check()
                self.state.goal_checked = True
                if check is None:
                    self.state.finished = True
                    self.state.stop_reason = "goal_check_failed"
                    break
                if check.get("goal_met"):
                    self.state.goal_met = True
                    self.state.phase = "final"
                    break
                if self.cfg.architecture == "planner_executor":
                    self.state.phase = "replan"
                else:
                    self.state.plan_iteration += 1
                    self.state.phase = "steps"
                self.state.goal_checked = False  # re-check after the next pass
                continue
            if phase == "replan":
                replan = self._span_replanning()
                if replan is not None and replan.get("steps") is not None:
                    self.state.plan = replan["steps"]
                self.state.plan_iteration += 1
                self.state.step_index = 0
                self.state.phase = "steps"
                if self.state.stop_requested:
                    break
                continue
            if phase == "final":
                break
            break
        return self.finish()

    def _completed_result(self) -> "RunResult":
        reached = (self._stop_node is None or
                   any(n.get("node") == self._stop_node for n in self.state.node_log))
        return RunResult(
            final_answer=self.state.final_answer, goal_met=self.state.goal_met,
            stop_node_reached=reached, stop_reason=self.state.stop_reason,
            node_log=list(self.state.node_log), aborted=self.state.aborted,
        )

    def finish(self) -> "RunResult":
        if not self.state.stop_requested and not self.state.final_emitted:
            final = self._span_final()
            if final:
                self.state.final_answer = final.get("final_answer", "")
            self.state.final_emitted = True
            self._emit_node("final_evaluator")
        return self._completed_result()


@dataclass
class RunResult:
    final_answer: str = ""
    goal_met: bool = False
    stop_node_reached: bool = False
    stop_reason: str = ""
    node_log: List[Dict[str, Any]] = field(default_factory=list)
    aborted: bool = False
