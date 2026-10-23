"""MockAppWorldRuntime: the AgentRuntime implementation over the pilot
harness (mock world + sample agents), plus the trace recorder and the drivers
used for baseline runs, single-span replay, and multi-span policy runs.

This file documents the exact contract a real AppWorld integration must
implement (restore / resolve_request / parse / schema_check / verify_local /
inject / run_to_node / run_to_completion / rollback).
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..llm_client import LMResponse
from ..runtime import (
    AgentRuntime, InjectedResponse, ParseResult, ResolvedRequest,
    RestoredSession, SpanContext, StopReplay,
)
from ..schemas import (
    AgentCondition, BaselineResponse, Continuation, LMRequest, OutputContractRuntime,
    RunMeta, SpanRecord, StateSnapshot, TaskCondition, TaskManifest, ToolExecution,
    TraceRun, TRACE_SCHEMA_VERSION,
)
from ..verifiers import VerifierVerdict, no_verifier_verdict, validate_schema
from .mock_world import (
    MockWorld, TOOL_SPECS, ToolResult, build_world, evaluate_task,
)
from .sample_agent import (
    AgentConfig, AgentState, SampleAgent, RunResult, extract_json,
)

from ..schemas import (
    DownstreamOutcome, FinalOutcome, ManifestTask, ReplayInput, SpanDescriptor,
)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

class PassthroughDriver:
    """Baseline runs: every span goes to the baseline LLM."""

    def __init__(self, client: Any):
        self.client = client
        self.usage: List[LMResponse] = []

    def complete(self, messages: List[Dict[str, Any]], ctx: SpanContext,
                 tools: Any = None) -> LMResponse:
        resp = self.client.complete(messages, ctx=ctx, temperature=ctx.temperature,
                                    max_tokens=ctx.max_output_tokens)
        self.usage.append(resp)
        return resp


class ReplayDriver:
    """Single-span counterfactual replay (14.4): span j consumes the injected
    parsed output; every later span goes to the baseline LLM."""

    def __init__(self, override_index: int, baseline_client: Any):
        self.override_index = override_index
        self.override_output: Any = None
        self.baseline_client = baseline_client
        self.usage: List[LMResponse] = []

    def complete(self, messages: List[Dict[str, Any]], ctx: SpanContext,
                 tools: Any = None) -> LMResponse:
        if ctx.invocation_index == self.override_index:
            if self.override_output is None:
                raise RuntimeError("replay override not injected for span %d"
                                   % self.override_index)
            return InjectedResponse(parsed=self.override_output, latency_ms=0.0)
        resp = self.baseline_client.complete(messages, ctx=ctx, temperature=ctx.temperature,
                                             max_tokens=ctx.max_output_tokens)
        self.usage.append(resp)
        return resp


class MultiSpanDriver:
    """Multi-span end-to-end sanity runs (14.8): a static policy decides per
    span whether the candidate SLM or the baseline LLM answers, with optional
    'simple schema/verifier fallback' to the baseline on parse failure."""

    def __init__(self, policy: Callable[[SpanContext], str],
                 baseline_client: Any, slm_client: Any, fallback: bool = False):
        self.policy = policy
        self.baseline_client = baseline_client
        self.slm_client = slm_client
        self.fallback_enabled = fallback
        self.stats: Dict[str, Any] = {
            "offloaded": 0, "baseline_calls": 0, "fallbacks": 0,
            "baseline_usage": [], "slm_usage": [], "fallback_usage": [],
        }

    def complete(self, messages: List[Dict[str, Any]], ctx: SpanContext,
                 tools: Any = None) -> LMResponse:
        if self.policy(ctx) == "slm":
            self.stats["offloaded"] += 1
            resp = self.slm_client.complete(messages, ctx=ctx, temperature=ctx.temperature,
                                            max_tokens=ctx.max_output_tokens)
            self.stats["slm_usage"].append(resp)
            return resp
        self.stats["baseline_calls"] += 1
        resp = self.baseline_client.complete(messages, ctx=ctx, temperature=ctx.temperature,
                                             max_tokens=ctx.max_output_tokens)
        self.stats["baseline_usage"].append(resp)
        return resp

    def fallback_complete(self, messages: List[Dict[str, Any]], ctx: SpanContext,
                          tools: Any = None) -> LMResponse:
        self.stats["fallbacks"] += 1
        resp = self.baseline_client.complete(messages, ctx=ctx, temperature=ctx.temperature,
                                             max_tokens=ctx.max_output_tokens)
        self.stats["fallback_usage"].append(resp)
        return resp


# ---------------------------------------------------------------------------
# Trace recorder
# ---------------------------------------------------------------------------

class TraceRecorder:
    """Records a live agent run into the canonical native-trace format.

    Bulky payloads live in `artifacts` keyed by refs; each span stores refs
    only (14.3).  Continuation points are finalized post-run from the agent's
    node log."""

    def __init__(self, task: ManifestTask, agent_cfg: AgentConfig,
                 baseline_model: str, cost_fn: Callable[[int, int], float]):
        self.task = task
        self.agent_cfg = agent_cfg
        self.baseline_model = baseline_model
        self.cost_fn = cost_fn
        self.artifacts: Dict[str, Any] = {
            "state_snapshots": {}, "messages": {}, "system_prompts": {},
            "tool_schemas": {}, "tool_lists": {}, "previous_tool_results": {},
            "conversation_histories": {}, "plan_states": {}, "schemas": {},
            "response_formats": {},
        }
        self.spans: List[SpanRecord] = []
        self.node_ids: List[str] = []

    def record_span(self, agent: SampleAgent, ctx: SpanContext,
                    messages: List[Dict[str, str]], state_before: Dict[str, Any],
                    parsed: Any, raw: str, resp: Optional[LMResponse],
                    ok: bool, error: str, injected: bool = False) -> None:
        i = ctx.invocation_index
        env_snap = agent.world.snapshot()
        self.artifacts["state_snapshots"]["env:%d" % i] = env_snap
        self.artifacts["state_snapshots"]["agent:%d" % i] = state_before
        self.artifacts["conversation_histories"]["conv:%d" % i] = state_before.get("conversation_history", [])
        self.artifacts["plan_states"]["plan:%d" % i] = {
            "plan": state_before.get("plan", []),
            "step_index": state_before.get("step_index", 0),
            "plan_iteration": state_before.get("plan_iteration", 0),
        }
        self.artifacts["messages"]["msg:%d" % i] = list(messages)
        self.artifacts["system_prompts"]["sys:%d" % i] = (
            messages[0]["content"] if messages and messages[0].get("role") == "system" else "")
        self.artifacts["previous_tool_results"]["prev:%d" % i] = state_before.get("recent_results", [])
        self.artifacts["response_formats"]["fmt:%d" % i] = {"type": "json_object"}
        schema = None
        if ctx.contract_type in CONTRACT_SCHEMAS_BY_TYPE:
            schema = CONTRACT_SCHEMAS_BY_TYPE[ctx.contract_type]
            self.artifacts["schemas"]["schema:%d" % i] = schema
        tools_schema = None
        if ctx.allowed_tools:
            tools_schema = {name: TOOL_SPECS[name].arg_schema
                            for name in ctx.allowed_tools if name in TOOL_SPECS}
            self.artifacts["tool_schemas"]["tools:%d" % i] = tools_schema
            self.artifacts["tool_lists"]["list:%d" % i] = list(ctx.allowed_tools)

        prompt_tokens = resp.prompt_tokens if resp else 0
        completion_tokens = resp.completion_tokens if resp else 0
        span = SpanRecord(
            invocation_index=i,
            node_id="lm:%d" % i,
            span_descriptor=ctx.descriptor,
            request=LMRequest(
                system_prompt_ref="sys:%d" % i,
                messages_ref="msg:%d" % i,
                previous_tool_results_ref="prev:%d" % i,
                available_tools_ref="list:%d" % i if ctx.allowed_tools else "",
                tool_schemas_ref="tools:%d" % i if tools_schema else "",
                response_format_ref="fmt:%d" % i,
                temperature=ctx.temperature,
                max_output_tokens=ctx.max_output_tokens,
            ),
            state_snapshot=StateSnapshot(
                agent_state_ref="agent:%d" % i,
                environment_snapshot_ref="env:%d" % i,
                conversation_history_ref="conv:%d" % i,
                plan_state_ref="plan:%d" % i,
                application_state_hash=MockWorld(env_snap).state_hash(),
            ),
            output_contract_runtime=OutputContractRuntime(
                contract_type=ctx.contract_type,
                schema_ref="schema:%d" % i if schema else "",
                allowed_tools=list(ctx.allowed_tools),
                semantic_constraints={},
            ),
            verifier_runtime=ctx.verifier_runtime,
            continuation=Continuation(),  # finalized post-run
            baseline_response=BaselineResponse(
                raw=raw or "", parsed=parsed, parse_success=ok, parse_error=error,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                latency_ms=resp.latency_ms if resp else 0.0,
                cost=self.cost_fn(prompt_tokens, completion_tokens),
            ),
            tool_execution=None,
            meta_hint=ctx.meta_hint,
        )
        self.spans.append(span)
        self.node_ids.append("lm:%d" % i)

    def record_tool(self, agent: SampleAgent, node_id: str, tool_name: str,
                    arguments: Dict[str, Any], result: ToolResult) -> None:
        self.node_ids.append(node_id)
        for span in reversed(self.spans):
            if span.span_descriptor.node_pattern == "tool_argument_generation" \
                    and span.tool_execution is None:
                span.tool_execution = ToolExecution(
                    tool_name=tool_name, arguments=arguments, success=result.ok,
                    result=result.result, error=result.error, node_id=node_id,
                )
                return

    def finalize(self, node_log: List[Dict[str, Any]],
                 eval_result: Dict[str, Any],
                 total_cost: float, total_latency_ms: float) -> TraceRun:
        node_ids = [n["node"] for n in node_log]
        for span in self.spans:
            i = span.invocation_index
            try:
                pos = node_ids.index("lm:%d" % i)
            except ValueError:
                pos = len(node_ids)
            resume = node_ids[pos + 1] if pos + 1 < len(node_ids) else "final_evaluator"
            remaining = max(0, len(node_ids) - pos - 1)
            span.continuation = Continuation(
                resume_node_id=resume,
                later_lm_model=self.baseline_model,
                evaluator_ref="%s#%s" % (self.task.task_id, self.task.evaluator_type),
                max_remaining_steps=remaining + 10,
            )
        trace = TraceRun(
            schema_version=TRACE_SCHEMA_VERSION,
            run_meta=RunMeta(
                agent_id=self.agent_cfg.agent_id,
                task_id=self.task.task_id,
                run_id="run1",
                baseline_model=self.baseline_model,
                created_at=datetime.datetime.now().isoformat(timespec="seconds"),
                final_evaluator_passed=bool(eval_result.get("goal_state_passed")),
                goal_state_passed=bool(eval_result.get("goal_state_passed")),
                collateral_damage=bool(eval_result.get("collateral_damage")),
                total_cost=total_cost,
                total_latency_ms=total_latency_ms,
                notes=json.dumps({"evaluator_details": eval_result.get("details", [])}),
            ),
            task_condition=TaskCondition(
                task_type=self.task.task_type,
                involved_apps=list(self.task.involved_apps),
                cross_app=bool(self.task.cross_app),
                horizon=self.task.horizon,
                task_state_effect=self.task.task_state_effect,
                branching=self.task.branching,
                evaluator_type=self.task.evaluator_type,
                collateral_risk=self.task.collateral_risk,
            ),
            agent_condition=AgentCondition(
                representative_id=self.agent_cfg.agent_id,
                source_cluster_id=self.agent_cfg.source_cluster_id,
                architecture_annotation=self.agent_cfg.architecture_annotation,
            ),
            artifacts=self.artifacts,
            node_ids=node_ids,
            spans=self.spans,
        )
        return trace


# Contract schemas must match the agent's CONTRACT_SCHEMAS; imported here to
# avoid a circular import at module level (agent imports mock_world only).
from .sample_agent import CONTRACT_SCHEMAS as _AGENT_SCHEMAS  # noqa: E402
CONTRACT_SCHEMAS_BY_TYPE = dict(_AGENT_SCHEMAS)


# ---------------------------------------------------------------------------
# Baseline / policy run helpers
# ---------------------------------------------------------------------------

def run_baseline_trace(task: ManifestTask, agent_cfg: AgentConfig,
                       client: Any) -> Tuple[TraceRun, RunResult, Dict[str, Any], MockWorld]:
    """Run the agent with the baseline LLM on a fresh world and record the
    native trace + final evaluator result."""
    world = build_world(task.task_id, task.world_variant)
    agent = SampleAgent(agent_cfg)
    agent.state.task_id = task.task_id
    agent.state.statement = task.statement
    recorder = TraceRecorder(task, agent_cfg, client.name, client.cost_for)
    driver = PassthroughDriver(client)
    before = world.snapshot()
    result = agent.resume(world, driver, recorder=recorder,
                          branching=task.branching == "conditional")
    eval_result = evaluate_task(task.task_id, MockWorld(before), world, result.final_answer)
    total_cost = sum(client.cost_for(r.prompt_tokens, r.completion_tokens)
                     for r in driver.usage)
    total_latency = sum(r.latency_ms for r in driver.usage)
    trace = recorder.finalize(agent.state.node_log, eval_result, total_cost, total_latency)
    return trace, result, eval_result, world


def run_policy_trace(task: ManifestTask, agent_cfg: AgentConfig,
                     baseline_client: Any, slm_client: Any,
                     policy: Callable[[SpanContext], str],
                     fallback: bool) -> Tuple[Dict[str, Any], RunResult, Dict[str, Any]]:
    """Multi-span end-to-end run under a static span policy (14.8)."""
    import time
    world = build_world(task.task_id, task.world_variant)
    agent = SampleAgent(agent_cfg)
    agent.state.task_id = task.task_id
    agent.state.statement = task.statement
    driver = MultiSpanDriver(policy, baseline_client, slm_client, fallback=fallback)
    before = world.snapshot()
    t0 = time.perf_counter()
    result = agent.resume(world, driver, recorder=None,
                          branching=task.branching == "conditional")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    eval_result = evaluate_task(task.task_id, MockWorld(before), world, result.final_answer)

    def cost_of(usage: List[LMResponse], price_fn: Callable[[int, int], float]) -> float:
        return sum(price_fn(r.prompt_tokens, r.completion_tokens) for r in usage)

    stats = {
        "task_id": task.task_id,
        "agent_id": agent_cfg.agent_id,
        "final_task_success": bool(eval_result.get("goal_state_passed")),
        "collateral_damage": bool(eval_result.get("collateral_damage")),
        "cost": (cost_of(driver.stats["baseline_usage"], baseline_client.cost_for)
                 + cost_of(driver.stats["slm_usage"], slm_client.cost_for)
                 + cost_of(driver.stats["fallback_usage"], baseline_client.cost_for)),
        "latency_ms": elapsed_ms,
        "n_offloaded": driver.stats["offloaded"],
        "n_fallbacks": driver.stats["fallbacks"],
        "n_spans": agent.state.next_invocation_index,
        "stop_reason": result.stop_reason,
        "evaluator_details": eval_result.get("details", []),
    }
    return stats, result, eval_result


# ---------------------------------------------------------------------------
# The AgentRuntime implementation
# ---------------------------------------------------------------------------

@dataclass
class MockSession(RestoredSession):
    agent: Optional[SampleAgent] = None
    world: Optional[MockWorld] = None
    driver: Optional[ReplayDriver] = None
    trace: Optional[TraceRun] = None
    original_world_snapshot: Optional[Dict[str, Any]] = None


class MockAppWorldRuntime(AgentRuntime):
    name = "mock-appworld"

    def __init__(self, agents: Dict[str, AgentConfig], manifest: TaskManifest):
        self.agents = agents
        self.manifest = manifest

    # -- 14.4 step 3 --------------------------------------------------------
    def restore(self, trace: TraceRun, span: SpanRecord,
                baseline_client: Any = None) -> MockSession:
        states = trace.artifacts.get("state_snapshots") or {}
        env_snap = states[span.state_snapshot.environment_snapshot_ref]
        agent_state_dict = states[span.state_snapshot.agent_state_ref]
        world = MockWorld.from_snapshot(env_snap)
        state = AgentState.from_dict(agent_state_dict)
        # the injected span re-emits its hint; rewind the ordinal counter so
        # later spans keep the trace-time ordinal sequence
        self._rewind_contract_count(state, span.meta_hint)
        agent_cfg = self.agents.get(trace.run_meta.agent_id)
        if agent_cfg is None:
            raise KeyError("unknown agent %r" % trace.run_meta.agent_id)
        agent = SampleAgent(agent_cfg, state=state)
        driver = ReplayDriver(override_index=span.invocation_index,
                              baseline_client=baseline_client)
        session = MockSession(
            runtime=self, agent=agent, world=world, driver=driver,
            trace=trace,
            original_world_snapshot=states.get("env:0", env_snap),
            snapshot_hash=world.state_hash(),
            next_invocation_index=state.next_invocation_index,
        )
        if state.next_invocation_index != span.invocation_index:
            raise RuntimeError("restored state next_invocation_index=%d != span index %d"
                               % (state.next_invocation_index, span.invocation_index))
        return session

    @staticmethod
    def _rewind_contract_count(state: AgentState, meta_hint: str) -> None:
        parts = (meta_hint or "").split("|")
        if len(parts) != 3:
            return
        task_id, kind, ordinal = parts[0], parts[1], int(parts[2])
        key = "%s:%s" % (task_id, kind)
        if key in state.contract_count and state.contract_count[key] == ordinal + 1:
            state.contract_count[key] = ordinal

    # -- 14.4 step 4 --------------------------------------------------------
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

    # -- 14.4 step 5 --------------------------------------------------------
    def parse(self, span: SpanRecord, raw: str) -> ParseResult:
        ok, parsed, error = extract_json(raw)
        return ParseResult(ok=ok, parsed=parsed, error=error)

    def schema_check(self, session: MockSession, span: SpanRecord,
                     parsed: Any) -> bool:
        schema_ref = span.output_contract_runtime.schema_ref
        if not schema_ref:
            return True
        schema = (session.trace.artifacts.get("schemas") or {}).get(schema_ref)
        if schema is None:
            return True
        return not validate_schema(parsed, schema)

    def verify_local(self, session: MockSession, span: SpanRecord,
                     parsed: Any) -> VerifierVerdict:
        """Local verifiers per span.verifier_runtime (pre-commit semantics;
        nothing is committed by an SLM output before injection)."""
        types = list(span.verifier_runtime.verifier_type or ["none"])
        verdict = VerifierVerdict()
        for vtype in types:
            verdict = verdict.merge(self._run_one_verifier(session, span, parsed, vtype))
        if not types or types == ["none"]:
            return no_verifier_verdict()
        return verdict

    def _run_one_verifier(self, session: MockSession, span: SpanRecord,
                          parsed: Any, vtype: str) -> VerifierVerdict:
        if vtype == "none":
            return no_verifier_verdict()
        if vtype == "schema_check":
            ok = self.schema_check(session, span, parsed)
            return VerifierVerdict(triggered=True, detected_failure=not ok,
                                   passed=ok, executed=["schema_check"],
                                   errors=[] if ok else ["schema invalid"])
        if vtype == "tool_exec_success":
            tool = (parsed or {}).get("tool", "")
            args = (parsed or {}).get("arguments") or {}
            allowed = list(span.output_contract_runtime.allowed_tools or [])
            if tool == "__done__":  # workflow sentinel, not a real tool
                return VerifierVerdict(triggered=True, detected_failure=False,
                                       passed=True, executed=["tool_exec_success"],
                                       errors=[])
            if tool not in allowed:
                return VerifierVerdict(triggered=True, detected_failure=True,
                                       passed=False, executed=["tool_exec_success"],
                                       errors=["tool %r not in allowed_tools" % tool])
            ok, err = session.world.dry_run(tool, args)
            return VerifierVerdict(triggered=True, detected_failure=not ok,
                                   passed=ok, executed=["tool_exec_success"],
                                   errors=[] if ok else ["dry_run: %s" % err])
        if vtype == "policy_check":
            decision = (parsed or {}).get("decision")
            if decision not in ("allow", "deny"):
                return VerifierVerdict(triggered=True, detected_failure=True,
                                       passed=False, executed=["policy_check"],
                                       errors=["decision must be allow/deny"])
            # ground-truth policy verdict for the proposed action (recovered
            # from the span's recorded request, which embeds the action JSON)
            action = self._proposed_action(session, span)
            expected = self._policy_verdict(session.world, action)
            ok = (decision == expected)
            return VerifierVerdict(triggered=True, detected_failure=not ok,
                                   passed=ok, executed=["policy_check"],
                                   errors=[] if ok else ["policy mismatch: expected %s" % expected])
        if vtype in ("goal_state_match", "db_state_match"):
            # verify the goal-check claim against the actual world (only for
            # tasks whose goal predicates are after-world functions)
            claim = bool((parsed or {}).get("goal_met"))
            actual = self._goal_actual(session)
            if actual is None:
                return VerifierVerdict(triggered=True, detected_failure=False,
                                       passed=True, executed=[vtype],
                                       errors=[])
            ok = (claim == actual)
            return VerifierVerdict(triggered=True, detected_failure=not ok,
                                   passed=ok, executed=[vtype],
                                   errors=[] if ok else ["goal claim %s vs actual %s" % (claim, actual)])
        return no_verifier_verdict()

    @staticmethod
    def _policy_verdict(world: MockWorld, action: Optional[Dict[str, Any]]) -> str:
        to = str((action or {}).get("arguments", {}).get("to", ""))
        from .mock_world import EXTERNAL_DOMAINS
        if not any(dom in to for dom in EXTERNAL_DOMAINS):
            return "allow"  # internal sends always allowed
        approved = bool(world.state.get("external_send_approved", False))
        return "allow" if approved else "deny"

    @staticmethod
    def _proposed_action(session: MockSession, span: SpanRecord) -> Optional[Dict[str, Any]]:
        """Recover the proposed action from the policy span's recorded request
        (the prompt embeds 'Proposed action: {...}')."""
        try:
            messages = session.trace.artifacts["messages"][span.request.messages_ref]
            last_user = next((m["content"] for m in reversed(messages)
                              if m.get("role") == "user"), "")
            marker = "Proposed action: "
            pos = last_user.rfind(marker)
            if pos < 0:
                return None
            tail = last_user[pos + len(marker):]
            # cut out the complete JSON by brace pairing: arguments itself has nested braces, so a regex
            # [^{}]* can never match -> this once made the policy determination constantly "allow"
            start = tail.find("{")
            if start < 0:
                return None
            depth = 0
            for i in range(start, len(tail)):
                if tail[i] == "{":
                    depth += 1
                elif tail[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(tail[start:i + 1])
            return None
        except (KeyError, ValueError, StopIteration):
            return None

    def _goal_actual(self, session: MockSession) -> Optional[bool]:
        from .mock_world import EVALUATOR_SPECS, GOAL_PREDICATES
        task_id = session.trace.run_meta.task_id
        task = self.manifest.get(task_id)
        if task is not None and task.evaluator_type == "answer_match":
            return None  # answer-dependent goals cannot pre-verify at span time
        spec = EVALUATOR_SPECS.get(task_id, {})
        names = spec.get("goal_predicates", [])
        if not names or not all(n in GOAL_PREDICATES for n in names):
            return None  # before-world-dependent predicates cannot pre-verify
        return all(bool(GOAL_PREDICATES[n](session.world, "")) for n in names)

    # -- 14.4 step 6 --------------------------------------------------------
    def inject(self, session: MockSession, span: SpanRecord, parsed: Any) -> bool:
        session.driver.override_output = parsed
        return True

    # -- 14.4 steps 7-8 -----------------------------------------------------
    def run_to_node(self, session: MockSession, span: SpanRecord,
                    client: Any, max_steps: int, timeout_s: float,
                    budget_usd: float) -> Tuple[DownstreamOutcome, List[LMResponse]]:
        task = self.manifest.get(session.trace.run_meta.task_id)
        start = len(session.driver.usage)
        outcome = DownstreamOutcome()
        try:
            result = session.agent.resume(
                session.world, session.driver,
                stop_node=span.continuation.resume_node_id,
                branching=task.branching == "conditional" if task else False,
            )
            outcome.next_node_reached = bool(result.stop_node_reached)
            outcome.downstream_checkpoint_reached = bool(result.stop_node_reached)
            outcome.next_tool_executed = any(
                n.get("node") == "tool:%d" % span.invocation_index and n.get("ok")
                for n in session.agent.state.node_log)
            outcome.downstream_execution_valid = bool(
                result.stop_node_reached) and not result.aborted
        except StopReplay:
            raise
        except Exception as exc:
            outcome.unhandled_exception = True
            outcome.downstream_execution_valid = False
            outcome.next_node_reached = False
            outcome.downstream_checkpoint_reached = False
        usage = session.driver.usage[start:]
        return outcome, usage

    def run_to_completion(self, session: MockSession, span: SpanRecord,
                          client: Any, max_steps: int, timeout_s: float,
                          budget_usd: float) -> Tuple[FinalOutcome, List[LMResponse]]:
        task = self.manifest.get(session.trace.run_meta.task_id)
        start = len(session.driver.usage)
        final = FinalOutcome()
        try:
            result = session.agent.resume(
                session.world, session.driver, stop_node=None,
                branching=task.branching == "conditional" if task else False,
            )
            eval_result = evaluate_task(
                session.trace.run_meta.task_id,
                MockWorld(session.original_world_snapshot),
                session.world, result.final_answer)
            final.goal_state_passed = bool(eval_result.get("goal_state_passed"))
            final.final_task_success = final.goal_state_passed
            final.collateral_damage = bool(eval_result.get("collateral_damage"))
            final.evaluator_details = list(eval_result.get("details", []))
        except StopReplay:
            raise
        except Exception as exc:
            final.evaluator_details.append("exception: %s" % exc)
        usage = session.driver.usage[start:]
        return final, usage

    def rollback(self, session: MockSession) -> bool:
        # pre-commit verifiers only in this pilot: nothing committed by the
        # SLM output before injection, so rollback is trivially successful.
        return True
