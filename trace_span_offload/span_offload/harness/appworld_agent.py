"""Policy-driven agent loop over the REAL AppWorld environment (14.8).

The AppWorld-backed analog of mock_runtime.run_policy_trace: a fresh episode
is run on an AppWorld task; every LM call is a span with a SpanContext
descriptor, a static policy decides per span whether the candidate SLM or the
baseline LLM answers (optional fallback to the baseline on parse failure),
tool calls execute in the real environment, and the AppWorld final evaluator
judges success and collateral damage.

The loop mirrors the span structure of the exported traces:

    span 0             tool_selection            (api prediction, freeform)
    spans 1..k         tool_argument_generation  (tool-call JSON loop,
                                                 repair_controller after a
                                                 failure observation)
    final step         final_synthesis           (supervisor__complete_task)

Prompt templates are taken from the check_point_trace exporter's AutoGen
agent so new runs stay comparable with the recorded ones.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..runtime import SpanContext
from ..schemas import (
    GraphContext, MechanismTags, SpanDescriptor, VerifierRuntime,
)
from ..appworld_adapter import parse_llm_output
from .appworld_env import AppWorldBridge, APP_API_SEPARATOR

#: The system prompt the AutoGen exporter's agent used for api prediction
#: (span 0 of the check_point traces).
API_PREDICTOR_SYSTEM = (
    "You are an AI Assistant. Your task is to analyze a given complex user "
    "request and determine which available APIs would be useful to accomplish "
    "it autonomously on behalf of the user (supervisor).\n\n"
    "Available APIs:\n{api_list}\n\n"
    "Only output the API names, one per line."
)

API_PREDICTOR_USER = (
    "# Task Instruction\n{instruction}\n\n"
    "List all APIs that may be needed to complete this task. "
    "If you are unsure whether a certain API is useful, include it."
)

#: The system prompt the AutoGen exporter's agent used for the tool-call loop
#: (spans 1..k of the check_point traces).
TOOL_LOOP_SYSTEM = (
    "You are an AI assistant. Call functions to complete the task. "
    "Always call supervisor__complete_task when done.\n\n"
    "Available tools:\n{tool_docs}\n\n"
    "For each step, output EXACTLY a JSON list of tool calls in the form: "
    '[{{"name": "<tool_name>", "arguments": {{<json object>}}}}].'
)

MAX_STEPS_DEFAULT = 64
TIMEOUT_S_DEFAULT = 600.0


class StopLoop(Exception):
    """Raised when the step/time budget trips."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Span context construction (descriptors match appworld_adapter.classify)
# ---------------------------------------------------------------------------

def _tool_effect(tool_name: str) -> str:
    from ..appworld_adapter import APWORLD_TOOL_EFFECTS
    return APWORLD_TOOL_EFFECTS.get(tool_name.replace("__", "."),
                                    "external_state_mutation")


def _span_ctx(invocation_index: int, node_pattern: str, control_role: str,
              phase: str, subtask: str, output_tags: List[str],
              verifier_types: List[str], allowed_tools: List[str],
              after_failure: bool = False) -> SpanContext:
    state_effect = ["read_only"]  # refined by the caller when known
    if node_pattern == "tool_argument_generation":
        state_effect = sorted({_tool_effect(t) for t in allowed_tools}) or ["read_only"]
    return SpanContext(
        invocation_index=invocation_index,
        node_pattern=node_pattern,
        control_role=control_role,
        contract_type=("tool_call_list" if node_pattern == "tool_argument_generation"
                       else "freeform_text"),
        span_subtask_type=subtask,
        descriptor=SpanDescriptor(
            node_pattern=node_pattern,
            control_role=control_role,
            span_subtask_type=subtask,
            mechanism_tags=MechanismTags(
                interface=["api_tool"],
                context_source=["conversation_history", "tool_schema",
                                "previous_tool_results"],
                output_contract=output_tags,
                state_effect=state_effect,
            ),
            verifier_type=list(verifier_types),
            graph_context=GraphContext(
                inside_loop=node_pattern == "tool_argument_generation",
                control_phase=phase,
                after_failure=after_failure,
                loop_iteration_bucket="1",
            ),
        ),
        verifier_runtime=VerifierRuntime(
            verifier_type=list(verifier_types),
            verifier_location="pre_commit",
            pre_commit=True,
            rollback_available=False,
        ),
        allowed_tools=list(allowed_tools),
        meta_hint="appworld|%s|%d" % (node_pattern, invocation_index),
        temperature=0.0,
        max_output_tokens=8192,
    )


# ---------------------------------------------------------------------------
# Policy caller: per-span model calls with policy + fallback + usage tracking
# ---------------------------------------------------------------------------

class PolicyCaller:
    """Routes every span to the SLM or the baseline per the static policy.
    When the SLM output cannot be parsed, on_local_failure decides the
    recovery strategy (mirrors mock_runtime.MultiSpanDriver):

      "fallback" -> the baseline re-computes THIS span (recorded fallback
                    cost), the workflow continues downstream.
      "skip"     -> the span is recorded as failed WITHOUT any baseline call;
                    the failure is fed back as an observation and the loop
                    moves on to the NEXT span's offload decision.
    """

    ON_LOCAL_FAILURE_MODES = {"fallback", "skip"}

    def __init__(self, policy: Callable[[SpanContext], str],
                 baseline_client: Any, slm_client: Any,
                 on_local_failure: str = "fallback"):
        if on_local_failure not in self.ON_LOCAL_FAILURE_MODES:
            raise ValueError("on_local_failure must be one of %s"
                             % sorted(self.ON_LOCAL_FAILURE_MODES))
        self.policy = policy
        self.baseline_client = baseline_client
        self.slm_client = slm_client
        self.on_local_failure = on_local_failure
        self.stats: Dict[str, Any] = {
            "offloaded": 0, "baseline_calls": 0, "fallbacks": 0, "skipped": 0,
            "baseline_usage": [], "slm_usage": [], "fallback_usage": [],
        }

    def complete(self, messages: List[Dict[str, str]],
                 ctx: SpanContext, tools: Any = None) -> Any:
        if self.policy(ctx) == "slm":
            self.stats["offloaded"] += 1
            resp = self.slm_client.complete(
                messages, ctx=ctx, tools=tools,
                temperature=ctx.temperature,
                max_tokens=ctx.max_output_tokens)
            self.stats["slm_usage"].append(resp)
            return resp
        self.stats["baseline_calls"] += 1
        resp = self.baseline_client.complete(
            messages, ctx=ctx, tools=tools,
            temperature=ctx.temperature,
            max_tokens=ctx.max_output_tokens)
        self.stats["baseline_usage"].append(resp)
        return resp

    def fallback_complete(self, messages: List[Dict[str, str]],
                          ctx: SpanContext, tools: Any = None) -> Any:
        self.stats["fallbacks"] += 1
        resp = self.baseline_client.complete(
            messages, ctx=ctx, tools=tools,
            temperature=ctx.temperature,
            max_tokens=ctx.max_output_tokens)
        self.stats["fallback_usage"].append(resp)
        return resp


# ---------------------------------------------------------------------------
# The episode loop
# ---------------------------------------------------------------------------

def _parse_turn(resp: Any) -> Tuple[Any, str]:
    """Parse one tool-loop turn: native tool_calls from the API protocol win
    (format enforced by the API); text output falls back to the prompt-format
    parser."""
    if getattr(resp, "tool_calls", None):
        return ({"tool_calls": [{"name": c["name"], "arguments": c["arguments"]}
                                for c in resp.tool_calls]}, "tool_call_list")
    return parse_llm_output(getattr(resp, "raw_text", "") or "")


def _tool_docs_text(docs: List[Dict[str, Any]]) -> str:
    lines = []
    for doc in docs:
        inputs = ", ".join(
            "%s: %s" % (k, v.get("type", "any"))
            for k, v in (doc.get("inputs") or {}).items())
        lines.append("- %s (%s): %s" % (doc["name"], inputs,
                                        doc.get("description", "")))
    return "\n".join(lines)


def _api_list_text(docs: List[Dict[str, Any]]) -> str:
    return "\n".join("- %s: %s" % (d["name"], d.get("description", ""))
                     for d in docs)


def _tool_result_text(res: Dict[str, Any]) -> str:
    if res["ok"]:
        return "Observation: %s" % json.dumps(res["result"], ensure_ascii=False)[:2000]
    return "Error: %s" % res["error"][:2000]


def run_appworld_task(
        task_id: str,
        baseline_client: Any,
        slm_client: Any,
        policy: Callable[[SpanContext], str],
        on_local_failure: str = "fallback",
        max_steps: int = MAX_STEPS_DEFAULT,
        timeout_s: float = TIMEOUT_S_DEFAULT,
) -> Dict[str, Any]:
    """One fresh episode on an AppWorld task under a span policy.

    on_local_failure: "fallback" (baseline re-computes the failed span and the
    workflow continues downstream) or "skip" (the failed span is recorded
    without any baseline call; the failure becomes an observation and the next
    span's offload decision proceeds).

    Returns the same stats shape as mock_runtime.run_policy_trace:
    {task_id, final_task_success, collateral_damage, cost, latency_ms,
     n_offloaded, n_fallbacks, n_skipped, n_spans, stop_reason,
     evaluator_details}."""
    caller = PolicyCaller(policy, baseline_client, slm_client,
                          on_local_failure=on_local_failure)
    # time.time() (wall clock) on purpose: this machine's perf_counter has
    # been observed to jump to epoch-like values intermittently, which would
    # trip the deadline guard spuriously.
    t0 = time.time()
    deadline = t0 + timeout_s
    invocation = 0
    stats: Dict[str, Any] = {
        "task_id": task_id,
        "stop_reason": "unknown",
        "evaluator_details": [],
    }

    def budget_guard() -> None:
        if invocation >= max_steps:
            raise StopLoop("max_steps")
        if time.time() > deadline:
            raise StopLoop("timeout")

    def cost_of(usage: List[Any], price_fn: Callable[..., float]) -> float:
        return sum(price_fn(r.prompt_tokens, r.completion_tokens) for r in usage)

    try:
        with AppWorldBridge(task_id) as bridge:
            docs = bridge.tool_docs()
            names = [d["name"] for d in docs]

            # ---- span 0: tool_selection (api prediction) -----------------
            budget_guard()
            ctx = _span_ctx(
                invocation, "tool_selection", "global_planner", "initial",
                "api_prediction", ["freeform_text"], ["none"], names)
            messages = [
                {"role": "system", "content": API_PREDICTOR_SYSTEM.format(
                    api_list=_api_list_text(docs))},
                {"role": "user", "content": API_PREDICTOR_USER.format(
                    instruction=bridge.instruction)},
            ]
            resp = caller.complete(messages, ctx)
            if resp.error:
                predicted: List[str] = []
            else:
                predicted = [n.replace("__", ".").strip() for n in
                             resp.raw_text.splitlines() if n.strip()]
            allowed = [n for n in names if n.replace("__", ".") in
                       set(predicted)] or names  # empty prediction -> all docs
            allowed = allowed[:32]  # keep the tool loop prompt bounded
            # native function-calling tool schemas for the tool loop: the
            # API enforces the tool-call format instead of prompt compliance
            from .appworld_runtime import _openai_tool_schemas
            tool_schemas = _openai_tool_schemas(bridge, allowed)
            invocation += 1

            # ---- spans 1..k: tool_argument_generation loop ---------------
            loop_messages = [
                {"role": "system", "content": TOOL_LOOP_SYSTEM.format(
                    tool_docs=_tool_docs_text(
                        [d for d in docs if d["name"] in allowed]))},
                {"role": "user", "content": bridge.instruction},
            ]
            after_failure = False
            completed = False
            last_calls_key = ""
            repeat_count = 0
            while True:
                budget_guard()
                ctx = _span_ctx(
                    invocation, "tool_argument_generation", "local_executor",
                    "recovery" if after_failure else "execution",
                    "error_recovery_retry" if after_failure else "tool_parameter_extraction",
                    ["tool_name", "tool_arguments", "json_schema"],
                    ["schema_check", "tool_exec_success"], allowed,
                    after_failure=after_failure)
                resp = caller.complete(loop_messages, ctx, tools=tool_schemas)
                parsed, kind = _parse_turn(resp)
                if kind != "tool_call_list":
                    # SLM output not parseable: per on_local_failure, either
                    # the baseline re-computes THIS span ("fallback") or the
                    # span is recorded as failed and the loop moves to the
                    # NEXT span's offload decision ("skip", no baseline call).
                    if caller.on_local_failure == "fallback":
                        resp = caller.fallback_complete(loop_messages, ctx,
                                                        tools=tool_schemas)
                        parsed, kind = _parse_turn(resp)
                    if kind != "tool_call_list":
                        caller.stats["skipped"] += 1
                        loop_messages.append({
                            "role": "assistant", "content": resp.raw_text})
                        loop_messages.append({
                            "role": "user",
                            "content": "Error: expected a JSON list of tool "
                                       "calls [{\"name\": ..., \"arguments\": {...}}]."})
                        after_failure = True
                        invocation += 1
                        continue
                calls = parsed.get("tool_calls", [])
                calls_key = json.dumps(
                    [(c.get("name"), c.get("arguments")) for c in calls],
                    sort_keys=True, ensure_ascii=False)
                if calls_key == last_calls_key:
                    repeat_count += 1
                    if repeat_count >= 3:
                        loop_messages.append({"role": "user", "content":
                                              "stagnation: identical tool calls repeated; loop stopped."})
                        break
                else:
                    last_calls_key = calls_key
                    repeat_count = 0
                assistant_text = json.dumps(calls, ensure_ascii=False)
                loop_messages.append({"role": "assistant", "content": assistant_text})
                span_failed = False
                for call in calls:
                    name = str(call.get("name", "")).replace("__", ".")
                    args = call.get("arguments")
                    if not isinstance(args, dict):
                        args = {}
                    if name.endswith("supervisor.complete_task") or \
                            name.endswith("complete_task"):
                        status = str(args.get("status", "success"))
                        if status not in ("success", "fail"):
                            status = "success"  # the supervisor API only
                            # accepts success/fail; normalize model output
                        try:
                            bridge.complete_task(args.get("answer"), status)
                            completed = True
                        except Exception as exc:
                            loop_messages.append({"role": "user", "content":
                                                  "Error: complete_task rejected: %s" % exc})
                            span_failed = True
                        break
                    if name.replace(".", "__") not in allowed:
                        loop_messages.append({"role": "user", "content":
                                              "Error: tool %r not in the allowed tool set." % name})
                        span_failed = True
                        continue
                    res = bridge.execute_tool(name, args)
                    loop_messages.append(
                        {"role": "user", "content": _tool_result_text(res)})
                    if not res["ok"]:
                        span_failed = True
                invocation += 1
                if completed:
                    break
                after_failure = span_failed

            # ---- final evaluation ----------------------------------------
            eval_result = bridge.evaluate()
            stats.update({
                "final_task_success": eval_result["final_task_success"],
                "goal_state_passed": eval_result["goal_state_passed"],
                "collateral_damage": eval_result["collateral_damage"],
                "evaluator_details": eval_result["evaluator_details"],
                "stop_reason": "complete_task" if completed else "max_steps",
            })
    except StopLoop as stop:
        stats.update({
            "final_task_success": False, "collateral_damage": False,
            "stop_reason": stop.reason,
            "evaluator_details": ["loop stopped: %s" % stop.reason],
        })
    except Exception as exc:  # one episode never kills the validation batch
        stats.update({
            "final_task_success": False, "collateral_damage": False,
            "stop_reason": "exception",
            "evaluator_details": ["exception: %s: %s" % (type(exc).__name__, exc)],
        })

    stats.update({
        "cost": (cost_of(caller.stats["baseline_usage"], baseline_client.cost_for)
                 + cost_of(caller.stats["slm_usage"], slm_client.cost_for)
                 + cost_of(caller.stats["fallback_usage"], baseline_client.cost_for)),
        "latency_ms": (time.time() - t0) * 1000.0,
        "n_offloaded": caller.stats["offloaded"],
        "n_fallbacks": caller.stats["fallbacks"],
        "n_skipped": caller.stats["skipped"],
        "on_local_failure": caller.on_local_failure,
        "n_spans": invocation,
    })
    return stats


# ---------------------------------------------------------------------------
# Allowlist validation on real AppWorld tasks (14.8)
# ---------------------------------------------------------------------------

CONFIGS = ["llm_only", "slm_only", "allowlist", "allowlist_fallback"]


def run_appworld_validation(
        task_ids: List[str],
        baseline_client: Any,
        slm_client: Any,
        allowlist: Optional[Any] = None,
        on_local_failure: str = "fallback",
        progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """Multi-span end-to-end sanity check on real AppWorld tasks across the
    four runnable configs (14.8).  Each config x task is run once; the
    allowlist configs require an Allowlist built from pilot profiles.

    on_local_failure: "fallback" (baseline re-computes a failed span,
    workflow continues downstream) or "skip" (failed span recorded without a
    baseline call, next span's offload proceeds).  Applies to every config."""
    results: Dict[str, Any] = {"configs": {}, "runs": []}

    def policy_llm_only(_ctx: SpanContext) -> str:
        return "baseline"

    def policy_slm_only(_ctx: SpanContext) -> str:
        return "slm"

    allow_policy: Callable[[SpanContext], str] = policy_llm_only
    if allowlist is not None:
        from ..allowlist import make_policy
        allow_policy = make_policy(allowlist)

    for task_id in task_ids:
        for config in CONFIGS:
            if progress:
                progress(task_id, config)
            if config == "llm_only":
                policy = policy_llm_only
            elif config == "slm_only":
                policy = policy_slm_only
            else:  # allowlist / allowlist_fallback
                policy = allow_policy
            stats = run_appworld_task(
                task_id, baseline_client, slm_client,
                policy=policy, on_local_failure=on_local_failure)
            stats["config"] = config
            results["runs"].append(stats)

    for config in CONFIGS:
        runs = [r for r in results["runs"] if r["config"] == config]
        n = len(runs)
        results["configs"][config] = {
            "n_runs": n,
            "n_success": sum(1 for r in runs if r["final_task_success"]),
            "n_collateral": sum(1 for r in runs if r["collateral_damage"]),
            "final_task_success_rate": (sum(1 for r in runs if r["final_task_success"]) / n) if n else float("nan"),
            "collateral_rate": (sum(1 for r in runs if r["collateral_damage"]) / n) if n else float("nan"),
            "total_cost": sum(r["cost"] for r in runs),
            "mean_latency_ms": (sum(r["latency_ms"] for r in runs) / n) if n else float("nan"),
            "n_offloaded": sum(r["n_offloaded"] for r in runs),
            "n_fallbacks": sum(r["n_fallbacks"] for r in runs),
            "n_skipped": sum(r["n_skipped"] for r in runs),
            "n_spans": sum(r["n_spans"] for r in runs),
        }
    return results
