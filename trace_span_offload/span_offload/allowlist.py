"""Lightweight static-profile utility validation (spec 14.8).

One lightweight check only: freeze a static allowlist from the pilot-split
span profiles, then run a multi-span end-to-end sanity check on held-out
tasks.  Rules (14.8):

    offload only if:
      read_only
      + structured_output
      + strong_pre_commit_verifier
      + lower_confidence_bound(O_m(phi)) >= threshold

Compared configurations: LLM-only, SLM-only, static profile allowlist,
allowlist + simple schema/verifier fallback, and the oracle single-span
opportunity upper bound.  No learned router, no policy optimization, no
cost/latency Pareto frontier as a main result (14.8).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .analysis import build_cells
from .harness.mock_runtime import run_policy_trace
from .harness.sample_agent import AgentConfig, AGENTS
from .runtime import SpanContext
from .schemas import ManifestTask, SpanDescriptor, VerifierRuntime

STRUCTURED_CONTRACTS = {"json_schema", "tool_name", "tool_arguments",
                        "structured_plan", "boolean_decision"}


@dataclass
class Allowlist:
    model: str = ""
    lcb_threshold: float = 0.8
    min_samples: int = 10
    created_from_split: str = "pilot"
    entries: Dict[str, bool] = field(default_factory=dict)  # unit_key -> allow

    def size(self) -> int:
        return sum(1 for v in self.entries.values() if v)


def build_allowlist(records: List[dict], model: str, min_samples: int,
                    lcb_threshold: float, split: str = "pilot") -> Allowlist:
    """Freeze a static allowlist from the pilot profiles: a unit cell is
    offloadable iff its Wilson lower confidence bound >= threshold (cells with
    insufficient support are NOT offloadable)."""
    cells = build_cells([r for r in records if r.get("model") == model],
                        mode="unit", min_samples=min_samples)
    entries: Dict[str, bool] = {}
    for key, stats in cells.items():
        entries[key] = (not stats["insufficient_support"]
                        and stats["ci_lo"] >= lcb_threshold)
    return Allowlist(model=model, lcb_threshold=lcb_threshold,
                     min_samples=min_samples, created_from_split=split,
                     entries=entries)


def rules_match(descriptor: SpanDescriptor,
                verifier_runtime: VerifierRuntime) -> Tuple[bool, List[str]]:
    """The 14.8 structural rules (independent of the profile threshold):
    read_only + structured_output + strong_pre_commit_verifier."""
    missing: List[str] = []
    effects = set(descriptor.mechanism_tags.state_effect)
    read_only = "read_only" in effects and "external_state_mutation" not in effects
    if not read_only:
        missing.append("read_only")
    contracts = set(descriptor.mechanism_tags.output_contract)
    structured = bool(contracts & STRUCTURED_CONTRACTS) and "freeform_text" not in contracts
    if not structured:
        missing.append("structured_output")
    vtypes = set(verifier_runtime.verifier_type)
    # strong pre-commit verifier = verifier runs before commit AND at least one
    # executable local check (schema / tool dry-run / policy / goal / db state)
    strong_verifiers = {"schema_check", "tool_exec_success", "policy_check",
                        "goal_state_match", "db_state_match"}
    strong_pre_commit = bool(verifier_runtime.pre_commit) and bool(vtypes & strong_verifiers)
    if not strong_pre_commit:
        missing.append("strong_pre_commit_verifier")
    return (not missing), missing


def is_offloadable(allowlist: Allowlist, unit_key: str,
                   descriptor: SpanDescriptor,
                   verifier_runtime: VerifierRuntime) -> Tuple[bool, str]:
    """Static decision: structural rules AND frozen profile LCB threshold."""
    rules_ok, missing = rules_match(descriptor, verifier_runtime)
    if not rules_ok:
        return False, "rules_failed:" + ",".join(missing)
    if not allowlist.entries.get(unit_key, False):
        return False, "profile_lcb_below_threshold_or_insufficient_support"
    return True, "allow"


def make_policy(allowlist: Allowlist) -> Callable[[SpanContext], str]:
    """Span-level policy closure for multi-span runs."""
    def policy(ctx: SpanContext) -> str:
        if ctx.verifier_runtime is None or ctx.descriptor is None:
            return "baseline"
        ok, _reason = is_offloadable(allowlist, _unit_key_of(ctx),
                                     ctx.descriptor, ctx.verifier_runtime)
        return "slm" if ok else "baseline"
    return policy


def _unit_key_of(ctx: SpanContext) -> str:
    import json
    d = ctx.descriptor
    unit = {
        "node_pattern": d.node_pattern,
        "control_role": d.control_role,
        "state_effect": sorted(d.mechanism_tags.state_effect),
        "verifier_type": sorted(d.verifier_type),
    }
    return json.dumps(unit, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Validation comparison (14.8)
# ---------------------------------------------------------------------------

CONFIGS = ["llm_only", "slm_only", "allowlist", "allowlist_fallback"]


def run_validation(heldout_tasks: List[ManifestTask],
                   agents: List[AgentConfig],
                   baseline_client: Any, slm_client: Any,
                   allowlist: Allowlist,
                   progress: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
    """Multi-span end-to-end sanity check on held-out tasks across the four
    runnable configs.  Each config x task x agent is run once."""
    results: Dict[str, Any] = {"configs": {}, "runs": []}

    def policy_llm_only(_ctx: SpanContext) -> str:
        return "baseline"

    def policy_slm_only(_ctx: SpanContext) -> str:
        return "slm"

    allow_policy = make_policy(allowlist)

    for agent_cfg in agents:
        for task in heldout_tasks:
            for config in CONFIGS:
                if progress:
                    progress(task.task_id, "%s/%s" % (agent_cfg.agent_id, config))
                if config == "llm_only":
                    policy, fallback = policy_llm_only, False
                elif config == "slm_only":
                    policy, fallback = policy_slm_only, False
                elif config == "allowlist":
                    policy, fallback = allow_policy, False
                else:  # allowlist_fallback
                    policy, fallback = allow_policy, True
                stats, _result, _eval = run_policy_trace(
                    task, agent_cfg, baseline_client, slm_client,
                    policy=policy, fallback=fallback)
                stats["config"] = config
                results["runs"].append(stats)

    # aggregate per config
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
            "n_spans": sum(r["n_spans"] for r in runs),
        }
    return results


def oracle_bound(heldout_baseline_spans: List[dict],
                 pilot_records: List[dict], model: str,
                 min_samples: int, threshold: float) -> Dict[str, Any]:
    """Oracle single-span opportunity upper bound: the maximum achievable cost
    saving if every span whose pilot profile reaches the threshold were
    offloaded (assumes SLM success and zero cost for those spans)."""
    cells = build_cells([r for r in pilot_records if r.get("model") == model],
                        mode="unit", min_samples=min_samples)
    offloadable = {key for key, stats in cells.items()
                   if not stats["insufficient_support"] and stats["point_estimate"] >= threshold}
    spans = [s for s in heldout_baseline_spans if s.get("eligible")]
    hit = [s for s in spans if s.get("unit_key") in offloadable]
    opportunity = sum(s.get("baseline_span_cost", 0.0) for s in hit)
    return {
        "model": model,
        "threshold": threshold,
        "n_heldout_eligible_spans": len(spans),
        "n_oracle_offloadable": len(hit),
        "oracle_opportunity_cost_bound": opportunity,
        "oracle_fraction_offloadable": (len(hit) / len(spans)) if spans else float("nan"),
    }
