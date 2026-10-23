"""Conditional offloadability analysis (spec 14.6-14.7).

Strict success:
    S_s,m^strict = I[ final_task_success = 1 AND collateral_damage = 0 ]

Conditional offloadability for candidate model m and span descriptor phi:
    O_m^strict(phi) = P( S = 1 | phi_s = phi, Y_s^LLM = 1, M = m )

Y_s^LLM = 1 is enforced at extraction time (only spans from baseline-successful
runs are eligible), so the estimator conditions on it by construction.

Everything below operates on plain dict records (replays.jsonl rows).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .schemas import MISSING_DESCRIPTOR_CHECKLIST


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mean_ci(values: List[float], z: float = 1.96) -> tuple:
    """Normal-approximation 95% CI for a mean."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    m = sum(values) / n
    if n == 1:
        return (m, m)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return (m - z * se, m + z * se)


def _rate(records: List[dict], key: str) -> float:
    if not records:
        return float("nan")
    return sum(1 for r in records if _get(r, key)) / len(records)


def _get(record: dict, dotted: str, default: Any = None) -> Any:
    node: Any = record
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def filter_source(records: List[dict], source: Optional[str]) -> List[dict]:
    """Keep only records of one outcome source.  None = no filtering.
    outcome_source is "live" (full replay) or "trace_local" (local layer only,
    never counted in O_strict profiles)."""
    if source is None:
        return records
    return [r for r in records
            if _get(r, "replay_meta.outcome_source", "live") == source]


def _final_success(r: dict) -> bool:
    return bool(_get(r, "final_outcome.final_task_success"))


def _collateral(r: dict) -> bool:
    return bool(_get(r, "final_outcome.collateral_damage"))


def _strict(r: dict) -> bool:
    return bool(r.get("strict_success"))


# ---------------------------------------------------------------------------
# Cell statistics (14.6) -- every cell reports at least:
# sample_count, point_estimate (of strict success) + 95% CI, local success,
# downstream success, final-task success, collateral-damage rate.
# ---------------------------------------------------------------------------

def cell_stats(records: List[dict], min_samples: int = 10) -> dict:
    n = len(records)
    k = sum(1 for r in records if _strict(r))
    lo, hi = wilson_ci(k, n)
    # verifier detection quality (documented operationalization):
    #   bad  = parse/schema invalid (ground-truth local failure)
    #   detect_rate       = P(verifier detects failure | output bad)
    #   false_reject_rate = P(verifier detects failure | output ok)
    bad = [r for r in records if not (_get(r, "local_outcome.parse_success") and _get(r, "local_outcome.schema_valid"))]
    good = [r for r in records if _get(r, "local_outcome.parse_success") and _get(r, "local_outcome.schema_valid")]
    detect = sum(1 for r in bad if _get(r, "fallback.verifier_detected_failure")) / len(bad) if bad else float("nan")
    false_reject = sum(1 for r in good if _get(r, "fallback.verifier_detected_failure")) / len(good) if good else float("nan")
    savings = [r["resource"]["cost_saving"] for r in records if r.get("resource")]
    latency = [r["resource"]["latency_saving"] for r in records if r.get("resource")]
    return {
        "sample_count": n,
        "strict_successes": k,
        "point_estimate": k / n if n else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
        "local_success": _rate(records, "local_outcome.local_output_valid"),
        "downstream_success": _rate(records, "downstream_outcome.downstream_checkpoint_reached"),
        "final_task_success": _rate(records, "final_outcome.final_task_success"),
        "collateral_damage_rate": _rate(records, "final_outcome.collateral_damage"),
        "fallback_rate": _rate(records, "fallback.fallback_triggered"),
        "detect_rate": detect,
        "false_reject_rate": false_reject,
        "mean_cost_saving": mean_ci(savings)[0] if savings else float("nan"),
        "mean_latency_saving": mean_ci(latency)[0] if latency else float("nan"),
        "insufficient_support": n < min_samples,
    }


def group_records(records: List[dict], mode: str = "unit") -> Dict[str, List[dict]]:
    """Group replay records by stratification key.

    mode: 'unit'  -> u = (node_pattern, control_role, state_effect, verifier_type)
          'full'  -> full intrinsic span descriptor phi_s
          or any single descriptor field name ('node_pattern', 'control_role', ...)
    """
    groups: Dict[str, List[dict]] = {}
    for r in records:
        if mode == "unit":
            key = r.get("unit_key") or ""
        elif mode == "full":
            key = _descriptor_key(r.get("span_descriptor") or {})
        else:
            d = r.get("span_descriptor") or {}
            key = str(d.get(mode, "")) if isinstance(d, dict) else str(d)
        groups.setdefault(key, []).append(r)
    return groups


def _descriptor_key(descriptor: dict) -> str:
    import json

    def canon(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: canon(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            try:
                return sorted(canon(x) for x in obj)
            except TypeError:
                return [canon(x) for x in obj]
        return obj

    return json.dumps(canon(descriptor), sort_keys=True, ensure_ascii=False)


def build_cells(records: List[dict], mode: str = "unit",
                min_samples: int = 10) -> Dict[str, dict]:
    """Per-cell stats incl. the full multidimensional profile (14.6)."""
    cells: Dict[str, dict] = {}
    for key, group in group_records(records, mode).items():
        stats = cell_stats(group, min_samples)
        stats["cell_key"] = key
        cells[key] = stats
    return cells


def profile_m(records: List[dict], model: str,
              mode: str = "unit", min_samples: int = 10,
              source: Optional[str] = "live") -> dict:
    """Profile_m(phi) = { p_local, p_downstream, p_final, p_collateral,
    p_fallback, DetectRate, FalseRejectRate, CostSaving, LatencySaving } plus
    the strict-success estimate with CI.  Never a single aggregate score."""
    records = filter_source(records, source)
    records = [r for r in records if r.get("model") == model]
    cells = build_cells(records, mode, min_samples)
    return {
        "model": model,
        "mode": mode,
        "min_samples": min_samples,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Task-conditioned analysis (14.7.1)
# ---------------------------------------------------------------------------

TASK_DIMENSIONS = [
    ("cross_app", lambda t: "cross_app" if t.get("cross_app") else "single_app"),
    ("horizon", lambda t: t.get("horizon", "short")),
    ("task_state_effect", lambda t: t.get("task_state_effect", "read_only")),
    ("collateral_risk", lambda t: t.get("collateral_risk", "low")),
    ("evaluator_type", lambda t: t.get("evaluator_type", "")),
    ("task_type", lambda t: t.get("task_type", "")),
]

TASK_DIMENSION_LABELS = {
    "cross_app": "single-app vs cross-app",
    "horizon": "horizon",
    "task_state_effect": "read-only vs state-mutating task",
    "collateral_risk": "collateral risk",
    "evaluator_type": "evaluator class",
    "task_type": "task type",
}


def task_dimension_tables(records: List[dict], min_samples: int = 10,
                           source: Optional[str] = "live") -> Dict[str, Any]:
    """One table per task-condition dimension: O_m(phi | z) for each level.
    Kept per-dimension on purpose -- no high-dimensional Cartesian products in
    round one (14.7.1)."""
    records = filter_source(records, source)
    out: Dict[str, Any] = {}
    for dim, fn in TASK_DIMENSIONS:
        levels: Dict[str, List[dict]] = {}
        for r in records:
            level = str(fn(r.get("task_condition") or {}))
            levels.setdefault(level, []).append(r)
        rows = {}
        for level, group in sorted(levels.items()):
            stats = cell_stats(group, min_samples)
            rows[level] = stats
        out[dim] = {"label": TASK_DIMENSION_LABELS.get(dim, dim), "levels": rows}
    return out


def delta_task(records: List[dict], unit_key: str = "",
               min_samples: int = 10, source: Optional[str] = "live") -> Optional[dict]:
    """Heterogeneity gap (14.7.1):
        Delta_task(phi) = max_z O_m(phi | z) - min_z O_m(phi | z)
    over task-condition levels with sufficient support.  Small Delta_task means
    phi generalizes across task contexts; large means descriptor fields may be
    missing (cross_app_dependency, context complexity, risk-related, ...)."""
    records = filter_source(records, source)
    scoped = [r for r in records if not unit_key or r.get("unit_key") == unit_key]
    per_level: Dict[str, dict] = {}
    for dim, fn in TASK_DIMENSIONS:
        levels: Dict[str, List[dict]] = {}
        for r in scoped:
            level = str(fn(r.get("task_condition") or {}))
            levels.setdefault(level, []).append(r)
        for level, group in levels.items():
            stats = cell_stats(group, min_samples)
            if stats["insufficient_support"]:
                continue
            per_level["%s=%s" % (dim, level)] = stats

    if len(per_level) < 2:
        return None
    best_key = max(per_level, key=lambda k: per_level[k]["point_estimate"])
    worst_key = min(per_level, key=lambda k: per_level[k]["point_estimate"])
    return {
        "unit_key": unit_key,
        "n_levels": len(per_level),
        "max_z": per_level[best_key]["point_estimate"],
        "max_z_level": best_key,
        "min_z": per_level[worst_key]["point_estimate"],
        "min_z_level": worst_key,
        "delta_task": per_level[best_key]["point_estimate"] - per_level[worst_key]["point_estimate"],
        "per_level": per_level,
    }


def delta_task_table(records: List[dict], min_samples: int = 10,
                      source: Optional[str] = "live") -> List[dict]:
    """Delta_task(phi) for every minimal stratification unit cell."""
    records = filter_source(records, source)
    rows = []
    units = group_records(records, "unit")
    for key in sorted(units):
        d = delta_task(records, unit_key=key, min_samples=min_samples)
        if d:
            d["unit_key"] = key
            rows.append(d)
    return rows


# ---------------------------------------------------------------------------
# Agent-conditioned analysis (14.7.2)
# ---------------------------------------------------------------------------

def baseline_success_tasks(records: List[dict]) -> Dict[str, set]:
    """T_H^LLM = {t : Y_LLM(H, t) = 1} per agent (eligibility implies run-level
    baseline success, so this equals the set of tasks the agent contributes
    eligible spans for)."""
    out: Dict[str, set] = {}
    for r in records:
        out.setdefault(r.get("agent_id", "?"), set()).add(r.get("task_id", "?"))
    return out


def matched_baseline_success_tasks(records: List[dict]) -> set:
    """T^matched = intersect_H { t : Y_LLM(H, t) = 1 } (14.7.2)."""
    sets = list(baseline_success_tasks(records).values())
    if not sets:
        return set()
    result = set(sets[0])
    for s in sets[1:]:
        result &= s
    return result


def within_agent_tables(records: List[dict], min_samples: int = 10,
                          source: Optional[str] = "live") -> Dict[str, Any]:
    """O_m(phi | H, Y_LLM(H,t)=1): per agent, per unit cell, on that agent's own
    baseline-successful tasks only."""
    records = filter_source(records, source)
    out: Dict[str, Any] = {}
    for agent_id in sorted({r.get("agent_id", "?") for r in records}):
        scoped = [r for r in records if r.get("agent_id") == agent_id]
        units = group_records(scoped, "unit")
        cells = {key: cell_stats(group, min_samples) for key, group in units.items()}
        out[agent_id] = cells
    return out


def matched_agent_comparison(records: List[dict], min_samples: int = 10,
                               source: Optional[str] = "live") -> Dict[str, Any]:
    """Compare agents on T^matched only (same task IDs, same task condition,
    same span class) -- the defensible implementation/architecture comparison
    (14.7.2)."""
    records = filter_source(records, source)
    matched_tasks = matched_baseline_success_tasks(records)
    agents = sorted({r.get("agent_id", "?") for r in records})
    units = sorted(group_records(records, "unit").keys())
    table: List[dict] = []
    for unit in units:
        row: Dict[str, Any] = {"unit_key": unit}
        any_supported = False
        for agent in agents:
            group = [r for r in records
                     if r.get("agent_id") == agent
                     and r.get("task_id") in matched_tasks
                     and r.get("unit_key") == unit]
            stats = cell_stats(group, min_samples)
            row[agent] = stats
            any_supported = any_supported or not stats["insufficient_support"]
        if any_supported:
            table.append(row)
    return {"matched_tasks": sorted(matched_tasks), "rows": table}


# ---------------------------------------------------------------------------
# Architecture-conditioned analysis (14.7.3)
# ---------------------------------------------------------------------------

def architecture_tables(records: List[dict], min_samples: int = 10,
                          source: Optional[str] = "live") -> Dict[str, Any]:
    """O_m(phi | K = k).  Clusters with a single representative MUST carry the
    caveat that implementation and architecture effects are not separated."""
    records = filter_source(records, source)
    clusters: Dict[str, Dict[str, Any]] = {}
    for r in records:
        cluster = r.get("agent_condition", {}).get("source_cluster_id", "unknown")
        agent = r.get("agent_id", "?")
        entry = clusters.setdefault(cluster, {"agents": set(), "records": []})
        entry["agents"].add(agent)
        entry["records"].append(r)
    out: Dict[str, Any] = {}
    for cluster, entry in sorted(clusters.items()):
        units = group_records(entry["records"], "unit")
        cells = {key: cell_stats(group, min_samples) for key, group in units.items()}
        out[cluster] = {
            "agents": sorted(entry["agents"]),
            "single_representative": len(entry["agents"]) == 1,
            "caveat": ("single representative: implementation and architecture "
                       "effects are NOT separated (14.7.3)") if len(entry["agents"]) == 1 else None,
            "cells": cells,
        }
    return out


def joint_matched_tables(records: List[dict], min_samples: int = 10,
                           source: Optional[str] = "live") -> List[dict]:
    """O_m(phi | z, H): same span descriptor, same task condition, different
    agents (14.7.4) -- controls task-distribution confounding better than
    O_m(phi | H) alone."""
    records = filter_source(records, source)
    matched_tasks = matched_baseline_success_tasks(records)
    rows = []
    units = group_records(records, "unit")
    for unit, group in units.items():
        # task-condition level = full task condition key
        by_task: Dict[str, List[dict]] = {}
        for r in group:
            key = _descriptor_key(r.get("task_condition") or {})
            by_task.setdefault(key, []).append(r)
        for tkey, tgroup in by_task.items():
            agents = {r.get("agent_id") for r in tgroup if r.get("task_id") in matched_tasks}
            if len(agents) < 2:
                continue
            row: Dict[str, Any] = {"unit_key": unit, "task_condition": tkey}
            any_supported = False
            for agent in sorted(agents):
                sub = [r for r in tgroup if r.get("agent_id") == agent
                       and r.get("task_id") in matched_tasks]
                stats = cell_stats(sub, min_samples)
                row[agent] = stats
                any_supported = any_supported or not stats["insufficient_support"]
            if any_supported:
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Descriptor completeness test (14.7.5)
# ---------------------------------------------------------------------------

def completeness_test(records: List[dict], min_samples: int = 10,
                      gap_threshold: float = 0.2,
                      source: Optional[str] = "live") -> Dict[str, Any]:
    """Core question: is O_m(phi | z, H) ~= O_m(phi)?

    For every unit cell with sufficient marginal support, compare the marginal
    strict-success estimate against every (z, H) conditional cell with
    sufficient support.  Cells whose conditionals deviate beyond the marginal
    CI indicate descriptor incompleteness; report the gap and the spec's
    checklist of candidate missing fields."""
    records = filter_source(records, source)
    units = group_records(records, "unit")
    rows = []
    n_complete = 0
    n_incomplete = 0
    for unit, group in sorted(units.items()):
        marginal = cell_stats(group, min_samples)
        if marginal["insufficient_support"]:
            continue
        # conditional cells: (task condition key, agent) pairs
        conds: Dict[str, List[dict]] = {}
        for r in group:
            key = "%s|%s" % (_descriptor_key(r.get("task_condition") or {}), r.get("agent_id", "?"))
            conds.setdefault(key, []).append(r)
        deviations = []
        worst = None
        for key, cgroup in conds.items():
            stats = cell_stats(cgroup, min_samples)
            if stats["insufficient_support"]:
                continue
            gap = stats["point_estimate"] - marginal["point_estimate"]
            outside_ci = stats["point_estimate"] < marginal["ci_lo"] or stats["point_estimate"] > marginal["ci_hi"]
            deviations.append({
                "condition": key,
                "n": stats["sample_count"],
                "point_estimate": stats["point_estimate"],
                "gap_to_marginal": gap,
                "outside_marginal_ci": outside_ci,
            })
            if worst is None or abs(gap) > abs(worst["gap_to_marginal"]):
                worst = {
                    "condition": key,
                    "gap_to_marginal": gap,
                    "outside_marginal_ci": outside_ci,
                }
        if worst is None:
            continue
        incomplete = abs(worst["gap_to_marginal"]) >= gap_threshold or worst["outside_marginal_ci"]
        if incomplete:
            n_incomplete += 1
        else:
            n_complete += 1
        rows.append({
            "unit_key": unit,
            "marginal": marginal,
            "max_abs_gap": worst["gap_to_marginal"],
            "worst_condition": worst["condition"],
            "outside_marginal_ci": worst["outside_marginal_ci"],
            "descriptor_complete": not incomplete,
            "suggested_missing_fields": MISSING_DESCRIPTOR_CHECKLIST if incomplete else [],
        })
    return {
        "n_units_tested": len(rows),
        "n_units_descriptor_complete": n_complete,
        "n_units_descriptor_incomplete": n_incomplete,
        "gap_threshold": gap_threshold,
        "min_samples": min_samples,
        "checklist": MISSING_DESCRIPTOR_CHECKLIST,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Local-only profiles for trace_local records (NOT strict success -- the
# downstream/final layers are not measured without an environment checkpoint)
# ---------------------------------------------------------------------------

def local_profile_m(records: List[dict], model: str,
                    min_samples: int = 10) -> Dict[str, Any]:
    """Per-unit-cell LOCAL layer profile for trace_local records:
    p_parse, p_schema, p_contract(=local_output_valid), p_fallback."""
    records = [r for r in filter_source(records, "trace_local")
               if r.get("model") == model]
    cells: Dict[str, Dict[str, Any]] = {}
    for key, group in group_records(records, "unit").items():
        n = len(group)
        cells[key] = {
            "sample_count": n,
            "p_parse": _rate(group, "local_outcome.parse_success"),
            "p_schema": _rate(group, "local_outcome.schema_valid"),
            "p_contract": _rate(group, "local_outcome.local_output_valid"),
            "p_verifier_passed": _rate(group, "local_outcome.local_verifier_passed"),
            "p_fallback": _rate(group, "fallback.fallback_triggered"),
            "p_fallback_recovered": _rate(group, "fallback.fallback_recovered"),
            "mean_token_delta": mean_ci([float(r["resource"]["token_delta"])
                                         for r in group if r.get("resource")])[0]
            if any(r.get("resource") for r in group) else float("nan"),
            "insufficient_support": n < min_samples,
        }
    return {"model": model, "mode": "unit", "min_samples": min_samples,
            "cells": cells}
