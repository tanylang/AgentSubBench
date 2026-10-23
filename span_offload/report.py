"""Markdown report writing for Stage 10.

The report is organized by spec section:
  1. census (14.1-14.2), 2. profiles (14.6), 3. task-conditioned (14.7.1),
  4. agent/architecture-conditioned (14.7.2-14.7.4), 5. descriptor
  completeness (14.7.5), 6. allowlist validation (14.8).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def pct(x: Any) -> str:
    if x is None or x != x:  # NaN check
        return "n/a"
    return "%.1f%%" % (100.0 * float(x))


def num(x: Any, nd: int = 3) -> str:
    if x is None or x != x:
        return "n/a"
    return ("%." + str(nd) + "f") % float(x)


def ci(stats: Dict[str, Any]) -> str:
    if stats.get("insufficient_support"):
        return "insufficient_support"
    return "%.2f [%.2f, %.2f]" % (stats["point_estimate"], stats["ci_lo"], stats["ci_hi"])


def _row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [_row(headers), _row(["---"] * len(headers))]
    for r in rows:
        lines.append(_row(r))
    return "\n".join(lines)


def _unit_label(key: str, mode: str) -> str:
    if mode == "unit":
        try:
            d = json.loads(key)
            return "%s / %s / %s / %s" % (
                d["node_pattern"], d["control_role"],
                ",".join(d["state_effect"]) or "-", ",".join(d["verifier_type"]) or "-")
        except (ValueError, KeyError):
            return key[:90]
    return key[:90]


def profiles_section(profiles: Dict[str, Any]) -> str:
    """Per model x per unit cell: the full 14.6 profile, never one score."""
    out: List[str] = []
    for model, prof in profiles.items():
        cells = prof.get("cells", {})
        out.append("### Model `%s` (mode=%s, min_samples=%d)" %
                   (model, prof.get("mode", "unit"), prof.get("min_samples", 10)))
        out.append("")
        rows = []
        for key, s in sorted(cells.items(),
                             key=lambda kv: (-(kv[1]["sample_count"] or 0), kv[0])):
            rows.append([
                _unit_label(key, prof.get("mode", "unit")),
                str(s["sample_count"]),
                ci(s),
                pct(s["local_success"]),
                pct(s["downstream_success"]),
                pct(s["final_task_success"]),
                pct(s["collateral_damage_rate"]),
                pct(s["fallback_rate"]),
                pct(s["detect_rate"]),
                pct(s["false_reject_rate"]),
                num(s["mean_cost_saving"], 4),
                num(s["mean_latency_saving"], 1),
            ])
        out.append(table(
            ["cell (u = node / role / state_effect / verifier)",
             "n", "O_strict + 95% CI", "p_local", "p_downstream",
             "p_final", "p_collateral", "p_fallback", "DetectRate",
             "FalseRejectRate", "CostSaving", "LatencySaving(ms)"],
            rows))
        out.append("")
        out.append("_insufficient_support cells are omitted from conclusions; "
                   "their point estimates are unstable._")
        out.append("")
    return "\n".join(out)


def task_conditioned_section(tables: Dict[str, Any]) -> str:
    out: List[str] = []
    for dim, entry in tables.items():
        out.append("### Dimension: %s" % entry.get("label", dim))
        out.append("")
        rows = []
        for level, s in sorted(entry.get("levels", {}).items()):
            rows.append([level, str(s["sample_count"]), ci(s),
                         pct(s["local_success"]), pct(s["final_task_success"]),
                         pct(s["collateral_damage_rate"])])
        out.append(table(["level", "n", "O_strict + 95% CI", "p_local",
                          "p_final", "p_collateral"], rows))
        out.append("")
    return "\n".join(out)


def delta_task_section(rows: List[Dict[str, Any]]) -> str:
    out: List[str] = ["Heterogeneity gap  Delta_task(phi) = max_z O(phi|z) - min_z O(phi|z):", ""]
    if not rows:
        out.append("_no unit cell has >= 2 task-condition levels with sufficient support_")
        return "\n".join(out)
    tr = []
    for d in sorted(rows, key=lambda x: -abs(x.get("delta_task", 0.0))):
        tr.append([_unit_label(d["unit_key"], "unit"),
                   str(d["n_levels"]),
                   "%s (%s)" % (num(d["max_z"]), d["max_z_level"]),
                   "%s (%s)" % (num(d["min_z"]), d["min_z_level"]),
                   num(d["delta_task"])])
    out.append(table(["unit cell", "n levels", "max_z O(phi|z)", "min_z O(phi|z)",
                      "Delta_task"], tr))
    out.append("")
    out.append("_Large Delta_task => check for missing descriptors: cross_app_dependency, "
               "context complexity, risk-related fields (14.7.1)._")
    return "\n".join(out)


def agent_section(within: Dict[str, Any], matched: Dict[str, Any],
                  architecture: Dict[str, Any], joint: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    out.append("### Matched baseline-success tasks")
    out.append("")
    out.append("T^matched = intersect_H {t : Y_LLM(H,t)=1}: `%s`" %
               ", ".join(matched.get("matched_tasks", [])))
    out.append("")
    out.append("### O(phi | H) comparison on matched tasks only")
    out.append("")
    rows = []
    for row in matched.get("rows", []):
        cells = [_unit_label(row["unit_key"], "unit")]
        for agent, s in sorted(row.items()):
            if agent in ("unit_key",):
                continue
            cells.append("n=%s %s" % (s["sample_count"], ci(s)))
        rows.append(cells)
    agents = sorted({k for row in matched.get("rows", []) for k in row.keys()
                     if k not in ("unit_key",)})
    if rows:
        out.append(table(["unit cell"] + agents, rows))
    else:
        out.append("_no unit cell has sufficient support on matched tasks_")
    out.append("")

    out.append("### Architecture-conditioned O(phi | K=k)")
    out.append("")
    for cluster, entry in sorted(architecture.items()):
        note = entry.get("caveat")
        out.append("**cluster `%s`** (agents: %s)%s" %
                   (cluster, ", ".join(entry.get("agents", [])),
                    " -- CAVEAT: %s" % note if note else ""))
        out.append("")
        rows = []
        for key, s in sorted(entry.get("cells", {}).items(),
                             key=lambda kv: -(kv[1]["sample_count"] or 0)):
            rows.append([_unit_label(key, "unit"), str(s["sample_count"]), ci(s),
                         pct(s["final_task_success"])])
        out.append(table(["unit cell", "n", "O_strict + 95% CI", "p_final"], rows))
        out.append("")
    out.append("### Joint matched O(phi | z, H)")
    out.append("")
    if joint:
        rows = []
        for row in joint[:30]:
            cells = [_unit_label(row["unit_key"], "unit"),
                     row["task_condition"][:40]]
            for agent, s in sorted(row.items()):
                if agent in ("unit_key", "task_condition"):
                    continue
                cells.append("n=%s %s" % (s["sample_count"], ci(s)))
            rows.append(cells)
        agents = sorted({k for row in joint for k in row.keys()
                         if k not in ("unit_key", "task_condition")})
        out.append(table(["unit cell", "task condition"] + agents, rows))
    else:
        out.append("_no joint (phi, z) cell has >= 2 agents with sufficient support_")
    out.append("")
    return "\n".join(out)


def completeness_section(result: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("Descriptor completeness test: O(phi | z, H) ~= O(phi)?")
    out.append("")
    out.append("- units tested: %d" % result.get("n_units_tested", 0))
    out.append("- descriptor complete: %d" % result.get("n_units_descriptor_complete", 0))
    out.append("- descriptor incomplete: %d" % result.get("n_units_descriptor_incomplete", 0))
    out.append("- gap threshold: %s" % num(result.get("gap_threshold", 0.2)))
    out.append("")
    rows = []
    for row in result.get("rows", []):
        marginal = row["marginal"]
        rows.append([
            _unit_label(row["unit_key"], "unit"),
            str(marginal["sample_count"]),
            ci(marginal),
            "%+s" % num(row["max_abs_gap"]),
            row["worst_condition"][:50],
            "yes" if row["outside_marginal_ci"] else "no",
            "complete" if row["descriptor_complete"] else
            "INCOMPLETE -> revisit: %s" % ", ".join(row.get("suggested_missing_fields", [])[:3]),
        ])
    out.append(table(["unit cell", "n marginal", "marginal O + CI",
                      "max |gap|", "worst (z,H) condition", "outside CI",
                      "verdict"], rows))
    out.append("")
    return "\n".join(out)


def local_profiles_section(profiles: Dict[str, Any]) -> str:
    """Local-layer profiles for trace_local records (parse/schema/contract
    rates per unit cell).  Explicitly NOT strict success -- downstream/final
    layers are not measured without an environment checkpoint."""
    out: List[str] = [
        "Local-layer measurement (outcome_source = trace_local): same recorded "
        "prompt -> SLM -> original parser -> local contract checks. "
        "Downstream/final layers are NOT measured; these records never enter "
        "O_strict profiles.",
        "",
    ]
    for model, prof in profiles.items():
        cells = prof.get("cells", {})
        out.append("### Model `%s` (trace_local)" % model)
        out.append("")
        rows = []
        for key, s in sorted(cells.items(),
                             key=lambda kv: -(kv[1]["sample_count"] or 0)):
            rows.append([
                _unit_label(key, "unit"),
                str(s["sample_count"]),
                pct(s["p_parse"]),
                pct(s["p_schema"]),
                pct(s["p_contract"]),
                pct(s["p_verifier_passed"]),
                pct(s["p_fallback"]),
                pct(s["p_fallback_recovered"]),
                num(s["mean_token_delta"], 1),
            ])
        out.append(table(["cell (u)", "n", "p_parse", "p_schema", "p_contract",
                          "p_verifier", "p_fallback", "p_fb_recovered",
                          "mean token_delta"], rows))
        out.append("")
        out.append("_insufficient_support cells are omitted from conclusions._")
        out.append("")
    return "\n".join(out)


def validation_section(results: Dict[str, Any], oracle: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("Multi-span end-to-end sanity check on held-out tasks (static "
               "profile allowlist, frozen from the pilot split).")
    out.append("")
    rows = []
    order = ["llm_only", "slm_only", "allowlist", "allowlist_fallback"]
    for config in order:
        agg = results.get("configs", {}).get(config, {})
        rows.append([
            config,
            str(agg.get("n_runs", 0)),
            pct(agg.get("final_task_success_rate")),
            pct(agg.get("collateral_rate")),
            num(agg.get("total_cost"), 5),
            num(agg.get("mean_latency_ms"), 1),
            str(agg.get("n_offloaded", 0)),
            str(agg.get("n_fallbacks", 0)),
        ])
    out.append(table(["config", "n runs", "final success", "collateral",
                      "total cost", "mean latency ms", "spans offloaded",
                      "fallbacks"], rows))
    out.append("")
    out.append("Oracle single-span opportunity upper bound: %d / %d held-out "
               "eligible spans offloadable, opportunity cost bound `%s`."
               % (oracle.get("n_oracle_offloadable", 0),
                  oracle.get("n_heldout_eligible_spans", 0),
                  num(oracle.get("oracle_opportunity_cost_bound"), 5)))
    out.append("")
    out.append("_This experiment only validates that the static profile has "
               "systematic predictive value.  It trains no learned router and "
               "performs no policy optimization (14.8)._")
    return "\n".join(out)


def write_report(path: str, sections: Dict[str, str], title: str = "Stage 10 pilot report") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s\n\n" % title)
        for heading, body in sections.items():
            f.write("## %s\n\n" % heading)
            f.write(body)
            f.write("\n\n")
    return path


def save_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path
