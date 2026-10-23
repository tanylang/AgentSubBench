#!/usr/bin/env python
"""Cross-agent consistency check: simplified (48-span) vs tau2 (22-span).

Three parts:
  1) metrics summary table: whether the sign/significance of the T vs B paired Δ is consistent across agents
     (simplified full batch 48 / simplified argument-generation subset 11 / tau2 full batch 22)
  2) success group vs failure group: success−failure signature direction + DiD ((Tsucc−Tfail)−(Isucc−Ifail)) compared across agents
  3) offloadability metrics: argument-generation layer gate profile + reproducibility of continuous features on like samples
Output: results/cross_agent/cross_agent_comparison.json + console tables
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from span_offload.controlflow_metrics import (  # noqa: E402
    CORE_TABLE, build_base_map, cf_metrics,
)

MIN_ROUNDS = 5
BOOT = 2000
SEED = 42

SIM_RECORD_DIR = "out_span_offload/simplified/screening"
SIM_TRACES_DIR = "check_point_traces/appworld_simplified_qwen3_8-27b_batch1"
SIM_SPANS = "out_span_offload/spans/spans_simplified_qwen27b_50.jsonl"
TAU2_RESULT = "results/tau2/tau2_analysis.json"
OUT = "results/cross_agent/cross_agent_comparison.json"


def _boot(diffs, nboot=BOOT, seed=SEED):
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                   for _ in range(nboot))
    return draws[int(0.025 * nboot)], draws[int(0.975 * nboot)]


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = ((sum((a - mx) ** 2 for a in rx)
            * sum((b - my) ** 2 for b in ry)) ** 0.5)
    return num / den if den else 0.0


def pick_files():
    files = sorted(glob.glob(os.path.join(SIM_RECORD_DIR, "*.jsonl")))
    chosen = {}
    for p in files:
        m = re.search(r"([0-9a-f]{6,8}_\d)\.jsonl$", os.path.basename(p))
        if not m:
            continue
        task = m.group(1)
        is_new = "qwen3_8-27b_Qwen" in os.path.basename(p)
        if task not in chosen or is_new:
            chosen[task] = p
    return sorted(chosen.values())


def main() -> int:
    # ---- recompute simplified per-span full metric means ----
    records = []
    for p in pick_files():
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                records.append(json.loads(l))
            except ValueError:
                continue
    arm = lambda r: (r.get("replay_meta") or {}).get("arm", "")  # noqa: E731
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        if arm(r) in ("T_span", "B"):
            by[(r["task_id"], r["invocation_index"])][arm(r)].append(r)
    spans = {}
    for l in open(SIM_SPANS, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        s = json.loads(l)
        spans[(s["task_id"], s["invocation_index"])] = s
    base_map = build_base_map(SIM_TRACES_DIR)
    sim = []
    for (t, i), arms in by.items():
        if len(arms.get("T_span", [])) < MIN_ROUNDS or len(arms.get("B", [])) < MIN_ROUNDS:
            continue
        sd = arms["T_span"][0].get("span_descriptor") or {}
        np_ = sd.get("node_pattern", "other")
        m = {"node": np_}
        for a, recs in (("T", arms["T_span"][:MIN_ROUNDS]),
                        ("B", arms["B"][:MIN_ROUNDS])):
            for v in CORE_TABLE:
                xs = [x for x in (cf_metrics(r, base_map.get((t, i)))[v]
                                  for r in recs) if x == x]
                m["%s_%s" % (v, a)] = statistics.mean(xs) if xs else float("nan")
        sim.append(m)
    print("simplified per-span recomputed: n=%d (argument-generation subset %d)" % (
        len(sim), sum(1 for s in sim if s["node"] == "tool_argument_generation")))

    # ---- tau2 data ----
    tau = json.load(open(TAU2_RESULT, encoding="utf-8"))
    tau_rows = tau["per_span"]
    # tau2 per-span CORE_TABLE values are not in per_span; aggregates come from total_table
    # but per-span is needed — recompute from records (same caliber as tau2_analysis)
    import importlib.util
    t2_records = []
    for p in sorted(glob.glob("out_span_offload/tau2/screening/*.jsonl")):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except ValueError:
                continue
            d = ((r.get("resource") or {}).get("downstream_prompt_tokens", 0) or 0) \
                + ((r.get("resource") or {}).get("downstream_completion_tokens", 0) or 0)
            if d > 0:
                t2_records.append(r)
    by2 = defaultdict(lambda: defaultdict(list))
    for r in t2_records:
        if arm(r) in ("T_span", "B"):
            by2[(r["task_id"], r["invocation_index"])][arm(r)].append(r)
    t2_base = build_base_map("check_point_traces/appworld_tau2_qwen3_8-27b_batch1")
    t2 = []
    for (t, i), arms2 in by2.items():
        if len(arms2.get("T_span", [])) < MIN_ROUNDS or len(arms2.get("B", [])) < MIN_ROUNDS:
            continue
        sd2 = arms2["T_span"][0].get("span_descriptor") or {}
        m2 = {"node": sd2.get("node_pattern", "other")}
        for a, recs in (("T", arms2["T_span"]), ("B", arms2["B"])):
            for v in CORE_TABLE:
                xs = [x for x in (cf_metrics(r, t2_base.get((t, i)))[v]
                                  for r in recs) if x == x]
                m2["%s_%s" % (v, a)] = statistics.mean(xs) if xs else float("nan")
        t2.append(m2)
    print("tau2 per-span recomputed: n=%d" % len(t2))

    # ============ 1) metrics summary table comparison ============
    print("\n" + "=" * 88)
    print("1. T vs B metric Δ: simplified (full batch 48 / argument subset 11) vs tau2 (22)")
    print("=" * 88)
    sim_arg = [s for s in sim if s["node"] == "tool_argument_generation"]
    compare_rows = []
    for v in CORE_TABLE:
        out = {"metric": v}
        for name, group in (("sim48", sim), ("sim_arg11", sim_arg), ("tau2_22", t2)):
            diffs = [s["%s_T" % v] - s["%s_B" % v] for s in group
                     if s["%s_T" % v] == s["%s_T" % v] and s["%s_B" % v] == s["%s_B" % v]]
            if len(diffs) < 3:
                out[name] = None
                continue
            mu = statistics.mean(diffs)
            lo, hi = _boot(diffs)
            out[name] = {"delta": mu, "ci_lo": lo, "ci_hi": hi,
                         "sig": bool(lo > 0 or hi < 0), "n": len(diffs)}
        # consistency verdict
        s48, s11, t22 = out.get("sim48"), out.get("sim_arg11"), out.get("tau2_22")
        verdict = ""
        if s11 and t22:
            same_sign = (s11["delta"] > 0) == (t22["delta"] > 0)
            if s11["sig"] and t22["sig"] and same_sign:
                verdict = "reproduced (both significant, same sign on like samples)"
            elif same_sign and (s11["sig"] or t22["sig"]):
                verdict = "direction same (only one side significant)"
            elif same_sign:
                verdict = "direction same (neither significant)"
            else:
                verdict = "opposite signs"
        elif s48 and t22:
            same_sign = (s48["delta"] > 0) == (t22["delta"] > 0)
            verdict = ("direction same" if same_sign else "opposite signs") + " (full batch vs tau2)"
        out["verdict"] = verdict
        compare_rows.append(out)
    print("%-24s | %-28s | %-28s | %-22s | %s" % (
        "metric", "sim48 Δ[sig]", "sim_arg11 Δ[sig]", "tau2 Δ[sig]", "verdict"))
    for r in compare_rows:
        def fmt(x):
            return "%+.3f%s" % (x["delta"], "★" if x["sig"] else "") if x else "—"
        print("%-24s | %-28s | %-28s | %-22s | %s" % (
            r["metric"], fmt(r.get("sim48")), fmt(r.get("sim_arg11")),
            fmt(r.get("tau2_22")), r["verdict"]))

    # ============ 2) success group vs failure group comparison ============
    print("\n" + "=" * 88)
    print("2. success−failure signature and DiD: simplified (265 rounds) vs tau2 (round level)")
    print("=" * 88)
    ss = json.load(open("results/appworld_53span/success_stratification.json", encoding="utf-8"))
    ss_map = {x["var"]: x for x in ss["per_variable"]}
    t2sf = {x["var"]: x for x in tau["success_failure"]}
    print("%-24s | %-30s | %-30s | %s" % ("variable", "simplified succ−fail (SLM/id)",
                                          "tau2 succ−fail (T/I)", "DiD direction same?"))
    sf_compare = []
    for v in CORE_TABLE:
        s_row, t_row = ss_map.get(v), t2sf.get(v)
        if not s_row or not t_row:
            continue
        s_slm = s_row["SLM_succ"] - s_row["SLM_fail"]
        s_id = s_row["id_succ"] - s_row["id_fail"]
        t_t = t_row["T_span成"] - t_row["T_span败"]
        t_i = t_row["I成"] - t_row["I败"]
        s_did = s_slm - s_id
        t_did = t_t - t_i
        same = (s_slm > 0) == (t_t > 0)
        did_same = (s_did > 0) == (t_did > 0)
        sf_compare.append({"var": v, "s_slm": s_slm, "s_id": s_id, "s_did": s_did,
                           "t_t": t_t, "t_i": t_i, "t_did": t_did,
                           "sign_same": same, "did_same": did_same})
        print("%-24s | %+8.3f / %+8.3f (DiD %+.3f) | %+8.3f / %+8.3f (DiD %+.3f) | %s%s" % (
            v, s_slm, s_id, s_did, t_t, t_i, t_did,
            "sign same " if same else "sign opposite ", "DiD same" if did_same else "DiD opposite"))
    n_same = sum(1 for x in sf_compare if x["sign_same"])
    n_did = sum(1 for x in sf_compare if x["did_same"])
    print("sign direction same: %d/%d | DiD direction same: %d/%d" % (
        n_same, len(sf_compare), n_did, len(sf_compare)))

    # ============ 3) offloadability metrics comparison ============
    print("\n" + "=" * 88)
    print("3. offloadability metrics: argument-generation layer profile + continuous features (like samples)")
    print("=" * 88)
    d48 = json.load(open("results/stratification/stratified_offloadability_48.json", encoding="utf-8"))
    arg48 = [s for s in d48["per_span"] if s["node"] == "tool_argument_generation"]
    t2sp = tau["per_span"]
    def profile(g):
        if not g:
            return None
        return {
            "n": len(g),
            "strict_T": statistics.mean(s["strict_T"] for s in g),
            "strict_B": statistics.mean(s["strict_B"] for s in g),
            "net_saving": statistics.mean(s["net_saving"] for s in g),
            "rho_median": statistics.median([s["rho_comp"] for s in g
                                             if abs(s["rho_comp"]) < 1e4]),
            "zero_detour": statistics.mean(1 if s["zero_detour_delta"] else 0 for s in g),
            "direct_mean": statistics.mean(s["direct_saving"] for s in g),
        }
    p48 = profile(arg48)
    p22 = profile(t2sp)
    print("argument-generation layer profile:")
    print("%-12s | %2s %6s %6s %9s %9s %9s %8s %8s" % (
        "sample", "n", "stT", "stB", "NetSav", "ρ_med", "direct", "zeroDet(delta)", ""))
    print("%-12s | %2d %6.2f %6.2f %+9.0f %9.1f %+9.0f %7.0f%%" % (
        "simplified", p48["n"], p48["strict_T"], p48["strict_B"],
        p48["net_saving"], p48["rho_median"], p48["direct_mean"],
        100 * p48["zero_detour"]))
    print("%-12s | %2d %6.2f %6.2f %+9.0f %9.1f %+9.0f %7.0f%%" % (
        "tau2", p22["n"], p22["strict_T"], p22["strict_B"],
        p22["net_saving"], p22["rho_median"], p22["direct_mean"],
        100 * p22["zero_detour"]))
    print("\ncontinuous features (Spearman on like samples):")
    print("%-24s | %-30s | %-30s" % ("feature×outcome", "simplified argument subset (n=11)",
                                     "tau2 (n=22)"))
    feats = {"baseline_tokens": "baseline_tokens", "max_remaining_steps": "max_remaining_steps"}
    targets = {"direct_saving": "direct_saving", "net_saving": "net_saving",
               "downstream_comp": "downstream_comp", "d_delta": "d_delta",
               "d_success": "d_success"}
    corr_compare = []
    for f in feats:
        for tg in targets:
            g48v = [(s[f], s[tg]) for s in arg48 if s.get(f) is not None]
            g22v = [(s[f], s[tg]) for s in t2sp if s.get(f) is not None]
            if len(g48v) < 6 or len(g22v) < 6:
                continue
            r48 = _spearman([x[0] for x in g48v], [x[1] for x in g48v])
            r22 = _spearman([x[0] for x in g22v], [x[1] for x in g22v])
            same = (r48 > 0) == (r22 > 0)
            corr_compare.append({"feature": f, "target": tg,
                                 "rho_sim_arg": r48, "rho_tau2": r22,
                                 "same_sign": same})
            print("%-24s | %+8.3f (n=%d)              | %+8.3f (n=%d)              %s" % (
                "%s × %s" % (f, tg), r48, len(g48v), r22, len(g22v),
                "same" if same else "opposite"))
    n_same_corr = sum(1 for x in corr_compare if x["same_sign"])
    print("continuous feature sign same: %d/%d" % (n_same_corr, len(corr_compare)))

    out = {"total_table": compare_rows, "success_failure": sf_compare,
           "arg_profile": {"simplified": p48, "tau2": p22},
           "corr": corr_compare,
           "summary": {
               "sign_same_rate": n_same / max(len(sf_compare), 1),
               "did_same_rate": n_did / max(len(sf_compare), 1),
               "corr_sign_same_rate": n_same_corr / max(len(corr_compare), 1),
           }}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
