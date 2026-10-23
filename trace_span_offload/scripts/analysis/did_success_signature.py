#!/usr/bin/env python
"""Success group vs failure group background-removed analysis (DiD caliber, span cluster bootstrap).

For each variable:
  raw signature: Tsucc−Tfail (SLM within-group success-failure difference) / Isucc−Ifail (identity within-group success-failure difference)
  background-removed: DiD = (Tsucc−Tfail) − (Isucc−Ifail)
  CI: span resampling × within-span round resampling (two-layer), 2000 draws, 2.5%/97.5% quantiles
Verdict:
  DiD significant (CI excludes 0) = SLM-specific success signature (beyond natural fluctuation)
  DiD not significant = natural success-failure signature (disappears after background removal)

Input: simplified 48-span records + tau2 screening records (valid rounds)
Output: results/cross_agent/did_success_signature.json + console tables
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


def _load_records(d, valid_filter=None):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if valid_filter and not valid_filter(r):
                continue
            recs.append(r)
    return recs


def _t2_valid(r):
    res = r.get("resource") or {}
    d = (res.get("downstream_prompt_tokens", 0) or 0) \
        + (res.get("downstream_completion_tokens", 0) or 0)
    return d > 0


def _pick_files():
    files = sorted(glob.glob("out_span_offload/simplified/screening/*.jsonl"))
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


def _span_rounds(records, base_map, min_rounds=MIN_ROUNDS):
    """Returns {(task,inv): {arm: [(strict, {metric: val}), ...]}}"""
    arm = lambda r: (r.get("replay_meta") or {}).get("arm", "")  # noqa: E731
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        a = arm(r)
        if a in ("T_span", "I"):
            by[(r["task_id"], r["invocation_index"])][a].append(r)
    out = {}
    for k, arms in by.items():
        if len(arms.get("T_span", [])) < min_rounds or len(arms.get("I", [])) < min_rounds:
            continue
        out[k] = {}
        for a in ("T_span", "I"):
            rows = []
            for r in arms[a][:10]:
                strict = int(bool(r["strict_success"]))
                m = {v: cf_metrics(r, base_map.get(k))[v] for v in CORE_TABLE}
                rows.append((strict, m))
            out[k][a] = rows
    return out


def _did_draw(groups, rng):
    """One two-layer draw: span resampling × round resampling, returns per-variable DiD and signatures."""
    keys = list(groups.keys())
    t_succ, t_fail = defaultdict(list), defaultdict(list)
    i_succ, i_fail = defaultdict(list), defaultdict(list)
    for _ in range(len(keys)):
        k = rng.choice(keys)
        g = groups[k]
        for a, dst_s, dst_f in (("T_span", t_succ, t_fail), ("I", i_succ, i_fail)):
            rows = g[a]
            for _ in range(len(rows)):
                strict, m = rng.choice(rows)
                if strict:
                    dst = dst_s
                else:
                    dst = dst_f
                for v in CORE_TABLE:
                    dst[v].append(m[v])
    out = {}
    for v in CORE_TABLE:
        ts = t_succ[v]; tf = t_fail[v]; isu = i_succ[v]; ifa = i_fail[v]
        sig_t = statistics.mean(ts) - statistics.mean(tf) if ts and tf else None
        sig_i = statistics.mean(isu) - statistics.mean(ifa) if isu and ifa else None
        did = sig_t - sig_i if (sig_t is not None and sig_i is not None) else None
        out[v] = (sig_t, sig_i, did)
    return out


def main() -> int:
    # ---- tau2 ----
    t2 = _load_records("out_span_offload/tau2/screening", _t2_valid)
    t2_base = build_base_map("check_point_traces/appworld_tau2_qwen3_8-27b_batch1")
    t2_groups = _span_rounds(t2, t2_base)
    print("tau2 spans: %d" % len(t2_groups))
    # ---- simplified (48-span caliber) ----
    sim_recs = []
    for p in _pick_files():
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                sim_recs.append(json.loads(l))
            except ValueError:
                continue
    sim_base = build_base_map("check_point_traces/appworld_simplified_qwen3_8-27b_batch1")
    sim_groups = _span_rounds(sim_recs, sim_base)
    print("simplified spans: %d" % len(sim_groups))

    # ---- two-layer bootstrap ----
    def boot_did(groups):
        rng = random.Random(SEED)
        draws = []
        for _ in range(BOOT):
            draws.append(_did_draw(groups, rng))
        res = {}
        for v in CORE_TABLE:
            sig_t = [d[v][0] for d in draws if d[v][0] is not None]
            sig_i = [d[v][1] for d in draws if d[v][1] is not None]
            dids = [d[v][2] for d in draws if d[v][2] is not None]
            if len(dids) < BOOT // 2:
                continue
            dids_s = sorted(dids)
            sig_t_s = sorted(sig_t)
            sig_i_s = sorted(sig_i)
            res[v] = {
                "sig_T": statistics.mean(sig_t), "sig_T_lo": sig_t_s[int(0.025 * BOOT)],
                "sig_T_hi": sig_t_s[int(0.975 * BOOT)],
                "sig_I": statistics.mean(sig_i), "sig_I_lo": sig_i_s[int(0.025 * BOOT)],
                "sig_I_hi": sig_i_s[int(0.975 * BOOT)],
                "did": statistics.mean(dids), "did_lo": dids_s[int(0.025 * BOOT)],
                "did_hi": dids_s[int(0.975 * BOOT)],
                "did_sig": bool(dids_s[int(0.025 * BOOT)] > 0
                                or dids_s[int(0.975 * BOOT)] < 0),
                "T_sig": bool(sig_t_s[int(0.025 * BOOT)] > 0
                               or sig_t_s[int(0.975 * BOOT)] < 0),
                "I_sig": bool(sig_i_s[int(0.025 * BOOT)] > 0
                               or sig_i_s[int(0.975 * BOOT)] < 0),
            }
        return res

    print("bootstrapping (2000 × 2 layers × 2 agents)...")
    t2_res = boot_did(t2_groups)
    sim_res = boot_did(sim_groups)

    # ---- output ----
    print("\n" + "=" * 96)
    print("tau2 background-removed results (DiD = (Tsucc−Tfail) − (Isucc−Ifail), two-layer bootstrap CI)")
    print("=" * 96)
    print("%-24s %9s %16s %9s %16s %10s %16s %s" % (
        "variable", "Tsucc−Tfail", "[CI]", "Isucc−Ifail", "[CI]", "DiD", "[CI]", "verdict"))
    for v in CORE_TABLE:
        r = t2_res.get(v)
        if not r:
            continue
        verdict = "★ SLM-specific signature" if r["did_sig"] else (
            "natural signature (disappears after background removal)" if (r["T_sig"] or r["I_sig"]) else "no significant signature")
        print("%-24s %+9.3f [%+.3f,%+.3f] %+9.3f [%+.3f,%+.3f] %+10.3f [%+.3f,%+.3f] %s" % (
            v, r["sig_T"], r["sig_T_lo"], r["sig_T_hi"],
            r["sig_I"], r["sig_I_lo"], r["sig_I_hi"],
            r["did"], r["did_lo"], r["did_hi"], verdict))

    print("\n" + "=" * 96)
    print("simplified (48-span) background-removed results")
    print("=" * 96)
    print("%-24s %9s %16s %9s %16s %10s %16s %s" % (
        "variable", "Tsucc−Tfail", "[CI]", "Isucc−Ifail", "[CI]", "DiD", "[CI]", "verdict"))
    for v in CORE_TABLE:
        r = sim_res.get(v)
        if not r:
            continue
        verdict = "★ SLM-specific signature" if r["did_sig"] else (
            "natural signature (disappears after background removal)" if (r["T_sig"] or r["I_sig"]) else "no significant signature")
        print("%-24s %+9.3f [%+.3f,%+.3f] %+9.3f [%+.3f,%+.3f] %+10.3f [%+.3f,%+.3f] %s" % (
            v, r["sig_T"], r["sig_T_lo"], r["sig_T_hi"],
            r["sig_I"], r["sig_I_lo"], r["sig_I_hi"],
            r["did"], r["did_lo"], r["did_hi"], verdict))

    # compare the DiD-significant sets of the two agents
    t2_did_sig = {v for v in t2_res if t2_res[v]["did_sig"]}
    sim_did_sig = {v for v in sim_res if sim_res[v]["did_sig"]}
    t2_nat = {v for v in t2_res if (t2_res[v]["T_sig"] or t2_res[v]["I_sig"])
              and not t2_res[v]["did_sig"]}
    sim_nat = {v for v in sim_res if (sim_res[v]["T_sig"] or sim_res[v]["I_sig"])
               and not sim_res[v]["did_sig"]}
    print("\nDiD-significant (SLM-specific) set comparison:")
    print("  tau2:      %s" % sorted(t2_did_sig))
    print("  simplified:%s" % sorted(sim_did_sig))
    print("  intersection: %s" % sorted(t2_did_sig & sim_did_sig))
    print("\nnatural signature (disappears after background removal) set comparison:")
    print("  tau2:      %s" % sorted(t2_nat))
    print("  simplified:%s" % sorted(sim_nat))
    print("  intersection: %s" % sorted(t2_nat & sim_nat))

    out = {"tau2": t2_res, "simplified": sim_res,
           "tau2_did_sig": sorted(t2_did_sig), "simplified_did_sig": sorted(sim_did_sig),
           "tau2_natural": sorted(t2_nat), "simplified_natural": sorted(sim_nat),
           "intersection_did": sorted(t2_did_sig & sim_did_sig),
           "intersection_natural": sorted(t2_nat & sim_nat)}
    with open("results/cross_agent/did_success_signature.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/cross_agent/did_success_signature.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
