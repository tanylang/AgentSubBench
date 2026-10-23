#!/usr/bin/env python
"""Three-caliber paired main analysis: T_span vs B (primary comparison) / T_span vs I / B vs I.

Computes the full control-flow metrics for each record (canonical implementation in
span_offload/controlflow_metrics.py, including base-dependent KeyChainCoverage/ParamMatchRate etc.),
pairs the three arms by (task, inv), and for each pair runs a span-level paired Δ + 2000 bootstrap
CIs + significance.

Usage (27B new batch):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/main_comparison.py \
      --files out_span_offload/offload_profile_simplified_qwen3_8-27b_Qwen3_8-27B_Qwen_Qwen3_5-9B.jsonl \
      --traces-dir check_point_traces/appworld_simplified_qwen3_8-27b_batch1 \
      --out results/main_comparisons/main_comparison_27b.json \
      --pairs T_span:B T_span:I B:I
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.controlflow_metrics import (  # noqa: E402
    CORE_TABLE, METRICS, build_base_map, cf_metrics,
)

ARM_ALIAS = {"identity": "I", "slm": "T_span"}


def record_arm(r: dict) -> str:
    m = r.get("replay_meta") or {}
    arm = m.get("arm") or ARM_ALIAS.get(m.get("injection", ""), "")
    return arm if arm in ("T_span", "B", "I") else ""


def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                try:
                    recs.append(json.loads(l))
                except ValueError:
                    continue  # leftover line
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--traces-dir", required=True,
                    help="trace batch directory for base remaining info")
    ap.add_argument("--out", default="results/main_comparisons/main_comparison.json")
    ap.add_argument("--pairs", nargs="+", default=["T_span:B", "T_span:I", "B:I"],
                    help="comparison pairs, armA:armB (default runs all three pairs)")
    ap.add_argument("--min-rounds", type=int, default=5,
                    help="minimum completed rounds per group; spans below it are dropped")
    args = ap.parse_args()

    records = load(args.files)
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        arm = record_arm(r)
        if arm:
            by[(r["task_id"], r["invocation_index"])][arm].append(r)
    complete = {k: v for k, v in by.items()
                if all(len(v.get(a, [])) >= args.min_rounds
                       for a in ("T_span", "B", "I"))}
    print("records %d | complete paired spans %d (each group ≥%d rounds)" % (
        len(records), len(complete), args.min_rounds))

    base_map = build_base_map(args.traces_dir)
    # metric means per span × per arm
    span_means = {}
    for k, arms in complete.items():
        span_means[k] = {}
        for arm in ("T_span", "B", "I"):
            recs = arms[arm][: args.min_rounds]
            span_means[k][arm] = {}
            for v in CORE_TABLE:
                xs = [x for x in (cf_metrics(r, base_map.get(k))[v]
                                  for r in recs) if x == x]
                span_means[k][arm][v] = statistics.mean(xs) if xs else float("nan")

    random.seed(42)
    all_rows = []
    l67_rows = []
    from trace_edit_distance import (  # noqa: E402
        edit_distance, first_divergence, signature,
    )

    def _sigs(rs):
        return [signature(r) for r in rs]

    def _cross(sa, sb):
        return statistics.mean(
            edit_distance(x, y) / max(len(x), len(y), 1)
            for x in sa for y in sb)

    def _self(ss):
        ds = [edit_distance(ss[i], ss[j]) / max(len(ss[i]), len(ss[j]), 1)
              for i in range(len(ss)) for j in range(i + 1, len(ss))]
        return statistics.mean(ds) if ds else 0.0

    def _dyn(sa, sb, theta=0.4):
        """Per-position divergence dynamics: net d_t = cross-group mismatch rate − within-group noise
        floor (take the smaller). Returns (tau_on, L_prop, L_rec, persistence, amplification,
        net). L_rec = recovery distance (everything after is ≤θ; None = never recovers)."""
        maxlen = max((len(s) for s in sa + sb), default=0)
        net = []
        for t in range(maxlen):
            pairs = [(x, y) for x in sa for y in sb
                     if t < len(x) and t < len(y)]
            if not pairs:
                net.append(0.0)
                continue
            mism = sum(1 for x, y in pairs if x[t] != y[t]) / len(pairs)
            wa = [(x, y) for i, x in enumerate(sa) for y in sa[i + 1:]
                  if t < len(x) and t < len(y)]
            wb = [(x, y) for i, x in enumerate(sb) for y in sb[i + 1:]
                  if t < len(x) and t < len(y)]
            n_a = sum(1 for x, y in wa if x[t] != y[t]) / len(wa) if wa else 0.0
            n_b = sum(1 for x, y in wb if x[t] != y[t]) / len(wb) if wb else 0.0
            net.append(max(0.0, mism - min(n_a, n_b)))
        on = next((t for t, v in enumerate(net) if v > theta), None)
        if on is None:
            return None, 0.0, None, 0.0, 0.0, net
        prop = next((t for t in range(len(net) - 1, -1, -1)
                     if net[t] > theta), on)
        # L_rec: first position after τ_on where "everything after is ≤θ"; none = never recovers
        rec = None
        for t in range(on, len(net)):
            if all(v <= theta for v in net[t:]):
                rec = t
                break
        persist = sum(1 for v in net if v > theta) / max(len(net), 1)
        amp = max(net) / max(net[on], 1e-9)
        return on, prop, rec, persist, amp, net

    for pair in args.pairs:
        a, b = pair.split(":")
        dcs, dss, deltas, tstars = [], [], [], []
        tau_ons, l_props, l_recs, persists, amps = [], [], [], [], []
        n_ident = 0
        net_curves = []
        for k in complete:
            sa, sb = _sigs(complete[k][a][: args.min_rounds]), \
                _sigs(complete[k][b][: args.min_rounds])
            dc, ds = _cross(sa, sb), min(_self(sa), _self(sb))
            dcs.append(dc)
            dss.append(ds)
            deltas.append(dc - ds)
            ts = [first_divergence(x, y) for x in sa for y in sb]
            n_ident += sum(1 for t in ts if t is None)
            ts = [t for t in ts if t is not None]
            tstars.append(statistics.median(ts) if ts else 999)
            on, prop, rec, persist, amp, net = _dyn(sa, sb)
            if on is not None:
                tau_ons.append(on)
                l_props.append(prop)
                l_recs.append(rec)
                persists.append(persist)
                amps.append(amp)
                net_curves.append(net)
        mu = statistics.mean(deltas)
        boots = sorted(statistics.mean(random.choices(deltas, k=len(deltas)))
                       for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        l_rec_never = sum(1 for r in l_recs if r is None)
        l_rec_vals = [r for r in l_recs if r is not None]
        # d_t curve: per-position mean across spans (aligned to the common length)
        curve_len = min((len(c) for c in net_curves), default=0)
        mean_curve = [statistics.mean(c[t] for c in net_curves
                                      if t < len(c))
                      for t in range(curve_len)] if curve_len else []
        l67_rows.append({
            "pair": pair,
            "d_cross": statistics.mean(dcs),
            "d_self": statistics.mean(dss),
            "d_delta": mu, "d_ci_lo": lo, "d_ci_hi": hi,
            "d_significant": bool(lo > 0 or hi < 0),
            "t_star_median": statistics.median(tstars),
            "identical_ratio": n_ident / (len(complete)
                                          * args.min_rounds ** 2),
            "tau_on_median": statistics.median(tau_ons) if tau_ons else None,
            "tau_on_never": (len(complete) - len(tau_ons)) / len(complete),
            "l_prop_median": statistics.median(l_props) if l_props else None,
            "l_rec_median": (statistics.median(l_rec_vals)
                             if l_rec_vals else None),
            "l_rec_never": (l_rec_never / len(complete)
                            if l_recs else None),
            "persistence_mean": statistics.mean(persists) if persists else None,
            "amplification_mean": statistics.mean(amps) if amps else None,
            "net_curve_mean": mean_curve,
        })
        print("L6/L7 %-4s vs %-4s(%d span):" % (a, b, len(complete)))
        print("  d_norm: d_cross=%.3f d_self=%.3f Δ=%+.3f [%+.3f,%+.3f] %s" % (
            l67_rows[-1]["d_cross"], l67_rows[-1]["d_self"],
            mu, lo, hi, "★" if l67_rows[-1]["d_significant"] else "n.s."))
        print("  t★: median=%.1f | fully identical round pairs=%.0f%%" % (
            l67_rows[-1]["t_star_median"],
            100 * l67_rows[-1]["identical_ratio"]))
        print("  τ_on: median=%s | spans that never diverged=%.0f%%" % (
            l67_rows[-1]["tau_on_median"],
            100 * l67_rows[-1]["tau_on_never"]))
        print("  L_prop: median=%s | L_rec: median=%s never recovers=%.0f%%" % (
            l67_rows[-1]["l_prop_median"],
            l67_rows[-1]["l_rec_median"],
            100 * (l67_rows[-1]["l_rec_never"] or 0)))
        print("  persistence: %.2f | amplification: %.2f" % (
            l67_rows[-1]["persistence_mean"],
            l67_rows[-1]["amplification_mean"]))
        print("  d_t curve (first 10 positions): %s" % ", ".join(
            "%.2f" % v for v in mean_curve[:10]))
    # ---- NetSaving (plan §9.4): per-span total-token paired difference, itemized caliber ----
    # total = target(prompt+completion) + downstream(prompt+completion);
    # NetSaving = E[total_B] − E[total_T] (token level; price list applied in post-processing)
    def _total_tokens(r):
        res = r.get("resource") or {}
        return ((res.get("target_prompt_tokens", 0) or 0)
                + (res.get("target_completion_tokens", 0) or 0)
                + (res.get("downstream_prompt_tokens", 0) or 0)
                + (res.get("downstream_completion_tokens", 0) or 0))

    if "B" in args.pairs[0].split(":") or any("B" in p.split(":")
                                               for p in args.pairs):
        for pair in args.pairs:
            a, b = pair.split(":")
            if "B" not in (a, b):
                continue
            base_arm = "B"
            other = a if b == "B" else b
            deltas = []
            for k in complete:
                tb = statistics.mean(_total_tokens(r)
                                     for r in complete[k][base_arm][: args.min_rounds])
                to = statistics.mean(_total_tokens(r)
                                     for r in complete[k][other][: args.min_rounds])
                deltas.append(tb - to)
            mu = statistics.mean(deltas)
            boots = sorted(statistics.mean(random.choices(deltas, k=len(deltas)))
                           for _ in range(2000))
            lo, hi = boots[50], boots[1949]
            lcb = boots[50]  # one-sided 2.5% lower bound
            print("NetSaving(%s vs %s, token/round): mean=%+.0f 95%%CI=[%+.0f,%+.0f] "
                  "LCB=%+.0f %s" % (other, base_arm, mu, lo, hi, lcb,
                                    "LCB>0 ✓" if lcb > 0 else "LCB≤0 (fails the utility gate)"))
            all_rows.append({
                "metric": "NetSaving_tokens", "layer": "utility",
                "tier": "cost", "pair": pair,
                "mu_a": None, "mu_b": None,
                "delta": mu, "ci_lo": lo, "ci_hi": hi,
                "significant": bool(lo > 0 or hi < 0),
                "lcb": lcb, "n_spans": len(deltas),
            })
    print()
    for pair in args.pairs:
        a, b = pair.split(":")
        print()
        print("=" * 74)
        print("=== %s vs %s (paired Δ, span level, 2000 bootstrap) ===" % (a, b))
        print("=" * 74)
        print("%-24s %8s %8s %+9s  %s" % ("metric", "μ_" + a, "μ_" + b, "Δ", "95%CI"))
        rows = []
        for v in CORE_TABLE:
            diffs = []
            for k in complete:
                va = span_means[k][a][v]
                vb = span_means[k][b][v]
                if va == va and vb == vb:
                    diffs.append(va - vb)
            if len(diffs) < 3:
                continue
            mu = statistics.mean(diffs)
            boots = sorted(statistics.mean(random.choices(diffs, k=len(diffs)))
                           for _ in range(2000))
            lo, hi = boots[50], boots[1949]
            sig = "★" if (lo > 0 or hi < 0) else ""
            layer, tier, definition = METRICS[v]
            rows.append({
                "metric": v, "layer": layer, "tier": tier,
                "mu_a": statistics.mean(span_means[k][a][v]
                                        for k in complete),
                "mu_b": statistics.mean(span_means[k][b][v]
                                        for k in complete),
                "delta": mu, "ci_lo": lo, "ci_hi": hi,
                "significant": bool(lo > 0 or hi < 0),
                "pair": pair, "n_spans": len(diffs),
            })
            print("%-24s %8.3f %8.3f %+9.3f  [%+.3f,%+.3f] %s" % (
                v, rows[-1]["mu_a"], rows[-1]["mu_b"], mu, lo, hi, sig))
        all_rows.extend(rows)

    out = {"n_spans": len(complete), "n_records": len(records),
           "rows": all_rows, "l67": l67_rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
