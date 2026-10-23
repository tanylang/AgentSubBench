#!/usr/bin/env python
"""Stratified offloadability analysis (plan §14 + preregistered Batch 3): stratify by span
descriptor, compute a quality-risk-utility profile per stratum, and find candidate strata
with positive NetSaving.

Input:
  confirm_merged20.jsonl (screening 5 + confirm 15 = 20 rounds/arm)
  spans file (descriptor source: node_pattern/state_effect/position)
Output (per stratum):
  n (span count), strict_T/strict_B/ΔSuccess (LCB), collateral (UCB),
  NetSaving (per-span B−T total tokens, mean+LCB), ρ_comp (compensation ratio),
  zero-detour span ratio (d_norm<0.1), positive-return span count (NetSaving>0)

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/stratified_offloadability.py \
      --records out_span_offload/simplified/confirm_merged20.jsonl \
      --spans out_span_offload/spans/spans_simplified_qwen27b.jsonl \
      --traces-dir check_point_traces/appworld_simplified_qwen3_8-27b_batch1 \
      --out results/stratification/stratified_offloadability.json
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from trace_edit_distance import edit_distance, signature  # noqa: E402
from span_offload.controlflow_metrics import (  # noqa: E402
    build_base_map, cf_metrics,
)

#: zero-detour threshold: spans whose three-arm trajectories are nearly identical (very small d_norm) have downstream compensation ≈0
ZERO_DETOUR_DNORM = 0.1


def _total(r):
    res = r.get("resource") or {}
    return ((res.get("target_prompt_tokens", 0) or 0)
            + (res.get("target_completion_tokens", 0) or 0)
            + (res.get("downstream_prompt_tokens", 0) or 0)
            + (res.get("downstream_completion_tokens", 0) or 0))


def _d_norm(recs_a, recs_b):
    sa = [signature(r) for r in recs_a]
    sb = [signature(r) for r in recs_b]
    return statistics.mean(
        edit_distance(x, y) / max(len(x), len(y), 1)
        for x in sa for y in sb)


def per_span_stats(records_path, spans_path, traces_dir, min_rounds=20,
                   manifest_path=None):
    by = defaultdict(lambda: defaultdict(list))
    with open(records_path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            by[(r["task_id"], r["invocation_index"])][
                r["replay_meta"].get("arm")].append(r)
    base_map = build_base_map(traces_dir) if traces_dir else {}
    # task-level category (locomo gold qa category 1-5; None for non-locomo tasks)
    category_of = {}
    if manifest_path:
        from span_offload.trace_io import load_manifest as _load_manifest
        _m = _load_manifest(manifest_path)
        for _tid in {k[0] for k in by.keys()}:
            _parts = _tid.split(":")
            if len(_parts) < 3:
                continue
            _sample, _qi = _parts[1], int(_parts[2][1:])
            _mt = _m.get(_sample)
            if _mt is None:
                continue
            _qa = list(_mt.evaluator.get("qa") or [])
            if _qi < len(_qa):
                category_of[_tid] = _qa[_qi].get("category")
    spans = {s["span_id"]: s for s in (
        json.loads(l) for l in open(spans_path, encoding="utf-8") if l.strip())}
    # per-task max inv (used for position tertiles)
    max_inv = defaultdict(int)
    for sid, s in spans.items():
        max_inv[s["task_id"]] = max(max_inv[s["task_id"]],
                                    s["invocation_index"])
    out = []
    for (t, i), arms in by.items():
        if not all(len(arms.get(a, [])) >= min_rounds
                   for a in ("T_span", "B", "I")):
            continue
        recT, recB, recI = (arms["T_span"][:min_rounds],
                            arms["B"][:min_rounds], arms["I"][:min_rounds])
        # descriptor: taken from span_descriptor in the record (consistent with the spans file)
        sd = recT[0]["span_descriptor"]
        np_ = sd["node_pattern"]
        se = tuple(sorted(sd["mechanism_tags"].get("state_effect", [])))[0] \
            if sd["mechanism_tags"].get("state_effect") else "read_only"
        pos = "early" if i / max(max_inv[t], 1) <= 0.34 else (
            "late" if i / max(max_inv[t], 1) >= 0.67 else "middle")
        stT = statistics.mean(int(bool(r["strict_success"])) for r in recT)
        stB = statistics.mean(int(bool(r["strict_success"])) for r in recB)
        stI = statistics.mean(int(bool(r["strict_success"])) for r in recI)
        totT = statistics.mean(_total(r) for r in recT)
        totB = statistics.mean(_total(r) for r in recB)
        # downstream impact metrics (mean per arm T/B/I)
        down_metrics = ("llm", "tools", "hall", "MaxFailRun", "RevisitRate",
                        "NewBranchRate", "ToolEntropy", "DirectActionRate")
        dm = {}
        for arm, recs_ in (("T", recT), ("B", recB), ("I", recI)):
            for v in down_metrics:
                dm["%s_%s" % (v, arm)] = statistics.mean(
                    cf_metrics(r, base_map.get((t, i)))[v] for r in recs_)
        downT = statistics.mean(
            (r["resource"].get("downstream_prompt_tokens", 0) or 0)
            + (r["resource"].get("downstream_completion_tokens", 0) or 0)
            for r in recT)
        downB = statistics.mean(
            (r["resource"].get("downstream_prompt_tokens", 0) or 0)
            + (r["resource"].get("downstream_completion_tokens", 0) or 0)
            for r in recB)
        tgtT = totT - downT
        tgtB = totB - downB
        direct = tgtB - tgtT          # direct saving (target layer)
        comp = downT - downB          # downstream compensation (positive = spending more)
        rho = comp / max(direct, 1e-9)
        dnorm = _d_norm(recT, recB)
        out.append({
            "task": t, "inv": i, "category": category_of.get(t),
            "node": np_, "state_effect": se,
            "position": pos, "cls": "%s_%s" % (np_, se),
            "strict_T": stT, "strict_B": stB, "strict_I": stI,
            "d_success": stT - stB,
            "total_T": totT, "total_B": totB,
            "total_I": statistics.mean(_total(r) for r in recI),
            "net_saving": totB - totT,   # positive = offload saves tokens
            "direct_saving": direct, "downstream_comp": comp, "rho_comp": rho,
            "d_norm": dnorm,
            "zero_detour": dnorm < ZERO_DETOUR_DNORM,
            "collateral_T": max(int(r["final_outcome"].get(
                "collateral_damage", False)) for r in recT),
            **dm,
        })
    return out


def _boot(deltas, n=2000, seed=42):
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(deltas, k=len(deltas)))
                   for _ in range(n))
    return boots[n // 40], boots[-n // 40 - 1], boots[n // 40]


def stratum_table(spans, key):
    groups = defaultdict(list)
    for s in spans:
        groups[s[key]].append(s)
    rows = []
    for name, g in sorted(groups.items()):
        d_succ = [s["d_success"] for s in g]
        ns = [s["net_saving"] for s in g]
        rho = [s["rho_comp"] for s in g if abs(s["rho_comp"]) < 1e4]
        lo_s, hi_s, lcb_s = _boot(d_succ)
        lo_n, hi_n, lcb_n = _boot(ns)
        row = {
            "stratum": name, "n": len(g),
            "strict_T": statistics.mean(s["strict_T"] for s in g),
            "strict_B": statistics.mean(s["strict_B"] for s in g),
            "d_success": statistics.mean(d_succ),
            "d_success_lcb": lcb_s,
            "total_T": statistics.mean(s["total_T"] for s in g),
            "total_B": statistics.mean(s["total_B"] for s in g),
            "net_saving": statistics.mean(ns),
            "net_saving_lcb": lcb_n,
            "rho_comp_median": statistics.median(rho) if rho else None,
            "zero_detour_ratio": sum(s["zero_detour"] for s in g) / len(g),
            "positive_ns": sum(1 for s in g if s["net_saving"] > 0),
        }
        # downstream impact metrics: T/B means + Δ
        for v in ("llm", "tools", "hall", "MaxFailRun", "RevisitRate",
                  "NewBranchRate", "ToolEntropy", "DirectActionRate"):
            row[v + "_T"] = statistics.mean(s[v + "_T"] for s in g)
            row[v + "_B"] = statistics.mean(s[v + "_B"] for s in g)
            row[v + "_d"] = row[v + "_T"] - row[v + "_B"]
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--traces-dir", default="")
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    ap.add_argument("--out", default="results/stratification/stratified_offloadability.json")
    ap.add_argument("--min-rounds", type=int, default=20,
                    help="minimum completed rounds per arm; spans below it are dropped (default 20;"
                         "use --min-rounds 10 for 10-round batches)")
    args = ap.parse_args()

    spans = per_span_stats(args.records, args.spans, args.traces_dir,
                           args.min_rounds, args.manifest)
    print("span count: %d" % len(spans))
    print()
    for key, title in (("category", "official task category (cat 1-5)"),
                       ("cls", "node_pattern × state_effect (mid-level 5 classes)"),
                       ("position", "position tertiles"),
                       ("node", "node_pattern (coarse)")):
        print("=" * 88)
        print("=== stratified by %s: quality/utility table ===" % title)
        print("=" * 88)
        print("%-40s %3s %6s %6s %+8s %+8s %8s %8s %+10s %6s %5s" % (
            "stratum", "n", "sT", "sB", "ΔSucc", "LCB", "TotalT", "TotalB",
            "NetSaving", "ρ_med", "zeroDet"))
        for r in stratum_table(spans, key):
            print("%-40s %3d %6.2f %6.2f %+8.3f %+8.3f %8.0f %8.0f %+10.0f %6.2f %5.0f%%" % (
                r["stratum"], r["n"], r["strict_T"], r["strict_B"],
                r["d_success"], r["d_success_lcb"],
                r["total_T"], r["total_B"],
                r["net_saving"], r["rho_comp_median"] or 0,
                100 * r["zero_detour_ratio"]))
        print()
        print("--- downstream impact metrics (T/B/Δ) ---")
        print("%-40s %14s %14s %14s %14s %14s" % (
            "stratum", "llm(T/B/Δ)", "tools", "hall", "MaxFailRun", "Revisit"))
        for r in stratum_table(spans, key):
            print("%-40s %5.1f/%5.1f/%+5.1f %5.1f/%5.1f/%+5.1f %5.2f/%5.2f/%+5.2f %5.1f/%5.1f/%+5.1f %5.2f/%5.2f/%+5.2f" % (
                r["stratum"], r["llm_T"], r["llm_B"], r["llm_d"],
                r["tools_T"], r["tools_B"], r["tools_d"],
                r["hall_T"], r["hall_B"], r["hall_d"],
                r["MaxFailRun_T"], r["MaxFailRun_B"], r["MaxFailRun_d"],
                r["RevisitRate_T"], r["RevisitRate_B"], r["RevisitRate_d"]))
        print()
        print("%-40s %14s %14s %14s %14s" % (
            "stratum", "NewBranch", "ToolEntropy", "DirectAction", "d_norm"))
        for r in stratum_table(spans, key):
            print("%-40s %5.2f/%5.2f/%+5.2f %5.2f/%5.2f/%+5.2f %5.2f/%5.2f/%+5.2f %5.2f" % (
                r["stratum"], r["NewBranchRate_T"], r["NewBranchRate_B"],
                r["NewBranchRate_d"],
                r["ToolEntropy_T"], r["ToolEntropy_B"], r["ToolEntropy_d"],
                r["DirectActionRate_T"], r["DirectActionRate_B"],
                r["DirectActionRate_d"],
                statistics.mean(s["d_norm"] for s in spans
                                if s[key] == r["stratum"])))
        print()
    # candidate strata summary
    rows = stratum_table(spans, "cls")
    print("=== offloadable candidate strata (quality LCB≥−0.05 and NetSaving mean>0) ===")
    for r in rows:
        if r["d_success_lcb"] >= -0.05 and r["net_saving"] > 0:
            print("  %-38s ΔSuccess=%+.3f(LCB %+.3f) NetSaving=%+.0f(LCB %+.0f)" % (
                r["stratum"], r["d_success"], r["d_success_lcb"],
                r["net_saving"], r["net_saving_lcb"]))
    print()
    # zero-detour span list
    zero = [s for s in spans if s["zero_detour"]]
    print("=== zero-detour spans (%d spans, d_norm<%.1f) ===" % (
        len(zero), ZERO_DETOUR_DNORM))
    for s in sorted(zero, key=lambda x: -x["net_saving"]):
        print("  %-12s inv=%d [%s] NetSaving=%+.0f direct saving=%+.0f" % (
            s["task"], s["inv"], s["cls"], s["net_saving"],
            s["direct_saving"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"spans": spans,
                   "by_category": stratum_table(spans, "category"),
                   "by_class": stratum_table(spans, "cls"),
                   "by_position": stratum_table(spans, "position"),
                   "by_node": stratum_table(spans, "node")},
                  f, ensure_ascii=False, indent=1)
    print("\nDetails: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
