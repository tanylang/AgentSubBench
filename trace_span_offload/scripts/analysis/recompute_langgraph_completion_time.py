#!/usr/bin/env python
"""Recompute unconditional mean completion/termination time (target-call start -> finish/termination)
for the LoCoMo + LangGraph ReAct agent cohorts, directly from the raw merged replay records.

口径 (paper Appendix E / v7_completion_time.tex):
  - metric: replay_meta.elapsed_ms, unconditional mean over all live records
    (failures and early terminations remain in both arms)
  - W_B, W_Tspan per conversation and overall; delta = W_Tspan - W_B (negative = faster)
  - also report the paired per-span delta (T_span - B averaged over rounds per span),
    which is the span-level matched estimate with a t-based CI.

Inputs:
  out_span_offload/merged/_tmp_loco10_recs_merged.jsonl      (10-conv cohort, 78,690 records)
  out_span_offload/merged/_tmp_loco_recs_merged.jsonl        (4-conv intermediate batch, 31,800 records)

Usage:
  PYTHONUTF8=1 python -I scripts/analysis/recompute_langgraph_completion_time.py
"""
from __future__ import annotations

import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILES = {
    "10conv": os.path.join(ROOT, "out_span_offload", "merged", "_tmp_loco10_recs_merged.jsonl"),
    "4conv": os.path.join(ROOT, "out_span_offload", "merged", "_tmp_loco_recs_merged.jsonl"),
}
ARMS = ["B", "T_span"]


def conv_of(task_id: str) -> str:
    # task_id format: "locomo:conv-26:q001"
    parts = task_id.split(":")
    return parts[1] if len(parts) >= 2 else task_id


def analyze(path: str, label: str) -> dict:
    per_conv = {}          # conv -> arm -> [ms]
    per_span = {}          # (conv, span_id) -> arm -> [ms]  (paired over rounds)
    n_missing = 0
    n_lines = 0
    arms_seen = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            r = json.loads(line)
            meta = r.get("replay_meta") or {}
            arm = meta.get("arm")
            ms = meta.get("elapsed_ms")
            arms_seen[arm] = arms_seen.get(arm, 0) + 1
            if arm not in ARMS or ms is None:
                if arm in ARMS and ms is None:
                    n_missing += 1
                continue
            conv = conv_of(r.get("task_id", ""))
            per_conv.setdefault(conv, {}).setdefault(arm, []).append(ms)
            per_span.setdefault((conv, r.get("span_id")), {}).setdefault(arm, []).append(ms)

    convs = sorted(per_conv)
    conv_rows = []
    for conv in convs:
        row = {"conversation": conv, "label": label}
        for arm in ARMS:
            vals = per_conv.get(conv, {}).get(arm, [])
            row[f"W_{arm}_s"] = round(statistics.mean(vals) / 1000.0, 2) if vals else None
            row[f"n_{arm}"] = len(vals)
        b = per_conv.get(conv, {}).get("B", [])
        t = per_conv.get(conv, {}).get("T_span", [])
        if b and t:
            row["delta_s"] = round((statistics.mean(t) - statistics.mean(b)) / 1000.0, 2)
            row["saving_pct"] = round(100.0 * (statistics.mean(b) - statistics.mean(t))
                                      / statistics.mean(b), 1)
        conv_rows.append(row)

    # overall unconditional means
    overall = {}
    for arm in ARMS:
        vals = [ms for conv in convs for ms in per_conv[conv].get(arm, [])]
        overall[f"W_{arm}_s"] = round(statistics.mean(vals) / 1000.0, 2)
        overall[f"n_{arm}"] = len(vals)
    overall["delta_s"] = round(overall["W_T_span_s"] - overall["W_B_s"], 2)
    overall["saving_pct"] = round(
        100.0 * (overall["W_B_s"] - overall["W_T_span_s"]) / overall["W_B_s"], 1)

    # paired per-span delta (T_span - B), mean of per-span round means
    deltas = []
    for (conv, sid), arms in per_span.items():
        b, t = arms.get("B", []), arms.get("T_span", [])
        if b and t:
            deltas.append((statistics.mean(t) - statistics.mean(b)) / 1000.0)
    paired = {"n_spans": len(deltas), "mean_delta_s": None, "ci95": None}
    if len(deltas) >= 2:
        m = statistics.mean(deltas)
        sd = statistics.stdev(deltas)
        import math
        tcrit = 1.96 if len(deltas) > 100 else 1.96  # normal approx fine at these n
        se = sd / math.sqrt(len(deltas))
        paired.update(mean_delta_s=round(m, 2),
                      ci95=[round(m - tcrit * se, 2), round(m + tcrit * se, 2)])
    return {"label": label, "lines": n_lines, "arms_seen": arms_seen,
            "n_missing_elapsed": n_missing, "conv_rows": conv_rows,
            "overall": overall, "paired": paired}


def main() -> int:
    for label, path in FILES.items():
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found")
            continue
        res = analyze(path, label)
        print("=" * 78)
        print(f"Cohort {label}: {res['lines']:,} lines, arms seen: {res['arms_seen']}, "
              f"missing elapsed_ms (B/T_span): {res['n_missing_elapsed']}")
        print("-" * 78)
        print(f"{'conv':<10} {'W_B (s)':>9} {'n_B':>7} {'W_T (s)':>9} {'n_T':>7} "
              f"{'delta (s)':>10} {'saving':>8}")
        for row in res["conv_rows"]:
            print(f"{row['conversation']:<10} {row['W_B_s']!s:>9} {row['n_B']:>7} "
                  f"{row['W_T_span_s']!s:>9} {row['n_T_span']:>7} "
                  f"{row['delta_s']!s:>10} {str(row['saving_pct'])+'%':>8}")
        print("-" * 78)
        o = res["overall"]
        print(f"OVERALL: W_B={o['W_B_s']}s (n={o['n_B']}), W_T_span={o['W_T_span_s']}s "
              f"(n={o['n_T_span']}), delta={o['delta_s']}s, saving={o['saving_pct']}%")
        p = res["paired"]
        print(f"PAIRED per-span: n={p['n_spans']}, mean delta={p['mean_delta_s']}s, "
              f"CI95={p['ci95']}")
        out = os.path.join(ROOT, "results", "locomo",
                           f"completion_time_langgraph_{label}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
