#!/usr/bin/env python
"""Minimal significant metric set: aggregate significance evidence across all analysis calibers,
rank by cross-analysis frequency, and output the minimal core set after applying redundancy rules
(definitional equivalence / derived quantities / same-source mirrors).

Analysis calibers (each line = one independent analysis):
  1-3. screening 48 spans: T vs B / T vs I / B vs I
  4-6. confirm 18 spans × 20 rounds: T vs B / T vs I / B vs I
  7-8. screening solvable/unsolvable groups (by strict_B == 1.0 / 0.0)
  9-12. screening 2×2 cells (success×success / failure×failure / failure×success / success×failure)

Redundancy rules (CONTROL_FLOW_METRICS.md §redundancy analysis + definitional equivalence):
  - AbnormalExit and FirstCompleteNormPos mirror each other (two sides of the same event) → keep AbnormalExit
  - DeadEndRate ≈ 1 − DirectActionRate (derived quantity) → keep DirectActionRate
  - StepsRatio = llm / baseline remaining steps (derived, strongly correlated with llm) → keep llm
  - TokenDelta is deprecated
Output: results/misc/minimal_metric_set.json + console stratified table.

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/minimal_metric_set.py
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from span_offload.controlflow_metrics import (  # noqa: E402
    CORE_TABLE, build_base_map, cf_metrics,
)

#: redundancy families: keep the representative, drop the rest from the "minimal set"
REDUNDANCY = {
    "AbnormalExit": {"FirstCompleteNormPos"},
    "DirectActionRate": {"DeadEndRate"},
    "llm": {"StepsRatio"},
}
#: metrics excluded from the minimal set (deprecated/mechanical)
EXCLUDE = {"TokenDelta"}


def sig_set_from_json(path, pair):
    d = json.load(open(path, encoding="utf-8"))
    return {r["metric"] for r in d["rows"]
            if r["pair"] == pair and r["significant"]}


def sig_set_paired(group, a, b, base_map, full, rounds=5):
    rng = random.Random(42)
    sig = set()
    for v in CORE_TABLE:
        deltas = []
        for k in group:
            va = statistics.mean(
                cf_metrics(r, base_map.get(k))[v]
                for r in full[k][a][:rounds])
            vb = statistics.mean(
                cf_metrics(r, base_map.get(k))[v]
                for r in full[k][b][:rounds])
            if va == va and vb == vb:
                deltas.append(va - vb)
        if len(deltas) < 2:
            continue
        mu = statistics.mean(deltas)
        boots = sorted(statistics.mean(rng.choices(deltas, k=len(deltas)))
                       for _ in range(2000))
        if boots[50] > 0 or boots[1949] < 0:
            sig.add(v)
    return sig


def main() -> int:
    p48 = ("out_span_offload/legacy/"
           "offload_profile_simplified_qwen3_8-27b_Qwen3_8-27B_Qwen_Qwen3_5-9B_dedup.jsonl")
    recs = [json.loads(l) for l in open(p48, encoding="utf-8") if l.strip()]
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[(r["task_id"], r["invocation_index"])][
            r["replay_meta"].get("arm")].append(r)
    full = {k: v for k, v in by.items()
            if all(len(v.get(a, [])) == 5 for a in ("T_span", "B", "I"))}
    base_map = build_base_map(
        "check_point_traces/appworld_simplified_qwen3_8-27b_batch1")
    sr = lambda k, a: statistics.mean(
        int(bool(r["strict_success"])) for r in full[k][a])

    freq = defaultdict(int)
    analyses = {}
    for pair, name in (("T_span:B", "screening TvsB"),
                       ("T_span:I", "screening TvsI"),
                       ("B:I", "screening BvsI")):
        s = sig_set_from_json("results/main_comparisons/main_comparison_27b_final.json", pair)
        analyses[name] = s
        for m in s:
            freq[m] += 1
    for pair, name in (("T_span:B", "confirm TvsB"),
                       ("T_span:I", "confirm TvsI"),
                       ("B:I", "confirm BvsI")):
        s = sig_set_from_json("results/main_comparisons/main_comparison_confirm20_final.json",
                              pair)
        analyses[name] = s
        for m in s:
            freq[m] += 1
    g1 = [k for k in full if sr(k, "B") == 1.0]
    g0 = [k for k in full if sr(k, "B") == 0.0]
    for g, name in ((g1, "可解组"), (g0, "不可解组")):
        s = sig_set_paired(g, "T_span", "B", base_map, full)
        analyses[name] = s
        for m in s:
            freq[m] += 1
    cells = defaultdict(list)
    for k in full:
        cells[(("成功" if sr(k, "B") == 1.0 else "失败"),
               ("成功" if sr(k, "T_span") == 1.0 else "失败"))].append(k)
    for (tb, tt), g in cells.items():
        if len(g) < 2:
            continue
        s = sig_set_paired(g, "T_span", "B", base_map, full)
        analyses["%s×%s" % (tb, tt)] = s
        for m in s:
            freq[m] += 1

    print("=== significance frequency across %d analyses (all metrics) ===" % len(analyses))
    for m, f in sorted(freq.items(), key=lambda x: -x[1]):
        if m in EXCLUDE:
            continue
        print("  %-24s %d times" % (m, f))

    # minimal set: frequency ≥2 and not among the dropped items of the redundancy families
    dropped = set().union(*REDUNDANCY.values())
    core = [m for m, f in freq.items()
            if f >= 2 and m not in EXCLUDE and m not in dropped]
    print()
    print("=== minimal core set (frequency ≥2, after redundancy removal, %d total) ===" % len(core))
    for m in sorted(core, key=lambda x: -freq[x]):
        print("  %-24s frequency %d/11" % (m, freq[m]))

    out = {"n_analyses": len(analyses), "frequency": dict(freq),
           "analyses": {k: sorted(v) for k, v in analyses.items()},
           "minimal_core": sorted(core, key=lambda x: -freq[x]),
           "redundancy_rule": {k: sorted(v) for k, v in REDUNDANCY.items()}}
    with open("results/misc/minimal_metric_set.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/misc/minimal_metric_set.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
