#!/usr/bin/env python
"""Normalized structural edit distance + first divergence point (following Mazhar et al., arXiv:2604.27586).

Paper definitions:
  - structural event signature σ(e): keeps only control-flow information —
    decision event = tool name list; tool execution = (tool name, success/failure); idle = R; protocol repair = Rp
  - signature sequence S(τ) = (σ(e_1), …, σ(e_T))
  - normalized structural edit distance d_norm = ED(S(τ), S(τ̃)) / max(|S|, |S̃|)
    (Wagner–Fischer dynamic programming, range 0=identical ~ 1=completely different)
  - first divergence point t★ = position where the two sequences first diverge under the optimal alignment
    (equivalent to longest common prefix length + 1; None when fully identical)

Control design (consistent with the apparatus): each span's SLM rounds are compared pairwise with 5
identity rounds, giving cross-group distance D_cross and identity within-group distance D_self (rerun
noise floor); intervention divergence = D_cross − D_self. t★ uses the median of cross-group alignments.

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/trace_edit_distance.py
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

SLM_FILES = [
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl",
]
ID_FILES = [
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_21abae1_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_2.jsonl",
]


def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs.extend(json.loads(l) for l in f if l.strip())
    return recs


def signature(record: dict) -> list:
    """Structural event signature sequence (paper σ(e): control-flow information, no text content)."""
    sig = []
    for e in (record.get("downstream_trace") or {}).get("spans", []):
        if e.get("source") == "repair":
            sig.append(("Rp",))
        elif e.get("span_type") == "llm.call":
            out = e.get("output") or {}
            calls = out.get("tool_calls")
            if calls:
                # decision event: which tools are chosen to be called (control-flow content of the routing decision)
                sig.append(("D", tuple(c.get("name", "") for c in calls)))
            else:
                sig.append(("R",))
        elif e.get("span_type") == "tool.execution":
            # tool execution: tool name + success/failure (paper example: tool name, operation,
            # success/failure)
            sig.append(("T", str(e.get("tool", "")), bool(e.get("ok"))))
    return sig


def edit_distance(a: list, b: list) -> int:
    """Wagner–Fischer edit distance (substitution/insertion/deletion each cost 1)."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + cost)  # substitution/match
        prev = cur
    return prev[m]


def first_divergence(a: list, b: list):
    """First divergence point t★ = longest common prefix length + 1 (None when fully identical).

    Under the optimal alignment, matching the whole prefix is the optimal strategy (matching
    costs zero), so the longest common prefix is "the step from which the two signature
    sequences start to diverge".
    """
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    if k == len(a) and k == len(b):
        return None  # fully identical
    return k + 1


def main() -> int:
    random.seed(42)
    slm = defaultdict(list)
    idd = defaultdict(list)
    for r in load(SLM_FILES + ID_FILES):
        k = (r["task_id"], r["invocation_index"])
        (idd if r.get("replay_meta", {}).get("injection") == "identity"
         else slm)[k].append(r)
    keys = sorted(set(slm) & set(idd))
    print("paired span count: %d" % len(keys))

    classes = {r["task"] + "__" + str(r["inv"]): r["response_class"]
               for r in json.load(open("results/appworld_53span/response_analysis_clean53.json",
                                       encoding="utf-8"))["per_span"]}

    rows = []
    by_cls = defaultdict(list)
    for k in keys:
        s_sigs = [signature(r) for r in slm[k]]
        i_sigs = [signature(r) for r in idd[k]]
        # cross-group: SLM × identity pairwise
        cross = [edit_distance(a, b) / max(len(a), len(b), 1)
                 for a in s_sigs for b in i_sigs]
        # within-group noise floor: identity pairwise
        selfd = [edit_distance(i_sigs[p], i_sigs[q]) /
                 max(len(i_sigs[p]), len(i_sigs[q]), 1)
                 for p in range(len(i_sigs)) for q in range(p + 1, len(i_sigs))]
        # first divergence point (SLM × identity alignments, take the median)
        fds = [first_divergence(a, b) for a in s_sigs for b in i_sigs]
        fds_finite = [f for f in fds if f is not None]
        cls = classes.get(k[0] + "__" + str(k[1]), "?")
        row = {"task": k[0], "inv": k[1], "class": cls,
               "d_cross": statistics.mean(cross),
               "d_self": statistics.mean(selfd),
               "delta": statistics.mean(cross) - statistics.mean(selfd),
               "t_star_median": statistics.median(fds_finite) if fds_finite else None,
               "t_star_1_frac": sum(1 for f in fds_finite if f <= 1) / max(len(fds_finite), 1),
               "n_identical": sum(1 for f in fds if f is None),
               "sig_len_s": statistics.mean(len(x) for x in s_sigs),
               "sig_len_i": statistics.mean(len(x) for x in i_sigs)}
        rows.append(row)
        by_cls[cls].append(row)

    print("\n=== per span (sorted by response class) ===")
    print("%-14s %-3s %-24s %8s %8s %+8s %6s %8s" % (
        "task", "inv", "class", "d_cross", "d_self", "Δ", "t★median", "t★≤1 frac"))
    for r in sorted(rows, key=lambda r: (r["class"], r["task"], r["inv"])):
        print("%-14s %-3d %-24s %8.3f %8.3f %+8.3f %6s %8.0f%%" % (
            r["task"], r["inv"], r["class"], r["d_cross"], r["d_self"],
            r["delta"],
            "∞" if r["t_star_median"] is None else r["t_star_median"],
            100 * r["t_star_1_frac"]))

    print("\n=== response class aggregates ===")
    for cls in sorted(by_cls):
        g = by_cls[cls]
        print("  %-24s n=%2d | d_cross %.3f | d_self %.3f | Δ %+.3f | t★median %s | t★≤1 %.0f%% | fully identical %.0f%%" % (
            cls, len(g),
            statistics.mean(r["d_cross"] for r in g),
            statistics.mean(r["d_self"] for r in g),
            statistics.mean(r["delta"] for r in g),
            ("∞" if statistics.mean(r["t_star_median"] for r in g if r["t_star_median"]) is None
             else "%.1f" % statistics.mean(r["t_star_median"] for r in g if r["t_star_median"])),
            100 * statistics.mean(r["t_star_1_frac"] for r in g),
            100 * statistics.mean(r["n_identical"] / 25 for r in g)))

    with open("results/appworld_53span/trace_edit_distance_53.json", "w", encoding="utf-8") as f:
        json.dump({"per_span": rows}, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/appworld_53span/trace_edit_distance_53.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
