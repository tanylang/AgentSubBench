#!/usr/bin/env python
"""Compute all numbers needed to update the paper's LoCoMo cohort from the
257-span conv-26 batch to the 10-conversation cohort (2,623 spans, 78,690 records).

Outputs (per cohort):
  - per-arm strict-success rates and counts
  - mixed-B count (checkpoints whose repeated B runs contain both success and failure)
  - direct / compensation / net token decomposition (k), time reduction %
  - control-phase strata (initial/execution/terminal): n, dQ pp, net tokens k
  - task-cluster bootstrap CIs (10,000 reps, seed 20260924) for dQ / net / dW

Validation: the same procedure run on the old batch1/257span cohort must reproduce
the paper's tab:v9-new-cohorts row (-10.6 [-14.9,-6.6], +2.19 [-0.07,+4.53],
-0.96 [-1.72,-0.19]) and the v9_detailed_tables decomposition (+0.26 / -1.93 / +2.19 / 8.3%).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = 20260924
N_REPS = 10_000


def records_from(path):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            yield json.loads(line)


def collect(paths):
    """Per-task aggregates from raw records: arm -> (n, sum_strict, sum_tokens, sum_ms)."""
    per_task = {}
    for p in paths:
        for r in records_from(p):
            m = r.get("replay_meta") or {}
            arm, ms = m.get("arm"), m.get("elapsed_ms")
            res = r.get("resource") or {}
            tok = ((res.get("target_prompt_tokens") or 0) + (res.get("target_completion_tokens") or 0)
                   + (res.get("downstream_prompt_tokens") or 0)
                   + (res.get("downstream_completion_tokens") or 0))
            # split for direct/compensation
            tgt = ((res.get("target_prompt_tokens") or 0) + (res.get("target_completion_tokens") or 0))
            dwn = ((res.get("downstream_prompt_tokens") or 0)
                   + (res.get("downstream_completion_tokens") or 0))
            if arm is None or ms is None:
                continue
            t = per_task.setdefault(r.get("task_id"), {})
            a = t.setdefault(arm, {"n": 0, "strict": 0.0, "tok": 0.0, "ms": 0.0,
                                   "tgt": 0.0, "dwn": 0.0})
            a["n"] += 1
            a["strict"] += float(bool(r.get("strict_success")))
            a["tok"] += tok
            a["ms"] += ms
            a["tgt"] += tgt
            a["dwn"] += dwn
    return per_task


def arm_mean(per_task, arm, key):
    n = sum(t[arm]["n"] for t in per_task.values() if arm in t)
    s = sum(t[arm][key] for t in per_task.values() if arm in t)
    return s / n if n else float("nan")


def bootstrap_ci(per_task, key_T, key_B, nkey="n", reps=N_REPS, seed=SEED):
    tasks = sorted(per_task)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tasks), size=(reps, len(tasks)))
    val_T = np.array([per_task[t].get("T_span", {}).get(key_T, 0.0) for t in tasks])
    val_B = np.array([per_task[t].get("B", {}).get(key_B, 0.0) for t in tasks])
    nT = np.array([per_task[t].get("T_span", {}).get(nkey, 1) for t in tasks])
    nB = np.array([per_task[t].get("B", {}).get(nkey, 1) for t in tasks])
    sT = val_T[idx].sum(axis=1) / nT[idx].sum(axis=1)
    sB = val_B[idx].sum(axis=1) / nB[idx].sum(axis=1)
    return sT, sB


def report(label, paths):
    per_task = collect(paths)
    tasks = sorted(per_task)
    print("=" * 78)
    print(f"### {label}: {len(tasks)} tasks")
    nB = sum(t["B"]["n"] for t in per_task.values() if "B" in t)
    print(f"records per arm: B={nB}  (spans = {sum(1 for t in per_task.values() if 'B' in t and 'T_span' in t)})")
    rates = {arm: arm_mean(per_task, arm, "strict") for arm in ("I", "B", "T_span")}
    counts = {arm: round(rates[arm] * nB) for arm in ("I", "B", "T_span")}
    print("strict rates:", {a: round(r, 4) for a, r in rates.items()}, "counts:", counts)
    print("contrasts: B-I=%.1f  T-B=%.1f  T-I=%.1f" % (
        100 * (rates["B"] - rates["I"]), 100 * (rates["T_span"] - rates["B"]),
        100 * (rates["T_span"] - rates["I"])))
    # mixed-B: spans with both success and failure among B runs
    mixed_fs = mixed_st = 0
    for t in per_task.values():
        pass
    # mixed-B needs span granularity; recompute from records
    span_B = {}
    for p in paths:
        for r in records_from(p):
            m = r.get("replay_meta") or {}
            if m.get("arm") == "B":
                fo = r.get("final_outcome") or {}
                s = span_B.setdefault(r.get("span_id"), set())
                s.add(bool(fo.get("final_task_success")))
                s2 = span_B.setdefault(r.get("span_id") + "|strict", set())
                s2.add(bool(r.get("strict_success")))
    mixed_fs = sum(1 for k, v in span_B.items() if not k.endswith("|strict") and len(v) > 1)
    mixed_st = sum(1 for k, v in span_B.items() if k.endswith("|strict") and len(v) > 1)
    n_spans = sum(1 for k in span_B if not k.endswith("|strict"))
    print(f"mixed-B: final_task_success {mixed_fs}/{n_spans}, strict {mixed_st}/{n_spans}")
    # token decomposition
    dir_ = (arm_mean(per_task, "B", "tgt") - arm_mean(per_task, "T_span", "tgt")) / 1000
    comp = (arm_mean(per_task, "T_span", "dwn") - arm_mean(per_task, "B", "dwn")) / 1000
    net = (arm_mean(per_task, "B", "tok") - arm_mean(per_task, "T_span", "tok")) / 1000
    print("tokens: direct=%+.2f  compensation=%+.2f  net=%+.2f" % (dir_, comp, net))
    wB = arm_mean(per_task, "B", "ms") / 1000
    wT = arm_mean(per_task, "T_span", "ms") / 1000
    print("time: W_B=%.2f  W_T=%.2f  dW=%+.2f  time_red=%.1f%%" % (wB, wT, wT - wB, 100 * (wB - wT) / wB))
    # control-phase strata
    phase = {}
    for p in paths:
        for r in records_from(p):
            m = r.get("replay_meta") or {}
            arm = m.get("arm")
            if arm not in ("B", "T_span"):
                continue
            ph = ((r.get("span_descriptor") or {}).get("graph_context") or {}).get("control_phase")
            res = r.get("resource") or {}
            tok = ((res.get("target_prompt_tokens") or 0) + (res.get("target_completion_tokens") or 0)
                   + (res.get("downstream_prompt_tokens") or 0)
                   + (res.get("downstream_completion_tokens") or 0))
            a = phase.setdefault(ph, {}).setdefault(arm, {"n": 0, "strict": 0.0, "tok": 0.0})
            a["n"] += 1
            a["strict"] += float(bool(r.get("strict_success")))
            a["tok"] += tok
    print("control-phase strata:")
    for ph in sorted(phase, key=str):
        n = phase[ph]["B"]["n"]
        dq = 100 * (phase[ph]["T_span"]["strict"] / phase[ph]["T_span"]["n"]
                    - phase[ph]["B"]["strict"] / phase[ph]["B"]["n"])
        nt = (phase[ph]["B"]["tok"] / phase[ph]["B"]["n"]
              - phase[ph]["T_span"]["tok"] / phase[ph]["T_span"]["n"]) / 1000
        print(f"  {ph}: spans={n//10}  dQ={dq:+.1f} pp  net_tok={nt:+.2f} k")
    # bootstrap CIs
    sT, sB = bootstrap_ci(per_task, "strict", "strict")
    dq = 100 * (sT - sB)
    tT, tB = bootstrap_ci(per_task, "tok", "tok")
    netk = (tB - tT) / 1000
    wTb, wBb = bootstrap_ci(per_task, "ms", "ms")
    dw = (wTb - wBb) / 1000
    for name, arr in (("dQ pp", dq), ("net k", netk), ("dW s", dw)):
        print(f"bootstrap {name}: point={np.mean(arr):+.4f}  ci=[{np.percentile(arr,2.5):+.2f}, {np.percentile(arr,97.5):+.2f}]")


if __name__ == "__main__":
    old = sorted(glob.glob(os.path.join(ROOT, "out_span_offload", "langgraph", "batch1",
                                        "257span", "*.jsonl")))
    new = [os.path.join(ROOT, "out_span_offload", "merged", "_tmp_loco10_recs_merged.jsonl")]
    report("OLD batch1/257span (paper cohort, validation)", old)
    report("NEW 10conv cohort", new)
