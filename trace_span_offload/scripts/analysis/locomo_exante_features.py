#!/usr/bin/env python
"""LoCoMo pre-offload (ex-ante) offloadability profile: upstream trajectory features × offloadable outcome.

All features are aggregated directly from the spans file (the recorded native trajectory), no replay needed:
  - baseline_tokens / n_tool_calls_self: the span's own size
  - max_remaining_steps: downstream length (remaining LLM rounds)
  - up_llm_steps / up_tool_calls: upstream LLM round count / cumulative tool calls
  - up_search_spans / up_write_spans: upstream memory-retrieval/write span counts
  - last_write_gap: round distance from the most recent memory write (memory freshness; no write = whole upstream)
  - after_failure / inside_loop: whether this span is inside a repair turn / loop
  - position (early/mid/late ordinal) / category (official category 1-5)

Outcome (from replay records, only used for profile fitting):
  - offload_ok: per-span offload success = DiD≥0 (T not worse than identity)
  - d_success: strict_T − strict_B

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \\
    scripts/analysis/locomo_exante_features.py \\
    --spans out_span_offload/merged/_tmp_loco_spans_merged.jsonl \\
    --records out_span_offload/merged/_tmp_loco_recs_merged.jsonl \\
    --out results/locomo/locomo_exante_features.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from span_offload.trace_io import load_manifest  # noqa: E402

BOOT = 2000


def _mw(x: list, y: list) -> float:
    """Mann-Whitney U two-tailed p (approximation, normal)."""
    from math import erf, sqrt
    m, n = len(x), len(y)
    if not m or not n:
        return 1.0
    pooled = sorted(list((v, 0) for v in x) + list((v, 1) for v in y))
    ranks = {}
    i = 0
    while i < len(pooled):
        j = i
        while j < len(pooled) and pooled[j][0] == pooled[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    rx = sum(ranks[k] for k in range(len(pooled)) if pooled[k][1] == 0)
    u = rx - m * (m + 1) / 2.0
    mu = m * n / 2.0
    sigma = sqrt(m * n * (m + n + 1) / 12.0)
    if sigma == 0:
        return 1.0
    z = abs(u - mu) / sigma
    return max(1e-12, 1 - erf(z / sqrt(2)))


def _boot_spearman(x: list, y: list, rng: random.Random) -> tuple:
    n = len(x)
    if n < 3:
        return 0.0, 0.0, 0.0
    idx = list(range(n))
    draws = []
    for _ in range(BOOT):
        s = [rng.randrange(n) for _ in range(n)]
        xs = [x[i] for i in s]
        ys = [y[i] for i in s]
        try:
            draws.append(statistics.correlation(xs, ys))
        except statistics.StatisticsError:
            continue
    draws.sort()
    if not draws:
        return 0.0, 0.0, 0.0
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    return statistics.correlation(x, y), lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    ap.add_argument("--out", default="results/locomo/locomo_exante_features.json")
    args = ap.parse_args()

    # ---- records → per-span outcome ----
    by = defaultdict(lambda: defaultdict(list))
    for l in open(args.records, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        by[(r["task_id"], r["invocation_index"])][
            r["replay_meta"].get("arm")].append(bool(r["strict_success"]))
    outcome = {}
    for (t, i), arms in by.items():
        if not all(len(arms.get(a, [])) >= 10 for a in ("T_span", "B", "I")):
            continue
        stT = statistics.mean(arms["T_span"][:10])
        stB = statistics.mean(arms["B"][:10])
        stI = statistics.mean(arms["I"][:10])
        outcome[(t, i)] = {"offload_ok": stT >= stI,
                           "d_success": stT - stB}

    # ---- spans → per-span upstream aggregation ----
    spans_by_task = defaultdict(list)
    for l in open(args.spans, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        s = json.loads(l)
        spans_by_task[s["task_id"]].append(s)
    manifest = load_manifest(args.manifest)
    rows = []
    for t, spans in sorted(spans_by_task.items()):
        spans.sort(key=lambda s: s["invocation_index"])
        parts = t.split(":")
        sample_id = parts[1] if len(parts) >= 3 else t
        qi = int(parts[2][1:]) if len(parts) >= 3 else 0
        mt = manifest.get(sample_id)
        qa = list((mt.evaluator.get("qa") or []) if mt else [])
        cat = qa[qi].get("category") if qi < len(qa) else None
        n_qa = len(qa)
        up_llm = 0
        up_tools = 0
        up_search = 0
        up_write = 0
        last_write_inv = -1
        for s in spans:
            inv = s["invocation_index"]
            if (t, inv) not in outcome:
                continue
            d = s.get("span_descriptor") or {}
            se = (d.get("mechanism_tags") or {}).get("state_effect") or []
            gc = d.get("graph_context") or {}
            sc = (s.get("replay_input") or {}).get(
                "output_contract_runtime", {}).get("semantic_constraints") or {}
            own_tools = int(sc.get("n_tool_calls", 0) or 0)
            cont = (s.get("replay_input") or {}).get("continuation") or {}
            is_tool_span = d.get("span_subtask_type") == "langgraph_memory_operation"
            rows.append({
                "task": t, "inv": inv,
                "category": cat,
                "position": (0 if qi < n_qa / 3 else (1 if qi < 2 * n_qa / 3
                            else 2)) if n_qa else 0,
                "baseline_tokens": int(s.get("baseline_tokens") or 0),
                "n_tool_calls_self": own_tools,
                "max_remaining_steps": int(cont.get("max_remaining_steps", 0)
                                           or (len(spans) - inv)),
                "up_llm_steps": up_llm,
                "up_tool_calls": up_tools,
                "up_search_spans": up_search,
                "up_write_spans": up_write,
                "last_write_gap": (inv - last_write_inv)
                                  if last_write_inv >= 0 else inv,
                "after_failure": int(bool(gc.get("after_failure"))),
                "inside_loop": int(bool(gc.get("inside_loop"))),
                "offload_ok": outcome[(t, inv)]["offload_ok"],
                "d_success": outcome[(t, inv)]["d_success"],
            })
            up_llm += 1
            up_tools += own_tools
            if is_tool_span:
                if "memory_write" in se:
                    up_write += 1
                    last_write_inv = inv
                else:
                    up_search += 1
    print("spans with outcome:", len(rows))

    rng = random.Random(42)
    feats = ["baseline_tokens", "n_tool_calls_self", "max_remaining_steps",
             "up_llm_steps", "up_tool_calls", "up_search_spans",
             "up_write_spans", "last_write_gap", "after_failure",
             "inside_loop", "position", "category"]
    table = []
    ok_rows = [r for r in rows if r["offload_ok"]]
    fail_rows = [r for r in rows if not r["offload_ok"]]
    for f in feats:
        x_ok = [r[f] for r in ok_rows if r[f] is not None]
        x_fail = [r[f] for r in fail_rows if r[f] is not None]
        p_mw = _mw(x_ok, x_fail)
        rho, lo, hi = _boot_spearman(
            [r[f] for r in rows if r[f] is not None],
            [r["d_success"] for r in rows if r[f] is not None], rng)
        table.append({
            "feature": f,
            "mean_ok": statistics.mean(x_ok) if x_ok else None,
            "mean_fail": statistics.mean(x_fail) if x_fail else None,
            "mw_p": p_mw,
            "rho_d_success": rho, "rho_ci_lo": lo, "rho_ci_hi": hi,
            "significant": (p_mw < 0.05) or (lo > 0 or hi < 0),
        })
    table.sort(key=lambda r: r["mw_p"])
    print()
    print("=== pre-offload features × offloadable outcome (sorted by discriminability) ===")
    print("%-22s %9s %9s %8s %9s [%s] %s" % (
        "feature", "mean_ok", "mean_fail", "MW p", "rho(d)", "CI", "★"))
    for r in table:
        print("%-22s %9.2f %9.2f %8.1e %+9.3f [%+.3f,%+.3f] %s" % (
            r["feature"],
            r["mean_ok"] if r["mean_ok"] is not None else float("nan"),
            r["mean_fail"] if r["mean_fail"] is not None else float("nan"),
            r["mw_p"], r["rho_d_success"], r["rho_ci_lo"], r["rho_ci_hi"],
            "★" if r["significant"] else ""))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "features": table}, f,
                  ensure_ascii=False, indent=1)
    print("\nDetails: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
