#!/usr/bin/env python
"""LoCoMo task-level (QA unit) classification.

Aggregates the spans file by task_id (locomo:<sample>:q<qi>), producing a task-level
classification table and distributions, and optionally merges offload replay records
for per-class offload statistics.

Classification dimensions:
  - memory_class: memory_write (any span in the unit has state_effect containing
    memory_write = manage_memory) / memory_read (has a langgraph_memory_operation
    tool span = search_memory) / no_memory (all langgraph_qa_answer, no tools)
  - baseline: pass / fail (any span baseline_failed)
  - category: official task category (gold qa integer 1-5, passed through without naming)
  - position: question position (early/mid/late tertile of qi within the sample)
  - n_spans / n_tool_calls: complexity

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \\
    scripts/analysis/locomo_task_classification.py \\
    --spans "out_span_offload/langgraph/*/spans_langgraph_*.jsonl" \\
    --records "out_span_offload/langgraph/conv-*/offload/offload_profile_langgraph_p*_*.jsonl" \\
    --out results/locomo/locomo_task_classification.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from span_offload.trace_io import load_manifest  # noqa: E402


def _expand(patterns: str) -> list:
    out: list = []
    for p in patterns.split():
        out.extend(sorted(glob.glob(p)))
    return out


def classify_task(spans: list, manifest, sample_id: str, qi: int) -> dict:
    """Produce task-level classification fields from the unit's span list."""
    n_spans = len(spans)
    n_tool_calls = 0
    has_write = False
    has_tool_span = False
    baseline_failed = False
    node_patterns = collections.Counter()
    for s in spans:
        d = s.get("span_descriptor") or {}
        np_ = d.get("node_pattern", "?")
        node_patterns[np_] += 1
        st = d.get("span_subtask_type", "")
        se = (d.get("mechanism_tags") or {}).get("state_effect") or []
        if "memory_write" in se:
            has_write = True
        if st == "langgraph_memory_operation":
            has_tool_span = True
        sc = (s.get("replay_input") or {}).get(
            "output_contract_runtime", {}).get("semantic_constraints") or {}
        n_tool_calls += int(sc.get("n_tool_calls", 0) or 0)
        if s.get("baseline_failed"):
            baseline_failed = True
    if has_write:
        memory_class = "memory_write"
    elif has_tool_span:
        memory_class = "memory_read"
    else:
        memory_class = "no_memory"

    gold_qa = []
    total_qa = 0
    mt = manifest.get(sample_id)
    if mt is not None:
        gold_qa = list(mt.evaluator.get("qa") or [])
        total_qa = len(gold_qa)
    gq = gold_qa[qi] if qi < len(gold_qa) else {}
    category = gq.get("category")
    position = "unknown"
    if total_qa:
        if qi < total_qa / 3:
            position = "early"
        elif qi < 2 * total_qa / 3:
            position = "mid"
        else:
            position = "late"
    return {
        "memory_class": memory_class,
        "baseline": "fail" if baseline_failed else "pass",
        "category": category,
        "position": position,
        "n_spans": n_spans,
        "n_tool_calls": n_tool_calls,
        "node_patterns": dict(node_patterns),
        "question": str(gq.get("question", ""))[:200],
        "evidence": gq.get("evidence"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True,
                    help="spans jsonl glob (space-separated, multiple)")
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    ap.add_argument("--records", default="",
                    help="optional: offload replay record glob (merges per-class strict statistics)")
    ap.add_argument("--out", default="results/locomo/locomo_task_classification.json")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    span_files = _expand(args.spans)
    if not span_files:
        raise SystemExit("no spans files matched: %s" % args.spans)
    tasks: dict = {}
    for sf in span_files:
        for line in open(sf, encoding="utf-8"):
            s = json.loads(line)
            tid = s.get("task_id", "")
            parts = tid.split(":")
            if len(parts) < 3 or not parts[2].startswith("q"):
                continue
            sample_id = parts[1]
            qi = int(parts[2][1:])
            tasks.setdefault(tid, []).append(s)

    rows = []
    for tid, spans in sorted(tasks.items()):
        sample_id = tid.split(":")[1]
        qi = int(tid.rsplit("q", 1)[-1])
        row = {"task_id": tid, "sample_id": sample_id, "qa_index": qi}
        row.update(classify_task(spans, manifest, sample_id, qi))
        rows.append(row)
    print("classified tasks:", len(rows), "| span files:", len(span_files))

    dist = collections.Counter()
    for r in rows:
        dist[(r["memory_class"], r["baseline"], str(r["category"]))] += 1
    print("\n== memory_class x baseline x category ==")
    for (mc, bl, cat), n in sorted(dist.items()):
        print("  %-12s %-5s cat=%-4s %d" % (mc, bl, cat, n))
    print("\n== memory_class ==")
    for mc, n in sorted(collections.Counter(
            r["memory_class"] for r in rows).items()):
        print("  %-12s %d" % (mc, n))
    print("\n== baseline ==", dict(collections.Counter(
        r["baseline"] for r in rows)))

    out = {"tasks": rows,
           "memory_class_counts": dict(collections.Counter(
               r["memory_class"] for r in rows)),
           "baseline_counts": dict(collections.Counter(
               r["baseline"] for r in rows)),
           "category_counts": {str(k): v for k, v in collections.Counter(
               r["category"] for r in rows).items()},
           "distribution": {("%s|%s|%s" % k): v for k, v in dist.items()},
           "offload_by_class": {}}

    if args.records:
        rec_files = _expand(args.records)
        per_task = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for rf in rec_files:
            for line in open(rf, encoding="utf-8"):
                r = json.loads(line)
                arm = (r.get("replay_meta") or {}).get("arm", "?")
                per_task[r.get("task_id", "?")][arm].append(
                    bool(r.get("strict_success")))
        stats = collections.defaultdict(
            lambda: collections.defaultdict(list))
        for r in rows:
            arms = per_task.get(r["task_id"], {})
            for arm in ("T_span", "I", "B"):
                v = arms.get(arm, [])
                stats[r["memory_class"]][arm].extend(v)
        obc = {}
        for mc, arms in sorted(stats.items()):
            obc[mc] = {}
            for arm, vals in sorted(arms.items()):
                if not vals:
                    continue
                obc[mc][arm] = {
                    "n": len(vals),
                    "strict_rate": sum(vals) / len(vals),
                }
        out["offload_by_class"] = obc
        print("\n== per-class offload strict (T / I / B) ==")
        for mc, arms in sorted(obc.items()):
            line = ["  %-12s" % mc]
            for arm in ("T_span", "I", "B"):
                v = arms.get(arm)
                line.append("%s=%s" % (
                    arm, "%.2f(n=%d)" % (v["strict_rate"], v["n"])
                    if v else "-"))
            print("  ".join(line))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
