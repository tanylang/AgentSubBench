#!/usr/bin/env python
"""Dedupe offload_profile + split by task.

Dedupe rule: for each (span_id, arm) keep the first --rounds records in
**file order** (duplicate rounds come from block plan drift on resume and are
appended at the file tail; the first N records in file order = the original
batch from the first submission, clean independent repeats). Skip leftover lines.

Output:
  - split by task: <base>_<task>.jsonl
  - deduped full:  <base>_dedup.jsonl
  - dedup report:  <base>_dedup_report.json (kept/dropped counts per arm per span)

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/dedupe_split_profile.py \
      --input out_span_offload/offload_profile_simplified_qwen3_8-27b_Qwen3_8-27B_Qwen_Qwen3_5-9B.jsonl \
      --out-dir out_span_offload \
      --base offload_profile_simplified_qwen3_8-27b_Qwen3_8-27B_Qwen_Qwen3_5-9B \
      --rounds 5
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

ARM_ALIAS = {"identity": "I", "slm": "T_span"}
ARMS = ("T_span", "B", "I")


def record_arm(r: dict) -> str:
    m = r.get("replay_meta") or {}
    arm = m.get("arm") or ARM_ALIAS.get(m.get("injection", ""), "")
    return arm if arm in ARMS else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", required=True,
                    help="output filename stem (same as input minus .jsonl)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="max rounds to keep per (span, arm)")
    ap.add_argument("--complete-only", action="store_true",
                    help="keep only complete spans with all three groups at --rounds rounds"
                         "(drop partial records of unfinished spans)")
    args = ap.parse_args()

    # read + dedupe (file order takes priority)
    kept = []
    seen = Counter()  # (span_id, arm) -> number already kept
    dropped = Counter()
    n_broken = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                n_broken += 1
                continue
            arm = record_arm(r)
            key = (r.get("span_id"), arm)
            if not arm or seen[key] >= args.rounds:
                dropped[key] += 1
                continue
            seen[key] += 1
            kept.append(r)
    print("input %d lines | kept %d | dropped duplicates %d | broken lines %d" % (
        seen.total() + sum(dropped.values()) + n_broken, len(kept),
        sum(dropped.values()), n_broken))

    # per-span completeness report
    span_rounds = defaultdict(Counter)
    for r in kept:
        span_rounds[r["span_id"]][record_arm(r)] += 1
    complete = sum(1 for c in span_rounds.values()
                   if all(c[a] >= args.rounds for a in ARMS))
    partial = sum(1 for c in span_rounds.values() if 0 < sum(c.values()) < args.rounds * 3)
    print("spans %d | complete (all three groups at %d rounds) %d | partial %d" % (
        len(span_rounds), args.rounds, complete, partial))
    if args.complete_only:
        complete_ids = {sid for sid, c in span_rounds.items()
                        if all(c[a] >= args.rounds for a in ARMS)}
        n_dropped = len(kept)
        kept = [r for r in kept if r["span_id"] in complete_ids]
        print("complete-only: dropped %d records of partial spans" % (n_dropped - len(kept)))

    # write per-task splits + deduped full output
    os.makedirs(args.out_dir, exist_ok=True)
    by_task = defaultdict(list)
    for r in kept:
        by_task[r["task_id"]].append(r)
    for task, recs in sorted(by_task.items()):
        p = os.path.join(args.out_dir, "%s_%s.jsonl" % (args.base, task))
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    full_path = os.path.join(args.out_dir, "%s_dedup.jsonl" % args.base)
    with open(full_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "kept": len(kept), "dropped_duplicates": sum(dropped.values()),
        "broken_lines": n_broken, "n_spans": len(span_rounds),
        "complete_spans": complete, "partial_spans": partial,
        "per_task": {t: len(v) for t, v in sorted(by_task.items())},
        "incomplete_spans": [
            {"span_id": sid, "arms": dict(c)}
            for sid, c in span_rounds.items()
            if not all(c[a] >= args.rounds for a in ARMS)],
    }
    rp = os.path.join(args.out_dir, "%s_dedup_report.json" % args.base)
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("split by task: %d files | deduped full: %s" % (len(by_task), full_path))
    print("report: %s" % rp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
