#!/usr/bin/env python
"""Filter spans of adversarial tasks (cat 5) and produce a standalone spans file for offload experiments.

Adversarial tasks are all marked ineligible because their baseline accuracy is 0, so they do not
enter the regular offload pool; this script picks all spans of the corresponding units from the full
spans file (including ineligible ones) by manifest gold qa category==5, for separate replay under
the RELAX=1 caliber.

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. .venv_appworld/Scripts/python.exe \\
    scripts/analysis/filter_adversarial_spans.py \\
    --spans out_span_offload/_tmp_loco10_spans_merged.jsonl \\
    --manifest data/locomo/task_manifest.json \\
    --out out_span_offload/langgraph/spans_cat5.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from span_offload.trace_io import load_manifest  # noqa: E402


def _expand(patterns: str) -> list:
    out = []
    for p in patterns.split():
        out.extend(sorted(glob.glob(p)) or [p])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True,
                    help="spans jsonl (space-separated multiple globs)")
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    ap.add_argument("--category", type=int, default=5,
                    help="category to filter (default 5 = adversarial)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    # set of cat N question ids per sample
    cat_units = set()
    cat_count = 0
    for mt in manifest.tasks:
        qa = list(mt.evaluator.get("qa") or [])
        for qi, q in enumerate(qa):
            if q.get("category") == args.category:
                cat_units.add("locomo:%s:q%03d" % (mt.task_id, qi))
                cat_count += 1
    print("cat%d total questions: %d" % (args.category, cat_count))

    files = _expand(args.spans)
    kept = []
    total = 0
    unit_hits = set()
    for f in files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            total += 1
            if s.get("task_id") in cat_units:
                kept.append(s)
                unit_hits.add(s["task_id"])
    print("total spans: %d | kept by filter: %d (involving %d questions)" % (
        total, len(kept), len(unit_hits)))
    if kept:
        elig = sum(1 for s in kept if s.get("eligible"))
        print("of which eligible: %d | ineligible (baseline failed, needs RELAX=1): %d"
              % (elig, len(kept) - elig))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as o:
        for s in kept:
            o.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("Details: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
