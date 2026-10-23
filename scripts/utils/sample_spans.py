#!/usr/bin/env python
"""Stratified deterministic span sampling CLI (command-line entry to span_offload/span_sampling.py).

Samples a fixed-size experiment subset from any batch's (agent x capture
baseline) spans file, zero randomness, same input = same output. To switch
agent batches just switch the --spans file:
  - simplified (deepseek capture): out_span_offload/spans.jsonl
  - simplified (27B capture):      out_span_offload/spans_simplified_qwen27b.jsonl
  - autogen (27B capture):         out_span_offload/spans_autogen_qwen27b.jsonl

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/utils/sample_spans.py \
      --spans out_span_offload/spans_simplified_qwen27b.jsonl \
      --out out_span_offload/spans_simplified_qwen27b_50.jsonl --n 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.span_sampling import stratified_sample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True, help="input spans jsonl")
    ap.add_argument("--out", required=True, help="output sampled spans jsonl")
    ap.add_argument("--n", type=int, default=50, help="target span count")
    args = ap.parse_args()

    spans = [json.loads(l) for l in open(args.spans, encoding="utf-8")
             if l.strip()]
    elig = [s for s in spans if s.get("eligible")]
    picked = stratified_sample(spans, n=args.n)
    print("input: %d spans (eligible %d) -> sampled %d" % (len(spans), len(elig), len(picked)))
    print("node_pattern:", dict(Counter(
        (s.get("span_descriptor") or {}).get("node_pattern") for s in picked)))
    print("task coverage: %d/%d" % (len(set(s["task_id"] for s in picked)),
                                    len(set(s["task_id"] for s in elig))))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in picked:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
