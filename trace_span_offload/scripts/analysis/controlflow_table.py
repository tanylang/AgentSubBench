#!/usr/bin/env python
"""Control-flow metrics summary table generator (single entry point for CONTROL_FLOW_METRICS.md).

Canonical implementation: span_offload/controlflow_metrics.py
Data: 53 spans × 10 rounds (SLM×5 + identity×5), all clean baselines
Output: results/appworld_53span/controlflow_table.json + console markdown summary table

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/controlflow_table.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.controlflow_metrics import (  # noqa: E402
    CORE_TABLE, METRICS, build_base_map, paired_span_means, paired_table,
    render_markdown,
)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir",
                    default="check_point_traces/appworld_simplified_batch1")
    ap.add_argument("--layers", default="",
                    help="only output the specified layers (comma-separated, e.g. L1,L2); empty = all core tables")
    ap.add_argument("--out", default="results/appworld_53span/controlflow_table.json")
    args = ap.parse_args()

    records = load(SLM_FILES + ID_FILES)
    base_map = build_base_map(args.traces_dir)
    keys, span_means = paired_span_means(records, base_map)
    print("paired span count: %d" % len(keys))

    variables = CORE_TABLE
    if args.layers:
        want = set(args.layers.split(","))
        variables = [v for v in CORE_TABLE if METRICS[v][0] in want]

    rows = paired_table(keys, span_means, variables=variables)
    print()
    for layer in sorted({METRICS[v][0] for v in variables},
                        key=lambda x: (len(x), x)):
        lr = [r for r in rows if r["layer"] == layer]
        if lr:
            print("## %s" % layer)
            print(render_markdown(lr))
            print()

    out = {"n_spans": len(keys), "rows": rows}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("Details: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
