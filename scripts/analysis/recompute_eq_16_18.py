#!/usr/bin/env python
"""Offline equivalence (eq) recomputation — engine 2026-09-16.18 caliber.

eq is a post-hoc judgment (does not participate in execution); all of its input material is in the record:
  - output layer: slm_output.slm_tool_calls / baseline_tool_calls
  - execution layer: tool executions with step=0/source=slm in downstream_trace (live results)
    vs the trace artifact span_tool_results (recorded results)
The eq computed during replay (old caliber) is overwritten by the recalculation under the 16.18
caliber (JWT exp exemption); all other fields are untouched. Zero model cost.

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/recompute_eq_16_18.py \
      --dir out_span_offload/tau2/screening [--trace-dir check_point_traces/appworld_tau2_qwen3_8-27b_batch1]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402
from span_offload.local_equivalence import (  # noqa: E402
    compare_execution_results, verify_local_equivalence,
)
from span_offload.trace_io import load_manifest  # noqa: E402


def contract_of(r: dict) -> str:
    eq = (r.get("local_outcome") or {}).get("equivalence") or {}
    if eq.get("contract"):
        return eq["contract"]
    np_ = (r.get("span_descriptor") or {}).get("node_pattern")
    if np_ == "tool_selection":
        return "freeform_text"
    if np_ == "final_synthesis":
        return "answer_json"
    return "tool_call_list"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--trace-dir",
                    default="check_point_traces/appworld_tau2_qwen3_8-27b_batch1")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite records in place (default: only counts, does not write)")
    args = ap.parse_args()

    adapter = AppWorldTraceAdapter(
        load_manifest(os.path.join(ROOT, "data", "appworld",
                                   "task_manifest.json")))
    trace_cache: dict = {}
    files = sorted(glob.glob(os.path.join(args.dir, "offload_profile_*.jsonl")))
    changed = 0
    total = 0
    per_file: dict = {}
    for f in files:
        lines = []
        for l in open(f, encoding="utf-8"):
            l = l.rstrip("\n")
            if not l.strip():
                lines.append(l)
                continue
            r = json.loads(l)
            total += 1
            tp = (r.get("replay_meta") or {}).get("trace_path", "")
            if tp and tp not in trace_cache:
                try:
                    trace_cache[tp] = adapter.load(
                        tp.replace("\\", "/"))
                except Exception:
                    trace_cache[tp] = None
            tr = trace_cache.get(tp)
            contract = contract_of(r)
            slm_o = r.get("slm_output") or {}
            if contract == "tool_call_list":
                slm_parsed = {"tool_calls": slm_o.get("slm_tool_calls") or []}
                base_parsed = {"tool_calls": slm_o.get("baseline_tool_calls") or []}
            else:
                slm_parsed = {"text": slm_o.get("slm_raw_text") or ""}
                base_parsed = {}
            try:
                new_eq = verify_local_equivalence(
                    contract, slm_parsed, base_parsed)
            except Exception:
                new_eq = {"contract": contract, "requirements": [],
                          "passed": False}
            if contract == "tool_call_list" and tr is not None:
                dt = (r.get("downstream_trace") or {}).get("spans") or []
                slm_exec = [e for e in dt
                            if e.get("span_type") == "tool.execution"
                            and e.get("step") == 0
                            and e.get("source") == "slm"]
                base_exec = (tr.artifacts.get("span_tool_results") or {}).get(
                    "span:%d" % r.get("invocation_index"), [])
                try:
                    exec_out = compare_execution_results(slm_exec, base_exec)
                    new_eq["requirements"] = list(
                        new_eq.get("requirements") or [])
                    new_eq["requirements"].extend(
                        exec_out["requirements"])
                    new_eq["passed"] = bool(new_eq.get("passed", True)) \
                        and bool(exec_out["passed"])
                except Exception:
                    pass
            old = (r.get("local_outcome") or {}).get("equivalence") or {}
            if json.dumps(old, sort_keys=True) != json.dumps(new_eq, sort_keys=True):
                changed += 1
                per_file[f] = per_file.get(f, 0) + 1
            if args.in_place:
                r["local_outcome"]["equivalence"] = new_eq
                lines.append(json.dumps(r, ensure_ascii=False))
            else:
                lines.append(l)
        if args.in_place:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
    print("total records:", total, "| eq changes:", changed)
    if args.in_place:
        for f, n in sorted(per_file.items()):
            print("  %s: %d records updated" % (os.path.basename(f), n))
        print("updated in place (engine 2026-09-16.18 caliber)")
    else:
        print("statistics only (--in-place required to write back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
