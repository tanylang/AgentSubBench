#!/usr/bin/env python
"""Offline recomputation of the local equivalence (eq) metric for langgraph-arm tool spans.

2026-09-22 fix: compare_values gains a skip_keys exemption -- the memory
records' created_at/updated_at in tool execution results are generated at
replay time and necessarily differ from capture-side records, which previously
made tool spans' eq misjudged 100% as False (even the I arm with injected
captured answers failed). Replay-side execution results live in the record's
downstream_trace.spans, capture-side results in the trace's artifacts["span_tool_results"]; both inputs survive, so no LLM rounds need rerunning.

Usage (server or desktop, screening dir synced):
  python scripts/utils/recompute_langgraph_tool_eq.py \
      --screening out_span_offload/langgraph/batch1/langgraph/screening \
      --traces-dir check_point_traces/locomo_langgraph_qwen3_8-27b \
      --manifest data/locomo/task_manifest.json

Idempotent: recomputed records carry the eq["recomputed"] marker and are
skipped on reruns; the original passed is kept in eq["original_passed"].
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from span_offload.langgraph_adapter import LangGraphTraceAdapter  # noqa: E402
from span_offload.local_equivalence import compare_execution_results  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402

RECOMPUTED_BY = "recompute_langgraph_tool_eq.py"


def _tool_exec_reqs(eq: dict):
    if not isinstance(eq, dict):
        return None
    reqs = eq.get("requirements") or []
    hits = [r for r in reqs
            if "execution results match" in str(r.get("requirement", ""))]
    return hits if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", required=True,
                    help="screening dir or *.jsonl glob")
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    args = ap.parse_args()

    if os.path.isdir(args.screening):
        files = sorted(glob.glob(os.path.join(
            args.screening, "offload_profile_langgraph_p*.jsonl")))
    else:
        files = sorted(glob.glob(args.screening))
    if not files:
        print("no matching files: %s" % args.screening)
        return 1

    manifest = load_manifest(args.manifest)
    adapter = LangGraphTraceAdapter(manifest)
    trace_cache = {}
    changed = {"false->true": 0, "false->false": 0, "true->true": 0,
               "skipped": 0, "no_base_exec": 0}
    for path in files:
        lines = [l for l in open(path, encoding="utf-8") if l.strip()]
        out = []
        for line in lines:
            rec = json.loads(line)
            eq = (rec.get("local_outcome") or {}).get("equivalence")
            if _tool_exec_reqs(eq) is None or eq.get("recomputed"):
                changed["skipped"] += 1
                out.append(line)
                continue
            task_id = rec.get("task_id", "")
            sample = task_id.split(":")[1] if task_id.count(":") >= 1 else ""
            inv = rec.get("invocation_index")
            # replay-side execution results (same filter convention as replay.py at runtime)
            slm_exec = [e for e in (rec.get("downstream_trace") or {})
                        .get("spans", [])
                        if e.get("span_type") == "tool.execution"
                        and e.get("step") == 0 and e.get("source") == "slm"]
            # capture-side recorded results (locate trace file by task_id, take by span index)
            key = (sample, task_id)
            if key not in trace_cache:
                trace_file = os.path.join(args.traces_dir, sample + ".json")
                if not os.path.exists(trace_file):
                    trace_cache[key] = None
                else:
                    try:
                        trace_cache[key] = adapter.load_unit(
                            trace_file, task_id)
                    except Exception as exc:
                        print("  skip %s: trace load failed %s" % (
                            task_id, exc), flush=True)
                        trace_cache[key] = None
            trace = trace_cache[key]
            if trace is None:
                changed["no_base_exec"] += 1
                out.append(line)
                continue
            base_exec = (trace.artifacts.get("span_tool_results") or {}) \
                .get("span:%d" % inv, [])
            old_passed = bool(eq.get("passed"))
            exec_out = compare_execution_results(slm_exec, base_exec)
            # replace the old execution-result subchecks (items 3/4 appended
            # at runtime), keep the call-name/args subchecks (verify_local products, unaffected by this bug)
            reqs = [r for r in (eq.get("requirements") or [])
                    if "execution results match" not in str(
                        r.get("requirement", ""))
                    and "executed tool names match" not in str(
                        r.get("requirement", ""))]
            reqs.extend(exec_out["requirements"])
            eq["requirements"] = reqs
            eq["original_passed"] = old_passed
            eq["recomputed"] = True
            eq["recomputed_by"] = RECOMPUTED_BY
            eq["passed"] = bool(
                all(bool(r.get("passed")) for r in reqs))
            new_passed = eq["passed"]
            changed[("%s->%s" % (
                "true" if old_passed else "false",
                "true" if new_passed else "false"))] += 1
            out.append(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(out)
        print("  updated %s" % path, flush=True)
    print("\nrecompute stats:")
    for k in sorted(changed):
        print("  %s: %d" % (k, changed[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
