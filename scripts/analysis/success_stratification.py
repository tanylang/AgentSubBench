#!/usr/bin/env python
"""Metric statistics stratified by final task success/failure (53 spans × 10 rounds, clean baseline).

Splits all 530 rounds into four groups by (injection group × strict_success), computes the means
of the existing metric system (22 downstream variables + 6 control-flow variables) per group,
and reports the within-group "success − failure" difference — revealing the metric signatures
of successful replays vs failed replays.

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/success_stratification.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import analyze_offloadability as AO  # noqa: E402  (round_variables + _events)
from controlflow_analysis import cf_vars, ID_FILES, SLM_FILES, load  # noqa: E402
from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402


def main() -> int:
    slm = defaultdict(list)
    idd = defaultdict(list)
    for r in load(SLM_FILES + ID_FILES):
        k = (r["task_id"], r["invocation_index"])
        (idd if r.get("replay_meta", {}).get("injection") == "identity"
         else slm)[k].append(r)
    keys = sorted(set(slm) & set(idd))
    print("paired span count: %d" % len(keys))

    adapter = AppWorldTraceAdapter()
    base = {}
    for f in glob.glob("check_point_traces/appworld_simplified_batch1/*.json"):
        tr = adapter.load(f)
        tid = tr.run_meta.task_id
        for j, s in enumerate(tr.spans):
            chain = [x.baseline_response.parsed.get("tool_calls", [{}])[0]["name"]
                     for x in tr.spans[j:]
                     if x.baseline_response.parsed.get("tool_calls")]
            all_calls = [{"name": q["name"], "arguments": q.get("arguments")}
                         for x in tr.spans[j:]
                         for q in x.baseline_response.parsed.get("tool_calls", [])]
            base[(tid, s.invocation_index)] = {
                "remaining_steps": len(tr.spans) - j,
                "remaining_tools": [c["name"] for c in all_calls],
                "remaining_calls": all_calls,
                "key_chain": chain,
            }

    VARS = ["tools", "llm", "hall", "ParamMatchRate", "KeyChainCoverage",
            "DirectActionRate", "WriteRatio", "ReasonOnlyRate",
            "DupCallRate", "FailChaining", "FailRetryRate", "MaxFailRun",
            "RepairRate", "ToolSeqJaccard", "StepsRatio",
            "ToolEntropy", "BranchWidth", "RevisitRate",
            "SameToolPersistence", "AbnormalExit", "FirstCompleteNormPos",
            "NewBranchRate", "MissingBranchRate", "BatchDecisionRate",
            "DeadEndRate", "LoopPatternCount"]

    groups = {"SLM成": [], "SLM败": [], "id成": [], "id败": []}
    for k in keys:
        for r in slm[k]:
            v = AO.round_variables(r, base[k])
            c = cf_vars(r)
            groups["SLM成" if r.get("strict_success") else "SLM败"].append(
                {**v, **c, "base_names": base[k]["remaining_tools"]})
        for r in idd[k]:
            v = AO.round_variables(r, base[k])
            c = cf_vars(r)
            groups["id成" if r.get("strict_success") else "id败"].append(
                {**v, **c, "base_names": base[k]["remaining_tools"]})

    for g, rs in groups.items():
        print("%s: %d rounds" % (g, len(rs)))

    print("\n%-22s %10s %10s %10s %10s %10s %10s" % (
        "variable", "SLMsucc", "SLMfail", "idsucc", "idfail", "ΔSLM", "Δid"))
    for v in VARS:
        row = []
        for g in ["SLM成", "SLM败", "id成", "id败"]:
            xs = [r[v] for r in groups[g] if r.get(v) is not None]
            row.append(statistics.mean(xs) if xs else float("nan"))
        d_slm = row[0] - row[1] if row[0] == row[0] and row[1] == row[1] else float("nan")
        d_id = row[2] - row[3] if row[2] == row[2] and row[3] == row[3] else float("nan")
        print("%-22s %10.3f %10.3f %10.3f %10.3f %+10.3f %+10.3f" % (
            v, row[0], row[1], row[2], row[3], d_slm, d_id))

    out = {"groups": {g: len(rs) for g, rs in groups.items()},
           "per_variable": [
               {"var": v,
                "SLM_succ": statistics.mean([r[v] for r in groups["SLM成"] if r.get(v) is not None]),
                "SLM_fail": statistics.mean([r[v] for r in groups["SLM败"] if r.get(v) is not None]),
                "id_succ": statistics.mean([r[v] for r in groups["id成"] if r.get(v) is not None]),
                "id_fail": statistics.mean([r[v] for r in groups["id败"] if r.get(v) is not None])}
               for v in VARS if any(r.get(v) is not None
                                    for g in groups.values() for r in g)]}
    with open("results/appworld_53span/success_stratification.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/appworld_53span/success_stratification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
