#!/usr/bin/env python
"""Full-metric analysis for the simplified confirmation set (18 spans × three arms × 15 rounds).

Computes the full metric set for each offload replay trace (record):
  - local layer: parse_success / schema_valid / local_verifier_passed /
    equivalence.passed (per requirement)
  - downstream layer: 22 variables (analyze_offloadability.round_variables) +
    6 control-flow variables (controlflow_analysis.cf_vars)
  - final layer: strict_success / final_task_success / collateral_damage
  - cost layer: itemized token / cost / elapsed_ms

Outputs:
  out_span_offload/simplified/confirm/full_metrics_per_record.jsonl
      one line per record = full metrics of that offload trace
  out_span_offload/simplified/confirm/full_metrics_per_span.json
      metric mean/median per span × arm (aggregated over the offload-point span)

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/confirm18_full_metrics.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import analyze_offloadability as AO  # noqa: E402
from controlflow_analysis import cf_vars  # noqa: E402
from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402

CONFIRM_DIR = os.path.join(ROOT, "out_span_offload", "simplified", "confirm")
TRACE_DIR = "check_point_traces/appworld_simplified_qwen3_8-27b_batch1"


def build_base(adapter: AppWorldTraceAdapter) -> dict:
    """Baseline remaining info per span (the base argument of round_variables)."""
    base = {}
    for f in glob.glob(os.path.join(TRACE_DIR, "*.json")):
        try:
            tr = adapter.load(f)
        except Exception:
            continue
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
    return base


def record_metrics(r: dict, base: dict) -> dict:
    """Full metrics of one offload trace."""
    m = r.get("replay_meta") or {}
    lo = r.get("local_outcome") or {}
    eq = lo.get("equivalence") or {}
    fo = r.get("final_outcome") or {}
    res = r.get("resource") or {}
    out = {
        "task_id": r.get("task_id"),
        "span_id": r.get("span_id"),
        "invocation_index": r.get("invocation_index"),
        "arm": m.get("arm"),
        "round": m.get("profile_round"),
        "engine_version": m.get("engine_version"),
        "node_pattern": (r.get("span_descriptor") or {}).get("node_pattern"),
        # local layer
        "parse_success": bool(lo.get("parse_success")),
        "schema_valid": bool(lo.get("schema_valid")),
        "local_verifier_passed": bool(lo.get("local_verifier_passed")),
        "eq_passed": bool(eq.get("passed")),
        "eq_contract": eq.get("contract"),
        # final layer
        "strict_success": bool(r.get("strict_success")),
        "final_task_success": bool(fo.get("final_task_success")),
        "collateral_damage": bool(fo.get("collateral_damage")),
        # cost layer
        "elapsed_ms": m.get("elapsed_ms"),
        "target_tokens": (res.get("target_prompt_tokens", 0) or 0)
                         + (res.get("target_completion_tokens", 0) or 0),
        "downstream_tokens": (res.get("downstream_prompt_tokens", 0) or 0)
                             + (res.get("downstream_completion_tokens", 0) or 0),
    }
    # downstream 22 variables + control-flow 6 variables
    key = (r.get("task_id"), r.get("invocation_index"))
    if key in base:
        try:
            out.update(AO.round_variables(r, base[key]))
            out.update(cf_vars(r))
        except Exception as e:
            out["_metric_error"] = "%s: %s" % (type(e).__name__, e)
    else:
        out["_metric_error"] = "base missing"
    return out


def main() -> int:
    adapter = AppWorldTraceAdapter(
        load_manifest(os.path.join(ROOT, "data", "appworld",
                                   "task_manifest.json")))
    base = build_base(adapter)
    print("spans covered by base:", len(base))

    records = []
    for f in sorted(glob.glob(os.path.join(CONFIRM_DIR, "offload_profile_*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                records.append(json.loads(l))
            except ValueError:
                continue
    print("record count:", len(records))

    per_record = []
    for r in records:
        per_record.append(record_metrics(r, base))
    out_rec = os.path.join(CONFIRM_DIR, "full_metrics_per_record.jsonl")
    with open(out_rec, "w", encoding="utf-8") as f:
        for row in per_record:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("per-record full metrics written:", out_rec)

    # aggregate by span × arm (means; numeric metrics also get medians)
    agg = {}
    for row in per_record:
        sid = row["span_id"]
        arm = row["arm"]
        key = (sid, arm)
        agg.setdefault(key, []).append(row)
    per_span = []
    numeric_keys = set()
    for rows in agg.values():
        numeric_keys |= {k for k, v in rows[0].items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for (sid, arm), rows in sorted(agg.items()):
        entry = {"span_id": sid, "arm": arm, "n": len(rows),
                 "task_id": rows[0]["task_id"],
                 "invocation_index": rows[0]["invocation_index"],
                 "node_pattern": rows[0]["node_pattern"]}
        for k in sorted(numeric_keys):
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))
                    and not isinstance(r.get(k), bool)]
            if not vals:
                continue
            entry[k + "_mean"] = statistics.mean(vals)
            if len(vals) >= 5:
                entry[k + "_median"] = statistics.median(vals)
        # boolean metrics take the rate
        for k in ("parse_success", "schema_valid", "local_verifier_passed",
                  "eq_passed", "strict_success", "final_task_success",
                  "collateral_damage"):
            entry[k + "_rate"] = sum(1 for r in rows if r.get(k)) / len(rows)
        per_span.append(entry)
    out_span = os.path.join(CONFIRM_DIR, "full_metrics_per_span.json")
    with open(out_span, "w", encoding="utf-8") as f:
        json.dump(per_span, f, ensure_ascii=False, indent=1)
    print("per-span aggregation written:", out_span, "(%d span×arm groups)" % len(per_span))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
