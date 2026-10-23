#!/usr/bin/env python
"""LangGraph (LoCoMo + prebuilt ReAct + LangMem) C0 frozen deterministic
replay batch runner (Phase A gate, zero LM calls).

Frozen replays each span: inject the recorded output for the target span,
inject trace-recorded values for all downstream LM outputs, only really
executing the langmem memory tools (local store, via the langgraph.func entrypoint), no model calls anywhere; verdict reuses run_c0_replay._span_verdict.

Differences from the locomo version:
  - adapter = LangGraphTraceAdapter (unitized: one unit trace per QA,
    task_id = locomo:<sample>:q<qi>), loaded by (trace_path, task_id)
  - runtime = LangGraphRuntime (needs langgraph/langmem deps)
  - --relax-baseline-pass: all 6 samples in this batch fail pass -> all depend on the exemption

Usage (server, inside the langgraph venv):
  python scripts/runners/run_langgraph_c0.py \
      --config OUT_DIR/config.json --spans OUT_DIR/spans_langgraph.jsonl \
      --manifest data/locomo/task_manifest.json --rounds 1 --relax-baseline-pass \
      --out OUT_DIR/c0_replay_langgraph.jsonl \
      --summary OUT_DIR/c0_fidelity_summary_langgraph.json

Local (Windows, zero LM calls):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/runners/run_langgraph_c0.py \
      --spans out_span_offload/langgraph/spans_langgraph.jsonl \
      --manifest data/locomo/task_manifest.json \
      --trace-id conv-30 --n-spans 3 --relax-baseline-pass
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_c0_replay import _span_verdict  # noqa: E402 (reuse verdict logic)
from run_offload_profile import make_client  # noqa: E402

from span_offload.config import load_config  # noqa: E402
from span_offload.harness.langgraph_runtime import LangGraphRuntime  # noqa: E402
from span_offload.langgraph_adapter import LangGraphTraceAdapter  # noqa: E402
from span_offload.replay import SingleSpanReplayEngine  # noqa: E402
from span_offload.schemas import SpanInstance, from_dict, to_dict  # noqa: E402
from span_offload.span_extraction import load_spans  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402

RELAXABLE = {"baseline_run_failed_final_evaluator", "baseline_goal_state_failed"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--spans", required=True)
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    ap.add_argument("--trace-id", default="", help="trace filename substring")
    ap.add_argument("--n-spans", type=int, default=0, help="0 = all")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--relax-baseline-pass", action="store_true",
                    help="relax the baseline evaluation threshold (all 6 samples in this batch fail, must enable)")
    ap.add_argument("--out",
                    default="out_span_offload/langgraph/c0_replay.jsonl")
    ap.add_argument("--summary",
                    default="out_span_offload/langgraph/c0_fidelity_summary.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    model_id = slm.name  # C0 makes no model calls; only used for registration

    spans_all = load_spans(args.spans)
    # filter by the raw dicts before deserializing (unitized batches can reach
    # millions of lines; a full from_dict costs tens of minutes + several GB
    # of memory per run)
    selected = [s for s in spans_all
                if not args.trace_id or args.trace_id in s.get("trace_path", "")]
    if args.n_spans > 0:
        selected = selected[: args.n_spans]
    todo = [from_dict(SpanInstance, s) for s in selected if s.get("eligible")]
    if args.relax_baseline_pass:
        relaxed = [s for s in selected
                   if not s.get("eligible")
                   and set(s.get("ineligibility_reasons") or [])
                   .issubset(RELAXABLE)]
        todo = todo + [from_dict(SpanInstance, s) for s in relaxed]
        print("relax-baseline-pass: +%d relaxed spans (total %d)" % (
            len(relaxed), len(todo)), flush=True)
    print("C0 span count: %d x %d rounds (no LM calls anywhere)" % (len(todo), args.rounds),
          flush=True)

    manifest = load_manifest(args.manifest)
    adapter = LangGraphTraceAdapter(manifest)
    traces = {}
    for s in todo:
        key = (s.trace_path, s.task_id)
        if key not in traces:
            traces[key] = adapter.load_unit(s.trace_path, s.task_id)

    runtime = LangGraphRuntime(manifest)
    engine = SingleSpanReplayEngine(
        cfg, runtime, baseline_client=baseline,
        slm_clients={model_id: slm}, fallback_mode=cfg.replay.fallback_mode)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    per_span = []
    with open(args.out, "w", encoding="utf-8") as f:
        for si, span in enumerate(todo):
            trace = traces[(span.trace_path, span.task_id)]
            native_final = {
                "final_evaluator_passed": bool(trace.run_meta.final_evaluator_passed),
                "goal_state_passed": bool(trace.run_meta.goal_state_passed),
                "collateral_damage": bool(trace.run_meta.collateral_damage),
            }
            rounds_meta = []
            last = None
            for rnd in range(args.rounds):
                record = engine.replay(span, trace, model_id, arm="C0")
                record.replay_meta["c0_round"] = rnd
                record.replay_meta["native_final"] = native_final
                rounds_meta.append(record.replay_meta)
                last = record
                f.write(json.dumps(to_dict(record), ensure_ascii=False) + "\n")
                f.flush()
            verdict = _span_verdict(rounds_meta, last)
            entry = {
                "task_id": span.task_id,
                "span_id": span.span_id,
                "invocation_index": span.invocation_index,
                "node_pattern": span.span_descriptor.node_pattern,
                "control_role": span.span_descriptor.control_role,
                "trace_path": span.trace_path,
                "rounds": args.rounds,
                "native_final": native_final,
                "replayed_final": {
                    "goal_state_passed": bool(last.final_outcome.goal_state_passed),
                    "final_task_success": bool(last.final_outcome.final_task_success),
                    "collateral_damage": bool(last.final_outcome.collateral_damage),
                },
            }
            entry.update(verdict)
            per_span.append(entry)
            print("span %d/%d [%s inv=%d] %s: %s%s" % (
                si + 1, len(todo), span.span_descriptor.node_pattern,
                span.invocation_index, verdict["verdict"], entry["tags"],
                " final_match=%s" % verdict["final_match"]
                if verdict["final_match"] is not None else ""), flush=True)

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(per_span, f, ensure_ascii=False, indent=1)

    n = len(per_span)
    passed = [p for p in per_span if p["verdict"] == "c0_pass"]
    print()
    print("=== C0 fidelity summary (%d spans x %d rounds) ===" % (n, args.rounds))
    print("c0_pass:           %d/%d (%.0f%%)" % (
        len(passed), n, 100 * len(passed) / max(1, n)))
    from collections import Counter
    tag_counts = Counter(t for p in per_span for t in p["tags"])
    for tag, c in tag_counts.most_common():
        print("  %-28s %d" % (tag, c))
    soft_counts = Counter(t for p in per_span for t in p.get("soft_tags", []))
    for tag, c in soft_counts.most_common():
        print("  [soft] %-24s %d" % (tag, c))
    if passed:
        print("S_valid(%d):" % len(passed))
        for p in passed:
            print("  %s [%s inv=%d]" % (p["span_id"], p["node_pattern"],
                                        p["invocation_index"]))
    print("details: %s" % args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
