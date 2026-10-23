#!/usr/bin/env python
"""LocoMo C0 frozen deterministic replay batch runner (Phase A gate, zero LM calls).

Frozen replays each span: inject the recorded output for the target span,
inject trace-recorded values for all downstream LM outputs, only re-execute
the retrieval tools (read-only), no model calls anywhere. Produces a fidelity summary and S_valid verdict (verdict reuses run_c0_replay._span_verdict).

Differences from the AppWorld version:
  - adapter = LocomoTraceAdapter, runtime = LocomoRuntime
  - --relax-baseline-pass (on by default): smoke-test conv-26 baseline fails (f1=0.2843),
    relaxed-pool spans also enter the C0 gate (Phase A verifies restore fidelity independent of baseline pass)

Usage (server, inside container):
  python scripts/runners/run_locomo_c0.py \
      --config OUT_DIR/config.json --spans OUT_DIR/spans_locomo.jsonl \
      --manifest data/locomo/task_manifest.json --rounds 1 --relax-baseline-pass \
      --out OUT_DIR/c0_replay_locomo.jsonl \
      --summary OUT_DIR/c0_fidelity_summary_locomo.json

Local (Windows, zero LM calls):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/runners/run_locomo_c0.py \
      --spans out_span_offload_locomo/spans_locomo.jsonl \
      --manifest data/locomo/task_manifest.json --relax-baseline-pass
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_c0_replay import _span_verdict  # noqa: E402 (reuse verdict logic)
from run_offload_profile import make_client  # noqa: E402

from span_offload.config import load_config  # noqa: E402
from span_offload.harness.locomo_runtime import LocomoRuntime  # noqa: E402
from span_offload.locomo_adapter import LocomoTraceAdapter  # noqa: E402
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
                    help="relax the baseline evaluation threshold (on by default for the smoke test)")
    ap.add_argument("--out", default="out_span_offload_locomo/c0_replay.jsonl")
    ap.add_argument("--summary",
                    default="out_span_offload_locomo/c0_fidelity_summary.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    # C0 makes no model calls; model_id is only used to register the engine's client
    model_id = slm.name

    spans_all = load_spans(args.spans)
    # filter by the raw dicts before deserializing (full from_dict is very heavy for large batches)
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
    adapter = LocomoTraceAdapter(manifest)
    traces = {}
    for s in todo:
        if s.trace_path not in traces:
            traces[s.trace_path] = adapter.load(s.trace_path)

    runtime = LocomoRuntime(manifest)
    engine = SingleSpanReplayEngine(
        cfg, runtime, baseline_client=baseline,
        slm_clients={model_id: slm}, fallback_mode=cfg.replay.fallback_mode)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    per_span = []
    with open(args.out, "w", encoding="utf-8") as f:
        for si, span in enumerate(todo):
            trace = traces[span.trace_path]
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
