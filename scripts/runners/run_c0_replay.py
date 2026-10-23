#!/usr/bin/env python
"""C0 frozen deterministic replay batch runner (research plan v0.5 §4, Phase A gate).

Frozen replays each eligible span: inject the recorded output y_j^0 for the
target span, inject trace-recorded values for all downstream LM outputs, only
re-execute tools/state transitions, no LM calls anywhere (zero model cost). Outputs a replay fidelity summary and the S_valid verdict:

  - pre-state hash (raw-to-raw, restore chain vs capture-side record)
  - prompt/schema hash (recorded consistency)
  - per-span tool execution results vs recorded observations (c0_fidelity)
  - final evaluator result vs native run_meta
  - pass threshold (plan §4.4) decides c0_pass / failure tags

Usage (server, inside container):
  python scripts/runners/run_c0_replay.py \
      --config OUT_DIR/config.json --spans OUT_DIR/spans.jsonl \
      --manifest data/appworld/task_manifest.json --rounds 1 \
      --out OUT_DIR/c0_replay.jsonl \
      --summary OUT_DIR/c0_fidelity_summary.json

Local (Windows):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/runners/run_c0_replay.py \
      --spans out_span_offload/spans.jsonl --n-spans 3 --trace-id 23cf851_2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402
from span_offload.config import load_config  # noqa: E402
from span_offload.harness.appworld_runtime import AppWorldAgentRuntime  # noqa: E402
from span_offload.llm_client import ChatModelClient  # noqa: E402
from span_offload.replay import SingleSpanReplayEngine  # noqa: E402
from span_offload.schemas import SpanInstance, from_dict, to_dict  # noqa: E402
from span_offload.span_extraction import load_spans  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402


def make_client(section):
    c = section
    return ChatModelClient(
        base_url=c.base_url, api_key=c.resolved_api_key,
        model=c.candidate_models[0], temperature=c.temperature,
        max_output_tokens=c.max_output_tokens,
        price_per_1k_input=c.price_per_1k_input,
        price_per_1k_output=c.price_per_1k_output,
        timeout_s=c.timeout_s, max_retries=c.max_retries,
        extra_body=c.extra_body,
    )


def _span_verdict(rounds_meta: list, record) -> dict:
    """Per-span C0 verdict: tags items one by one against the plan §4.4 threshold.

    rounds_meta: replay_meta of each round; record: the last full round record.
    tags = hard failure tags (any present means excluded from S_valid);
    soft_tags = observations (ephemeral value diffs etc., do not block the gate)."""
    tags = []
    soft_tags = []
    restore_ok = all(bool(m.get("restore_ok")) for m in rounds_meta)
    if not restore_ok:
        tags.append("replay_invalid")
    # 1) pre-state hash: raw-to-raw comparison (same space); tool_replay mode
    #    cannot compare in the same space as the recorded hash -> state_unverified, excluded from S_valid
    any_state_err = any("snapshot hash mismatch" in str(m.get("error") or "")
                        for m in rounds_meta)
    if any_state_err:
        tags.append("state_mismatch")
    modes = set()
    for m in rounds_meta:
        sh = str(m.get("snapshot_hash_restored") or "")
        modes.add(sh.split(":", 1)[0] if ":" in sh else "unknown")
    if "tool_replay" in modes and "env_delta" not in modes:
        tags.append("state_unverified(tool_replay)")
    # 2) prompt/schema hash: missing means we cannot claim prompt consistency
    if not all(m.get("prompt_hash") for m in rounds_meta):
        tags.append("prompt_mismatch")
    # 3) per-span tool execution comparison: name/count/ok flag = hard;
    #    result values = soft (login token/timestamp etc. ephemeral fields are inherently non-reproducible)
    fidelity = []
    for m in rounds_meta:
        fidelity.extend(m.get("c0_fidelity") or [])
    if fidelity:
        if any((f.get("tool_match") is False) for f in fidelity):
            tags.append("tool_nondeterministic")
        elif any((f.get("value_match") is False) for f in fidelity):
            soft_tags.append("ephemeral_value_diff")
    # frozen walk exhausted but still incomplete (recorded sequence finished but world incomplete): terminal status check
    dt = (record.downstream_trace or {}).get("terminal") or {}
    if "exhausted without completion" in str(dt.get("status", "")):
        tags.append("incomplete_frozen_walk")
    # 4) final evaluator vs native run_meta (excluding ephemeral business-state comparison)
    final = record.final_outcome
    native = record.replay_meta.get("native_final") or {}
    if native:
        replayed_strict = bool(final.final_task_success) and not bool(final.collateral_damage)
        native_strict = bool(native.get("final_evaluator_passed")) \
            and not bool(native.get("collateral_damage"))
        final_match = (bool(final.goal_state_passed) == bool(native.get("goal_state_passed"))
                       and bool(final.final_task_success) == bool(native.get("final_evaluator_passed"))
                       and bool(final.collateral_damage) == bool(native.get("collateral_damage")))
        if not final_match:
            tags.append("evaluator_mismatch")
    else:
        final_match = None
        replayed_strict = native_strict = None
    # 5) engine-level errors (exception/stop/injection failure)
    if any(m.get("error") for m in rounds_meta):
        tags.append("replay_error")
    return {
        "verdict": "c0_pass" if not tags else "c0_fail",
        "tags": tags,
        "soft_tags": soft_tags,
        "final_match": final_match,
        "replayed_strict": replayed_strict,
        "native_strict": native_strict,
        "fidelity_steps": len(fidelity),
        "fidelity": fidelity,
        "restore_modes": sorted(modes),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--spans", required=True)
    ap.add_argument("--manifest", default="data/appworld/task_manifest.json")
    ap.add_argument("--trace-id", default="", help="trace filename substring")
    ap.add_argument("--n-spans", type=int, default=0, help="0 = all")
    ap.add_argument("--rounds", type=int, default=1,
                    help="C0 rerun count per span (extend when nondeterminism appears)"
                         "; pass verdict = all rounds pass")
    ap.add_argument("--out", default="out_span_offload/c0_replay.jsonl")
    ap.add_argument("--summary", default="out_span_offload/c0_fidelity_summary.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    # C0 makes no model calls; model_id is only used to register the engine's client
    model_id = slm.name

    spans_all = load_spans(args.spans)
    todo = [from_dict(SpanInstance, s) for s in spans_all if s.get("eligible")]
    if args.trace_id:
        todo = [s for s in todo if args.trace_id in s.trace_path]
    if args.n_spans > 0:
        todo = todo[: args.n_spans]
    print("C0 span count: %d x %d rounds (no LM calls anywhere)" % (len(todo), args.rounds),
          flush=True)

    manifest = load_manifest(args.manifest)
    adapter = AppWorldTraceAdapter(manifest)
    traces = {}
    for s in todo:
        if s.trace_path not in traces:
            traces[s.trace_path] = adapter.load(s.trace_path)

    runtime = AppWorldAgentRuntime()
    engine = SingleSpanReplayEngine(
        cfg, runtime, baseline_client=baseline,
        slm_clients={model_id: slm}, fallback_mode=cfg.replay.fallback_mode)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    per_span = []
    with open(args.out, "w", encoding="utf-8") as f:
        for si, span in enumerate(todo):
            trace = traces[span.trace_path]
            # final result of the native run (τ0 reference, from trace capture records)
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

    # ---- summary ----
    n = len(per_span)
    passed = [p for p in per_span if p["verdict"] == "c0_pass"]
    print()
    print("=== C0 fidelity summary (%d spans x %d rounds) ===" % (n, args.rounds))
    print("c0_pass:           %d/%d (%.0f%%)" % (len(passed), n, 100 * len(passed) / max(1, n)))
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
