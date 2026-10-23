#!/usr/bin/env python
"""Continuation variance experiment: for each selected span, replay N times
with identity injection (recorded_baseline), quantifying the variance of
"same input replay" -- to tell whether the ceiling gap is dominated by model nondeterminism or context perturbation.

Usage (server, inside container):
  python scripts/runners/run_variance.py --config OUT_DIR/config.json \
      --spans OUT_DIR/spans.jsonl --trace-id 21abae1_1 \
      --n-spans 5 --rounds 3 --out OUT_DIR/variance.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--spans", required=True)
    ap.add_argument("--manifest", default="data/appworld/task_manifest.json")
    ap.add_argument("--trace-id", default="", help="trace filename substring")
    ap.add_argument("--n-spans", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="out_span_offload/variance.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    model_id = slm.name

    spans_all = load_spans(args.spans)
    todo = [from_dict(SpanInstance, s) for s in spans_all if s.get("eligible")]
    if args.trace_id:
        todo = [s for s in todo if args.trace_id in s.trace_path]
    todo = todo[: args.n_spans]
    print("selected %d spans x %d replay rounds (identity)" % (len(todo), args.rounds),
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
        slm_clients={model_id: slm}, fallback_mode="recorded_baseline")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    summary = []
    with open(args.out, "w", encoding="utf-8") as f:
        for si, span in enumerate(todo):
            seq = []
            for rnd in range(args.rounds):
                record = engine.replay(span, traces[span.trace_path], model_id)
                record.replay_meta["variance_round"] = rnd
                record.replay_meta["variance_span"] = span.span_id
                f.write(json.dumps(to_dict(record), ensure_ascii=False) + "\n")
                f.flush()
                seq.append(bool(record.strict_success))
                print("span %d [%s inv=%d] round %d/%d: strict=%s" % (
                    si + 1, span.span_descriptor.node_pattern,
                    span.invocation_index, rnd + 1, args.rounds, seq[-1]),
                    flush=True)
            summary.append((span, seq))
            print("  -> span inv=%d: %s (%d/%d)" % (
                span.invocation_index, seq, sum(seq), args.rounds), flush=True)

    print()
    print("=== variance summary ===")
    all_flat = [v for _, seq in summary for v in seq]
    for span, seq in summary:
        stable = len(set(seq)) == 1
        print("span inv=%2d [%-28s] %s (%d/%d) %s" % (
            span.invocation_index, span.span_descriptor.node_pattern, seq,
            sum(seq), args.rounds, "stable" if stable else "unstable"))
    print("overall success rate: %d/%d (%.0f%%)" % (sum(all_flat), len(all_flat),
          100 * sum(all_flat) / max(1, len(all_flat))))
    print("stable span ratio: %d/%d" % (
        sum(1 for _, seq in summary if len(set(seq)) == 1), len(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
