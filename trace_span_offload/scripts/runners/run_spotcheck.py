#!/usr/bin/env python
"""Small-scale smoke test: run full replays on a few spans to validate the local
equivalence validator and the downstream structured trace (no full pipeline needed).

Usage (server, inside container):
  python scripts/runners/run_spotcheck.py --config OUT_DIR/config.json \
      --spans OUT_DIR/spans.jsonl --trace-id 21abae1_1 --n-spans 3 \
      --out OUT_DIR/spotcheck.jsonl

Output:
  - strict_success and local_outcome (parse/schema/verifier) of each span;
  - local_outcome.equivalence: per-requirement verdict of SLM output vs
    baseline output (tool name/args/execution result/text semantics);
  - downstream_trace summary: meta/prefix sizes, span count, terminal status;
  - full record written to --out (jsonl, same structure as full replays).
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
from span_offload.replay import (  # noqa: E402
    SingleSpanReplayEngine, record_contract_problems,
)
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


def _print_span(record, span: SpanInstance) -> None:
    lo = record.local_outcome
    print("=" * 72)
    print("span %s inv=%d node=%s" % (
        span.span_id, span.invocation_index,
        span.span_descriptor.node_pattern))
    print("  local: parse=%s schema=%s verifier=%s (%s)" % (
        lo.parse_success, lo.schema_valid, lo.local_verifier_passed,
        lo.parse_error or "ok"))
    eq = lo.equivalence or {}
    print("  equivalence passed=%s (contract=%s)" % (
        eq.get("passed"), eq.get("contract")))
    for req in eq.get("requirements", []):
        detail = (str(req.get("details", "")) or "")[:160]
        print("    [%s] %s" % ("PASS" if req.get("passed") else "FAIL",
                               str(req.get("requirement", ""))[:80]))
        if detail:
            print("         detail: %s" % detail)
    dt = record.downstream_trace or {}
    spans = dt.get("spans", [])
    print("  downstream_trace: prefix_msgs=%d prev_tools=%d spans=%d terminal=%s" % (
        len((dt.get("prefix") or {}).get("messages", [])),
        len((dt.get("prefix") or {}).get("previous_tool_results", [])),
        len(spans),
        json.dumps(dt.get("terminal", {}), ensure_ascii=False)[:120]))
    for e in spans[:6]:
        if e.get("span_type") == "llm.call":
            out = e.get("output") or {}
            if "tool_calls" in out:
                print("    step=%d %s/%s calls=%s" % (
                    e.get("step"), e.get("span_type"), e.get("source"),
                    [c.get("name") for c in out["tool_calls"]]))
            else:
                print("    step=%d %s/%s text=%r" % (
                    e.get("step"), e.get("span_type"), e.get("source"),
                    str(out.get("text", ""))[:60]))
        else:
            print("    step=%d %s/%s tool=%s ok=%s" % (
                e.get("step"), e.get("span_type"), e.get("source"),
                e.get("tool"), e.get("ok")))
    if len(spans) > 6:
        print("    ... %d spans total" % len(spans))
    print("  strict_success=%s final_task_success=%s goal=%s collateral=%s" % (
        record.strict_success,
        record.final_outcome.final_task_success,
        record.final_outcome.goal_state_passed,
        record.final_outcome.collateral_damage))
    if record.final_outcome.evaluator_details:
        print("  evaluator_details: %s" % str(
            record.final_outcome.evaluator_details)[:200])
    if record.replay_meta.get("error"):
        print("  meta.error: %s" % str(record.replay_meta["error"])[:200])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--spans", required=True)
    ap.add_argument("--manifest", default="data/appworld/task_manifest.json")
    ap.add_argument("--trace-id", default="", help="trace filename substring")
    ap.add_argument("--n-spans", type=int, default=3)
    ap.add_argument("--fallback-mode", default="",
                    help="override fallback_mode from config (none/recorded_baseline/live_baseline)")
    ap.add_argument("--out", default="out_span_offload/spotcheck.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    model_id = slm.name
    fallback_mode = args.fallback_mode or cfg.replay.fallback_mode

    spans_all = load_spans(args.spans)
    todo = [from_dict(SpanInstance, s) for s in spans_all if s.get("eligible")]
    if args.trace_id:
        todo = [s for s in todo if args.trace_id in s.trace_path]
    todo = todo[: args.n_spans]
    print("selected %d spans, SLM=%s baseline=%s fallback=%s backend=%s" % (
        len(todo), model_id, baseline.name, fallback_mode,
        (AppWorldAgentRuntime.continuation_backend or "auto(by trace agent)")), flush=True)

    manifest = load_manifest(args.manifest)
    adapter = AppWorldTraceAdapter(manifest)
    traces = {}
    for s in todo:
        if s.trace_path not in traces:
            traces[s.trace_path] = adapter.load(s.trace_path)

    runtime = AppWorldAgentRuntime()
    engine = SingleSpanReplayEngine(
        cfg, runtime, baseline_client=baseline,
        slm_clients={model_id: slm}, fallback_mode=fallback_mode)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_ok = 0
    contract_fails = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, span in enumerate(todo, 1):
            record = engine.replay(span, traces[span.trace_path], model_id)
            f.write(json.dumps(to_dict(record), ensure_ascii=False) + "\n")
            f.flush()
            _print_span(record, span)
            n_ok += int(record.strict_success)
            cprobs = record_contract_problems(record)
            if cprobs:
                contract_fails += 1
                print("  [contract FAIL] %s" % "; ".join(cprobs[:3]))

    print()
    print("=== smoke test summary: strict_success %d/%d ===" % (n_ok, len(todo)))
    print("full records written to %s" % args.out)
    if contract_fails:
        print("=== data-consistency self-check FAIL: %d records violate the "
              "reasoning/engine contract (incomplete code sync or output from an old process) ===" % contract_fails)
        return 2
    print("=== data-consistency self-check PASS: all records satisfy the reasoning/engine contract ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
