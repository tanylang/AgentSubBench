#!/usr/bin/env python
"""Single-span SLM request diagnostic: prints the request structure sent to
the SLM + the SLM's raw response, to locate the root cause of local-layer
parse failures. Makes a single SLM call only, no continuation replay.

Usage (server, inside container):
  python scripts/diag/diag_slm_span.py --config OUT_DIR/config.json \
      --spans OUT_DIR/spans.jsonl --trace-id <filename substring> --inv <span index>
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
from span_offload.schemas import SpanInstance, from_dict  # noqa: E402
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
    ap.add_argument("--inv", type=int, default=1, help="span invocation_index")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)

    spans_all = load_spans(args.spans)
    todo = [from_dict(SpanInstance, s) for s in spans_all]
    if args.trace_id:
        todo = [s for s in todo if args.trace_id in s.trace_path]
    todo = [s for s in todo if s.invocation_index == args.inv]
    if not todo:
        print("no matching span found"); return 1
    span = todo[0]
    print("=== span:", span.span_id, "|", span.span_descriptor.node_pattern,
          "| contract:", span.replay_input.output_contract_runtime.contract_type)

    adapter = AppWorldTraceAdapter(load_manifest(args.manifest))
    trace = adapter.load(span.trace_path)
    span_rec = None
    for rec in trace.spans:
        if rec.invocation_index == span.invocation_index:
            span_rec = rec
            break
    assert span_rec is not None

    runtime = AppWorldAgentRuntime()
    req = runtime.resolve_request(trace, span_rec)
    print("=== request message count:", len(req.messages))
    for i, m in enumerate(req.messages):
        keys = list(m.keys())
        head = str(m.get("content", ""))[:100].replace("\n", " | ")
        tc = m.get("tool_calls")
        tc_info = ("tool_calls=%d" % len(tc)) if tc else ""
        print("  [%2d] %-9s %-20s %s" % (i, m.get("role"), tc_info, head))
    print("=== tools params:", ("%d tools" % len(req.tools)) if req.tools else "None")
    if req.tools:
        t0 = req.tools[0].get("function", req.tools[0])
        print("  tools[0] keys:", sorted(t0.keys()))
        print("  tools[0]:", json.dumps(t0, ensure_ascii=False)[:200])

    print("=== SLM call ...")
    resp = slm.complete(
        messages=req.messages, tools=req.tools,
        temperature=req.temperature, max_tokens=req.max_output_tokens)
    print("=== SLM response:")
    print("  error:", resp.error or "-")
    print("  tool_calls:", len(resp.tool_calls))
    for tc in resp.tool_calls[:3]:
        print("    -", tc["name"], str(tc["arguments"])[:100])
    print("  first 600 chars of raw_text:")
    print("    " + (resp.raw_text or "<empty>")[:600].replace("\n", " | "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
