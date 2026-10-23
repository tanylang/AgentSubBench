#!/usr/bin/env python
"""DeepSeek reasoning_content validation probe: pinpoints the exact trigger
condition of the "must be passed back" 400.

Builds a span's round-0 continuation request (same rebuild logic as our
harness), then sends several variants (ablations) of the same request directly
to the baseline API (max_tokens=1), printing the HTTP status and error
message of each variant -- one run determines exactly what the server checks.

Usage (server, inside container):
  python scripts/diag_rc_probe.py --config OUT_DIR/config.json \
      --trace-path check_point_traces/appworld_simplified_batch1/21abae1_1.json \
      --inv 2
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402
from span_offload.config import load_config  # noqa: E402
from span_offload.harness.appworld_runtime import (  # noqa: E402
    AppWorldAgentRuntime, RC_PLACEHOLDER,
)
from span_offload.llm_client import ChatModelClient  # noqa: E402


def _strip_rc(messages):
    out = []
    for m in messages:
        mm = dict(m)
        mm.pop("reasoning_content", None)
        out.append(mm)
    return out


def _set_injected_rc(messages, rc):
    out = copy.deepcopy(messages)
    # the injected message = the last assistant message carrying tool_calls
    for m in reversed(out):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m["reasoning_content"] = rc
            return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--trace-path", required=True)
    ap.add_argument("--inv", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bc = cfg.baseline_llm
    adapter = AppWorldTraceAdapter()
    rt = AppWorldAgentRuntime()

    tr = adapter.load(args.trace_path)
    span = next(s for s in tr.spans if s.invocation_index == args.inv)
    conv = tr.artifacts["messages"]["msg:%d" % args.inv]

    # ---- build the same round-0 request as our harness ----
    seed = rt._recorded_to_dict_messages(conv)
    narration = rt._span_narration(type("S", (), {"trace": tr})(), span)[:2000]
    calls = span.baseline_response.parsed.get("tool_calls", [])
    tcs = [{"id": "call_t%d" % i, "call_id": "call_t%d" % i, "type": "function",
            "function": {"name": c.get("name", ""),
                         "arguments": json.dumps(c.get("arguments") or {},
                                                 ensure_ascii=False)}}
           for i, c in enumerate(calls)]
    inj = [{"role": "assistant", "content": narration or None,
            "tool_calls": tcs, "reasoning_content": narration or RC_PLACEHOLDER}]
    tool_msgs = [{"role": "tool", "tool_call_id": "call_t%d" % i,
                  "name": c.get("name", ""), "content": "{}"}
                 for i, c in enumerate(calls)]
    base = seed + inj + tool_msgs

    # variants: ablate one by one to find what the server actually checks
    variants = [
        ("current (all carry rc, injected rc = narration/placeholder)", base),
        ("v1: strip all rc", _strip_rc(base)),
        ("v2: injected rc set to empty string", _set_injected_rc(base, "")),
        ("v3: injected rc empty, seed has no rc", _set_injected_rc(_strip_rc(base), "")),
        ("v4: insert a user message before the injected message (replicating pre-B4)", None),
    ]
    # v4: insert a user message between the seed's tool results and the
    # injected message (replicating the pre-B4 structure: the injected assistant no longer directly follows tool messages)
    v4 = []
    inserted = False
    for i, m in enumerate(base):
        if not inserted and i > 0 and base[i - 1].get("role") == "tool" \
                and m.get("role") == "assistant" and m.get("tool_calls"):
            v4.append({"role": "user", "content": "# Task Instruction placeholder"})
            inserted = True
        v4.append(m)
    variants[4] = ("v4: insert a user message before the injected message (replicating pre-B4)", v4)

    # minimal tool schema (names only, used when the server validates tool calls)
    tools = [{"type": "function", "function": {"name": c.get("name", ""),
                                               "parameters": {"type": "object",
                                                              "properties": {}}}}
             for c in calls] or None

    import requests
    url = (bc.base_url or "").rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if bc.api_key:
        headers["Authorization"] = "Bearer %s" % bc.api_key

    print("baseline: %s model=%s span inv=%d seed_msgs=%d" % (
        url, bc.model, args.inv, len(seed)))
    for name, msgs in variants:
        payload = {"model": bc.model, "messages": msgs,
                   "temperature": 0.0, "max_tokens": 1}
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            body = resp.text[:220].replace("\n", " ")
        except Exception as exc:
            body = "transport error: %s" % exc
            resp = None
        status = getattr(resp, "status_code", "ERR")
        print("\n[%s]" % name)
        print("  HTTP %s: %s" % (status, body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
