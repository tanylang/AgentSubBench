#!/usr/bin/env python
"""autogen continuation trigger-phrase diagnostic: sends several variants
of the same continuation context to the 27B baseline endpoint, compares them with the original response, and pinpoints the "empty reply in thinking mode" trigger condition.

Background: in continuation (engine 2026-09-16.3) the model returns empty
content + finish_reason=stop for the continue prompt, completion≈60 tokens
(nothing left after vLLM strips the thinking). Native capture with the same
endpoint and same thinking calls tools fine, so the difference converges to the trigger phrase x thinking interaction.

Usage (server, inside container):
  python scripts/diag/diag_continuation.py \
      --trace check_point_traces/appworld_autogen_qwen3_8-27b_batch1/23cf851_1.json \
      --inv 1
  BASELINE_BASE_URL / BASELINE_MODEL env vars default to
  192.168.10.26:8101/v1 / Qwen3.8-27B.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.environ.get("BASELINE_BASE_URL", "http://192.168.10.26:8101/v1")
MODEL = os.environ.get("BASELINE_MODEL", "Qwen3.8-27B")

PROMPT_FULL = (
    "Continue executing the task based on the conversation history. "
    "Output the next tool call. If the task is complete, call "
    "supervisor__complete_task. Do not output plain text.")
PROMPT_PLAIN = "Continue executing the task."


def _conv_to_openai(conv):
    out = []
    for m in conv:
        out.append({"role": m.get("role", "user"),
                    "content": str(m.get("content", ""))})
    return out


def _tools_from_trace(adapter, tr, inv):
    """Uses the span's native tool schemas as recorded in the trace, as-is."""
    rec = (tr.artifacts.get("recorded_tools") or {}).get("tools:%d" % inv)
    if rec:
        return [{"type": "function", "function": f["function"]} for f in rec
                if f.get("type") == "function" and "function" in f]
    allowed = []
    for s in tr.spans:
        if s.invocation_index == inv:
            allowed = list(s.output_contract_runtime.allowed_tools or [])
            break
    docs = []
    seen = set()
    for doc in _api_docs():
        fn = doc["function"]
        name = fn["name"].replace("__", ".")
        if name in allowed and name not in seen:
            seen.add(name)
            docs.append({"type": "function", "function": fn})
    return docs


def _api_docs():
    try:
        import appworld
        from appworld.task import Task
        # needs the world context; the diagnostic uses recorded_tools by default and never takes this branch
    except Exception:
        pass
    return []


def _post(payload, tag):
    import requests
    url = BASE_URL.rstrip("/") + "/chat/completions"
    try:
        r = requests.post(url, json=payload, timeout=300)
        body = r.json()
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        print("=== %s ===" % tag)
        print("status=%d finish=%s" % (r.status_code, choice.get("finish_reason")))
        print("content=%r" % str(msg.get("content", ""))[:200])
        print("tool_calls=%s" % json.dumps(msg.get("tool_calls"), ensure_ascii=False)[:200])
        print("reasoning=%r" % str(msg.get("reasoning_content", ""))[:200])
        print("extra_fields=%s" % [k for k in msg.keys()
                                   if k not in ("role", "content", "tool_calls")])
        print("usage=%s" % json.dumps(body.get("usage", {}), ensure_ascii=False))
    except Exception as e:
        print("=== %s ===\nREQUEST FAILED: %s: %s" % (tag, type(e).__name__, e))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--inv", type=int, default=1)
    args = ap.parse_args()

    from span_offload.appworld_adapter import AppWorldTraceAdapter
    from span_offload.trace_io import load_manifest
    adapter = AppWorldTraceAdapter(load_manifest("data/appworld/task_manifest.json"))
    tr = adapter.load(args.trace)
    span = next(s for s in tr.spans if s.invocation_index == args.inv)
    conv = list((tr.artifacts.get("messages") or {}).get(
        span.request.messages_ref, []))
    msgs = _conv_to_openai(conv)
    tools = _tools_from_trace(adapter, tr, args.inv)
    print("context: %d messages, %d tools" % (len(msgs), len(tools)))

    def payload(extra_user=None, tool_choice=None, think=None, seed=100):
        m = list(msgs)
        if extra_user:
            m.append({"role": "user", "content": extra_user})
        p = {"model": MODEL, "messages": m, "tools": tools,
             "temperature": 0.0}
        if seed is not None:
            p["seed"] = seed
        if tool_choice:
            p["tool_choice"] = tool_choice
        if think is not None:
            p["chat_template_kwargs"] = {"enable_thinking": think}
        return p

    _post(payload(extra_user=None, seed=100),
          "V1 no trigger phrase + seed=100 (current harness state, should reproduce the empty reply)")
    _post(payload(extra_user=None, seed=None),
          "V2 no trigger phrase + no seed (native capture convention)")
    _post(payload(extra_user=None, seed=1), "V3 no trigger phrase + seed=1")
    _post(payload(extra_user=None, seed=42), "V4 no trigger phrase + seed=42")
    _post(payload(extra_user=None, seed=100, tool_choice="required"),
          "V5 seed=100 + tool_choice=required")
    _post(payload(extra_user=None, seed=100, think=False),
          "V6 seed=100 + thinking off")
    _post(payload(extra_user=None, seed=None, think=False),
          "V7 no seed + thinking off")
    # V8/V9: harness client serialization A/B -- V8 serializes tools with
    # autogen_ext's convert_tools (the continuation client's outbound
    # convention), V9 additionally applies autogen message conversion. If V8 replies empty while V1 is fine -> the difference is in tool serialization
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for _pkg in ("autogen-core", "autogen-ext", "autogen-agentchat"):
            sys.path.insert(0, os.path.join(
                _root, "source_code", "autogen", "python", "packages", _pkg, "src"))
        from autogen_ext.models.openai._openai_client import convert_tools
        _fn_schemas = [f["function"] for f in tools
                       if isinstance(f, dict)
                       and isinstance(f.get("function"), dict)]
        _conv = convert_tools(_fn_schemas)
        p8 = payload(extra_user=None, seed=100)
        p8["tools"] = [dict(t) for t in _conv]
        _post(p8, "V8 harness tool serialization (convert_tools) + seed=100")
    except Exception as e:
        print("V8 skipped (import failed): %s: %s" % (type(e).__name__, e))
    # V9/V10: field-level A/B between harness outbound and the bare request
    # -- V9 adds name="user" to user messages (autogen serialization
    # behavior), V10 explicitly sends tool_choice=null. Whichever variant reproduces the empty reply, that field is the culprit.
    def p9():
        p = payload(extra_user=None, seed=100)
        p["messages"] = [dict(m) for m in p["messages"]]
        for m in p["messages"]:
            if m.get("role") == "user":
                m["name"] = "user"
        return p
    _post(p9(), "V9 bare request + name='user' on user messages")
    p10 = payload(extra_user=None, seed=100)
    p10["tool_choice"] = None
    _post(p10, "V10 bare request + tool_choice=null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
