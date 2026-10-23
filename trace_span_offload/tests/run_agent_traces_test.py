"""Offline adapter/runtime tests for the four non-autogen/simplified agents.

Covers the trace formats collected by trace_record's OrderedTraceCollector for
  - AppWorld-Legacy          (native function-calling, full conversation)
  - AppWorld-OpenAI-Agents   (SDK agent; conversation NOT recorded)
  - AppWorld-Smolagents      (text tool protocol: "Calling tools:"/"Observation:")
  - Tau2-LLMAgent            (native tool calls, driver-loop conversation)

Assertions are split into two layers:
  1. ADAPTER layer (always runs, no AppWorld dependency):
     annotations, span extraction, per-agent conversation splicing
     (tool_call_id linkage / text protocol / transcript reconstruction).
  2. RUNTIME layer (runs only when the appworld package is importable):
     backend resolution per agent identity + restore + resolve_request
     with a stub client (no real API calls).

Usage:  python tests/run_agent_traces_test.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    print("%s %s %s" % ("PASS" if condition else "FAIL", name, detail))


def main() -> int:
    from span_offload.appworld_adapter import AppWorldTraceAdapter

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapter = AppWorldTraceAdapter(None)

    # ---------------------------------------------------------------- adapter
    AGENT_DIRS = {
        "AppWorld-Legacy": "appworld_legacy_batch1",
        "AppWorld-OpenAI-Agents": "appworld_openai_agents_batch1",
        "AppWorld-Smolagents": "appworld_smolagents_batch1",
        "Tau2-LLMAgent": "appworld_tau2_batch1",
    }
    CLUSTERS = {
        "AppWorld-Legacy": "appworld-legacy",
        "AppWorld-OpenAI-Agents": "appworld-openai-agents",
        "AppWorld-Smolagents": "appworld-smolagents",
        "Tau2-LLMAgent": "tau2-llmagent",
    }
    traces = {}
    for agent_id, d in AGENT_DIRS.items():
        tdir = os.path.join(root, "check_point_traces", d)
        if not os.path.isdir(tdir):
            print("SKIP: trace dir missing: %s" % tdir)
            continue
        path = os.path.join(tdir, "0d8a4ee_1.json")
        if not os.path.exists(path):
            print("SKIP: trace missing: %s" % path)
            continue
        t = adapter.load(path)
        traces[agent_id] = t
        check("%s: agent annotation" % agent_id,
              t.agent_condition.source_cluster_id == CLUSTERS[agent_id],
              t.agent_condition.source_cluster_id)
        check("%s: spans extracted" % agent_id, len(t.spans) > 0,
              "n=%d" % len(t.spans))
        check("%s: span 0 is api prediction (tool_selection)" % agent_id,
              t.spans[0].span_descriptor.node_pattern == "tool_selection",
              t.spans[0].span_descriptor.node_pattern)
        # no private keys may leak into prompt artifacts
        leaked = [k for s in t.spans
                  for m in (t.artifacts.get("messages") or {}).get(
                      s.request.messages_ref, [])
                  for k in m if str(k).startswith("_")]
        check("%s: no private keys in messages" % agent_id, not leaked,
              str(leaked[:3]))

    # -- per-agent splicing --------------------------------------------------
    if "AppWorld-Legacy" in traces:
        t = traces["AppWorld-Legacy"]
        span = next(s for s in t.spans if s.invocation_index == 2)
        msgs = t.artifacts["messages"][span.request.messages_ref]
        n_tc = sum(1 for m in msgs if m.get("tool_calls"))
        n_linked = sum(1 for m in msgs
                       if m.get("role") == "tool" and m.get("tool_call_id"))
        check("legacy: history turn spliced with native tool_calls",
              n_tc >= 1, "assistant turns with tool_calls=%d" % n_tc)
        check("legacy: tool messages linked to call ids",
              n_linked >= 1, "linked=%d" % n_linked)
        # every spliced call must have a tool response (OpenAI protocol)
        total_calls = sum(len(m["tool_calls"]) for m in msgs if m.get("tool_calls"))
        check("legacy: every spliced call answered by a tool message",
              total_calls == n_linked, "calls=%d linked=%d" % (total_calls, n_linked))

    if "Tau2-LLMAgent" in traces:
        t = traces["Tau2-LLMAgent"]
        span = next(s for s in t.spans if s.invocation_index == 2)
        msgs = t.artifacts["messages"][span.request.messages_ref]
        n_tc = sum(1 for m in msgs if m.get("tool_calls"))
        n_linked = sum(1 for m in msgs
                       if m.get("role") == "tool" and m.get("tool_call_id"))
        check("tau2: history turn spliced with native tool_calls", n_tc >= 1,
              "turns=%d" % n_tc)
        check("tau2: tool messages linked to call ids", n_linked >= 1,
              "linked=%d" % n_linked)
        check("tau2: system prompt preserved",
              msgs and msgs[0].get("role") == "system"
              and "instructions" in str(msgs[0].get("content", "")),
              str(msgs[0].get("content", ""))[:40] if msgs else "none")

    if "AppWorld-Smolagents" in traces:
        t = traces["AppWorld-Smolagents"]
        span = next(s for s in t.spans if s.invocation_index == 2)
        msgs = t.artifacts["messages"][span.request.messages_ref]
        # smolagents keeps the TEXT tool protocol (faithful to the original
        # request shape); roles must be plain system/user/assistant
        roles = {m.get("role") for m in msgs}
        check("smolagents: text protocol (no native tool_calls)",
              all(not m.get("tool_calls") for m in msgs))
        check("smolagents: normalized plain roles",
              roles <= {"system", "user", "assistant"}, str(roles))
        has_call_text = any("Calling tools:" in str(m.get("content", ""))
                            for m in msgs)
        has_obs = any(str(m.get("content", "")).startswith("Observation:")
                      for m in msgs)
        check("smolagents: Calling tools text preserved", has_call_text)
        check("smolagents: Observation text preserved", has_obs)

    if "AppWorld-OpenAI-Agents" in traces:
        t = traces["AppWorld-OpenAI-Agents"]
        span = next(s for s in t.spans if s.invocation_index == 3)
        msgs = t.artifacts["messages"][span.request.messages_ref]
        # transcript: j=3 -> turns 1..2 spliced (span 0 is freeform api
        # prediction and must NOT be consumed)
        n_tc = sum(1 for m in msgs if m.get("tool_calls"))
        check("openai: transcript reconstructed from recorded tool calls",
              n_tc == 2, "assistant turns=%d (expected 2)" % n_tc)
        n_linked = sum(1 for m in msgs
                       if m.get("role") == "tool" and m.get("tool_call_id"))
        total_calls = sum(len(m["tool_calls"]) for m in msgs if m.get("tool_calls"))
        check("openai: transcript calls fully paired with tool messages",
              total_calls == n_linked, "calls=%d linked=%d" % (total_calls, n_linked))
        check("openai: reconstruction flag recorded",
              span.output_contract_runtime.semantic_constraints.get(
                  "openai_prompt_reconstructed") is True)
        check("openai: span 0 api-prediction freeform",
              t.spans[0].output_contract_runtime.contract_type == "freeform_text")

    # ---------------------------------------------------------------- runtime
    try:
        import appworld  # noqa: F401
        appworld_ok = True
    except ImportError:
        appworld_ok = False
    if not appworld_ok:
        print("SKIP: runtime-layer checks need the appworld package")
        return finish()

    from span_offload.harness.appworld_runtime import AppWorldAgentRuntime

    runtime = AppWorldAgentRuntime()

    class _StubClient:
        base_url = "x"
        api_key = "k"
        model = "m"
        temperature = 0.0
        max_output_tokens = 4096

    EXPECTED = {
        "AppWorld-Simplified": "simplified",
        "AutoGen-AssistantAgent": "autogen",
        "AppWorld-Legacy": "legacy",
        "Tau2-LLMAgent": None,  # framework missing -> generic fallback
    }
    for agent_id, d in AGENT_DIRS.items():
        tdir = os.path.join(root, "check_point_traces", d)
        if not os.path.isdir(tdir):
            continue
        path = os.path.join(tdir, "0d8a4ee_1.json")
        trace = adapter.load(path)
        span = next(s for s in trace.spans
                    if s.output_contract_runtime.contract_type == "tool_call_list"
                    and s.invocation_index >= 2)
        try:
            session = runtime.restore(trace, span, _StubClient())
            req = runtime.resolve_request(trace, span, session)
            check("%s: restore + resolve ok" % agent_id, True,
                  "backend=%s msgs=%d" % (session.backend, len(req.messages)))
            # protocol flags
            if agent_id == "AppWorld-Legacy":
                check("legacy: tool_choice=required (native protocol)",
                      session.tool_choice == "required", str(session.tool_choice))
            if agent_id == "AppWorld-Smolagents":
                check("smolagents: text_protocol flag",
                      session.text_protocol is True)
                check("smolagents: tool_choice=required",
                      session.tool_choice == "required")
            if agent_id == "AppWorld-OpenAI-Agents":
                # live layer: runtime re-renders the real prompt with task context
                check("openai: live prompt re-rendered",
                      session.openai_messages is not None
                      and len(session.openai_messages) > len(
                          trace.artifacts["messages"][span.request.messages_ref]),
                      "rendered=%d transcript=%d" % (
                          len(session.openai_messages or []),
                          len(trace.artifacts["messages"][span.request.messages_ref])))
        except Exception as exc:  # noqa: BLE001
            check("%s: restore + resolve ok" % agent_id, False,
                  "%s: %s" % (type(exc).__name__, exc))
    return finish()


def finish() -> int:
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print("\n%d/%d checks passed" % (len(CHECKS) - n_fail, len(CHECKS)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
