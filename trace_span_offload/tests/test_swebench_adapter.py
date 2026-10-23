#!/usr/bin/env python
"""Golden tests for SwebenchTraceAdapter (pure JSON, runs locally on Windows, no bash).

Golden data: psf__requests-1142.json from trace_record + full regression over all 58 traces.
Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_swebench_adapter.py -v
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.schemas import TraceRun  # noqa: E402
from span_offload.swebench_adapter import (  # noqa: E402
    AGENT_ID, WORKSPACE_TOKEN, SwebenchTraceAdapter,
)

TRACE_ROOT = os.environ.get(
    "SWE_TRACE_ROOT",
    "C:/Users/Lenovo/Desktop/trace_record/traces/swebench_mini-swe-agent")
GOLDEN_TRACE = os.path.join(TRACE_ROOT, "psf__requests-1142.json")
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture(scope="module")
def adapter():
    from span_offload.trace_io import load_manifest
    m = load_manifest(os.path.join(ROOT, "data", "swebench", "task_manifest.json"))
    return SwebenchTraceAdapter(m)


@pytest.fixture(scope="module")
def golden(adapter):
    return adapter.load(GOLDEN_TRACE)


def test_golden_trace_meta(golden: TraceRun):
    assert golden.run_meta.task_id == "psf__requests-1142"
    assert golden.run_meta.agent_id == AGENT_ID
    assert golden.run_meta.baseline_model == "deepseek-chat"
    assert golden.run_meta.final_evaluator_passed is False
    assert len(golden.spans) == 50  # 50 llm.call spans
    assert golden.task_condition.task_type == "swebench"
    assert golden.task_condition.involved_apps == ["psf/requests"]


def test_golden_hash_chain(golden: TraceRun):
    log = golden.artifacts["env_delta_log"]
    # total (non-root) spans = 50 llm + 53 tool = 103
    assert len(log) == 103
    # the first hash_before is a legitimate empty state (chain start); real file-change hashes appear afterwards
    assert log[0]["hash_before"] == EMPTY_SHA
    assert any(e["hash_before"] != EMPTY_SHA for e in log)
    # span0's env snapshot pointer is correct
    s0 = golden.spans[0]
    ref = s0.state_snapshot.environment_snapshot_ref
    snap = golden.artifacts["state_snapshots"][ref]
    assert snap["checkpointed"] is True
    assert log[snap["delta_log_index"]]["type"] == "llm.call"


def test_golden_splice_structure(golden: TraceRun):
    """msg:j's assistant carries tool_calls with real call ids; tool messages attach tool_call_id."""
    msgs = golden.artifacts["messages"]["msg:1"]
    roles = [m.get("role") for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    assistant = msgs[2]
    assert assistant.get("content") is None
    tcs = assistant.get("tool_calls") or []
    assert len(tcs) == 2
    # real call ids are preserved (no synthetic prefix)
    for tc in tcs:
        assert tc["id"].startswith("call_")
        assert tc["function"]["name"] == "bash"
    # tool messages carry tool_call_id
    assert msgs[3].get("tool_call_id") == tcs[0]["id"]
    assert msgs[4].get("tool_call_id") == tcs[1]["id"]


def test_golden_span0_contract(golden: TraceRun):
    s0 = golden.spans[0]
    assert s0.output_contract_runtime.contract_type == "tool_call_list"
    assert s0.baseline_response.parse_success is True
    calls = s0.baseline_response.parsed["tool_calls"]
    assert len(calls) == 2
    assert all(c["name"] == "bash" for c in calls)
    assert s0.continuation.resume_node_id == "tool:0:0"
    assert s0.continuation.max_remaining_steps == 50
    # attached tool results
    assert len(golden.artifacts["span_tool_results"]["span:0"]) == 2
    # descriptor classification: bash command generation
    assert s0.span_descriptor.node_pattern == "tool_argument_generation"
    assert "schema_check" in s0.span_descriptor.verifier_type


def test_golden_path_rewrite(golden: TraceRun):
    """Windows workspace absolute paths in the recorded commands are normalized to WORKSPACE_TOKEN."""
    prefixes = golden.artifacts["workspace_prefixes"]
    assert prefixes  # this trace has path pollution
    hit = False
    for s in golden.spans:
        sc = s.output_contract_runtime.semantic_constraints
        if sc.get("path_rewritten"):
            hit = True
            # after normalization the original prefixes no longer appear
            for p in prefixes:
                for c in s.baseline_response.parsed.get("tool_calls", []):
                    assert p not in json.dumps(c.get("arguments"))
    assert hit


def test_golden_no_degraded_splice(golden: TraceRun):
    degraded = [s for s in golden.spans
                if s.output_contract_runtime.semantic_constraints.get(
                    "splice_degraded")]
    assert len(degraded) == 0


def test_all_58_traces_load(adapter):
    """Full regression: all 58 traces load successfully, no exceptions."""
    n = 0
    for p in glob.glob(os.path.join(TRACE_ROOT, "*.json")):
        tr = adapter.load(p)
        n += 1
        assert tr.spans  # non-empty
        assert tr.run_meta.agent_id == AGENT_ID
    assert n == 58


def test_textonly_spans_marked_unparseable(adapter):
    """Text-only llm spans (no tool_calls) are faithfully mapped to parse_success=False."""
    n_textonly = 0
    for p in glob.glob(os.path.join(TRACE_ROOT, "*.json")):
        tr = adapter.load(p)
        for s in tr.spans:
            if s.output_contract_runtime.contract_type == "freeform_text":
                n_textonly += 1
                assert s.baseline_response.parse_success is False
    assert n_textonly >= 19  # 19 observed across all 58 traces


def test_recorded_tools_schema(adapter):
    tr = adapter.load(GOLDEN_TRACE)
    tools = tr.artifacts["recorded_tools"]["tools:0"]
    assert any(t.get("function", {}).get("name") == "bash" for t in tools)
