#!/usr/bin/env python
"""LangGraphRuntime offline end-to-end tests: C0 frozen arm (zero LM calls) closed-loop fidelity.

The C0 walkthrough injects trace-recorded outputs throughout and only really executes the langmem memory tools (local store),
covering all ingestion and QA spans; it runs offline locally on Windows and reproduces the original evaluation
(smoke single-question f1).

Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_langgraph_runtime.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LOCOMO_DATA_FILE",
                      os.path.join(ROOT, "data", "locomo", "locomo10.json"))

from span_offload.config import load_config  # noqa: E402
from span_offload.harness.langgraph_runtime import (  # noqa: E402
    AGENT_ID, LangGraphRuntime,
)
from span_offload.langgraph_adapter import LangGraphTraceAdapter  # noqa: E402
from span_offload.llm_client import ChatModelClient  # noqa: E402
from span_offload.replay import SingleSpanReplayEngine  # noqa: E402
from span_offload.span_extraction import extract_span  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402

GOLDEN_TRACE = ("C:/Users/Lenovo/Desktop/trace_record/trace_collect/"
                "langgraph_locomo/traces/schema_smoke/conv-26/q000.json")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(os.path.join(ROOT, "data", "locomo", "task_manifest.json"))


@pytest.fixture(scope="module")
def trace(manifest):
    return LangGraphTraceAdapter(manifest).load(GOLDEN_TRACE)


@pytest.fixture(scope="module")
def instances(trace, manifest):
    return [extract_span(trace, s, GOLDEN_TRACE, manifest)
            for s in trace.spans]


@pytest.fixture(scope="module")
def engine(manifest):
    cfg = load_config(os.path.join(ROOT, "config.json"))
    cfg.offline = True
    dummy = ChatModelClient(base_url="http://127.0.0.1:1", api_key="",
                            model="dummy", temperature=0.0,
                            max_output_tokens=8192)
    return SingleSpanReplayEngine(
        cfg, LangGraphRuntime(manifest), baseline_client=dummy,
        slm_clients={"dummy": dummy}, fallback_mode="none")


@pytest.mark.parametrize("inv", [0, 1])
def test_c0_closed_loop(engine, trace, instances, inv):
    """C0 frozen replay (QA-phase spans): zero LM calls, tool matches True,
    single-question evaluation reproduced."""
    record = engine.replay(instances[inv], trace, "dummy", arm="C0")
    assert record.replay_meta.get("injection") == "identity"
    fidelity = record.replay_meta.get("c0_fidelity") or []
    if inv == 1:
        # the last answer span: the frozen walkthrough has no downstream (empty plan); empty fidelity is the correct semantics
        pass
    else:
        assert fidelity, (inv, "c0_fidelity is empty")
    for f in fidelity:
        if f.get("tool_match") is not None:
            assert f["tool_match"] is True, (inv, f)
    dt_spans = (record.downstream_trace or {}).get("spans", [])
    assert not any(e.get("span_type") == "llm.call"
                   and e.get("source") == "baseline" for e in dt_spans)
    details = record.final_outcome.evaluator_details
    assert any("pass=" in d for d in details), (inv, details)
    # new verifier flavor: the smoke-recorded answer contains the gold text verbatim (format false-conviction rescue) -> pass
    assert record.final_outcome.final_task_success is True


def test_c0_qa_tool_span_value_match(engine, trace, instances):
    """C0 for the QA tool span (inv 0, search_memory): target execution vs recorded
    soft/hard comparisons all True."""
    record = engine.replay(instances[0], trace, "dummy", arm="C0")
    fidelity = record.replay_meta.get("c0_fidelity") or []
    target = next(f for f in fidelity if f["invocation_index"] == 0)
    assert target["tool_match"] is True
    assert target["value_match"] is True


def test_restore_snapshot_hash(engine, trace, instances):
    """restore's raw-to-raw hash gate: replayed chain hash == recorded hash (the chain contains
    ingestion diffs; the QA span's context = the full session memory)."""
    runtime = LangGraphRuntime(None)
    dummy = ChatModelClient(base_url="http://127.0.0.1:1", api_key="",
                            model="dummy", temperature=0.0,
                            max_output_tokens=8192)
    # QA tool span (first)
    s0 = trace.spans[0]
    session = runtime.restore(trace, s0, dummy)
    assert session.snapshot_hash.startswith("memory:")
    assert session.restored_env_hash_raw == \
        s0.state_snapshot.application_state_hash
    assert session.num_questions == 1
    assert session.phase == "qa"
    assert session.question_texts[0]
    assert session.ingestion_remaining == []
    # QA answer span (last)
    s1 = trace.spans[1]
    session2 = runtime.restore(trace, s1, dummy)
    assert session2.phase == "qa"
    assert session2.restored_env_hash_raw == \
        s1.state_snapshot.application_state_hash


def test_unsupported_agent_identity_raises(manifest, trace):
    """Non-LangGraph identities must raise an honest error; never degrade to an approximate continuation."""
    dummy = ChatModelClient(base_url="http://127.0.0.1:1", api_key="",
                            model="dummy", temperature=0.0,
                            max_output_tokens=8192)
    runtime = LangGraphRuntime(manifest)
    with open(GOLDEN_TRACE, encoding="utf-8") as f:
        raw = json.load(f)
    raw["agent_name"] = "LocoMo-Simplified"
    with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8") as tf:
        json.dump(raw, tf, ensure_ascii=False)
        tmp_path = tf.name
    try:
        tr2 = LangGraphTraceAdapter(manifest).load(tmp_path)
        with pytest.raises(RuntimeError, match="backend unavailable"):
            runtime.restore(tr2, tr2.spans[0], dummy)
    finally:
        os.unlink(tmp_path)


def test_agent_id_constant():
    assert AGENT_ID == "LoCoMo-LangGraph-ReAct-LangMem"
