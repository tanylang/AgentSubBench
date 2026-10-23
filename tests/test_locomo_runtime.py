#!/usr/bin/env python
"""LocomoRuntime offline end-to-end tests: C0 frozen arm (zero LM calls) closed-loop fidelity.

The C0 walkthrough injects trace-recorded outputs throughout and only really executes read-only tools -> no model calls at any point;
it runs offline locally on Windows and must reproduce the original evaluation (f1=0.2843, fail).

Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_locomo_runtime.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LOCOMO_DATA_FILE",
                      os.path.join(ROOT, "data", "locomo", "locomo10.json"))

from span_offload.config import load_config  # noqa: E402
from span_offload.harness.locomo_runtime import LocomoRuntime  # noqa: E402
from span_offload.llm_client import ChatModelClient  # noqa: E402
from span_offload.locomo_adapter import LocomoTraceAdapter  # noqa: E402
from span_offload.replay import SingleSpanReplayEngine  # noqa: E402
from span_offload.span_extraction import extract_span  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402

GOLDEN_TRACE = os.path.join(
    ROOT, "check_point_traces", "locomo_simplified_qwen3_8-27b_batch1",
    "conv-26.json")


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(os.path.join(ROOT, "data", "locomo", "task_manifest.json"))


@pytest.fixture(scope="module")
def trace(manifest):
    return LocomoTraceAdapter(manifest).load(GOLDEN_TRACE)


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
                            max_output_tokens=4096)
    return SingleSpanReplayEngine(
        cfg, LocomoRuntime(manifest), baseline_client=dummy,
        slm_clients={"dummy": dummy}, fallback_mode="none")


@pytest.mark.parametrize("inv", [0, 1, 4, 5, 8, 11])
def test_c0_closed_loop(engine, trace, instances, inv):
    """C0 frozen replay: zero LM calls, reproduces the original evaluation (f1=0.2843, fail)."""
    record = engine.replay(instances[inv], trace, "dummy", arm="C0")
    # the frozen arm makes no model calls throughout: usage is empty, injection source is identity
    assert record.replay_meta.get("injection") == "identity"
    meta = record.replay_meta
    assert meta.get("c0_fidelity") is not None
    # the final evaluation reproduces the official flavor
    assert record.final_outcome.final_task_success is False
    assert record.final_outcome.goal_state_passed is False
    assert any("f1=0.2843" in d for d in record.final_outcome.evaluator_details)
    # the system prompt goes through the real build path with no drift from the record
    dt_meta = (record.downstream_trace or {}).get("meta", {})
    assert dt_meta.get("system_prompt_source") == "rebuilt_from_agent_template"
    assert dt_meta.get("system_prompt_drift") is False
    # the downstream trace has no baseline-sourced llm.call (zero real calls)
    dt_spans = (record.downstream_trace or {}).get("spans", [])
    assert not any(e.get("span_type") == "llm.call"
                   and e.get("source") == "baseline" for e in dt_spans)


def test_c0_all_spans_closed_loop(engine, trace, instances):
    """C0 walkthrough of all 12 spans: tool spans' tool_match all True."""
    for i, inst in enumerate(instances):
        record = engine.replay(inst, trace, "dummy", arm="C0")
        fidelity = record.replay_meta.get("c0_fidelity") or []
        for f in fidelity:
            if f.get("tool_match") is not None:
                assert f["tool_match"] is True, (i, f)
        assert record.final_outcome.final_task_success is False


def test_c0_answer_span_bookkeeping(engine, trace, instances):
    """C0 for the answer span (inv 4, Q1 answer): injected answer bookkeeping + frozen walkthrough of subsequent questions."""
    record = engine.replay(instances[4], trace, "dummy", arm="C0")
    dt_spans = (record.downstream_trace or {}).get("spans", [])
    # the frozen walkthrough recorded Q2/Q3's recorded answers (downstream freeform llm.call records)
    frozen_answers = [e for e in dt_spans
                      if e.get("span_type") == "llm.call"
                      and e.get("source") == "frozen"
                      and "text" in (e.get("output") or {})]
    assert len(frozen_answers) >= 2  # recorded answers for Q2 and Q3


def test_c0_tool_span_fidelity_detail(engine, trace, instances):
    """C0 for the tool span (inv 1): comparison records for the target span and downstream spans are complete."""
    record = engine.replay(instances[1], trace, "dummy", arm="C0")
    fidelity = record.replay_meta.get("c0_fidelity") or []
    assert any(f["invocation_index"] == 1 for f in fidelity)  # target span
    assert any(f["invocation_index"] > 1 for f in fidelity)   # downstream span
    for f in fidelity:
        if f.get("tool_match") is None:
            continue
        assert f["tool_match"] is True


def test_question_prompt_parsing():
    from span_offload.harness.locomo_env import QUESTION_RE
    m = QUESTION_RE.search("Question 2/3: What inspired Caroline?")
    assert m is not None
    assert int(m.group(1)) == 2 and int(m.group(2)) == 3
    assert QUESTION_RE.search("no question here") is None


def test_unsupported_agent_identity_raises(manifest, trace):
    """Non-simplified identities (e.g. AutoGen) must raise an honest error; never degrade to an approximate continuation."""
    import json
    import tempfile
    dummy = ChatModelClient(base_url="http://127.0.0.1:1", api_key="",
                            model="dummy", temperature=0.0,
                            max_output_tokens=4096)
    runtime = LocomoRuntime(manifest)
    with open(GOLDEN_TRACE, encoding="utf-8") as f:
        raw = json.load(f)
    raw["agent_name"] = "LocoMo-AutoGen"
    with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False,
            encoding="utf-8") as tf:
        json.dump(raw, tf, ensure_ascii=False)
        tmp_path = tf.name
    try:
        tr2 = LocomoTraceAdapter(manifest).load(tmp_path)
        assert tr2.run_meta.agent_id == "LocoMo-AutoGen"
        with pytest.raises(RuntimeError, match="backend unavailable"):
            runtime.restore(tr2, tr2.spans[0], dummy)
    finally:
        os.unlink(tmp_path)
