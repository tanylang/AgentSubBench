#!/usr/bin/env python
"""Golden tests for LocomoTraceAdapter (pure JSON, runs locally on Windows).

Golden data: the conv-26.json smoke trace from trace_record (3 questions / 12 llm +
15 tool spans, pass=fail f1=0.2843). Read from the in-repo copy by default; use
LOC_TRACE_ROOT to point at the original trace_record location.

Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_locomo_adapter.py -v
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.locomo_adapter import (  # noqa: E402
    AGENT_ID, LocomoTraceAdapter,
)
from span_offload.schemas import TraceRun  # noqa: E402

# By default use the in-repo smoke copy (3 questions, verified golden assertions);
# the same-named file in trace_record has been overwritten by the full 50-question batch, so the golden assertions do not apply --
# for full-batch regression point LOC_TRACE_ROOT explicitly at the trace_record directory
TRACE_ROOT = os.environ.get(
    "LOC_TRACE_ROOT",
    os.path.join(ROOT, "check_point_traces",
                 "locomo_simplified_qwen3_8-27b_batch1"))
GOLDEN_TRACE = os.path.join(TRACE_ROOT, "conv-26.json")
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


@pytest.fixture(scope="module")
def adapter():
    from span_offload.trace_io import load_manifest
    m = load_manifest(os.path.join(ROOT, "data", "locomo", "task_manifest.json"))
    return LocomoTraceAdapter(m)


@pytest.fixture(scope="module")
def golden(adapter):
    return adapter.load(GOLDEN_TRACE)


def test_golden_trace_meta(golden: TraceRun):
    assert golden.run_meta.task_id == "conv-26"
    assert golden.run_meta.agent_id == AGENT_ID
    assert golden.run_meta.baseline_model  # Qwen3.8
    assert golden.run_meta.final_evaluator_passed is False
    assert golden.run_meta.goal_state_passed is False
    assert len(golden.spans) == 12  # 12 llm.call spans
    assert golden.task_condition.task_type == "locomo"
    assert golden.task_condition.task_state_effect == "read_only"
    assert golden.task_condition.evaluator_type == "answer_match"


def test_golden_contract_sequence(golden: TraceRun):
    kinds = [s.output_contract_runtime.contract_type for s in golden.spans]
    assert kinds == [
        "tool_call_list", "tool_call_list", "tool_call_list", "tool_call_list",
        "freeform_text",   # Q1 answer
        "tool_call_list", "tool_call_list", "tool_call_list",
        "freeform_text",   # Q2 answer
        "tool_call_list", "tool_call_list",
        "freeform_text",   # Q3 answer
    ]
    for s in golden.spans:
        assert s.baseline_response.parse_success is True  # freeform text is a valid answer


def test_golden_question_boundaries(golden: TraceRun):
    # question-boundary spans (inv 0/5/9) have their raw message list reset to [system,user]
    for inv in (0, 5, 9):
        msgs = golden.artifacts["messages"]["msg:%d" % inv]
        assert len(msgs) == 2, (inv, len(msgs))
        assert [m["role"] for m in msgs] == ["system", "user"]
        sc = golden.spans[inv].output_contract_runtime.semantic_constraints
        # inv 0 is the first span (no preceding turns, no splice needed); for inv 5/9 the preceding
        # tool_calls splice fails -> degraded (same flavor as swebench)
        assert sc["splice_degraded"] is {0: False, 5: True, 9: True}[inv]
        q_idx = golden.artifacts["state_snapshots"]["agent:%d" % inv]["q_idx"]
        assert q_idx == {0: 0, 5: 1, 9: 2}[inv]


def test_golden_splice_structure(golden: TraceRun):
    # inv 1 (step 4): splices back the 2 real tool_calls from inv 0
    msgs = golden.artifacts["messages"]["msg:1"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant",
                                         "tool", "tool"]
    assistant = msgs[2]
    assert assistant["content"] is None
    assert len(assistant["tool_calls"]) == 2
    for tc in assistant["tool_calls"]:
        # the splice uses the real call ids recorded in the trace (the collector keeps OpenAI native ids)
        assert tc["id"]
    for i, m in enumerate(msgs[3:]):
        assert m["tool_call_id"] == assistant["tool_calls"][i]["id"]
    # non-boundary spans are not marked degraded
    sc = golden.spans[1].output_contract_runtime.semantic_constraints
    assert sc["splice_degraded"] is False


def test_golden_hash_chain(golden: TraceRun):
    log = golden.artifacts["span_log"]
    assert len(log) == 27  # 12 llm + 15 tool
    hashes = {s.meta_hint.split("step=")[-1].split("|")[0]: None
              for s in golden.spans}
    by_step = {int(s.output_contract_runtime.semantic_constraints["source_step"]):
               s.state_snapshot.application_state_hash
               for s in golden.spans}
    # chain starts at e3b0 (a legitimate empty read set, not a placeholder); it advances only after retrieval
    assert by_step[1] == EMPTY_SHA
    assert by_step[4] == EMPTY_SHA
    assert by_step[6][:10] == "32caec0fc9"
    assert by_step[9][:10] == "c76662d349"
    assert by_step[12][:10] == "e983e1585f"
    assert by_step[13][:10] == "e983e1585f"  # question boundary does not advance
    assert by_step[19][:10] == "454989e7e3"
    assert by_step[21][:10] == "97ca802d99"
    assert by_step[27][:10] == "fdef8e526b"
    # no env snapshot carries a placeholder flag (e3b0 is a legitimate state)
    for ref, snap in golden.artifacts["state_snapshots"].items():
        if ref.startswith("env:"):
            assert snap.get("checkpointed") is True
            assert snap.get("application_state_hash_is_placeholder") is None


def test_golden_continuation_and_evaluator(golden: TraceRun):
    for s in golden.spans:
        assert s.continuation.evaluator_ref == "conv-26#locomo_qa"
        if s.continuation.resume_node_id != "final_evaluator":
            assert s.continuation.resume_node_id in golden.node_ids
    # correct q_idx for each answer span
    for inv in (4, 8, 11):
        snap = golden.artifacts["state_snapshots"]["agent:%d" % inv]
        assert snap["q_idx"] == {4: 0, 8: 1, 11: 2}[inv]
        assert snap["num_questions"] == 3


def test_golden_eligibility_relaxable(golden: TraceRun):
    from span_offload.span_extraction import check_eligibility
    from span_offload.trace_io import load_manifest
    m = load_manifest(os.path.join(ROOT, "data", "locomo", "task_manifest.json"))
    relaxable = {"baseline_run_failed_final_evaluator",
                 "baseline_goal_state_failed"}
    for s in golden.spans:
        reasons = set(check_eligibility(golden, s))
        assert reasons == relaxable, (s.invocation_index, reasons)
    # the extract_span product is serializable and carries the baseline_failed flag
    from span_offload.span_extraction import extract_span
    inst = extract_span(golden, golden.spans[0], GOLDEN_TRACE, m)
    assert inst.eligible is False
    assert inst.baseline_failed is True
    assert inst.task_id == "conv-26"


def test_golden_tool_key_names():
    # locomo tool spans use gen_ai.tool.name (not swebench's call.name)
    d = json.load(open(GOLDEN_TRACE, encoding="utf-8"))
    tool_spans = [s for s in d["spans"] if s.get("type") == "tool.execution"]
    assert len(tool_spans) == 15
    for t in tool_spans:
        attrs = t["attributes"]
        assert attrs.get("gen_ai.tool.name") in (
            "get_person_id", "search_messages", "all_messages", "time_range")
