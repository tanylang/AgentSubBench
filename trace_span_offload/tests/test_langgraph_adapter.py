#!/usr/bin/env python
"""Golden tests for LangGraphTraceAdapter (pure JSON, runs locally on Windows).

Golden data: schema_smoke/conv-26/q000.json from trace_record langgraph_locomo
(small smoke test, 1 question / 17 spans = 13 ingestion + 4 QA) and the production samples/conv-30.json
(105 questions / 2281 spans, spot-checked unit segmentation). Read from trace_record by default, overridable with
LANGGRAPH_TRACE_ROOT.

Offload unit semantics (2026-09-21 decision): one unit trace per QA = all ingestion
spans + that QA's spans; QA context = the full session (complete messages + full memory prefix).
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.langgraph_adapter import (  # noqa: E402
    AGENT_ID, LangGraphTraceAdapter,
)
from span_offload.schemas import TraceRun  # noqa: E402

TRACE_ROOT = os.environ.get(
    "LANGGRAPH_TRACE_ROOT",
    "C:/Users/Lenovo/Desktop/trace_record/trace_collect/langgraph_locomo/traces")
GOLDEN_TRACE = os.path.join(TRACE_ROOT, "schema_smoke", "conv-26", "q000.json")
FULL_TRACE = os.path.join(TRACE_ROOT, "locomo_native", "samples", "conv-30.json")


@pytest.fixture(scope="module")
def adapter():
    from span_offload.trace_io import load_manifest
    m = load_manifest(os.path.join(ROOT, "data", "locomo", "task_manifest.json"))
    return LangGraphTraceAdapter(m)


@pytest.fixture(scope="module")
def golden(adapter):
    return adapter.load(GOLDEN_TRACE)


def test_golden_trace_meta(golden: TraceRun):
    # unit task_id = locomo:<sample>:q<qi>
    assert golden.run_meta.task_id == "locomo:conv-26:q000"
    assert golden.run_meta.agent_id == AGENT_ID
    assert golden.run_meta.baseline_model == "Qwen3.8-27B"
    # collector side did not finalize -> pass missing -> baseline_failed (exemption-pool flavor)
    assert golden.run_meta.final_evaluator_passed is False
    assert golden.task_condition.task_type == "locomo"
    assert golden.task_condition.evaluator_type == "answer_match"


def test_golden_unit_spans(golden: TraceRun):
    """A unit = only that QA's spans (user decision: offload only the QA part)."""
    # smoke q000: 2 QA llm calls (search_memory + answer)
    assert len(golden.spans) == 2
    phases = [s.output_contract_runtime.semantic_constraints["locomo_phase"]
              for s in golden.spans]
    assert phases == ["qa", "qa"]
    kinds = [s.output_contract_runtime.contract_type for s in golden.spans]
    assert kinds == ["tool_call_list", "freeform_text"]
    for s in golden.spans:
        sc = s.output_contract_runtime.semantic_constraints
        assert sc["num_questions"] == 1
        assert sc["qa_index"] == 0
        assert s.baseline_response.parse_success is True
        assert s.continuation.evaluator_ref == "locomo:conv-26:q000#locomo_qa"
    # delta_log covers all non-root spans (16 entries: ingestion diffs kept for chain replay)
    assert len(golden.artifacts["memory_delta_log"]) == 16
    # the unit artifact carries the global question index and question text
    assert golden.artifacts["langgraph"]["qa_index"] == 0
    assert golden.artifacts["langgraph"]["question_text"]


def test_golden_hash_chain(golden: TraceRun):
    from span_offload.harness import langgraph_env as env
    for s in golden.spans:
        step = s.output_contract_runtime.semantic_constraints["source_step"]
        records, h, _ = env.replay_memory_log(
            "conv-26", golden.artifacts["memory_delta_log"], up_to_step=step)
        assert h == s.state_snapshot.application_state_hash
        assert h  # non-empty (contains ingestion memories)
    for ref, snap in golden.artifacts["state_snapshots"].items():
        if ref.startswith("env:"):
            assert snap.get("checkpointed") is True
            assert snap.get("application_state_hash_is_placeholder") is None


def test_golden_eligibility_relaxable(golden: TraceRun):
    from span_offload.span_extraction import check_eligibility
    relaxable = {"baseline_run_failed_final_evaluator",
                 "baseline_goal_state_failed"}
    for s in golden.spans:
        reasons = set(check_eligibility(golden, s))
        assert reasons == relaxable, (s.invocation_index, reasons)


def test_golden_messages_complete_no_splice(golden: TraceRun):
    # collector-side trace_message keeps the tool_calls structure -> no splicing needed;
    # the first QA span's input = [system, question]; the answer span's input contains
    # assistant tool_calls (dict args) and tool receipts
    msgs0 = golden.artifacts["messages"]["msg:0"]
    assert [m["role"] for m in msgs0] == ["system", "user"]
    msgs1 = golden.artifacts["messages"]["msg:1"]
    assistant_msgs = [m for m in msgs1 if m.get("tool_calls")]
    assert assistant_msgs, "the answer span's input should contain the previous step's assistant tool_calls"
    for m in assistant_msgs:
        for tc in m["tool_calls"]:
            assert isinstance(tc.get("args"), dict)
            assert tc.get("id")
    tool_msgs = [m for m in msgs1 if m.get("role") == "tool"]
    for m in tool_msgs:
        assert m.get("tool_call_id")
    for s in golden.spans:
        sc = s.output_contract_runtime.semantic_constraints
        assert sc["splice_degraded"] is False


@pytest.mark.skipif(not os.path.exists(FULL_TRACE), reason="conv-30 not available locally")
def test_full_trace_unit_segmentation(adapter):
    """Production scale: conv-30 is split into 105 units by QA, each unit = ingestion +
    that QA; hash chain spot-checked; load_unit fetches on demand."""
    unit_ids = [tid for tid, _ in adapter.iter_units(FULL_TRACE)]
    assert len(unit_ids) == 105
    assert unit_ids[0] == "locomo:conv-30:q000"
    assert unit_ids[-1] == "locomo:conv-30:q104"
    # each unit = only that QA's spans (no ingestion duplication; QA-only decision)
    tr0 = adapter.load_unit(FULL_TRACE, unit_ids[0])
    phases0 = [s.output_contract_runtime.semantic_constraints["locomo_phase"]
               for s in tr0.spans]
    assert phases0 and set(phases0) == {"qa"}
    tr_last = adapter.load_unit(FULL_TRACE, unit_ids[-1])
    phases_last = [s.output_contract_runtime.semantic_constraints["locomo_phase"]
                   for s in tr_last.spans]
    assert phases_last and set(phases_last) == {"qa"}
    # the last unit contains only q104's spans (global question index consistent)
    assert all(s.output_contract_runtime.semantic_constraints["qa_index"] == 104
               for s in tr_last.spans)
    # per-unit hash chain spot check (first/middle/last span)
    from span_offload.harness import langgraph_env as env
    for tr in (tr0, tr_last):
        for j in (0, len(tr.spans) // 2, len(tr.spans) - 1):
            s = tr.spans[j]
            step = s.output_contract_runtime.semantic_constraints["source_step"]
            _, h, _ = env.replay_memory_log(
                "conv-30", tr.artifacts["memory_delta_log"], up_to_step=step)
            assert h == s.state_snapshot.application_state_hash, j
