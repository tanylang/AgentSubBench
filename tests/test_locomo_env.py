#!/usr/bin/env python
"""Golden-vector tests for locomo_env: hash semantics cross-validated against
the collector-side authoritative implementation, hash-chain replay, byte-exact
tool re-execution reproduction, and official scoring wiring.

Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_locomo_env.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# vendored copy (LOCOMO_DATA_FILE points at the in-repo data)
os.environ.setdefault("LOCOMO_DATA_FILE",
                      os.path.join(ROOT, "data", "locomo", "locomo10.json"))
from span_offload.harness import locomo_env as env  # noqa: E402

TRACE_COLLECT = "C:/Users/Lenovo/Desktop/trace_record/trace_collect"
GOLDEN_TRACE = os.path.join(
    ROOT, "check_point_traces", "locomo_simplified_qwen3_8-27b_batch1",
    "conv-26.json")


@pytest.fixture(scope="module")
def sample():
    return env.load_sample("conv-26")


@pytest.fixture(scope="module")
def golden_trace_run():
    from span_offload.locomo_adapter import LocomoTraceAdapter
    from span_offload.trace_io import load_manifest
    m = load_manifest(os.path.join(ROOT, "data", "locomo", "task_manifest.json"))
    return LocomoTraceAdapter(m).load(GOLDEN_TRACE)


def test_memory_hash_semantics(sample):
    mem = env.new_memory(sample)
    assert mem.state_hash() == hashlib.sha256().hexdigest()
    # an empty delta does not advance
    mem.mark_read([])
    assert mem.state_hash() == hashlib.sha256().hexdigest()
    # sorted serialization advances
    mem.mark_read(["D3:2", "D1:1"])
    h1 = mem.state_hash()
    mem.mark_read(["D1:1", "D3:2"])  # already read -> no advance
    assert mem.state_hash() == h1
    mem2 = env.new_memory(sample)
    mem2.mark_read(["D1:1", "D3:2"])  # order-independent (identical after sorting)
    assert mem2.state_hash() == h1
    mem.mark_read(["D2:1"])
    h2 = mem.state_hash()
    assert h2 != h1
    # cross-validation against the collector-side authoritative implementation (when TRACE_COLLECT exists): same mark_read sequence
    if os.path.isdir(TRACE_COLLECT):
        sys.path.insert(0, TRACE_COLLECT)
        from adapter import locomot as auth  # noqa: F401
        auth_mem = auth.LocomoMemory(None)
        auth_mem.mark_read(["D3:2", "D1:1"])
        auth_mem.mark_read(["D2:1"])
        assert auth_mem.state_hash() == h2


def test_replay_span_log_chain(sample, golden_trace_run):
    tr = golden_trace_run
    log = tr.artifacts["span_log"]
    # full-chain replay: final hash = the last span's recorded hash (fdef8e...)
    mem, final, notes = env.replay_span_log(
        sample, log, up_to_step=10**9)
    assert final == tr.spans[-1].state_snapshot.application_state_hash
    # replay up to each target span: compared against the recorded hash
    for s in tr.spans:
        step = int(s.output_contract_runtime.semantic_constraints["source_step"])
        _, h, _ = env.replay_span_log(sample, log, up_to_step=step)
        assert h == s.state_snapshot.application_state_hash, (
            s.invocation_index, h[:12])
    # a tampered entry raises an honest error
    bad = [dict(e) for e in log]
    bad[5] = dict(bad[5], hash_before="00" * 32)
    with pytest.raises(RuntimeError):
        env.replay_span_log(sample, bad, up_to_step=10**9)


def test_tool_reexecution_reproduces_records(sample, golden_trace_run):
    tr = golden_trace_run
    log = tr.artifacts["span_log"]
    mem, _, _ = env.replay_span_log(sample, log, up_to_step=10**9)
    # re-execute each tool: results byte-identical to the records (including the TypeError quirk)
    n_checked = 0
    for entry in log:
        if entry.get("type") != "tool.execution":
            continue
        tool = entry["tool"]
        # replay a single entry with independent memory (no dependence on read-set advance order)
        mem2, _, _ = env.replay_span_log(sample, log, up_to_step=entry["step"])
        result = env.execute_tool(
            tool["name"], dict(tool["arguments"]), sample, mem2)
        # the recorded result was truncated at 5000 chars on the collector side (log_tool flavor)
        assert result[:5000] == tool["result"], tool["name"]
        assert (not result.startswith('{"error"')) == bool(tool["ok"])
        n_checked += 1
    assert n_checked == 15


def test_search_messages_typeerror_quirk(sample):
    # collector quirk: search_messages without person_id -> TypeError -> success=False
    mem = env.new_memory(sample)
    result = env.execute_tool("search_messages", {"query": "Melanie"},
                              sample, mem)
    assert result.startswith('{"error": "TypeError:')
    assert mem.state_hash() == hashlib.sha256().hexdigest()  # failure does not advance


def test_evaluate_answers_reproduces_official(sample, golden_trace_run):
    tr = golden_trace_run
    loco = tr.artifacts["locomo"]
    ev = env.evaluate_answers(
        sample, loco["answers"], set(loco.get("read_ids") or []))
    assert ev["f1"] == pytest.approx(0.2843, abs=1e-4)
    assert ev["success"] is False
    assert ev["num_questions"] == 3
    assert len(ev["per_question"]) == 3
    # the official scorer is usable (lazy bert_score change effective: no torch, no crash)
    scorer = env._load_scorer()
    assert callable(scorer)


def test_system_prompt_template_matches_recorded(sample, golden_trace_run):
    """The replay side's real build function is verbatim-identical to the system prompt recorded on the collector side."""
    tr = golden_trace_run
    rebuilt = env.build_system_prompt(sample)
    for j in range(len(tr.spans)):
        recorded = tr.artifacts["system_prompts"]["sys:%d" % j]
        assert rebuilt == recorded, (
            "system prompt template drift at span %d" % j)
