#!/usr/bin/env python
"""Unit tests for local_equivalence: timestamp exemption in structured-value comparison + tool execution
result comparison (2026-09-22 fix: created_at/updated_at differences between replay time and collection time
must not be judged as inconsistent tool execution results).

Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_local_equivalence.py -v
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.local_equivalence import (  # noqa: E402
    compare_execution_results,
    compare_values,
)


def _search_result(content: str, created_at: str) -> str:
    """Simulate a search_memory execution result (memory records carry timestamps)."""
    return json.dumps([{
        "namespace": ["locomo", "conv-26", "memories"],
        "key": "k1",
        "value": {"content": content},
        "created_at": created_at,
        "updated_at": created_at,
    }])


def test_compare_values_skip_keys():
    a = _search_result("fact A", "2026-01-01T00:00:00")
    b = _search_result("fact A", "2026-09-22T10:00:00")
    # not exempted: timestamp difference -> inconsistent (old behavior)
    ok, det = compare_values(a, b)
    assert not ok
    # timestamp exempted: consistent (post-fix behavior)
    ok, det = compare_values(
        a, b, skip_keys={"created_at", "updated_at"})
    assert ok, det
    # real value differences must still be caught (the exemption does not widen)
    c = _search_result("fact B", "2026-09-22T10:00:00")
    ok, det = compare_values(
        a, c, skip_keys={"created_at", "updated_at"})
    assert not ok
    # the dict key-set check likewise exempts the skip keys (a one-sided missing timestamp key is not a false positive)
    d = json.dumps({"value": {"content": "fact A"},
                    "created_at": "2026-01-01T00:00:00"})
    e = json.dumps({"value": {"content": "fact A"}})
    ok, det = compare_values(d, e, skip_keys={"created_at"})
    assert ok, det


def test_compare_execution_results_timestamps():
    """I-arm scenario: same tool calls and execution results, only timestamps differ -> passed."""
    base = [{
        "tool": "search_memory",
        "ok": True,
        "result": _search_result("fact A", "2026-01-01T00:00:00"),
    }]
    slm = [{
        "tool": "search_memory",
        "ok": True,
        "result": _search_result("fact A", "2026-09-22T10:00:00"),
    }]
    out = compare_execution_results(slm, base)
    assert out["passed"], out
    # real content difference: still judged inconsistent
    slm2 = [{
        "tool": "search_memory",
        "ok": True,
        "result": _search_result("fact B", "2026-09-22T10:00:00"),
    }]
    out2 = compare_execution_results(slm2, base)
    assert not out2["passed"]


def test_compare_execution_results_truncated():
    """Truncation-asymmetry rescue: collector-side full JSON vs replay-side 2000-char truncation
    (TRACE_RESULT_MAX_LEN), content and timestamps identical outside the window -> passed."""
    base_full = _search_result("x" * 5000, "2026-01-01T00:00:00")
    slm_trunc = _search_result("x" * 5000, "2026-09-22T10:00:00")[:2000]
    base = [{"tool": "search_memory", "ok": True, "result": base_full}]
    slm = [{"tool": "search_memory", "ok": True, "result": slm_trunc}]
    out = compare_execution_results(slm, base)
    assert out["passed"], out
    # real content difference within the window: still judged inconsistent
    slm_bad = [{"tool": "search_memory", "ok": True,
                "result": _search_result("y" * 5000,
                                         "2026-09-22T10:00:00")[:2000]}]
    out2 = compare_execution_results(slm_bad, base)
    assert not out2["passed"]
