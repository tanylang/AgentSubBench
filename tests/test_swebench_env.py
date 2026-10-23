#!/usr/bin/env python
"""Golden-vector tests for swebench_env: cross-comparison against trace_record's authoritative implementation.

The serialization rules for state_hash/chained accumulation must be verbatim-identical to the collector side (zero drift in hash semantics),
otherwise restore's hash-chain verification will inevitably mismatch. Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_swebench_env.py -v
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.harness import swebench_env as env  # noqa: E402

TRACE_RECORD_ROOT = os.environ.get(
    "TRACE_RECORD_ROOT", "C:/Users/Lenovo/Desktop/trace_record")


@pytest.fixture(scope="module")
def auth():
    """trace_record's authoritative implementation (package import; on the server use SWE_COLLECT_ROOT)."""
    if env._authoritative is not None:
        return env._authoritative
    sys.path.insert(0, TRACE_RECORD_ROOT)
    from adapter import swebench as mod  # noqa: E402
    return mod


CASES = [
    {},
    {"a.py": "print(1)\n"},
    {"a.py": "x", "b/c.py": "y", "d": None},
    {"__skipped__: big.bin": "9999999 bytes"},
    {"同一": "中文内容", "文件": None},
]


def test_state_hash_matches_authoritative(auth):
    for case in CASES:
        assert env.state_hash(case) == auth.state_hash(case), case


def test_chain_update_matches_authoritative(auth):
    """Chained accumulation: json.dumps(dict, ensure_ascii=False, default=str), insertion-order sensitive."""
    seq = [{"x.py": "a"}, {"y.py": None}, {}, {"x.py": "b", "z": "c"}]
    for s in seq:
        r1, r2 = hashlib.sha256(), hashlib.sha256()
        env._chain_update(r1, s)
        # the authoritative implementation's chained update is embedded in the provider closure;
        # here we replicate its _chain_update logic verbatim for cross-validation
        if s:
            r2.update(json.dumps(s, ensure_ascii=False, default=str)
                      .encode("utf-8"))
        assert r1.hexdigest() == r2.hexdigest()


def test_chain_insertion_order_sensitive():
    r1, r2 = hashlib.sha256(), hashlib.sha256()
    env._chain_update(r1, {"a": "1", "b": "2"})
    env._chain_update(r2, {"b": "2", "a": "1"})
    assert r1.hexdigest() != r2.hexdigest()  # same sensitivity as the collector side


def test_apply_env_changes_roundtrip(tmp_path):
    env.apply_env_changes(tmp_path, {
        "a/b.py": "content1",
        "c.txt": "content2",
    })
    assert (tmp_path / "a" / "b.py").read_text(encoding="utf-8") == "content1"
    assert (tmp_path / "c.txt").read_text(encoding="utf-8") == "content2"
    # deletion
    env.apply_env_changes(tmp_path, {"c.txt": None})
    assert not (tmp_path / "c.txt").exists()
    # __skipped__ entries are skipped
    env.apply_env_changes(tmp_path, {"__skipped__: huge": "1 bytes"})
    assert not (tmp_path / "__skipped__: huge").exists()


def test_restore_deltas_chain_verification(tmp_path):
    """restore_deltas: apply in order + verify the next hash_before (mismatch raises RuntimeError).

    Chain semantics: entry k's hash_before = the accumulated hash after applying entries 0..k-1's deltas.
    """
    h0 = hashlib.sha256().hexdigest()          # chain start (empty)
    r = hashlib.sha256()
    env._chain_update(r, {"f.py": "v1"})
    h1 = r.hexdigest()                          # accumulation after applying entry 1's delta
    log = [
        {"step": 1, "type": "llm.call", "hash_before": h0,
         "env_changes": {}},                     # entry 0: no delta
        {"step": 2, "type": "tool.execution", "hash_before": h0,
         "env_changes": {"f.py": "v1"}},         # entry 1: hash_before = state after entry 0 = h0
    ]
    final, notes = env.restore_deltas(tmp_path, log, up_to_index=1)
    assert final == h0
    assert not (tmp_path / "f.py").exists()
    final, notes = env.restore_deltas(tmp_path, log, up_to_index=2)
    assert final == h1
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "v1"
    # inconsistent chain -> error
    bad = [dict(log[0]), {"step": 2, "type": "tool.execution",
                          "hash_before": "deadbeef", "env_changes": {}}]
    with pytest.raises(RuntimeError):
        env.restore_deltas(tmp_path, bad, up_to_index=2)
