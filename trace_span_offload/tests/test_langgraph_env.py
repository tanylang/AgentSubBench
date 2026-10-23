#!/usr/bin/env python
"""Golden-vector tests for langgraph_env: memory hash/diff/embedder cross-checked
against the collector-side authoritative implementation, memory-chain replay,
store rebuild, and question prompt verbatim-identical to the record.

Run:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_langgraph_env.py -v
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("LOCOMO_DATA_FILE",
                      os.path.join(ROOT, "data", "locomo", "locomo10.json"))
from span_offload.harness import langgraph_env as env  # noqa: E402

TRACE_ROOT = "C:/Users/Lenovo/Desktop/trace_record/trace_collect/langgraph_locomo"
GOLDEN_TRACE = os.path.join(
    TRACE_ROOT, "traces", "schema_smoke", "conv-26", "q000.json")


@pytest.fixture(scope="module")
def golden_trace_run():
    from span_offload.langgraph_adapter import LangGraphTraceAdapter
    from span_offload.trace_io import load_manifest
    m = load_manifest(os.path.join(ROOT, "data", "locomo", "task_manifest.json"))
    return LangGraphTraceAdapter(m).load(GOLDEN_TRACE)


def test_semantic_hash_golden_vectors():
    empty = env.semantic_memory_hash([])
    # sha256("[]") (json.dumps of an empty semantic list, sort_keys + compact separators)
    assert empty == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    # created_at/updated_at do not affect the hash; sorted keys are significant
    r1 = [{"namespace": ["locomo", "conv-26", "memories"], "key": "k1",
           "value": {"content": "fact A"}, "created_at": "2026-01-01T00:00:00",
           "updated_at": "2026-01-01T00:00:00"}]
    r2 = [{"namespace": ["locomo", "conv-26", "memories"], "key": "k1",
           "value": {"content": "fact A"}}]
    assert env.semantic_memory_hash(r1) == env.semantic_memory_hash(r2)
    # Cross-validation: verbatim-identical to the collector-side authoritative implementation
    scripts = os.path.join(TRACE_ROOT, "scripts")
    sys.path.insert(0, scripts)
    from locomo_trace_core import (  # noqa: F401
        embed_texts as auth_embed,
        memory_diff as auth_diff,
        semantic_memory_hash as auth_hash,
    )
    assert env.semantic_memory_hash(r1) == auth_hash(r2)
    assert env.memory_diff([], r2) == auth_diff([], r2)
    v1 = env.embed_texts(["hello world"] * 2)
    v2 = auth_embed(["hello world"] * 2)
    assert v1 == v2
    # unit vector
    import math
    norm = math.sqrt(sum(x * x for x in v1[0]))
    assert abs(norm - 1.0) < 1e-9


def test_memory_diff_semantics():
    before = [{"namespace": ["locomo", "x", "memories"], "key": "a",
               "value": {"content": "v1"}},
              {"namespace": ["locomo", "x", "memories"], "key": "b",
               "value": {"content": "v2"}}]
    after = [{"namespace": ["locomo", "x", "memories"], "key": "b",
              "value": {"content": "v2b"}},
             {"namespace": ["locomo", "x", "memories"], "key": "c",
              "value": {"content": "v3"}}]
    changes = env.memory_diff(before, after)
    assert changes == [
        {"operation": "delete", "key": "a", "before": {"content": "v1"}},
        {"operation": "update", "key": "b", "before": {"content": "v2"},
         "after": {"content": "v2b"}},
        {"operation": "create", "key": "c", "after": {"content": "v3"}},
    ]


def test_replay_memory_log_chain(golden_trace_run):
    tr = golden_trace_run
    log = tr.artifacts["memory_delta_log"]
    for s in tr.spans:
        step = s.output_contract_runtime.semantic_constraints["source_step"]
        records, h, _ = env.replay_memory_log(
            "conv-26", log, up_to_step=step)
        assert h == s.state_snapshot.application_state_hash
    # Full-chain replay: final hash = recorded hash of the last span
    records, final, _ = env.replay_memory_log("conv-26", log, up_to_step=10**9)
    assert final == tr.spans[-1].state_snapshot.application_state_hash
    # tampering raises an honest error
    bad = [dict(e) for e in log]
    bad[3] = dict(bad[3], hash_before="00" * 32)
    with pytest.raises(RuntimeError):
        env.replay_memory_log("conv-26", bad, up_to_step=10**9)


def test_rebuild_store(golden_trace_run):
    tr = golden_trace_run
    log = tr.artifacts["memory_delta_log"]
    records, _, _ = env.replay_memory_log("conv-26", log, up_to_step=10**9)
    store = env.rebuild_store(records, ["locomo", "conv-26", "memories"])
    items = list(store.search(("locomo", "conv-26", "memories"), limit=10000))
    assert len(items) == len(records)
    assert len(records) > 0  # memories were written during the ingestion phase


def test_verifier_deterministic_metrics():
    """Golden vectors for deterministic metrics + cross-validation against the authoritative validate_locomot."""
    # number normalization
    assert env._num_f1("ten", "10") > 0.9
    assert env._num_f1("10th May", "10 May") > 0.9  # ordinal -> number
    assert env._num_f1("10th of May", "10 May") > 0.7  # extra word "of" legitimately lowers the score
    # containment match (contiguous fragment) + negation guard
    assert env.contains_score(
        "she went on 7 May 2023 to the group", "7 May 2023") == 1.0
    assert env.contains_score(
        "not on 7 May 2023", "7 May 2023") is None  # negation word within a 2-word window
    assert env.contains_score("totally different", "7 May 2023") == 0.0
    # rouge_l (word-order sensitive)
    assert env.rouge_l_f1("in 2022", "2022") > 0.5
    # shortcuts: empty answer -> 0; cat5 -> f1; strong hit -> 1
    assert env._shortcut_score(2, 0.1, None, "  ") == 0.0
    assert env._shortcut_score(5, 0.8, None, "no information available") == 0.8
    assert env._shortcut_score(2, 0.6, 1.0, "x") == 1.0
    # cross-validation against the authoritative module
    runners = os.path.join(TRACE_ROOT, "..", "runners")
    runners = os.path.abspath(runners)
    if os.path.isdir(runners):
        sys.path.insert(0, runners)
        from validate_locomot import (  # noqa: F401
            _num_f1 as a_num, contains_score as a_contains,
            rouge_l_f1 as a_rouge,
        )
        cases = [("Melanie painted in 2022", "2022"),
                 ("not in 2022", "2022"),
                 ("went on the 10th", "went on 10 May")]
        for pred, gold in cases:
            assert env._num_f1(pred, gold) == a_num(pred, gold)
            assert env.contains_score(pred, gold) == a_contains(pred, gold)
            assert env.rouge_l_f1(pred, gold) == a_rouge(pred, gold)


def test_question_prompt_matches_recorded(golden_trace_run):
    """The question prompt is verbatim-identical to the question messages recorded in the trace (smoke test = core flavor)."""
    tr = golden_trace_run
    recorded = json.load(open(GOLDEN_TRACE, encoding="utf-8"))
    sample_id = tr.artifacts["langgraph"]["sample_id"]
    sample = json.load(open(
        os.path.join(ROOT, "data", "locomo", "locomo10.json"),
        encoding="utf-8"))
    qa = [s for s in sample if s["sample_id"] == sample_id][0]["qa"]
    # smoke-test single question: core flavor (production traces use the full flavor)
    prompt = env.question_prompt_core(qa[0]["question"])
    found = False
    for s in recorded["spans"]:
        if s.get("type") != "llm.call":
            continue
        for m in (s.get("check_point") or {}).get("conversation", {}).get(
                "messages", []):
            if m.get("role") == "user" and prompt in str(m.get("content", "")):
                found = True
    assert found, "vendored question_prompt_core does not match the recorded question messages"
    # template detection
    assert env.detect_qa_template(prompt) == "core"
    assert env.detect_qa_template(
        env.question_prompt(0, qa[0]["question"])) == "full"


def test_judge_resolution_offload(monkeypatch):
    """Judge flavor (offload side): JUDGE_* env > offload-side config judge section > built-in defaults;
    no longer depends on the collector-side $TRACE_COLLECT/config.json."""
    from span_offload.config import ModelConfig, Stage10Config
    for k in ("JUDGE_MODEL", "JUDGE_BASE_URL", "JUDGE_API_KEY",
              "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    # nothing injected: built-in defaults
    env.set_judge_config(Stage10Config())
    assert env._resolve_judge()["model"] == "deepseek-v4-flash"
    assert env._resolve_judge()["base_url"] == "https://api.deepseek.com/v1"
    # offload-side config judge section
    cfg = Stage10Config()
    cfg.judge = ModelConfig(model="m-off", base_url="https://off.example/v1",
                            api_key="k-off")
    env.set_judge_config(cfg)
    assert env._resolve_judge() == {
        "model": "m-off", "base_url": "https://off.example/v1",
        "api_key": "k-off"}
    # environment variables take the highest precedence
    monkeypatch.setenv("JUDGE_MODEL", "m-env")
    assert env._resolve_judge()["model"] == "m-env"
    monkeypatch.delenv("JUDGE_MODEL")
    # inject the key into the litellm environment (does not override explicit settings)
    env._ensure_judge_key()
    assert os.environ["OPENAI_API_KEY"] == "k-off"
    assert os.environ["DEEPSEEK_API_KEY"] == "k-off"
    env.set_judge_config(Stage10Config())


def test_judge_json_parse_robust():
    """Judge response parsing compatibility: trailing content after JSON (sporadic when thinking is on), code
    fences, leading reasoning text, bare objects."""
    j = '{"judgments": [{"q": 0, "verdict": "correct"}]}'
    assert env._parse_judge_json(j)["judgments"][0]["verdict"] == "correct"
    # Extra data: trailing junk after JSON (observed failure: line 2 column 1)
    assert env._parse_judge_json(
        j + "\n(extra)")["judgments"][0]["verdict"] == "correct"
    # code fence
    assert env._parse_judge_json(
        "```json\n" + j + "\n```")["judgments"][0]["verdict"] == "correct"
    # leading reasoning text
    assert env._parse_judge_json(
        "reasoning...\n" + j)["judgments"][0]["verdict"] == "correct"
    # bare object (missing the judgments wrapper)
    assert env._parse_judge_json(
        '{"q": 0, "verdict": "partial"}')["verdict"] == "partial"


def test_judge_vendored_matches_authoritative(monkeypatch):
    """The vendored judge's call parameters are verbatim-identical to the collector-side
    validate_locomot._judge_batch (system/few-shot/temperature/seed/thinking flags; litellm is intercepted offline,
    no real API calls)."""
    import types
    litellm = pytest.importorskip("litellm")
    runners = os.path.abspath(os.path.join(TRACE_ROOT, "..", "runners"))
    if not os.path.isdir(runners):
        pytest.skip("collector-side runners unavailable")
    sys.path.insert(0, runners)
    from validate_locomot import _judge_batch as auth_judge_batch
    captured = []

    def _fake_completion(**kwargs):
        captured.append(kwargs)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content='{"judgments": '
                '[{"q": 0, "verdict": "correct", "reason": "ok"}]}'))])

    monkeypatch.setattr(litellm, "completion", _fake_completion)
    item = {"idx": 0, "category": 5,
            "question": "What did they buy in May?",
            "gold": "no information available",
            "pred": "The conversation does not mention what they bought in May."}
    env._vendored_judge([item])
    auth_judge_batch([item], model="deepseek-v4-flash",
                     base_url="https://api.deepseek.com/v1")
    assert len(captured) == 2
    assert captured[0] == captured[1]
