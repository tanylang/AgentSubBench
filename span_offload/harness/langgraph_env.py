"""LoCoMo + LangGraph ReAct + LangMem environment bridge: memory-state hash/differential semantics, memory-chain replay, real agent build
elements.

Import strategy (collection side / offload side fully separated): this
module is the sole implementation and imports NO source/config from
TRACE_COLLECT (the server cannot obtain the collection side's deploy
package). Every collection-side basis element (system prompt / semantic
hash / diff / embedder / compact_history / deterministic metrics / LLM
judge) is a verbatim vendored copy, pinned to the collection side by the
golden-vector tests + judge-parameter comparison tests
(tests/test_langgraph_env.py).

State semantics (locomo_trace_core.py, must match verbatim): - the only mutable state = the
LangMem InMemoryStore record set (namespace=("locomo", <sample_id>, "memories")); the
conversation is compressed into extracted memories, no read positions -
application_state_hash = sha256(json.dumps([{namespace,key,value}], sort_keys=True,
separators=(",",":"))) -- created_at/updated_at excluded - env_changes: llm spans always [];
tool spans = the memory diff [{operation: create|update|delete, key, before?, after?}]
(before/after value only); a tool span's check_point is recorded after execute (same basis as
the locomo arm: execute first, verify after) - chain start = the empty-store hash (a legal
state, not a placeholder)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Dict, List, Tuple

from .locomo_env import (  # reuse: corpus loading and official scoring (same locomo10.json)
    LOCOMO_DATA_FILE, evaluate_answers, load_sample,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

DIMS = 384

#: the collection side's SYSTEM_PROMPT (verbatim from locomo_trace_core.py)
SYSTEM_PROMPT = """You are a memory agent processing a chronological conversation for later question answering.
During ingestion, store only durable facts, events, preferences, relationships, plans, and dated information that may matter later. Resolve relative dates using the supplied session date when possible. Use manage_memory to create, update, or delete memories and search_memory when existing memory may need checking. Do not role-play the speakers. After processing an ingestion message, reply briefly with ACK.
During QA, use the memory tools when relevant and answer only from stored memory. Give a concise answer. Never invent missing facts."""


# ---------------------------------------------------------------------------
# collection-side basis copies (verbatim identical to locomo_trace_core.py;
# sole implementation, pinned by the golden-vector tests)
# ---------------------------------------------------------------------------
def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    return repr(value)


def embed_one(text: str) -> List[float]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    features = list(words)
    for word in words:
        padded = f"^{word}$"
        features.extend(padded[index: index + 3]
                        for index in range(max(1, len(padded) - 2)))
    vector = [0.0] * DIMS
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        vector[number % DIMS] += 1.0 if ((number >> 9) & 1) == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_texts(texts: List[str]) -> List[List[float]]:
    return [embed_one(text) for text in texts]


def semantic_memory_hash(records: List[Dict[str, Any]]) -> str:
    semantic = [{"namespace": item["namespace"], "key": item["key"],
                 "value": item["value"]} for item in records]
    encoded = json.dumps(semantic, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def memory_diff(before: List[Dict[str, Any]],
                after: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    before_map = {item["key"]: item for item in before}
    after_map = {item["key"]: item for item in after}
    changes = []
    for key in sorted(before_map.keys() | after_map.keys()):
        if key not in before_map:
            changes.append({"operation": "create", "key": key,
                            "after": after_map[key]["value"]})
        elif key not in after_map:
            changes.append({"operation": "delete", "key": key,
                            "before": before_map[key]["value"]})
        elif before_map[key]["value"] != after_map[key]["value"]:
            changes.append({"operation": "update", "key": key,
                            "before": before_map[key]["value"],
                            "after": after_map[key]["value"]})
    return changes


def compact_history_messages(history: List[Dict[str, Any]]) -> List[Any]:
    """Prior Q/A continuity: carry only questions and predicted answers, no old tool payloads (verbatim from 10_run_locomo_sample.py; langchain message
    objects)."""
    from langchain_core.messages import AIMessage, HumanMessage
    messages: List[Any] = []
    for item in history:
        messages.append(HumanMessage(
            content=f"LoCoMo question {item['question_index']}:\n"
                    f"{item['question']}"))
        messages.append(AIMessage(content=item["predicted_answer"]))
    return messages


def question_prompt(question_index: int, question: str) -> str:
    """The user message text for a QA question (verbatim, the production basis of 10_run_locomo_sample.py)."""
    return (
        f"LoCoMo question {question_index}:\n{question}\n\n"
        "Answer using the memory tools when relevant. Do not assume "
        "that earlier model answers are correct; verify against memory."
    )


def question_prompt_core(question: str) -> str:
    """The user message text for a QA question (verbatim, the smoke-test
    basis of locomo_trace_core.main; used by single-question / no-history
    traces)."""
    return ("Answer this LoCoMo question using the memory tools "
            "when relevant:\n" + question)


def detect_qa_template(recorded_user_content: str) -> str:
    """Detect this trace's QA template basis from the recorded question message prefix."""
    if "Answer this LoCoMo question" in recorded_user_content:
        return "core"
    return "full"


def ingestion_turn_prompt(date, dia_id: str, speaker: str, text: str) -> str:
    """The ingestion single-turn message text (verbatim from locomo_trace_core.main; pinned by the golden-vector tests against the recorded
    messages)."""
    return (
        f"Session date and time: {date}\n"
        f"Dialogue ID: {dia_id}\n"
        f"Speaker: {speaker}\n"
        f"Message: {text}\n"
        "Process this message for long-term memory."
    )


def empty_store_hash() -> str:
    return semantic_memory_hash([])


# ---------------------------------------------------------------------------
# vendored verifier deterministic metrics (verbatim identical to trace_record runners/validate_locomot.py; authoritative import preferred, pinned by the golden-vector
# tests)
# ---------------------------------------------------------------------------
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
             "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
_NEG_WORDS = {"not", "no", "never", "wasnt", "didnt", "doesnt", "isnt",
              "arent", "couldnt", "wouldnt", "shouldnt", "without", "except"}


def _norm_numbers(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    def _tens_ones(m):
        return str(_TENS[m.group(1).lower()] + _ONES[m.group(2).lower()])
    text = re.sub(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
                  r"(one|two|three|four|five|six|seven|eight|nine)\b",
                  _tens_ones, text, flags=re.I)
    for w, v in {**_ONES, **_TENS}.items():
        text = re.sub(rf"\b{w}\b", str(v), text, flags=re.I)

    def _ord(m):
        return str(_ORDINALS.get(m.group(1).lower(), 0))
    text = re.sub(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b",
                  _ord, text, flags=re.I)
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", r"\1", text)
    return text


def _evaluation_module():
    """Lazy load of the official scorer module (_load_scorer mounts the task_eval directory)."""
    from .locomo_env import _load_scorer
    _load_scorer()
    import evaluation  # noqa: F401
    return evaluation


def _official_f1(pred: str, gold: str) -> float:
    from .locomo_env import _load_scorer
    return float(_load_scorer()([{
        "answer": str(gold), "category": 0, "prediction": str(pred),
    }], eval_key="prediction", metric="f1")[0][0])


def _num_f1(pred: str, gold: str) -> float:
    evaluation = _evaluation_module()
    return evaluation.f1_score(_norm_numbers(pred), _norm_numbers(gold))


def _stem_tokens(text: str) -> list:
    evaluation = _evaluation_module()
    from nltk.stem import PorterStemmer
    toks = evaluation.SimpleTokenizer().tokenize(
        evaluation.normalize_answer(text), uncased=True)
    return [PorterStemmer().stem(t) for t in toks]


def contains_score(pred: str, gold: str) -> float:
    """Whether the gold word sequence appears as a contiguous fragment in pred; None = suspected negation."""
    g = _stem_tokens(gold)
    p = _stem_tokens(pred)
    if not g or not p:
        return 0.0
    for i in range(len(p) - len(g) + 1):
        if p[i:i + len(g)] == g:
            window = set(p[max(0, i - 2):i])
            if window & _NEG_WORDS:
                return None
            return 1.0
    return 0.0


def rouge_l_f1(pred: str, gold: str) -> float:
    evaluation = _evaluation_module()
    p = evaluation.normalize_answer(pred).split()
    g = evaluation.normalize_answer(gold).split()
    if not p or not g:
        return 0.0
    dp = [[0] * (len(g) + 1) for _ in range(len(p) + 1)]
    for i in range(1, len(p) + 1):
        for j in range(1, len(g) + 1):
            dp[i][j] = (dp[i - 1][j - 1] + 1 if p[i - 1] == g[j - 1]
                        else max(dp[i - 1][j], dp[i][j - 1]))
    lcs = dp[len(p)][len(g)]
    prec, rec = lcs / len(p), lcs / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _shortcut_score(category: int, f1: float, contains, pred: str) -> float:
    """Deterministic shortcut (verbatim like validate_locomot._shortcut_score)."""
    if not str(pred).strip():
        return 0.0
    if category == 5:
        return float(f1)
    if contains == 1 and f1 >= 0.4:
        return 1.0
    return None


# ── LLM judge (the offload side's sole implementation, verbatim identical to the collection side's
# validate_locomot._judge_batch: system prompt / few-shot / temperature seed / thinking switch; call parameters
# pinned by test_judge_vendored_matches_authoritative) ────────────────────────── judge basis (offload side): JUDGE_*
# env vars > the judge section of the offload side's config.json (injected by run_langgraph_offload via
# set_judge_config) > built-in defaults. collection/offload fully separated: no function/config imported from
# TRACE_COLLECT.
_JUDGE_DEFAULTS = {
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com/v1",
}
_JUDGE_VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
_JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
_JUDGE_SEED = int(os.environ.get("JUDGE_SEED", "100"))
_JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))

#: the collection side's JUDGE_SYSTEM verbatim (validate_locomot.py)
_JUDGE_SYSTEM = """You are a strict but fair grader for the LoCoMo long-term conversational memory benchmark. You judge whether a model's answer to a question is semantically correct given the gold answer.

Rules:
- Score "correct" if the model's answer contains the correct information, even if it adds extra explanation, hedging, or a longer narrative. Exact wording, number formatting ("ten" vs "10"), or answer order do not matter.
- Score "incorrect" if the core fact is wrong, contradicts the gold answer, or the model answers something else.
- Score "partial" only when the answer is right but incomplete (e.g., gold has several sub-answers and the model gives only some).
- Beware of negation: if the model says "not X" while gold is "X", that is incorrect, even though it contains the words of the gold answer.
- For questions that ask about information NOT present in the conversation (adversarial), the correct behavior is to say the information is unavailable; score "correct" for such refusals and "incorrect" for making up an answer.
- Dates/names/numbers must be factually equivalent to the gold (approximate year-only vs exact date is "partial" at best; a wrong number is "incorrect").

Output strict JSON only: {"judgments": [{"q": <question index>, "verdict": "correct"|"partial"|"incorrect", "reason": "<one short sentence>"}]}"""

#: the collection side's _JUDGE_EXAMPLES verbatim (validate_locomot.py)
_JUDGE_EXAMPLES = [
    ("Question: When did Melanie paint a sunrise?\nGold: 2022\nAnswer: Melanie said she painted the lake sunrise last year, i.e., in 2022, since she mentioned it on 8 May, 2023.",
     '{"q": 0, "verdict": "correct", "reason": "Verbose but states 2022 correctly."}'),
    ("Question: Where did they go?\nGold: Paris\nAnswer: They did not go to Paris; they went to Berlin instead.",
     '{"q": 1, "verdict": "incorrect", "reason": "Negates the gold answer."}'),
    ("Question: What is the name of Sam's dog?\nGold: Max\nAnswer: Sam has a cat named Max and a dog named Rex.",
     '{"q": 2, "verdict": "incorrect", "reason": "Max is the cat, dog is Rex."}'),
    ("Question: When was the wedding?\nGold: 7 May 2023\nAnswer: In May 2023.",
     '{"q": 3, "verdict": "partial", "reason": "Right month/year but missing the day."}'),
    ("Question: What did they buy in May?\nGold: no information available\nAnswer: The conversation does not mention what they bought in May.",
     '{"q": 4, "verdict": "correct", "reason": "Correct refusal: no information available."}'),
]


def _parse_judge_json(content: str) -> Any:
    """Judge response parsing (compatibility hardening): after stripping code fences, take only the first complete
    JSON value -- with thinking on, the model occasionally appends text / a second object after the JSON (json.loads
    reports Extra data, and retries under a fixed seed produce nearly the same output); when there is leading thinking
    text,
    parse from the first '{' onward."""
    content = re.sub(r"^```(json)?|```$", "", content.strip()).strip()
    decoder = json.JSONDecoder()
    starts = [0]
    brace = content.find("{")
    if brace > 0:
        starts.append(brace)
    for start in starts:
        try:
            return decoder.raw_decode(content[start:])[0]
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no JSON object", content, 0)


def _judge_thinking_mode(model: str) -> Dict[str, Any]:
    """The judge thinking switch (same basis as the collection side's adapter/model_config.py: JUDGE_THINKING=on/off, default on; the switch body is chosen by model
    prefix)."""
    on = os.environ.get("JUDGE_THINKING", "on").lower() != "off"
    if on:
        if model.lower().startswith("qwen"):
            return {"enable_thinking": True}
        if model.lower().startswith("deepseek"):
            return {"thinking": {"type": "enabled"}}
        return {}
    if model.lower().startswith("deepseek"):
        return {"thinking": {"type": "disabled"}}
    if model.lower().startswith("qwen"):
        return {"enable_thinking": False}
    return {}

#: the offload-side config's judge section (injected by set_judge_config)
_OFFLOAD_JUDGE: Dict[str, str] = {}


def set_judge_config(cfg: Any) -> None:
    """Inject the offload-side config's judge section (called after load_config). Stays empty when no judge section is configured, falling back to env vars / built-in
    defaults."""
    _OFFLOAD_JUDGE.clear()
    j = getattr(cfg, "judge", None)
    if j is None:
        return
    _OFFLOAD_JUDGE["model"] = getattr(j, "model", "") or ""
    _OFFLOAD_JUDGE["base_url"] = getattr(j, "base_url", "") or ""
    _OFFLOAD_JUDGE["api_key"] = (j.resolved_api_key
                                 if hasattr(j, "resolved_api_key") else "")


def _resolve_judge() -> Dict[str, str]:
    """Judge model/endpoint/key (offload-side basis, env vars have top priority)."""
    return {
        "model": (os.environ.get("JUDGE_MODEL", "")
                  or _OFFLOAD_JUDGE.get("model", "")
                  or _JUDGE_DEFAULTS["model"]),
        "base_url": (os.environ.get("JUDGE_BASE_URL", "")
                     or _OFFLOAD_JUDGE.get("base_url", "")
                     or _JUDGE_DEFAULTS["base_url"]),
        "api_key": (os.environ.get("JUDGE_API_KEY", "")
                    or _OFFLOAD_JUDGE.get("api_key", "")),
    }


def _ensure_judge_key() -> None:
    """Judge key (offload-side basis): JUDGE_API_KEY > the offload-side config's judge section >
    DEEPSEEK_API_KEY/OPENAI_API_KEY. The collection side's $TRACE_COLLECT/config.json is no longer read (the server cannot
    obtain that file)."""
    key = _resolve_judge()["api_key"]
    if not key:
        key = (os.environ.get("DEEPSEEK_API_KEY", "")
               or os.environ.get("OPENAI_API_KEY", ""))
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)
        os.environ.setdefault("DEEPSEEK_API_KEY", key)


def _vendored_judge(items: List[Dict[str, Any]],
                    retries: int = 3) -> Dict[int, Dict[str, Any]]:
    """Batch judging (the offload side's sole implementation; verbatim identical to the collection side's validate_locomot._judge_batch: system prompt
    + 5 few-shot examples + fixed seed/temperature + thinking switch, the last attempt drops the thinking switch). Degrades to {} after failed
    retries."""
    try:
        import litellm
    except Exception as e:
        print("    [judge] litellm unavailable: %s: %s" % (
            type(e).__name__, str(e)[:120]), flush=True)
        return {}
    user_lines = ["Judge the following answers. For each, output the verdict JSON."
                  "\n\nExamples:\n"]
    for qa, out in _JUDGE_EXAMPLES:
        user_lines.append(f"{qa}\nExpected: {out}\n")
    user_lines.append("Now judge these (same JSON format, one entry per question):\n")
    for it in items:
        cat_note = (" (adversarial: asks about info possibly absent from the "
                    "conversation)" if it["category"] == 5 else "")
        user_lines.append(f'--- q={it["idx"]} category={it["category"]}{cat_note}\n'
                          f'Question: {it["question"]}\nGold: {it["gold"]}\n'
                          f'Answer: {it["pred"][:800]}\n')
    messages = [{"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": "\n".join(user_lines)}]
    j = _resolve_judge()
    for attempt in range(retries):
        # temperature/seed fixed (reproducible), thinking mode on by default (config-injected); the last
        # attempt drops the thinking switch (fallback when some gateways' json mode is incompatible with
        # thinking)
        extra = _judge_thinking_mode(j["model"]) if attempt < retries - 1 else None
        try:
            resp = litellm.completion(
                model=f"openai/{j['model']}", api_base=j["base_url"],
                messages=messages,
                temperature=_JUDGE_TEMPERATURE, seed=_JUDGE_SEED,
                max_completion_tokens=_JUDGE_MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body=extra)
            content = resp.choices[0].message.content or ""
            data = _parse_judge_json(content)
            raw = data.get("judgments")
            if raw is None and isinstance(data, dict) and "verdict" in data:
                raw = [data]  # the model occasionally emits a bare object (missing the
                # judgments wrapper; retries useless under a fixed seed, parse compatibly directly)
            if not raw:
                print("    [judge] empty judgment payload: %s" % content[:200],
                      flush=True)
            out = {}
            for jd in raw or []:
                v = jd.get("verdict", "")
                if v in _JUDGE_VERDICT_SCORE:
                    out[int(jd["q"])] = {
                        "verdict": v,
                        "reason": str(jd.get("reason", ""))[:200]}
            return out
        except Exception as e:
            print("    [judge] attempt %d/%d failed: %s: %s" % (
                attempt + 1, retries, type(e).__name__, str(e)[:120]),
                flush=True)
            import time as _time
            _time.sleep(10 * (attempt + 1))
    return {}


def _judge_one(item: Dict[str, Any]) -> Dict[str, Any]:
    """Judge one question: the offload-side vendored judge (sole implementation). Collection/offload fully separated; no collection-side determination function is
    imported."""
    _ensure_judge_key()
    if not os.environ.get("OPENAI_API_KEY") and \
            not os.environ.get("DEEPSEEK_API_KEY"):
        return {}  # no key: do not spin network retries
    return _vendored_judge([item])


def judge_selfcheck() -> None:
    """Startup self-check (no API call): print the judge model / endpoint / key / litellm availability, so run logs can directly locate why the judge is
    unavailable."""
    j = _resolve_judge()
    _ensure_judge_key()
    try:
        import litellm  # noqa: F401
        llm_ok = "litellm OK"
    except Exception as e:
        llm_ok = "litellm missing: %s: %s" % (type(e).__name__, str(e)[:80])
    key_ok = bool(os.environ.get("OPENAI_API_KEY")
                  or os.environ.get("DEEPSEEK_API_KEY"))
    print("[judge] selfcheck model=%s base_url=%s key=%s %s" % (
        j["model"], j["base_url"],
        "yes" if key_ok else "NO", llm_ok), flush=True)


def verify_qa_answer(question: str, gold: str, category: Any,
                     pred: str, use_llm: bool = True) -> Dict[str, Any]:
    """Single-question determination (same basis as validate_locomot.evaluate_trace): deterministic metrics -> the det shortcut settles it;
    ambiguous questions go to the authoritative judge when use_llm (failures degrade to metric-fallback); pass = score == 1.0 (strict
    basis)."""
    pred = str(pred or "")
    row: Dict[str, Any] = {"question": question, "gold": str(gold),
                           "category": int(category), "pred": pred[:600]}
    if not pred.strip():
        row.update({"f1": 0.0, "num_f1": 0.0, "contains": 0.0,
                    "rouge_l": 0.0, "score": 0.0, "source": "rule",
                    "verdict": "incorrect", "reason": "empty answer",
                    "pass": "fail"})
        return row
    try:
        f1 = _official_f1(pred, str(gold))
    except Exception:
        f1 = 0.0
    try:
        num_f1 = _num_f1(pred, str(gold))
        contains = contains_score(pred, str(gold))
        rouge = rouge_l_f1(pred, str(gold))
    except Exception:
        # when scorer/nltk dependencies are missing, degrade: determine by stripped exact equality only
        num_f1 = 0.0
        contains = None
        rouge = 0.0
        if pred.strip() == str(gold).strip():
            row.update({"f1": f1, "num_f1": num_f1, "contains": contains,
                        "rouge_l": rouge, "score": 1.0,
                        "source": "metric-fallback",
                        "verdict": "correct",
                        "reason": "scorer unavailable; exact match"})
            row["pass"] = "success"
            return row
    det = _shortcut_score(int(category), f1, contains, pred)
    row.update({"f1": round(f1, 4), "num_f1": round(num_f1, 4),
                "contains": contains, "rouge_l": round(rouge, 4),
                "det": det})
    if det is not None:
        row.update({"score": det, "source": "rule",
                    "verdict": ("correct" if det == 1.0 else
                                ("partial" if det == 0.5 else "incorrect")),
                    "reason": "deterministic shortcut"})
        row["pass"] = "success" if det == 1.0 else "fail"
        return row
    if use_llm:
        try:
            verdicts = _judge_one({
                "idx": 0, "category": int(category),
                "question": question, "gold": str(gold),
                "pred": pred[:800]})
            v = verdicts.get(0)
            if v is not None:
                score = float(_JUDGE_VERDICT_SCORE.get(v["verdict"], 0.0))
                row.update({"score": score, "source": "llm",
                            "verdict": v["verdict"],
                            "reason": str(v.get("reason", ""))[:200]})
                row["pass"] = "success" if score == 1.0 else "fail"
                return row
        except Exception:
            pass
    score = max(f1, num_f1, contains or 0.0)
    row.update({"score": score, "source": "metric-fallback",
                "verdict": ("correct" if score == 1.0 else
                            ("partial" if score == 0.5 else "incorrect")),
                "reason": "no judge; deterministic metrics"})
    row["pass"] = "success" if score == 1.0 else "fail"
    return row


# ---------------------------------------------------------------------------
# memory chain replay (the collection side's replay_env mode: tool entries that execute first and verify after)
# ---------------------------------------------------------------------------
def _apply_diff(records: List[Dict[str, Any]], namespace: List[str],
                changes: List[Dict[str, Any]]) -> None:
    by_key = {r["key"]: r for r in records}
    for change in changes or []:
        key = change["key"]
        op = change.get("operation")
        if op == "create":
            by_key[key] = {"namespace": list(namespace), "key": key,
                           "value": change.get("after")}
        elif op == "update":
            if key in by_key:
                by_key[key] = dict(by_key[key], value=change.get("after"))
        elif op == "delete":
            by_key.pop(key, None)
    records[:] = sorted(by_key.values(), key=lambda r: r["key"])


def replay_memory_log(sample_id: str, delta_log: List[Dict[str, Any]],
                      up_to_step: int
                      ) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    """Replay the memory deltas with step < up_to_step to rebuild the record set, verifying the hash chain entry by
    entry.

    Semantics same as the locomo arm: llm entries verify first (env_changes always empty); tool entries apply the diff first, then
    verify (the checkpoint is recorded after execute). Returns (records, final hash, notes). Tampering/drift raises RuntimeError
    honestly.
    """
    namespace = ["locomo", sample_id, "memories"]
    records: List[Dict[str, Any]] = []
    notes: List[str] = []
    for entry in delta_log:
        step = int(entry.get("step", 0) or 0)
        if step >= up_to_step:
            break
        expected = str(entry.get("hash_before", "") or "")
        if entry.get("type") == "tool.execution":
            _apply_diff(records, namespace, entry.get("env_changes") or [])
            if expected and expected != semantic_memory_hash(records):
                raise RuntimeError(
                    "langgraph restore chain mismatch at step %d: recorded "
                    "%s != applied %s" % (
                        step, expected[:12],
                        semantic_memory_hash(records)[:12]))
        else:
            if expected and expected != semantic_memory_hash(records):
                raise RuntimeError(
                    "langgraph restore chain mismatch at step %d: recorded "
                    "%s != applied %s" % (
                        step, expected[:12],
                        semantic_memory_hash(records)[:12]))
    final = semantic_memory_hash(records)
    notes.append("replayed %d entries, final hash %s" % (
        len([e for e in delta_log
             if int(e.get("step", 0) or 0) < up_to_step]), final[:12]))
    return records, final, notes


def rebuild_store(records: List[Dict[str, Any]], namespace: List[str]):
    """Rebuild the real InMemoryStore from the record set (embedder = the collection side's deterministic hash vectorizer; required for the search_memory
    continuation)."""
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore(index={"dims": DIMS, "embed": embed_texts})
    for record in records:
        store.put(tuple(namespace), record["key"], record["value"])
    return store
