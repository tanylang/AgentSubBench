"""LocoMo environment bridge: corpus loading / memory read-set hash-chain restore / official F1 evaluation.

Import strategy (modeled on swebench_env.py): when TRACE_COLLECT (default /share/home/tanylang/trace_collect) is
importable, prefer the collection side's own source (`from adapter import locomot`), zero hash/tool-semantics drift; when
not importable, use this module's vendored copy (verbatim identical to the collection side, backed by the golden-vector
tests).

Hash semantics (trace_record adapter/locomot.py, must match verbatim): -
LocomoMemory.mark_read: merge new dia_ids (sorted serialization) and advance the sha256 chain -
the chain advances only when the retrieval tool calls mark_read (the only "environment state"
in this read-only baseline) - application_state_hash: llm.call = the accumulated hash before
the call; tool.execution = the accumulated hash after execution (the checkpoint is recorded at
log_tool time, later than execute) - chain start = e3b0..., a legal empty state, not a
placeholder - env_changes always an empty dict (a pure read-only baseline, no replayable
environment delta)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

#: trace_record deploy package root (server): prefer importing the authoritative implementation from it
TRACE_COLLECT = os.environ.get(
    "TRACE_COLLECT", "/share/home/tanylang/trace_collect")
#: this repo's locomo10.json (vendored path; overridable via LOCOMO_DATA_FILE)
REPO_ROOT = Path(__file__).parent.parent.parent
LOCOMO_DATA_FILE = os.environ.get(
    "LOCOMO_DATA_FILE", str(REPO_ROOT / "data" / "locomo" / "locomo10.json"))
PASS_THRESHOLD = 0.5

#: question boundary: "Question i/n: ..." (the collect side's build_question_prompt basis)
QUESTION_RE = re.compile(r"Question (\d+)/(\d+):")

_authoritative: Optional[Any] = None
_common_mod: Optional[Any] = None
if os.path.isdir(TRACE_COLLECT):
    _adapter_dir = os.path.join(TRACE_COLLECT, "adapter")
    if os.path.isdir(_adapter_dir):
        try:
            sys.path.insert(0, TRACE_COLLECT)
            from adapter import locomot as _authoritative  # type: ignore
        except Exception:
            _authoritative = None
    # authoritative locomot_common: the real source for agent prompt build / question construction /
    # the whole-question loop (the replay side never re-implements the agent's internal logic; it only
    # keeps the minimal "mid-question resume" extension that the collection side does not have). On
    # import failure (missing deps etc.) fall back to the vendored copy, backed by the golden-vector
    # tests.
    _scripts_dir = os.path.join(TRACE_COLLECT, "scripts")
    if os.path.isdir(_scripts_dir):
        try:
            if _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            from locomot_common import (  # type: ignore  # noqa: F401
                build_system_prompt as _auth_build_system_prompt,
                build_question_prompt as _auth_build_question_prompt,
                run_question_loop as _auth_run_question_loop,
                MAX_TOOL_STEPS as _auth_max_tool_steps,
            )
            _common_mod = True
        except Exception:
            _common_mod = None


# ---------------------------------------------------------------------------
# vendored copies (verbatim identical to trace_record adapter/locomot.py; the only difference: DATA_FILE supports a LOCOMO_DATA_FILE override and defaults to this repo's
# data/)
# ---------------------------------------------------------------------------
DATE_FORMAT = "%I:%M %p on %d %B, %Y"
RESULT_CHAR_CAP = 50_000


@dataclass
class LocomoTurn:
    dia_id: str
    speaker: str
    text: str
    session_key: str
    session_date: str
    img_url: str = ""
    blip_caption: str = ""


@dataclass
class LocomoQA:
    question: str
    answer: str
    evidence: list
    category: Any


@dataclass
class LocomoSample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list
    sessions: dict = field(default_factory=dict)
    qas: list = field(default_factory=list)
    event_summary: dict = field(default_factory=dict)
    session_summary: dict = field(default_factory=dict)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def num_turns(self) -> int:
        return len(self.turns)


def load_locomo_samples(limit: int = 0,
                        sample_ids: List[str] | None = None) -> List[LocomoSample]:
    """Load locomo10.json (limit=0 means all; sample_ids filters, keeping file order)."""
    data_file = Path(LOCOMO_DATA_FILE)
    if not data_file.exists():
        raise RuntimeError(
            "LocoMo data missing: %s -- please confirm data/locomo/locomo10.json "
            "exists or set LOCOMO_DATA_FILE" % data_file)
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    if sample_ids:
        raw = [s for s in raw if s.get("sample_id") in set(sample_ids)]
    if limit:
        raw = raw[:limit]
    samples: List[LocomoSample] = []
    for s in raw:
        conv = s["conversation"]
        turns: List[LocomoTurn] = []
        sessions: Dict[str, List[LocomoTurn]] = {}
        for key, value in conv.items():
            if not (key.startswith("session_") and isinstance(value, list)):
                continue
            date = conv.get(key + "_date_time", "")
            tlist: List[LocomoTurn] = []
            for t in value:
                turn = LocomoTurn(
                    dia_id=t.get("dia_id", ""),
                    speaker=t.get("speaker", ""),
                    text=t.get("text", ""),
                    session_key=key,
                    session_date=date,
                    img_url=", ".join(t.get("img_url") or []),
                    blip_caption=t.get("blip_caption", ""),
                )
                tlist.append(turn)
                turns.append(turn)
            sessions[key] = tlist
        qas = [LocomoQA(question=q.get("question", ""),
                        answer=q.get("answer", ""),
                        evidence=list(q.get("evidence") or []),
                        category=q.get("category"))
               for q in s.get("qa", [])]
        samples.append(LocomoSample(
            sample_id=s["sample_id"],
            speaker_a=conv.get("speaker_a", ""),
            speaker_b=conv.get("speaker_b", ""),
            turns=turns,
            sessions=sessions,
            qas=qas,
            event_summary=s.get("event_summary", {}),
            session_summary=s.get("session_summary", {}),
        ))
    return samples


class LocomoMemory:
    """Retrieval history (the read message set) -- the only "environment state" in this read-only baseline."""

    def __init__(self, sample: Any):
        self.sample = sample
        self.read_dia_ids: set = set()
        self._chain = hashlib.sha256()

    def mark_read(self, dia_ids) -> None:
        """Merge the read set and advance the hash chain (new ids only; sorted serialization ensures determinism)."""
        new = sorted(set(dia_ids) - self.read_dia_ids)
        if not new:
            return
        self.read_dia_ids.update(new)
        self._chain.update(json.dumps(new, ensure_ascii=False).encode("utf-8"))

    def state_hash(self) -> str:
        return self._chain.hexdigest()


def _parse_session_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s.strip(), DATE_FORMAT)
    except Exception:
        return None


def _serialize(results: List[dict]) -> str:
    out: List[dict] = []
    total = 0
    truncated = False
    for r in results:
        chunk = json.dumps(r, ensure_ascii=False)
        if total + len(chunk) > RESULT_CHAR_CAP:
            truncated = True
            break
        out.append(r)
        total += len(chunk)
    return json.dumps({"results": out, "count": len(out), "truncated": truncated},
                      ensure_ascii=False)


def make_tool_core(sample: Any, memory: LocomoMemory) -> Dict[str, Callable]:
    """Return {name: core implementation (**kwargs) -> JSON string} (no recording logic)."""

    def _get_person_id(name: str) -> str:
        n = name.strip().lower()
        if n == sample.speaker_a.lower():
            return json.dumps({"person_id": sample.speaker_a, "matched": "speaker_a"},
                              ensure_ascii=False)
        if n == sample.speaker_b.lower():
            return json.dumps({"person_id": sample.speaker_b, "matched": "speaker_b"},
                              ensure_ascii=False)
        return json.dumps({"error": f"person not found: {name!r}"}, ensure_ascii=False)

    def _filter_turns(person_id: str, session: str) -> List[Any]:
        turns = sample.sessions.get(session) if session else sample.turns
        if person_id:
            turns = [t for t in turns if t.speaker == person_id]
        return turns

    def _search_messages(person_id: str, query: str, session: str = "",
                         top_k: int = 10) -> str:
        q = query.strip()
        if q.startswith("regex:"):
            try:
                pat = re.compile(q[len("regex:"):], re.IGNORECASE)
                hit = lambda t: bool(pat.search(t.text))  # noqa: E731
            except re.error as e:
                return json.dumps({"error": f"bad regex: {e}"}, ensure_ascii=False)
        else:
            hit = lambda t: q.lower() in t.text.lower()  # noqa: E731
        matched = [t for t in _filter_turns(person_id, session) if hit(t)][:top_k]
        memory.mark_read(t.dia_id for t in matched)
        results = [{"dia_id": t.dia_id, "session": t.session_key,
                    "date": t.session_date, "speaker": t.speaker,
                    "text": t.text,
                    **({"blip_caption": t.blip_caption} if t.blip_caption else {})}
                   for t in matched]
        return _serialize(results)

    def _all_messages(person_id: str = "", session: str = "", limit: int = 200) -> str:
        turns = _filter_turns(person_id, session)[:limit]
        memory.mark_read(t.dia_id for t in turns)
        results = [{"dia_id": t.dia_id, "session": t.session_key,
                    "date": t.session_date, "speaker": t.speaker, "text": t.text}
                   for t in turns]
        return _serialize(results)

    def _time_range(person_id: str = "", start_date: str = "",
                    end_date: str = "") -> str:
        start = _parse_session_date(start_date) if start_date.strip() else None
        end = _parse_session_date(end_date) if end_date.strip() else None
        matched: List[Any] = []
        skipped_no_date = 0
        for t in _filter_turns(person_id, ""):
            d = _parse_session_date(t.session_date)
            if d is None:
                skipped_no_date += 1
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            matched.append(t)
        memory.mark_read(t.dia_id for t in matched)
        sessions_in_range = sorted({t.session_key for t in matched},
                                   key=lambda k: int(k.split("_")[-1]))
        results = [{"dia_id": t.dia_id, "session": t.session_key,
                    "date": t.session_date, "speaker": t.speaker, "text": t.text}
                   for t in matched]
        payload = json.loads(_serialize(results))
        payload["sessions_in_range"] = sessions_in_range
        payload["skipped_no_date"] = skipped_no_date
        return json.dumps(payload, ensure_ascii=False)

    return {
        "get_person_id": _get_person_id,
        "search_messages": _search_messages,
        "all_messages": _all_messages,
        "time_range": _time_range,
    }


OAI_TOOL_SCHEMAS: List[dict] = [
    {"type": "function", "function": {
        "name": "get_person_id",
        "description": "Resolve a person name to their person_id "
                       "(use this first before searching messages).",
        "parameters": {"type": "object",
                       "properties": {"name": {"type": "string",
                                               "description": "Person name"}},
                       "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "search_messages",
        "description": "Search conversation messages by text query "
                       "(substring match; prefix query with 'regex:' for regex). "
                       "Optionally filter by person_id and/or session.",
        "parameters": {"type": "object", "properties": {
            "person_id": {"type": "string", "description": "Optional person filter"},
            "query": {"type": "string", "description": "Search text or regex:"},
            "session": {"type": "string", "description": "Optional session_N filter"},
            "top_k": {"type": "integer", "description": "Max results (default 10)"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "all_messages",
        "description": "Return conversation messages in chronological order "
                       "(optionally filtered by person/session).",
        "parameters": {"type": "object", "properties": {
            "person_id": {"type": "string", "description": "Optional person filter"},
            "session": {"type": "string", "description": "Optional session_N filter"},
            "limit": {"type": "integer", "description": "Max messages (default 200)"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "time_range",
        "description": "Return messages whose session date falls in [start_date, end_date] "
                       "(dates like '8 May, 2023'; empty bound = open).",
        "parameters": {"type": "object", "properties": {
            "person_id": {"type": "string", "description": "Optional person filter"},
            "start_date": {"type": "string", "description": "Inclusive start"},
            "end_date": {"type": "string", "description": "Inclusive end"}},
            "required": []}}},
]

TOOL_DESCRIPTIONS = "\n".join(
    f"- {s['function']['name']}: {s['function']['description']}"
    for s in OAI_TOOL_SCHEMAS)

LOC_TOOL_NAMES = tuple(s["function"]["name"] for s in OAI_TOOL_SCHEMAS)

#: refusal phrases (the official cat5 basis, goes into the system prompt)
REFUSAL_PHRASES = ["No information available", "Not mentioned"]


def build_system_prompt(sample: Any) -> str:
    """The English system prompt (vendored copy, verbatim identical to the collection side's locomot_common.build_system_prompt; when
    TRACE_COLLECT is importable the module tail overrides this function with the authoritative implementation -- agent prompt build logic
    always comes from the agent source, the replay side never re-implements it. The trace-recorded sys:j is only a cross-check
    snapshot)."""
    return (
        "You are a helpful assistant answering questions about a long "
        "multi-session conversation between two people.\n\n"
        f"The conversation involves {sample.speaker_a!r} and {sample.speaker_b!r} "
        f"across {sample.num_sessions} sessions ({sample.num_turns} messages) "
        "spanning several months. Each message has a unique id (like 'D3:5') "
        "and belongs to a session with a date. Images are described by their "
        "captions.\n\n"
        "You have read-only tools to search the conversation database:\n"
        f"{TOOL_DESCRIPTIONS}\n\n"
        "Guidelines:\n"
        "1. Use the tools to retrieve relevant messages before answering.\n"
        "2. Answer concisely, using the exact information from the conversation.\n"
        "3. If the conversation does not contain the information, answer exactly "
        f"'{REFUSAL_PHRASES[0]}' or '{REFUSAL_PHRASES[1]}'.\n"
    )


def execute_tool(name: str, kwargs: dict, sample: Any, memory: Any,
                 collector=None) -> str:
    """The sole execute + record entry (the counterpart of the collection side's adapter.locomot.execute_tool)."""
    import time as _time
    if _authoritative is not None:
        return _authoritative.execute_tool(name, kwargs, sample, memory,
                                           collector)
    core = make_tool_core(sample, memory)
    t0 = _time.perf_counter()
    try:
        result = core[name](**kwargs)
        success = True
    except Exception as e:
        result = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        success = False
    if collector is not None:
        collector.log_tool(name, kwargs, str(result)[:5000],
                           (_time.perf_counter() - t0) * 1000, success)
    return result


# ---------------------------------------------------------------------------
# public API (shared by runtime / adapter)
# ---------------------------------------------------------------------------
_sample_cache: Dict[str, Any] = {}


def load_sample(sample_id: str) -> Any:
    """Load a single sample (cached); the authoritative implementation is preferred (the data file is the copy under the collection side's tasks/)."""
    if sample_id in _sample_cache:
        return _sample_cache[sample_id]
    if _authoritative is not None:
        sample = _authoritative.load_locomo_samples(
            sample_ids=[sample_id])[0]
    else:
        sample = load_locomo_samples(sample_ids=[sample_id])[0]
    _sample_cache[sample_id] = sample
    return sample


def new_memory(sample: Any) -> Any:
    """A LocomoMemory of the same origin as the sample (authoritative / vendored each paired)."""
    if _authoritative is not None and not isinstance(sample, LocomoSample):
        return _authoritative.LocomoMemory(sample)
    return LocomoMemory(sample)


def replay_span_log(sample: Any, span_log: List[Dict[str, Any]],
                    up_to_step: int) -> Tuple[Any, str, List[str]]:
    """Replay the records with step < up_to_step to rebuild the read set, verifying the hash chain entry by entry.

    Hash semantics (measured on the collection side, adapter/locomot.py + trace_collector.py): - llm.call
    entries: hash_before = the chain value BEFORE that call (recording performs no state change) -> verify
    first, leave the chain untouched - tool.execution entries: the check_point is recorded AFTER execution
    (log_tool is called after execute) -> really execute first (chain advances), then verify hash_before
    Chain updates come only from the tool mark_read (a pure function of the retrieval results), env_changes
    always empty. Returns (memory, final accumulated hash, note list). Tampering/drift raises RuntimeError
    honestly.
    """
    memory = new_memory(sample)
    notes: List[str] = []
    for entry in span_log:
        step = int(entry.get("step", 0) or 0)
        if step >= up_to_step:
            break
        expected = str(entry.get("hash_before", "") or "")
        if entry.get("type") == "tool.execution":
            tool = entry.get("tool") or {}
            args = tool.get("arguments")
            if not isinstance(args, dict):
                args = {}
            execute_tool(str(tool.get("name", "")), args, sample, memory)
            if expected and expected != memory.state_hash():
                raise RuntimeError(
                    "locomo restore chain mismatch at step %d: recorded %s != "
                    "applied %s" % (step, expected[:12],
                                    memory.state_hash()[:12]))
        else:
            if expected and expected != memory.state_hash():
                raise RuntimeError(
                    "locomo restore chain mismatch at step %d: recorded %s != "
                    "applied %s" % (step, expected[:12],
                                    memory.state_hash()[:12]))
    final = memory.state_hash()
    notes.append("replayed %d entries, final hash %s" % (
        len([e for e in span_log
             if int(e.get("step", 0) or 0) < up_to_step]), final[:12]))
    return memory, final, notes


def _load_scorer():
    """The official scorer: prefer the collection side's tasks/locomo_tasks/task_eval, else this repo's benchmarks/locomo/task_eval (lazy bert_score already patched, torch optional)."""
    candidates = []
    if _authoritative is not None and os.path.isdir(TRACE_COLLECT):
        candidates.append(os.path.join(
            TRACE_COLLECT, "tasks", "locomo_tasks", "task_eval"))
    candidates.append(str(REPO_ROOT / "benchmarks" / "locomo" / "task_eval"))
    for cand in candidates:
        if not os.path.isdir(cand):
            continue
        if cand not in sys.path:
            sys.path.insert(0, cand)
        try:
            from evaluation import eval_question_answering  # noqa: F401
            return eval_question_answering
        except Exception:
            continue
    raise RuntimeError(
        "locomo official scorer unavailable; check nltk/regex installed and "
        "benchmarks/locomo/task_eval/evaluation.py importable")


def build_question_prompt(i: int, n: int, question: str) -> str:
    return f"Question {i + 1}/{n}: {question}"


def evaluate_answers(sample: Any, answers: List[Dict[str, Any]],
                     read_ids: Any) -> Dict[str, Any]:
    """Official-basis evaluation (an in-memory version of finalize_locomot.evaluate_locomo_trace).

    Returns {success, f1, threshold, num_questions, category_counts, category_avg_f1, overall_recall, per_question}.
    """
    eval_question_answering = _load_scorer()
    gold_by_q = {q.question: q for q in sample.qas}
    read_set = set(read_ids or [])

    qas: List[dict] = []
    per_q: List[dict] = []
    recall_hits = recall_total = 0
    for a in answers:
        gold = gold_by_q.get(a.get("question", ""))
        if gold is None:
            continue
        qas.append({
            "answer": str(gold.answer),  # the official bert_score has no str guard, numeric answers would crash (same basis as the collection side's finalize_locomot.py)
            "category": int(gold.category),
            "prediction": str(a.get("pred", "")),
            "evidence": gold.evidence,
        })
        recall = None
        if gold.evidence:
            hits = len(set(gold.evidence) & read_set)
            recall = hits / len(gold.evidence)
            recall_hits += hits
            recall_total += len(gold.evidence)
        per_q.append({"idx": a.get("idx"), "category": gold.category,
                      "pred": str(a.get("pred", ""))[:300],
                      "gold": str(gold.answer)[:300],
                      "f1": None, "recall": recall})

    if not qas:
        return {"success": False, "f1": 0.0, "threshold": PASS_THRESHOLD,
                "num_questions": 0, "category_counts": {},
                "category_avg_f1": {}, "overall_recall": None,
                "per_question": per_q, "error": "no questions to align (gold missing)"}

    scores = eval_question_answering(
        qas, eval_key="prediction", metric="f1")[0]
    avg_f1 = sum(scores) / len(scores)
    success = avg_f1 >= PASS_THRESHOLD

    cat_scores: Dict[str, List[float]] = {}
    for q, s in zip(per_q, scores):
        cat_scores.setdefault(str(q["category"]), []).append(s)
        q["f1"] = round(s, 4)

    return {
        "success": bool(success),
        "f1": round(avg_f1, 4),
        "threshold": PASS_THRESHOLD,
        "num_questions": len(per_q),
        "category_counts": {k: len(v) for k, v in cat_scores.items()},
        "category_avg_f1": {k: round(sum(v) / len(v), 4)
                            for k, v in cat_scores.items()},
        "overall_recall": round(recall_hits / recall_total, 4)
        if recall_total else None,
        "per_question": per_q,
    }


# ---------------------------------------------------------------------------
# authoritative-first dispatch: agent prompt build / question construction / the whole-question loop always go through the collection side's real
# source (when TRACE_COLLECT is importable); the vendored copies are only an offline fallback, pinned to the records verbatim by the
# golden-vector
# tests.
# ---------------------------------------------------------------------------
if _common_mod:
    build_system_prompt = _auth_build_system_prompt  # type: ignore
    build_question_prompt = _auth_build_question_prompt  # type: ignore
    run_question_loop = _auth_run_question_loop  # type: ignore
    MAX_TOOL_STEPS = int(_auth_max_tool_steps or 10)  # type: ignore
else:
    # offline fallback: a vendored copy verbatim identical to the collection side's locomot_common
    def run_question_loop(env, collector, question, step_fn,
                          max_tool_steps: int = 10):
        """The generic answering sub-loop (a verbatim copy of the collection side's locomot_common.run_question_loop; used only when TRACE_COLLECT is not importable)."""
        sample = env.sample
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": question}]
        answer = ""

        def _norm_tc(tc: dict) -> dict:
            if "function" in tc:
                fn = tc["function"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                return {"id": tc.get("id", ""), "name": fn.get("name", ""),
                        "arguments": args or {}}
            return tc

        for _ in range(max_tool_steps):
            out, tool_calls = step_fn(messages)
            if not tool_calls:
                answer = out
                break
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                ntc = _norm_tc(tc)
                result = execute_tool(ntc["name"], ntc.get("arguments") or {},
                                      sample, env, collector)
                messages.append({"role": "tool",
                                 "tool_call_id": ntc.get("id", "")
                                 or tc.get("id", ""),
                                 "name": ntc["name"],
                                 "content": result})
            answer = out
        return answer, _

    MAX_TOOL_STEPS = 10
