"""LocoMo benchmark adapter layer — data model / retrieval memory / tool set / checkpoint provider.

LocoMo (snap-research/LoCoMo, ACL 2024) is a long-context multi-session dialogue
memory QA benchmark, **strictly read-only** (the agent only searches the dialogue
corpus and does not modify the environment). There is no official agent tool
schema; this layer builds 4 read-only retrieval tools on top of the official
data/locomo10.json fields:

- get_person_id: speaker name → person id (metadata; does not mark as read)
- search_messages: text / regex substring retrieval (can filter by person/session)
- all_messages: return all / paged messages in conversation order
- time_range: filter by session date closed interval

check_point semantics (the four-field contract aligned with AppWorld/SWE):
- application_state_hash: chained hash of the "read message set" (advanced when a retrieval tool marks read)
- env_changes: always an empty dict (read-only benchmark, no replayable environment delta)
- env_changes_baseline: the first span carries an empty dict (contract alignment)

Constraint: do not touch agent framework source; this module only does read-only
data access + state hashing, with zero network dependencies.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "tasks" / "locomo_tasks" / "locomo10.json"

# official session_N_date_time date format (e.g. "1:56 pm on 8 May, 2023")
DATE_FORMAT = "%I:%M %p on %d %B, %Y"
# official refusal phrases for cat5 (adversarial): a hit is judged correct
REFUSAL_PHRASES = ["no information available", "not mentioned"]
# hard cap on per-tool-result serialization (prevents token explosion)
RESULT_CHAR_CAP = 50_000


# ── Data model ──────────────────────────────────────────────────────
@dataclass
class LocomoTurn:
    dia_id: str                 # official dialogue id with "D" prefix (unique event locator)
    speaker: str
    text: str
    session_key: str            # the owning "session_N"
    session_date: str           # copied from session_N_date_time
    img_url: str = ""
    blip_caption: str = ""


@dataclass
class LocomoQA:
    question: str
    answer: str                # gold (for evaluation; not written into the trace)
    evidence: list[str]        # list of dia_ids (for evaluation retrieval recall)
    category: Any              # raw category value (ecosystem mappings are inconsistent; passed through unchanged)


@dataclass
class LocomoSample:
    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[LocomoTurn]                     # all sessions flattened, in official order
    sessions: dict[str, list[LocomoTurn]] = field(default_factory=dict)
    qas: list[LocomoQA] = field(default_factory=list)
    event_summary: dict = field(default_factory=dict)
    session_summary: dict = field(default_factory=dict)

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def num_turns(self) -> int:
        return len(self.turns)


def load_locomo_samples(limit: int = 0,
                        sample_ids: list[str] | None = None) -> list[LocomoSample]:
    """Load locomo10.json (limit=0 means all; filter by sample_ids, preserving file order)."""
    if not DATA_FILE.exists():
        raise RuntimeError(
            f"LocoMo data missing: {DATA_FILE} -- run python scripts/fetch_locomot.py")
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    if sample_ids:
        raw = [s for s in raw if s.get("sample_id") in set(sample_ids)]
    if limit:
        raw = raw[:limit]
    samples: list[LocomoSample] = []
    for s in raw:
        conv = s["conversation"]
        turns: list[LocomoTurn] = []
        sessions: dict[str, list[LocomoTurn]] = {}
        for key, value in conv.items():
            if not (key.startswith("session_") and isinstance(value, list)):
                continue
            date = conv.get(key + "_date_time", "")
            tlist: list[LocomoTurn] = []
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


# ── Retrieval memory (carrier of the application_state_hash hash chain) ──────────────────
class LocomoMemory:
    """Retrieval history (the set of read messages) — the only "environment state" in this read-only benchmark.

    The chain advances only when a tool marks read (the retrieval tools are the only
    state-changing points); state_hash() is O(1) (called for every span inside the
    collector's lock, so serialization is forbidden).
    """

    def __init__(self, sample: LocomoSample):
        self.sample = sample
        self.read_dia_ids: set[str] = set()
        self._chain = hashlib.sha256()

    def mark_read(self, dia_ids) -> None:
        """Merge the read set and advance the hash chain (only new ids; sorted serialization ensures determinism)."""
        new = sorted(set(dia_ids) - self.read_dia_ids)
        if not new:
            return
        self.read_dia_ids.update(new)
        self._chain.update(json.dumps(new, ensure_ascii=False).encode("utf-8"))

    def state_hash(self) -> str:
        return self._chain.hexdigest()


# ── checkpoint provider (aligned with the make_swe_checkpoint_provider contract) ────
def make_locomo_checkpoint_provider(memory: LocomoMemory, task_id: str):
    """LocoMo checkpoint collector — provide(span_type, span_id=None) -> dict.

    - application_state_hash: chained hash of the read set at span creation time (replay anchor)
    - env_changes: always an empty dict (strictly read-only benchmark)
    - the first span also carries env_changes_baseline = {} (contract alignment)
    """
    _init_done = [False]

    def provide(span_type: str, span_id: str | None = None) -> dict:
        first = not _init_done[0]
        if first:
            _init_done[0] = True
        cp = {
            "task_id": task_id,
            "application_state_hash": memory.state_hash(),
            "env_changes": {},
        }
        if first:
            cp["env_changes_baseline"] = {}
        return cp

    return provide


# ── Tool core (all in-memory retrieval, zero heavy dependencies) ────────────────────────────────
def _parse_session_date(s: str) -> datetime | None:
    """strptime(DATE_FORMAT); returns None on failure (time_range skips and counts them)."""
    try:
        return datetime.strptime(s.strip(), DATE_FORMAT)
    except Exception:
        return None


def _serialize(results: list[dict]) -> str:
    """Serialize the result list; truncate and mark truncated when the hard cap is exceeded."""
    out: list[dict] = []
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


def make_tool_core(sample: LocomoSample, memory: LocomoMemory) -> dict[str, Callable]:
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

    def _filter_turns(person_id: str, session: str) -> list[LocomoTurn]:
        turns = sample.sessions.get(session) if session else sample.turns
        if person_id:
            turns = [t for t in turns if t.speaker == person_id]
        return turns

    def _search_messages(query: str, person_id: str = "", session: str = "",
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
        matched: list[LocomoTurn] = []
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


# ── Tool schema (single source of truth: used directly by simplified/legacy/oai;
#    the autogen/tau2 wrapper layers generate signatures from it) ────────────────
OAI_TOOL_SCHEMAS: list[dict] = [
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


# ── Unified execute/record entry point + multi-framework wrappers ─────────────────────────────────
def record_locomo_tool(collector, tool_name: str, arguments: dict, result: Any,
                       duration_ms: float = 0.0, success: bool = True) -> None:
    """One LocoMo tool execution → canonical tool.execution span (records only, does not execute)."""
    if collector is None:
        return
    collector.log_tool(tool_name, arguments, str(result),
                       duration_ms, success)


def execute_tool(name: str, kwargs: dict, sample: LocomoSample,
                 memory: LocomoMemory, collector=None) -> str:
    """The single execute + record entry point (mirrors adapter.tools.appworld_execute_tool)."""
    import time as _time
    core = make_tool_core(sample, memory)
    t0 = _time.perf_counter()
    try:
        result = core[name](**kwargs)
        success = True
    except Exception as e:
        result = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        success = False
    record_locomo_tool(collector, name, kwargs, result,
                       (_time.perf_counter() - t0) * 1000, success)
    return result


def build_locomo_tools(sample: LocomoSample, memory: LocomoMemory,
                       collector=None) -> dict[str, Callable]:
    """{name: **kwargs -> JSON string}, already wrapped with execute_tool recording."""
    return {
        name: (lambda n: lambda **kw: execute_tool(n, kw, sample, memory, collector))(name)
        for name in ("get_person_id", "search_messages", "all_messages", "time_range")
    }


def build_autogen_tool_funcs(sample: LocomoSample, memory: LocomoMemory,
                             collector=None) -> list:
    """AutoGen native async tool functions (signatures generated from OAI_TOOL_SCHEMAS)."""
    import inspect

    def _make(name: str, schema: dict) -> Callable:
        props = schema["function"]["parameters"]["properties"]
        args = []
        call_kwargs = []
        for pname, pspec in props.items():
            # give all params defaults (required-first signatures cannot be written when required params follow optional ones)
            typ = "int" if pspec.get("type") == "integer" else "str"
            default = "0" if typ == "int" else '""'
            args.append(f"{pname}: {typ} = {default}")
            call_kwargs.append(f"'{pname}': {pname}")
        src = (
            f"async def {name}({', '.join(args)}):\n"
            f"    from adapter.locomot import execute_tool\n"
            f"    return await _run_sync(execute_tool, '{name}', "
            f"{{{', '.join(call_kwargs)}}}, _SAMPLE, _MEMORY, _COLLECTOR)\n"
        )
        ns = {"_SAMPLE": sample, "_MEMORY": memory, "_COLLECTOR": collector,
              "_run_sync": _run_sync}
        exec(src, ns)
        fn = ns[name]
        fn.__doc__ = schema["function"]["description"]
        # autogen generates the tool schema from type annotations (missing annotations would raise KeyError)
        fn.__annotations__ = {p: (int if pspec.get("type") == "integer" else str)
                              for p, pspec in props.items()}
        fn.__annotations__["return"] = str
        return fn

    funcs = [_make(s["function"]["name"], s) for s in OAI_TOOL_SCHEMAS]
    return funcs


def _run_sync(fn: Callable, *args) -> Any:
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: fn(*args))


def build_smolagents_tools(sample: LocomoSample, memory: LocomoMemory,
                           collector=None) -> list:
    """smolagents Tool objects (forward calls execute_tool).

    smolagents requires forward's named parameters to match the inputs keys —
    each tool's forward signature is generated from the schema.
    """
    try:
        from smolagents import Tool
    except ImportError:
        raise RuntimeError("smolagents is not installed")

    tools = []
    for s in OAI_TOOL_SCHEMAS:
        fn = s["function"]
        name = fn["name"]
        props = fn["parameters"]["properties"]
        fwd_args = ["self"] + [p for p in props]
        call_kwargs = ", ".join(f"'{p}': {p}" for p in props)
        src = (
            f"def forward({', '.join(fwd_args)}):\n"
            f"    from adapter.locomot import execute_tool\n"
            f"    return execute_tool('{name}', {{{call_kwargs}}}, "
            f"_SAMPLE, _MEMORY, _COLLECTOR)\n"
        )
        ns = {"_SAMPLE": sample, "_MEMORY": memory, "_COLLECTOR": collector}
        exec(src, ns)
        inputs = {k: {"type": "string" if v.get("type") == "string" else "integer",
                      "description": v.get("description", "")}
                  for k, v in props.items()}
        ToolCls = type(
            name, (Tool,),
            {"name": name, "description": fn["description"],
             "inputs": inputs, "output_type": "string",
             "forward": ns["forward"]})
        tools.append(ToolCls())
    return tools


def build_tau2_tools(sample: LocomoSample, memory: LocomoMemory,
                     collector=None) -> tuple[list, dict]:
    """tau2 as_tool wrappers + {name: tool} mapping.

    as_tool builds the parameter model from the function signature, so an explicit
    signature must be generated (**kwargs would produce an empty-field model);
    generated from OAI_TOOL_SCHEMAS.
    """
    from tau2.environment.tool import as_tool

    tools = []
    mapping = {}
    for s in OAI_TOOL_SCHEMAS:
        fn = s["function"]
        name = fn["name"]
        props = fn["parameters"]["properties"]
        required = set(fn["parameters"].get("required", []))
        params = []
        call_kwargs = []
        for pname, pspec in props.items():
            # give all params defaults (avoids syntax errors from required/optional
            # ordering conflicts; the required constraint is conveyed to the model
            # via the tool schema layer)
            typ = "int" if pspec.get("type") == "integer" else "str"
            default = "0" if typ == "int" else '""'
            params.append(f"{pname}: {typ} = {default}")
            call_kwargs.append(f"'{pname}': {pname}")
        src = (
            f"def {name}({', '.join(params)}):\n"
            f"    from adapter.locomot import execute_tool\n"
            f"    return execute_tool('{name}', "
            f"{{{', '.join(call_kwargs)}}}, _SAMPLE, _MEMORY, _COLLECTOR)\n"
        )
        ns = {"_SAMPLE": sample, "_MEMORY": memory, "_COLLECTOR": collector}
        exec(src, ns)
        impl = ns[name]
        impl.__doc__ = fn["description"]
        tool = as_tool(impl)
        tools.append(tool)
        mapping[name] = tool
    return tools, mapping


def build_oai_tools(sample: LocomoSample, memory: LocomoMemory,
                    collector=None) -> list:
    """openai-agents SDK function_tool wrappers (local execution, no MCP)."""
    from agents import function_tool

    tools = []
    for s in OAI_TOOL_SCHEMAS:
        fn = s["function"]
        name = fn["name"]

        def _make(n: str, doc: str):
            @function_tool
            def impl(**kwargs) -> str:
                return execute_tool(n, kwargs, sample, memory, collector)
            impl.__doc__ = doc
            return impl

        tools.append(_make(name, fn["description"]))
    return tools
