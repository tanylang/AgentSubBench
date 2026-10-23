"""AppWorldAgentRuntime: the AgentRuntime (14.4) for single-span counterfactual
replay against the REAL AppWorld environment -- the live layer that unlocks
when traces carry checkpoints (check_point_trace format, see
appworld_adapter._read_check_point).

State reconstruction strategy (restore): the trace's recorded tool calls are
the ground truth of how the environment reached the span-before state.  A
fresh task world is started and every tool call recorded before span j
(artifacts.previous_tool_results["prev:%d"]) is re-executed in order.
AppWorld's simulator is deterministic, so the reconstructed state is
semantically identical to the recorded one; this avoids depending on internal
SQL-level checkpoint formats and uses only public AppWorld APIs.

Honest limitations, documented in the records:
  - session.snapshot_hash is a hash over the replayed tool-call sequence
    ("tool_replay:<sha256>"), NOT the recorded application_state_hash.  When
    the exporter writes real hashes, the mismatch is recorded by the replay
    engine as designed; with the current placeholder hash it surfaces the
    placeholder instead of silently passing.
  - continuation runs the same tool-call loop as appworld_agent (the AutoGen
    agent loop re-implemented over the recorded conversation); the original
    framework (AutoGen) is not re-instantiated.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass

from span_offload.timing_util import monotonic_ms
from typing import Any, Dict, List, Optional, Tuple

# Make all later `autogen_*` imports (including
# appworld_agents.code.autogen.agent) resolve to the vendored microsoft/autogen
# source in this repo instead of the installed wheel; see span_offload/vendor_autogen.py
from .. import vendor_autogen  # noqa: E402
vendor_autogen.activate()

from ..appworld_adapter import (
    ANSWER_SCHEMA, TOOL_CALL_SCHEMA, parse_llm_output, _synth_call_id,
)
from ..llm_client import LMResponse
from ..runtime import (
    AgentRuntime, ParseResult, ResolvedRequest, RestoredSession, SpanContext,
)
from ..schemas import (
    DownstreamOutcome, FinalOutcome, SpanRecord, TraceRun, state_hash,
)
from ..verifiers import VerifierVerdict, validate_schema
from .appworld_env import AppWorldBridge

#: sys.path snapshot at interpreter startup (vendored autogen activated):
#: backend probing trusts only these roots + sys.prefix/sys.base_prefix
#: packages -- appworld's import chain pollutes sys.path with unrelated
#: projects' conda env dirs whose packages clash with this venv yet resolve via find_spec, so reject them (see _backend_available._top_ok)
_INITIAL_SYSPATH = tuple(os.path.abspath(p) for p in sys.path if p)

MAX_LOOP_STEPS = 128

_FUNC_CALL_BLOCK_RE = re.compile(r"FunctionCall\((.*?)\)", re.S)

_JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


def _openai_tool_schemas(bridge: AppWorldBridge,
                         allowed: List[str]) -> List[Dict[str, Any]]:
    """OpenAI-format tool schemas for the span's allowed tools, built from the
    task's real API docs (the recorded agent got the same schemas via native
    function calling; the trace does not carry them, the environment does).
    supervisor__complete_task is added explicitly."""
    tools: List[Dict[str, Any]] = []
    allowed_set = {t.replace("__", ".") for t in allowed}
    for doc in bridge.tool_docs():
        if doc["name"].replace("__", ".") not in allowed_set:
            continue
        props: Dict[str, Any] = {}
        required: List[str] = []
        for k, v in (doc.get("inputs") or {}).items():
            t = v.get("type", "string")
            if t not in _JSON_TYPES:
                t = "string"  # "any" and friends: keep the API open
            props[k] = {"type": t, "description": v.get("description", "")}
            if v.get("required"):
                required.append(k)
        tools.append({"type": "function", "function": {
            "name": doc["name"],
            "description": doc.get("description", ""),
            "parameters": {"type": "object", "properties": props,
                           "required": required, "additionalProperties": True},
        }})
    tools.append({"type": "function", "function": {
        "name": "supervisor__complete_task",
        "description": "Mark the active task as complete with the final answer.",
        "parameters": {"type": "object",
                       "properties": {"answer": {}, "status": {"type": "string"}},
                       "required": ["answer", "status"]},
    }})
    return tools


def _parse_continuation_output(raw: str) -> Tuple[Any, bool]:
    """Parse a continuation turn's output.  Accepts both the span_offload
    contract (plain JSON tool-call list) and AutoGen's
    [FunctionCall(name='...', arguments='...')] text, which is the format the
    recorded conversation history uses and the model tends to continue."""
    parsed, kind = parse_llm_output(raw)
    if kind == "tool_call_list":
        return parsed, True
    text = (raw or "").strip()
    if "[FunctionCall(" in text:
        calls = []
        for block in _FUNC_CALL_BLOCK_RE.findall(text):
            name_m = re.search(r"name\s*=\s*'([^']*)'", block)
            args_m = re.search(r"arguments\s*=\s*'((?:[^'\\]|\\.)*)'", block)
            name = name_m.group(1) if name_m else ""
            args: Dict[str, Any] = {}
            if args_m:
                try:
                    args = json.loads(args_m.group(1).replace("\\'", "'"))
                except ValueError:
                    args = {"_raw": args_m.group(1)}
            if name:
                calls.append({"name": name, "arguments": args})
        if calls:
            return {"tool_calls": calls}, True
    return parsed, False


@dataclass
class AppWorldSession(RestoredSession):
    bridge: Optional[AppWorldBridge] = None
    conversation: List[Dict[str, str]] = None  # type: ignore[assignment]
    span: Optional[SpanRecord] = None
    trace: Optional[TraceRun] = None
    injected: Any = None
    baseline_client: Any = None
    #: the trace-recorded native function-calling tool schemas (preferred
    #: over api-doc rebuilds when present)
    recorded_tools: Optional[List[Dict[str, Any]]] = None
    # continuation loop state, persisted across run_to_node -> run_to_completion
    loop_messages: Optional[List[Dict[str, str]]] = None
    loop_index: int = 0
    loop_completed: bool = False
    #: real-AutoGen continuation state (built in run_to_node, consumed in
    #: run_to_completion)
    autogen_messages: Optional[List[Any]] = None  # autogen_core LLMMessage list
    #: simplified-agent continuation state (same split)
    simplified_agent: Optional[Any] = None
    simplified_outputs: Optional[List[Any]] = None
    simplified_completed: bool = False
    #: per-round continuation trajectory (tool names + status), to locate why a run did not finish
    simplified_trajectory: Optional[List[str]] = None
    #: structured full trace of injection + continuation (the entire trace
    #: continued by the baseline LLM after offload); accumulated by each backend at runtime, written to ReplayRecord.downstream_trace)
    downstream_trace: Optional[List[Dict[str, Any]]] = None
    #: continuation backend (resolved from the trace's agent identity at restore time):
    #: autogen | simplified | legacy | smolagents | tau2 | openai | generic
    backend: str = ""
    #: smolagents' text tool protocol: history recorded in "Calling tools:" /
    #: "Observation:" text form (the original agent's real request shape), not native tool_calls
    text_protocol: bool = False
    #: native tool_choice (legacy / smolagents original protocol: "required")
    tool_choice: Optional[str] = None
    #: OpenAI-SDK reconstruction: re-rendered messages (with the real
    #: system prompt) for SLM requests and continuation; the trace itself does not record the session
    openai_messages: Optional[List[Dict[str, Any]]] = None
    openai_items: Optional[List[Dict[str, Any]]] = None
    #: legacy continuation state (real legacy LanguageModel + message list)
    legacy_lm: Optional[Any] = None
    legacy_messages: Optional[List[Dict[str, Any]]] = None
    legacy_completed: bool = False
    #: tau2 continuation state
    tau2_agent: Optional[Any] = None
    tau2_state: Optional[Any] = None
    tau2_completed: bool = False
    #: per-backend failure/trajectory notes (shared by non-autogen/simplified backends)
    backend_info: Optional[List[str]] = None
    #: C0 frozen replay (plan §4): target span injects the recorded output; downstream LM outputs all inject
    #: trace-recorded values, only tools / state transitions are re-executed. frozen_plan = the recorded
    #: span sequence after the target span (invocation_index ascending); frozen_index = current walk
    #: position (persisted across run_to_node/run_to_completion); c0_fidelity accumulates the per-span
    #: tool execution result comparison.
    frozen_downstream: bool = False
    frozen_plan: Optional[List[Any]] = None
    frozen_index: Optional[int] = None
    frozen_completed: bool = False
    c0_fidelity: Optional[List[Dict[str, Any]]] = None
    #: raw hash of the restore chain (env_delta mode); empty in tool_replay mode
    restored_env_hash_raw: str = ""

    def __post_init__(self) -> None:
        if self.conversation is None:
            self.conversation = []


#: The recorded agent used NATIVE function calling (the API tools protocol),
#: which the trace does not record.  This continuation approximates it with an
#: explicit output-format instruction plus the exact tool catalog (from the
#: span's allowed_tools); the note keeps the adaptation visible.
#: NOTE: {tool_catalog} is substituted via .replace(), NOT str.format() --
#: the JSON braces in the template would otherwise be parsed as format fields.
CONTINUATION_FORMAT_NOTE = (
    "\n\n[span_offload bridge note: the recorded agent used native function "
    "calling; this continuation approximates it with an explicit format "
    "instruction.]\n"
    "For each step, output EXACTLY a JSON list of tool calls in the form: "
    '[{"name": "<tool_name>", "arguments": {<json object>}}]. '
    "Do not narrate or explain, only output the JSON."
    "\n\nAvailable tools (call these EXACT names only):\n{tool_catalog}"
)

NUDGE_MESSAGE = (
    "You must output the next step as a JSON list of tool calls: "
    '[{"name": "<tool_name>", "arguments": {<json object>}}]. '
    "Do not narrate, only output the JSON."
)


def _json_args_or_empty(args: Any) -> Dict[str, Any]:
    """Parse tool call arguments (possibly a JSON string) into a dict; return {} on failure."""
    if not args:
        return {}
    if isinstance(args, dict):
        return args
    try:
        v = json.loads(str(args))
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def _parse_command_args(content: str) -> Dict[str, Any]:
    """Parse the argument dict from the agent's ExecutionIO command string print(name(**{args}))."""
    import ast as _ast
    m = re.search(r"\*\*(\{.*\})\s*\)\s*$", (content or "").strip())
    if not m:
        return {}
    try:
        args = _ast.literal_eval(m.group(1))
        return args if isinstance(args, dict) else {}
    except (ValueError, SyntaxError):
        return {}


#: max length of result content in the structured trace (controls record size)
TRACE_RESULT_MAX_LEN = None  # no truncation (full retention as required by the user)
#: max length of a single message's content in the trace prefix (the session window before the span)
TRACE_PREFIX_MSG_MAX = None  # no truncation
#: fallback text for thinking content: the trace does not record
#: reasoning_content, yet DeepSeek thinking mode requires assistant tool_calls msgs to carry non-empty rc (original run has real thinking text on each)
RC_PLACEHOLDER = "Determining the next tool call to continue the task."


def _span_ctx(invocation_index: int, allowed_tools: List[str]) -> SpanContext:
    from .appworld_agent import _span_ctx as _make
    return _make(
        invocation_index, "tool_argument_generation", "local_executor",
        "execution", "tool_parameter_extraction",
        ["tool_name", "tool_arguments", "json_schema"],
        ["schema_check", "tool_exec_success"], allowed_tools,
    )


def _replay_recorded_tools(bridge: AppWorldBridge, trace: TraceRun,
                           span: SpanRecord) -> List[Dict[str, Any]]:
    """Re-execute every recorded tool call before span j to rebuild the
    span-before environment state (fallback when an old trace has no checkpoint delta)."""
    prev = (trace.artifacts.get("previous_tool_results") or {}).get(
        span.request.previous_tool_results_ref, [])
    replayed = []
    for entry in prev:
        # recorded names use the dotted form ("supervisor.show_profile");
        # execute_tool expects the app__api separator
        name = str(entry.get("tool", "")).replace(".", "__")
        arguments = entry.get("arguments") or {}
        res = bridge.execute_tool(name, arguments)
        replayed.append({"tool": name, "arguments": arguments, "ok": res["ok"]})
    return replayed


def _chain_update_env(running: Any, env_changes: Dict[str, Any]) -> None:
    """Feed env_changes into the hash in recorded order (exactly as the collection side does)."""
    import json as _json
    for app_changes in env_changes.values():
        if app_changes:
            running.update(_json.dumps(
                app_changes, default=str, ensure_ascii=False).encode("utf-8"))


def _apply_env_changes(world: Any, env_changes: Dict[str, Any]) -> None:
    """Apply Change tuples (statement, parameters, executemany) to the world's per-app DBs
    (same mechanism as apply_changes in the trace_record collection project).

    The write target is anchored explicitly to the world's output home (world.models' storage):
    module-level db bindings may be redirected by evaluator/other-world access; SQL landing
    elsewhere would split "restored writes" and "walk-through writes" across two stores and
    distort the evaluator comparison (this is why c0 failed in practice on late spans like final_synthesis)."""
    from appworld.common.imports import import_models_module
    from appworld.apps.lib.apis.local_remote import set_local_dbs
    from appworld.apps.lib.models.db import ModelHashHandler
    from appworld.common.naming import modelize
    import re as _re
    set_local_dbs(
        to_db_home_path=world.models_to_db_home_path,
        from_db_home_path=world.models_to_db_home_path)
    for app, changes in env_changes.items():
        if not changes:
            continue
        mod = import_models_module(app)
        conn = mod.SQLModel.db.connection
        for statement, parameters, executemany in changes:
            if executemany:
                conn.executemany(statement, parameters)
            else:
                conn.execute(statement, parameters)
            # raw SQL bypasses the ORM: the evaluator's "model hash" is actually
            # a write counter (orm.py:860 +1 per model-layer write), SQL direct
            # writes do not update it, so the changed_model_names pre-filter
            # (dropping apps with equal counts) misjudges restored written rows
            # as "unchanged". Pad counts manually by the statement's target table, matching the ORM write path semantics.
            _m = _re.match(
                r'\s*(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+["`]?(\w+)',
                statement, _re.I)
            if _m:
                ModelHashHandler.data[
                    world.models_to_db_home_path][app][
                        modelize(_m.group(1))] += 1
        conn.commit()


class AppWorldAgentRuntime(AgentRuntime):
    """Live replay backend over the real AppWorld environment.

    The downstream continuation backend is resolved PER TRACE from the
    recorded agent identity: the SAME agent source that produced the trace
    drives the continuation (autogen / simplified / legacy / smolagents /
    tau2 / openai-SDK).

    Hard rule: the continuation must be driven by the REAL agent source; a generic approximation
    loop produces no experimental results. When framework deps are missing / agent identity unknown,
    this span errors out honestly (backend_unavailable), never silently degrading."""

    name = "appworld-live"

    #: continuation backend resolved by trace agent identity; SPAN_OFFLOAD_CONT_BACKEND may
    #: force-override (only to a real agent backend; generic is rejected --
    #: the continuation must be real agent source, approximation loops yield no results).
    continuation_backend: str = os.environ.get("SPAN_OFFLOAD_CONT_BACKEND", "")

    #: trace agent identity -> preferred continuation backend (same agent source as the trace producer)
    _AGENT_BACKEND: Dict[str, str] = {
        "AppWorld-Simplified": "simplified",
        "AutoGen-AssistantAgent": "autogen",
        "AppWorld-Legacy": "legacy",
        "AppWorld-OpenAI-Agents": "openai",
        "AppWorld-Smolagents": "smolagents",
        "Tau2-LLMAgent": "tau2",
    }

    #: native tool_choice protocol (matches the collection scripts' agent
    #: config): legacy's generate(messages, functions, "required") and
    #: smolagents' tool_choice="required" force tool calls; other agents do not pass it (server default)
    _AGENT_TOOL_CHOICE: Dict[str, str] = {
        "AppWorld-Legacy": "required",
        "AppWorld-Smolagents": "required",
    }

    _autogen_cache: Optional[bool] = None

    @classmethod
    def _autogen_available(cls) -> bool:
        # pure probing (top-level find_spec + path checks); do not import
        # appworld_agents submodules (that would pin APPWORLD_ROOT to cwd, see _backend_available)
        if cls._autogen_cache is None:
            import importlib.util as _ilu
            cls._autogen_cache = (
                cls._appworld_agents_has("autogen")
                and all((_ilu.find_spec(p) is not None)
                        for p in ("autogen_agentchat", "autogen_core",
                                  "autogen_ext")))
        return cls._autogen_cache

    @classmethod
    def _backend_available(cls, backend: str) -> bool:
        """Whether the backend's dependencies are importable (falls back to the generic loop when the framework is missing).

        Only TOP-LEVEL find_spec + pure path checks, never import appworld_agents submodules:
        find_spec imports the parent package to resolve a submodule, and appworld_agents'
        import chain instantiates AppWorld's path_store and pins APPWORLD_ROOT to cwd,
        breaking the subsequent environment init (see appworld_env._ensure_appworld_root).
        The real backend import happens only after the bridge is built, at use time."""
        if backend == "generic":
            return True
        if backend == "autogen":
            return cls._autogen_available()
        import importlib.util as _ilu

        def _top_ok(name: str) -> bool:
            # trust only packages inside this interpreter's environment
            # (prefix / initial sys.path, incl. the vendored autogen source
            # tree): appworld's import chain pollutes sys.path with unrelated projects' conda env dirs, whose packages clash with this venv and blow up on import
            try:
                spec = _ilu.find_spec(name)
            except (ModuleNotFoundError, ValueError, ImportError):
                return False
            if spec is None or not spec.origin:
                return False
            origin = os.path.abspath(spec.origin)
            roots = ((os.path.abspath(sys.prefix), os.path.abspath(sys.base_prefix))
                     + _INITIAL_SYSPATH)
            return any(origin == r or origin.startswith(r + os.sep)
                       for r in roots)

        if backend == "simplified":
            return cls._appworld_agents_has("simplified")
        if backend == "legacy":
            # legacy's OpenAILanguageModel imports numpy/diskcache at module level
            return (cls._appworld_agents_has("legacy")
                    and _top_ok("numpy") and _top_ok("diskcache"))
        if backend == "smolagents":
            return _top_ok("smolagents") and cls._appworld_agents_has("smolagents")
        if backend == "tau2":
            return _top_ok("tau2")
        if backend == "openai":
            return _top_ok("agents")
        return False

    #: cache of framework import-chain preload results (must happen before
    #: any world creation -- worlds freeze the clock, and litellm's
    #: import-time price-table fetch never times out under a frozen clock -> infinite loop + memory explosion)
    _prewarm_cache: Dict[str, bool] = {}

    @classmethod
    def _prewarm_framework(cls, name: str, import_target: str) -> bool:
        """Preload the framework import chain BEFORE ANY WORLD CREATION (idempotent).

        Under the real clock, litellm's price-table fetch falls back to the
        local table after one failure; afterwards all imports hit the module
        cache with zero network. Returns whether the load succeeded.
        """
        if name not in cls._prewarm_cache:
            try:
                import importlib
                importlib.import_module(import_target)
                cls._prewarm_cache[name] = True
            except Exception:
                cls._prewarm_cache[name] = False
        return cls._prewarm_cache[name]

    @classmethod
    def _prewarm_tau2(cls) -> bool:
        return cls._prewarm_framework("tau2", "tau2.agent.llm_agent")

    @classmethod
    def _prewarm_smolagents(cls) -> bool:
        # smolagents' model layer imports litellm, same frozen-clock hazard as tau2
        return cls._prewarm_framework("smolagents", "smolagents")

    @staticmethod
    def _appworld_agents_has(name: str) -> bool:
        """Pure path check that the appworld_agents.code.<name> directory
        exists (appworld's experiments editable install;
        <root>/experiments/code/<name>), without triggering any import."""
        try:
            import importlib.util as _ilu
            spec = _ilu.find_spec("appworld")
            if spec is None or not spec.origin:
                return False
            # <root>/src/appworld/__init__.py -> <root>
            root = os.path.dirname(os.path.dirname(os.path.dirname(spec.origin)))
            return os.path.isdir(os.path.join(root, "experiments", "code", name))
        except Exception:
            return False

    @classmethod
    def _backend_for(cls, agent_id: str) -> str:
        """Resolve the continuation backend from the trace's agent identity.

        Hard rule: the continuation must be driven by the REAL agent source;
        a generic approximation loop produces no experimental results.
        Hence: missing deps / unknown identity / force-override unavailable
        -> always return "unavailable", restore reports the error honestly
        (the span records backend_unavailable, never silently degraded).
        """
        if cls.continuation_backend:
            if cls.continuation_backend == "generic":
                return "unavailable"
            return (cls.continuation_backend
                    if cls._backend_available(cls.continuation_backend)
                    else "unavailable")
        preferred = cls._AGENT_BACKEND.get(agent_id)
        if preferred is None:
            return "unavailable"
        if cls._backend_available(preferred):
            return preferred
        return "unavailable"

    # -- 14.4 step 3 --------------------------------------------------------
    # -- 14.4 step 2: environment restore ----------------------------------
    @staticmethod
    def _restore_span_env(bridge: AppWorldBridge, trace: TraceRun,
                          span: SpanRecord) -> Tuple[Any, bool]:
        """Restore the environment state before span j. Returns (payload, delta_mode).

        Prefer applying the env_changes delta (same mechanism as the trace_record collection project): the state is restored
        byte for byte, session-record values naturally agree with world state (eliminating unreproducible-value mismatches from
        tool replay), and the application_state_hash chain is verified per span (any mismatch raises directly -- restore
        becomes a hard check, not an approximation). Fall back to tool replay when the trace has no checkpoint delta (old
        traces)."""
        snapshots = trace.artifacts.get("state_snapshots") or {}
        prior = [s for s in trace.spans
                 if s.invocation_index < span.invocation_index]
        if not prior:
            # the pre-state of inv=0 (first span) = a fresh world + the initial baseline
            # recorded by the collection side on the span 0 snapshot (env_changes_baseline,
            # see adapter §check_point semantics). The baseline already exists in the fresh world;
            # only feed the hash chain for verification, don't re-apply to the world; fall back to tool replay when the collection side recorded a placeholder hash (EMPTY_SHA256).
            import hashlib
            from ..appworld_adapter import EMPTY_SHA256
            snap = snapshots.get(
                span.state_snapshot.environment_snapshot_ref) or {}
            if not snap.get("checkpointed"):
                return _replay_recorded_tools(bridge, trace, span), False
            running = hashlib.sha256()
            baseline = snap.get("env_changes_baseline") or {}
            if baseline:
                _chain_update_env(running, baseline)
            recorded = str(span.state_snapshot.application_state_hash or "")
            if not recorded or recorded == EMPTY_SHA256:
                return _replay_recorded_tools(bridge, trace, span), False
            if recorded != running.hexdigest():
                raise RuntimeError(
                    "restore baseline mismatch at span %d: recorded hash "
                    "%s != applied baseline %s"
                    % (span.invocation_index, recorded[:12],
                       running.hexdigest()[:12]))
            return running.hexdigest(), True
        # the fallback condition only checks whether the checkpoint structure exists, not the delta content:
        # env_changes is a "write delta"; read-only spans / llm.call spans are always {},
        # an empty delta = the valid semantics of "state unchanged" (the hash chain naturally unchanged), not missing data
        for s in prior:
            snap = snapshots.get(s.state_snapshot.environment_snapshot_ref)
            if not snap or not snap.get("checkpointed"):
                return _replay_recorded_tools(bridge, trace, span), False
        import hashlib
        from ..appworld_adapter import EMPTY_SHA256
        running = hashlib.sha256()
        base = snapshots.get(
            prior[0].state_snapshot.environment_snapshot_ref, {})
        baseline = base.get("env_changes_baseline") or {}
        if baseline:
            _chain_update_env(running, baseline)
        applied = []
        for s in prior:
            snap = snapshots.get(s.state_snapshot.environment_snapshot_ref, {})
            changes = snap.get("env_changes") or {}
            recorded = str(s.state_snapshot.application_state_hash or "")
            if recorded and recorded != EMPTY_SHA256:
                got = running.hexdigest()
                if recorded != got:
                    raise RuntimeError(
                        "restore delta mismatch at span %d: recorded hash "
                        "%s != applied chain %s"
                        % (s.invocation_index, recorded[:12], got[:12]))
            if changes:
                _apply_env_changes(bridge.world, changes)
                applied.append(s.invocation_index)
            _chain_update_env(running, changes)
            # the write delta from executing this span's requested tools (exporter attaches
            # it to the tool.execution check_point; adapter collects it per requesting span, in order):
            # feed it into the hash chain and apply it to the world after the recorded-hash
            # verification point and before the next span's verification, matching the collection side's accounting order
            for td in snap.get("tool_env_changes") or []:
                if td:
                    _apply_env_changes(bridge.world, td)
                    applied.append(s.invocation_index)
                _chain_update_env(running, td)
        return running.hexdigest(), True

    @staticmethod
    def _split_app(name: str) -> str:
        return str(name or "").split("__")[0].split(".")[0]

    @staticmethod
    def _api_accepts_token(bridge: AppWorldBridge, name: str) -> bool:
        """Collection-side convention: inject the token only into APIs that
        originally have an access_token parameter (the has_token_param
        condition in run_benchmark_autogen.py); APIs like login have no such parameter, injection would fail on the extra keyword argument."""
        try:
            app, api = str(name or "").split("__", 1)
        except ValueError:
            return False
        target = app + "__" + api
        for doc in (bridge.world.task.api_docs.function_calling() or []):
            fn = doc.get("function") or {}
            if fn.get("name") == target:
                return "access_token" in (
                    (fn.get("parameters") or {}).get("properties") or {})
        return False

    @staticmethod
    def _maybe_inject_token(bridge: AppWorldBridge, name: str,
                            args: Dict[str, Any]) -> Dict[str, Any]:
        """Inject the per-app cached login token into calls missing access_token.

        The native agent framework auto-injects the token between model output and execution
        (visible in recorded execution args; the tool schema has no access_token, the model
        cannot pass it itself); if the harness does not reproduce this, all protected APIs
        401 -> the model re-logs-in in a dead loop (observed: venmo__login x16). Inject only
        into APIs whose schema contains access_token (same as the collection side's has_token_param condition)."""
        if isinstance(args, dict) and "access_token" not in args \
                and AppWorldAgentRuntime._api_accepts_token(bridge, name):
            tok = bridge.auth_tokens.get(
                AppWorldAgentRuntime._split_app(name))
            if tok:
                args = dict(args)
                args["access_token"] = tok
        return args

    @staticmethod
    def _maybe_cache_token(bridge: AppWorldBridge, name: str,
                           res: Dict[str, Any]) -> None:
        """Cache the returned access_token in the per-app table when login succeeds."""
        if not str(name or "").endswith("__login") or not res.get("ok"):
            return
        try:
            import ast as _ast
            val = _ast.literal_eval(str(res.get("result", "")))
        except Exception:
            return
        if isinstance(val, dict) and val.get("access_token"):
            bridge.auth_tokens[AppWorldAgentRuntime._split_app(name)] = \
                val["access_token"]

    @staticmethod
    def _prefetch_auth_tokens(bridge: AppWorldBridge) -> None:
        """The same main-user token prefetch as the collection side (run_benchmark_autogen.py):
        worlds with add_login_shortcut=True provide access_token_from per app; log in as the
        main user and store tokens in the per-app cache, later calls get them injected by
        _maybe_inject_token, invisible to the model."""
        try:
            from appworld.apps.admin.models import MainUser
            _main = MainUser.find_one(**bridge.world.task.supervisor)
            for _app in {d["function"]["name"].split("__", 1)[0]
                         for d in bridge.world.task.api_docs.function_calling()}:
                try:
                    bridge.auth_tokens[_app] = \
                        bridge.world.apis[_app]["access_token_from"](
                            user=_main, disable_expiry=False)
                except Exception:
                    pass  # apps without a login API (e.g. supervisor) need no token
        except Exception:
            pass

    @staticmethod
    def _replay_auth_tools(bridge: AppWorldBridge, trace: TraceRun,
                           span: SpanRecord) -> None:
        """Replay the recorded login tool calls before span j to rebuild the in-memory session state.

        env_changes covers only DB write deltas; sessions established by login (token verification state) live
        in process memory and the delta chain cannot carry them; without replay all pre-span logins are void and
        later protected APIs all 401 (observed: venmo.show_transactions). Credentials are fixed and the world clock
        is frozen, so logins reproduce; the new token text is random (exp sampled randomly) but outside the
        hash-chain/comparison scope. Replay failure does not block restore (honestly left for the C0 comparison to expose)."""
        snap = (trace.artifacts.get("state_snapshots") or {}).get(
            span.state_snapshot.environment_snapshot_ref) or {}
        for entry in snap.get("recorded_tool_results") or []:
            name = str(entry.get("tool", ""))
            if name.rsplit(".", 1)[-1] != "login":
                continue
            args = entry.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            try:
                res = bridge.execute_tool(name.replace(".", "__"), args)
                AppWorldAgentRuntime._maybe_cache_token(
                    bridge, name.replace(".", "__"), res)
            except Exception:
                pass

    def restore(self, trace: TraceRun, span: SpanRecord,
                baseline_client: Any = None) -> AppWorldSession:
        # the framework backend must finish preloading BEFORE ANY WORLD CREATION: worlds
        # freeze the clock (freezegun), and the tau2 / smolagents import chain triggers
        # litellm's price-table network fetch; under a frozen clock a failing fetch never
        # times out -> infinite loop eating memory (server-measured cgroup usage ~2.4GB/s
        # to OOM). The previous round's world is already closed here; under the real clock litellm falls back to the local table after one failure.
        agent_id = trace.run_meta.agent_id
        backend = self._backend_for(agent_id)
        if backend == "unavailable":
            raise RuntimeError(
                "backend unavailable for agent %r: the downstream "
                "continuation must run the REAL agent source; generic "
                "approximation is not an acceptable fallback" % agent_id)
        prewarm_ok = True
        if backend == "tau2":
            prewarm_ok = self._prewarm_tau2()
        elif backend == "smolagents":
            prewarm_ok = self._prewarm_smolagents()
        if not prewarm_ok:
            raise RuntimeError(
                "backend prewarm failed for agent %r (backend=%s): the "
                "downstream continuation must run the REAL agent source; "
                "generic approximation is not an acceptable fallback"
                % (agent_id, backend))
        bridge = AppWorldBridge(trace.run_meta.task_id)
        bridge.__enter__()
        try:
            replayed, delta_mode = self._restore_span_env(bridge, trace, span)
            if delta_mode:
                # delta-chain restore excludes session state: replay pre-span logins to rebuild the in-memory session
                self._replay_auth_tools(bridge, trace, span)
            # the same token prefetch as the collection side (uniform across all restore modes; the model never sees the token)
            self._prefetch_auth_tokens(bridge)
        except Exception:
            bridge.__exit__(None, None, None)
            raise
        conversation = list(
            (trace.artifacts.get("messages") or {}).get(span.request.messages_ref, []))
        recorded_tools = (trace.artifacts.get("recorded_tools") or {}).get(
            "tools:%d" % span.invocation_index)
        # structured downstream trace: init meta + upstream info (the
        # session window before the span and tools already executed), structure aligned with the original trace
        prev_results = (trace.artifacts.get("previous_tool_results") or {}).get(
            span.request.previous_tool_results_ref, [])
        prefix_messages = []
        for m in conversation:
            mm = dict(m)
            if (TRACE_PREFIX_MSG_MAX is not None
                    and isinstance(mm.get("content"), str)
                    and len(mm["content"]) > TRACE_PREFIX_MSG_MAX):
                mm["content"] = mm["content"][:TRACE_PREFIX_MSG_MAX]
            prefix_messages.append(mm)
        downstream_trace = {
            "meta": {
                "task_id": trace.run_meta.task_id,
                "run_id": trace.run_meta.run_id,
                "baseline_model": trace.run_meta.baseline_model,
                # the model actually used by the downstream continuation (distinct
                # from baseline_model: baseline_model is the model at trace collection
                # time; this field is the big model in this replay run's config -- when
                # switching local/cloud models, the data split relies on this field)
                "continuation_model": (getattr(baseline_client, "model", "")
                                       if baseline_client else ""),
                "offloaded_span": {
                    "invocation_index": span.invocation_index,
                    "node_id": span.node_id,
                    "node_pattern": span.span_descriptor.node_pattern,
                    "control_role": span.span_descriptor.control_role,
                    "contract_type": span.output_contract_runtime.contract_type,
                },
            },
            "prefix": {
                "messages": prefix_messages,
                "previous_tool_results": list(prev_results),
            },
            "spans": [],
            "terminal": {},
        }
        # per-agent continuation backend and protocol flags (driven by the trace's
        # agent identity, harness-generic); backend resolved before world creation (prewarm failure / unavailable already raised)
        info: List[str] = ["backend=%s agent=%s" % (backend, agent_id)]
        text_protocol = agent_id == "AppWorld-Smolagents"
        tool_choice = self._AGENT_TOOL_CHOICE.get(agent_id)
        # OpenAI-SDK traces record no session: re-render the real request
        # from task context (system prompt + demos + task input + trajectory items up to span j)
        openai_messages: Optional[List[Dict[str, Any]]] = None
        openai_items: Optional[List[Dict[str, Any]]] = None
        if agent_id == "AppWorld-OpenAI-Agents" and span.invocation_index >= 1:
            try:
                openai_messages, openai_items = self._render_openai_prompt(
                    bridge, trace, span, conversation)
            except Exception as exc:  # re-render failed: fall back to the transcript (recorded honestly)
                info.append("openai prompt render failed: %s: %s"
                            % (type(exc).__name__, exc))
        return AppWorldSession(
            runtime=self,
            bridge=bridge,
            conversation=conversation,
            span=span,
            trace=trace,
            injected=None,
            baseline_client=baseline_client,
            recorded_tools=recorded_tools,
            downstream_trace=downstream_trace,
            backend=backend,
            text_protocol=text_protocol,
            tool_choice=tool_choice,
            openai_messages=openai_messages,
            openai_items=openai_items,
            backend_info=info,
            # record the restore method honestly: env_delta = delta applied
            # (hash chain verified, consistent with recorded values -> no mismatch anymore); tool_replay = old-trace fallback
            snapshot_hash=("env_delta:" + replayed
                           if delta_mode
                           else "tool_replay:" + state_hash(replayed)),
            # raw hash of the restore chain (under env_delta it shares the
            # space of the collection-side recorded hash, for raw-to-raw
            # comparison by the engine / C0 gate; under tool_replay it is the hash of tool-replay results, not part of that comparison)
            restored_env_hash_raw=(replayed if delta_mode else ""),
            next_invocation_index=span.invocation_index,
        )

    # -- 14.4 step 4 --------------------------------------------------------
    def resolve_request(self, trace: TraceRun, span: SpanRecord,
                        session: Optional[AppWorldSession] = None) -> ResolvedRequest:
        messages = (trace.artifacts.get("messages") or {}).get(span.request.messages_ref, [])
        # OpenAI-SDK trace: the session was re-rendered at restore from
        # task context (real system prompt + trajectory), shared by the SLM
        # request and the continuation; if re-render is unavailable, fall back to the adapter's transcript (marked honestly)
        if session is not None and session.openai_messages:
            messages = session.openai_messages
        tools = None
        # give the SLM the SAME native function-calling protocol the recorded
        # agent used (the trace-recorded tool schemas win; generic fallback
        # otherwise).  Freeform spans keep tools=None.
        if span.output_contract_runtime.contract_type == "tool_call_list":
            recorded = (trace.artifacts.get("recorded_tools") or {}).get(
                "tools:%d" % span.invocation_index)
            if recorded:
                # two recorded formats: flat {name,...} (AutoGen export)
                # and pre-wrapped {type, function{name,...}} (simplified export)
                def _to_native(t):
                    if isinstance(t, dict) and isinstance(t.get("function"), dict)                             and t["function"].get("name"):
                        return t
                    if isinstance(t, dict) and t.get("name"):
                        return {"type": "function", "function": t}
                    return None
                tools = [x for x in (_to_native(t) for t in recorded) if x]
            else:
                allowed = span.output_contract_runtime.allowed_tools or []
                tools = [{"type": "function", "function": {
                    "name": t, "description": t,
                    "parameters": {"type": "object", "properties": {},
                                   "additionalProperties": True},
                }} for t in allowed]
        return ResolvedRequest(
            messages=list(messages),
            tools=tools,
            response_format=None,
            temperature=span.request.temperature,
            max_output_tokens=span.request.max_output_tokens,
            meta_hint=span.meta_hint,
            # legacy / smolagents native protocol forces tool_choice="required"
            tool_choice=session.tool_choice if session is not None else None,
        )

    # -- 14.4 step 5 --------------------------------------------------------
    def parse(self, span: SpanRecord, raw: str) -> ParseResult:
        parsed, kind = parse_llm_output(raw)
        expected = span.output_contract_runtime.contract_type
        if kind == "freeform_text" and expected != "freeform_text":
            return ParseResult(ok=False, parsed=None,
                               error="expected %s output, got freeform text" % expected)
        if kind != "freeform_text" and expected == "freeform_text":
            return ParseResult(ok=False, parsed=None,
                               error="expected freeform text, got structured output")
        return ParseResult(ok=True, parsed=parsed, error="")

    def schema_check(self, session: AppWorldSession, span: SpanRecord,
                     parsed: Any) -> bool:
        contract = span.output_contract_runtime.contract_type
        if contract == "tool_call_list":
            return not validate_schema(parsed, TOOL_CALL_SCHEMA)
        if contract == "answer_json":
            return not validate_schema(parsed, ANSWER_SCHEMA)
        return True

    def verify_local(self, session: AppWorldSession, span: SpanRecord,
                     parsed: Any) -> VerifierVerdict:
        verdict = VerifierVerdict()
        contract = span.output_contract_runtime.contract_type
        if contract == "tool_call_list":
            allowed = {t.replace("__", ".") for t in span.output_contract_runtime.allowed_tools}
            ok = True
            errors: List[str] = []
            for call in (parsed or {}).get("tool_calls", []):
                name = str(call.get("name", "")).replace("__", ".")
                if name not in allowed:
                    ok = False
                    errors.append("tool %r not in the allowed tool set" % name)
                if not isinstance(call.get("arguments"), dict):
                    ok = False
                    errors.append("tool %r arguments not a JSON object" % name)
            verdict = VerifierVerdict(triggered=True, detected_failure=not ok,
                                      passed=ok, executed=["contract_check"], errors=errors)
        if contract == "answer_json":
            ok = "status" in (parsed or {})
            verdict = VerifierVerdict(triggered=True, detected_failure=not ok,
                                      passed=ok, executed=["contract_check"],
                                      errors=[] if ok else ["answer_json missing 'status'"])
        return verdict

    # -- 14.4 step 6 --------------------------------------------------------
    def inject(self, session: AppWorldSession, span: SpanRecord, parsed: Any) -> bool:
        session.injected = parsed
        return True

    # -- 14.4 steps 7-8 -----------------------------------------------------
    @staticmethod
    def _assistant_calls_msg(calls: List[Dict[str, Any]], text_protocol: bool,
                             tag: str = "loop") -> Dict[str, Any]:
        """assistant message for a tool-call turn:
        - native protocol: assistant + tool_calls structure (synthesized deterministic ids)
        - text protocol (smolagents): "Calling tools:" text (the original agent's real shape)"""
        if text_protocol:
            return {"role": "assistant", "content": "Calling tools:\n" + json.dumps(
                [{"name": str(c.get("name", "")),
                  "arguments": c.get("arguments") if isinstance(c.get("arguments"), dict) else {}}
                 for c in calls], ensure_ascii=False)}
        tcs = []
        for i, c in enumerate(calls):
            args = c.get("arguments")
            if not isinstance(args, dict):
                args = {}
            tcs.append({
                "id": _synth_call_id(tag, i),
                "type": "function",
                "function": {"name": str(c.get("name", "")),
                             "arguments": json.dumps(args, ensure_ascii=False)},
            })
        return {"role": "assistant", "content": None, "tool_calls": tcs}

    @staticmethod
    def _tool_obs_msg(name: str, res: Dict[str, Any], call_id: str,
                      text_protocol: bool) -> Dict[str, Any]:
        """tool-execution observation message: native protocol pairs role=tool +
        tool_call_id; text protocol (smolagents) is role=user + "Observation:" prefix (original shape)."""
        text = ("Observation: %s"
                % json.dumps(res["result"], ensure_ascii=False)
                if res["ok"] else "Error: %s" % res["error"])
        if text_protocol:
            return {"role": "user", "content": text}
        return {"role": "tool", "tool_call_id": call_id, "name": name,
                "content": text}

    def _continue_loop(self, session: AppWorldSession,
                       stop_after_span: bool,
                       max_steps: int = MAX_LOOP_STEPS) -> Tuple[List[LMResponse], bool]:
        self._ensure_recorded_tokens(session)
        """Resume the tool-call loop from the restored conversation.  The
        intervened span consumes the injected parsed output; every later span
        calls the baseline client.  Returns (later_usage, completed).

        Protocol-aware (harness-generic):
        - native sessions (legacy / tau2 / openai spliced sessions): assistant
          carries a tool_calls structure, tool messages carry tool_call_id pairs
        - text protocol (smolagents): "Calling tools:" / "Observation:" text form,
          requests carry tool_choice="required" (the original agent's real request shape)
        - the rest (autogen/simplified fallback): keep the existing JSON text + bridge format note

        max_steps bounds the number of baseline turns (the engine passes the
        span's recorded remaining-step budget plus a margin).  Consecutive
        identical tool-call sets signal a stuck loop and stop it honestly."""
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        is_prediction = span.span_descriptor.node_pattern == "tool_selection"
        allowed = span.output_contract_runtime.allowed_tools or []
        text_protocol = bool(session.text_protocol)
        # resume persisted loop state (run_to_node -> run_to_completion share
        # one continuation; restarting would re-execute the injected span)
        if session.loop_messages is None:
            # init order matters: index first, so an exception inside the
            # note/catalog step cannot leave a half-initialized loop state
            session.loop_index = span.invocation_index
            if session.openai_messages is not None:
                # OpenAI-SDK: the runtime re-rendered the session from task
                # context (with the real system prompt), preferred over the adapter transcript
                session.loop_messages = [dict(m) for m in session.openai_messages]
            else:
                session.loop_messages = [dict(m) for m in session.conversation]
            # native / text-protocol sessions need no JSON format note (the
            # model outputs via the native tools protocol); only the text-JSON fallback path keeps the note
            native = any(m.get("tool_calls") for m in session.loop_messages)
            if not is_prediction and not native and not text_protocol:
                session.loop_messages = self._with_format_note(
                    session.loop_messages, allowed)
        messages = session.loop_messages
        usage: List[LMResponse] = []
        i = session.loop_index
        completed = session.loop_completed
        last_calls_key = ""
        repeat_count = 0
        steps_left = max(1, max_steps)
        while steps_left > 0:
            ctx = _span_ctx(i, span.output_contract_runtime.allowed_tools)
            turn_llm_ms = 0.0
            if i == span.invocation_index and not completed:
                if session.injected is None:
                    messages.append({"role": "assistant", "content":
                                     json.dumps({"__span_failure__": "not injected"}, ensure_ascii=False)})
                    break
                parsed = session.injected
                if not isinstance(parsed, dict):
                    break
                if is_prediction:
                    # the api-prediction turn is consumed without executing
                    # tools; the real agent then started a fresh tool-loop
                    # conversation -- switch to the next span's recorded window
                    next_conv = self._next_span_conversation(session, span)
                    if (session.trace is not None
                            and session.trace.run_meta.agent_id == "AppWorld-OpenAI-Agents"):
                        # openai: the next span's session is likewise re-rendered by the runtime
                        try:
                            next_span = next(
                                (sp for sp in session.trace.spans
                                 if sp.invocation_index == span.invocation_index + 1),
                                None)
                            if next_span is not None:
                                rendered, _items = self._render_openai_prompt(
                                    bridge, session.trace, next_span, next_conv)
                                session.loop_messages = [dict(m) for m in rendered]
                                next_conv = session.loop_messages
                        except Exception:
                            pass
                    messages = self._with_format_note(next_conv, allowed) \
                        if not any(m.get("tool_calls") for m in next_conv) \
                        and not text_protocol else [dict(m) for m in next_conv]
            else:
                t_llm0 = monotonic_ms()
                parsed, ok = self._complete_parseable(session, messages, ctx, usage)
                turn_llm_ms = monotonic_ms() - t_llm0
                if not ok:
                    break
            calls = parsed.get("tool_calls", [])
            calls_key = json.dumps(
                [(c.get("name"), c.get("arguments")) for c in calls],
                sort_keys=True, ensure_ascii=False)
            if calls_key == last_calls_key:
                repeat_count += 1
                if repeat_count >= 3:
                    messages.append({"role": "user", "content":
                                     "stagnation: identical tool calls repeated; loop stopped."})
                    self._trace_terminal(session, {
                        "status": "stopped: stagnation",
                        "world_completed": completed})
                    break
            else:
                last_calls_key = calls_key
                repeat_count = 0
            assistant_msg = self._assistant_calls_msg(
                calls, text_protocol, tag="loop")
            call_ids = [tc["id"] for tc in assistant_msg.get("tool_calls", [])]
            messages.append(assistant_msg)
            self._trace_step(session, {
                "span_type": "llm.call", "step": i - span.invocation_index,
                "source": "baseline",
                "elapsed_ms": turn_llm_ms,
                "output": {"tool_calls": [{"name": c.get("name", ""),
                                           "arguments": c.get("arguments")}
                                          for c in calls]}})
            for ci, call in enumerate(calls):
                name = str(call.get("name", "")).replace("__", ".")
                args = call.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                if name.endswith("complete_task"):
                    t_tool0 = monotonic_ms()
                    bridge.complete_task(args.get("answer"), args.get("status", "success"))
                    tool_ms = monotonic_ms() - t_tool0
                    completed = True
                    self._trace_step(session, {
                        "span_type": "tool.execution",
                        "step": i - span.invocation_index, "source": "baseline",
                        "tool": name.replace(".", "__"), "arguments": args,
                        "ok": True, "elapsed_ms": tool_ms,
                        "result": "Marked the active task complete."})
                    break
                t_tool0 = monotonic_ms()
                res = bridge.execute_tool(name, args)
                tool_ms = monotonic_ms() - t_tool0
                if res["ok"]:
                    res["result"] = self._sub_obs(session, name, res["result"])
                else:
                    res["error"] = self._sub_obs(session, name, res["error"])
                messages.append(self._tool_obs_msg(
                    name, res,
                    call_ids[ci] if ci < len(call_ids) else "",
                    text_protocol))
                self._trace_step(session, {
                    "span_type": "tool.execution",
                    "step": i - span.invocation_index, "source": "baseline",
                    "tool": name.replace(".", "__"), "arguments": args,
                    "ok": bool(res["ok"]), "elapsed_ms": tool_ms,
                    "result": (str(res["result"]) if res["ok"]
                               else str(res["error"]))[:TRACE_RESULT_MAX_LEN]})
            i += 1
            steps_left -= 1
            if stop_after_span and i > span.invocation_index:
                break
            if completed:
                break
        # persist the loop state for the next runtime call
        session.loop_messages = messages
        session.loop_index = i
        session.loop_completed = completed
        return usage, completed

    @staticmethod
    def _with_format_note(messages: List[Dict[str, str]],
                          allowed_tools: List[str]) -> List[Dict[str, str]]:
        """Append the bridge's output-format instruction + exact tool catalog
        to the system message (approximation of the recorded agent's native
        function calling, which the trace does not carry)."""
        catalog = "\n".join("- %s" % t for t in allowed_tools) or "(none recorded)"
        note = CONTINUATION_FORMAT_NOTE.replace("{tool_catalog}", catalog)
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m.get("role") == "system":
                m["content"] = m.get("content", "") + note
                return out
        out.append({"role": "system", "content": note})
        return out

    @staticmethod
    def _next_span_conversation(session: AppWorldSession,
                                span: SpanRecord) -> List[Dict[str, str]]:
        trace = session.trace
        if trace is None:
            return [dict(m) for m in session.conversation]
        next_ref = "msg:%d" % (span.invocation_index + 1)
        msgs = (trace.artifacts.get("messages") or {}).get(next_ref)
        if not msgs:
            return [dict(m) for m in session.conversation]
        return [dict(m) for m in msgs]

    def _complete_parseable(self, session: AppWorldSession,
                            messages: List[Dict[str, str]], ctx: SpanContext,
                            usage: List[LMResponse]) -> Tuple[Any, bool]:
        """One baseline completion that must yield a tool-call list.

        The request uses NATIVE function calling (OpenAI tools protocol) --
        the same protocol the recorded agent used -- so the model's tool
        format is enforced by the API, not by prompt compliance.  Text
        outputs still fall back to prompt-format parsing with one nudge
        retry before the loop gives up on the turn."""
        tool_schemas = self._tool_schemas(session)
        resp = session.baseline_client.complete(
            messages, ctx=ctx, tools=tool_schemas,
            temperature=ctx.temperature, max_tokens=ctx.max_output_tokens,
            tool_choice=session.tool_choice)
        usage.append(resp)
        if resp.error:
            return None, False
        if resp.tool_calls:
            return {"tool_calls": [{
                "name": c["name"], "arguments": c["arguments"],
            } for c in resp.tool_calls]}, True
        parsed, ok = _parse_continuation_output(resp.raw_text)
        if ok:
            return parsed, True
        nudged = [dict(m) for m in messages] + [
            {"role": "user", "content": NUDGE_MESSAGE}]
        resp2 = session.baseline_client.complete(
            nudged, ctx=ctx, tools=tool_schemas,
            temperature=ctx.temperature, max_tokens=ctx.max_output_tokens,
            tool_choice=session.tool_choice)
        usage.append(resp2)
        if resp2.error:
            return None, False
        if resp2.tool_calls:
            return {"tool_calls": [{
                "name": c["name"], "arguments": c["arguments"],
            } for c in resp2.tool_calls]}, True
        parsed2, ok2 = _parse_continuation_output(resp2.raw_text)
        return parsed2, ok2

    def _tool_schemas(self, session: AppWorldSession) -> List[Dict[str, Any]]:
        """Native tool schemas for the continuation: the trace-recorded ones
        (what the baseline model actually saw) win; otherwise rebuild them
        from the task's API docs."""
        span = session.span
        bridge = session.bridge
        if span is None or bridge is None:
            return []
        if session.recorded_tools:
            # compatible with both recorded formats: flat {name,...} or
            # pre-wrapped {type, function{name,...}}
            out = []
            for t in session.recorded_tools:
                if isinstance(t, dict) and isinstance(t.get("function"), dict)                         and t["function"].get("name"):
                    out.append(t)
                elif isinstance(t, dict) and t.get("name"):
                    out.append({"type": "function", "function": t})
            if out:
                return out
        return _openai_tool_schemas(
            bridge, span.output_contract_runtime.allowed_tools or [])

    # ------------------------------------------------------------------
    # OpenAI-SDK prompt reconstruction: the OpenAI collector records no
    # conversation, so the live runtime re-renders the REAL request with the
    # same templates the collection script used (instructions.txt +
    # app_descriptions + demos.json), then appends the recorded trajectory.
    # ------------------------------------------------------------------

    @staticmethod
    def _openai_prompt_files() -> Tuple[str, str, str]:
        """(instructions.txt, demos.json, task prompt template dir) -- exactly
        the same as the collection script run_benchmark_builtin_oai.py."""
        import appworld_agents.code.openai_agents as _oai_mod
        code_root = os.path.dirname(os.path.abspath(_oai_mod.__file__))
        exp_root = os.path.dirname(os.path.dirname(code_root))
        prompts = os.path.join(exp_root, "prompts", "function_calling_agent")
        return (os.path.join(prompts, "instructions.txt"),
                os.path.join(prompts, "demos.json"), prompts)

    @staticmethod
    def _convert_fc_demos_to_items(messages: List[Dict[str, Any]]
                                   ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """A local replica of openai_agents.run.convert_fc_demos_to_response_format
        (that module loads the openai-agents SDK on import; without the SDK the whole module
        cannot be imported, while this function itself is a pure conversion whose behavior matches
        the official implementation line by line): system -> instructions; assistant tool_calls ->
        function_call items; tool -> function_call_output items; other pure role/content -> as-is."""
        from appworld.common.random import get_unique_id
        system_prompt: Optional[str] = None
        items: List[Dict[str, Any]] = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
                continue
            if set(message.keys()) == {"role", "content"}:
                items.append(message)
            elif message["role"] == "assistant":
                for tool_call in message["tool_calls"]:
                    items.append({
                        "arguments": tool_call["function"]["arguments"],
                        "name": tool_call["function"]["name"],
                        "call_id": tool_call["id"],
                        "type": "function_call",
                        "id": "fc_" + get_unique_id(32),
                        "status": "completed",
                    })
            elif message["role"] == "tool":
                items.append({
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                    "type": "function_call_output",
                })
            else:
                raise ValueError("Unexpected message role: %s" % message["role"])
        return system_prompt, items

    @classmethod
    def _render_openai_prompt(cls, bridge: "AppWorldBridge",
                              trace: TraceRun, span: SpanRecord,
                              conversation: List[Dict[str, Any]]
                              ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Re-render the OpenAI-SDK agent's real request and append the trajectory up to span j.

        Returns (dict message list, SDK item list). The dict messages serve the SLM
        request and the generic loop; the item list serves the openai-SDK backend
        (Runner.run_streamed). span 0 (api prediction) is not on this path (callers
        use it only for j>=1) -- a prediction session's real prompt needs demo task
        loading, which is costly, and span 0 is freeform, so keep the adapter transcript."""
        from appworld.common.io import dump_yaml, read_file, read_json
        from appworld.common.prompts import load_prompt_to_chat_messages
        from appworld.common.text import render_template
        template_path, demos_path, _ = cls._openai_prompt_files()
        prompt_template = read_file(template_path)
        demo_messages = read_json(demos_path)
        task = bridge.world.task
        app_descriptions = dict(task.app_descriptions)
        app_descriptions.pop("api_docs", None)
        desc_string = dump_yaml(app_descriptions)
        header_content = render_template(
            prompt_template, instruction=task.instruction,
            main_user=task.supervisor, app_descriptions=desc_string,
            max_steps=70)
        header_messages = load_prompt_to_chat_messages(
            header_content, skip_system_message=False, only_header=True)
        test_input_content = render_template(
            prompt_template, instruction=task.instruction,
            main_user=task.supervisor, app_descriptions=desc_string,
            max_steps=70)
        test_input_messages = load_prompt_to_chat_messages(
            test_input_content, skip_system_message=True, only_body=True,
            end_at=1)
        system_prompt, items = cls._convert_fc_demos_to_items(
            header_messages + demo_messages + test_input_messages)
        # append the trajectory: assistant tool_calls and tool results of
        # spans 0..j-1 from the adapter transcript -> SDK items; placeholder
        # system/user messages (with the span_offload note marker) are skipped
        for m in conversation:
            role = m.get("role", "")
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    items.append({
                        "type": "function_call",
                        "name": str(fn.get("name", "")),
                        "arguments": str(fn.get("arguments", "{}")),
                        "call_id": str(tc.get("id") or _synth_call_id(
                            "oai", len(items))),
                        "id": "fc_" + str(tc.get("id") or len(items)),
                        "status": "completed",
                    })
            elif role == "tool" and m.get("tool_call_id"):
                items.append({
                    "type": "function_call_output",
                    "call_id": str(m["tool_call_id"]),
                    "output": str(m.get("content", "")),
                })
            elif role in ("system", "user") and "span_offload note" in \
                    str(m.get("content", "")):
                continue  # placeholder message: the real prompt was already re-rendered above
        return cls._items_to_dict_messages(system_prompt, items), items

    @staticmethod
    def _items_to_dict_messages(system_prompt: Optional[str],
                                items: List[Dict[str, Any]]
                                ) -> List[Dict[str, Any]]:
        """SDK items -> OpenAI dict messages (consecutive function_calls merge
        into one assistant message; function_call_output -> tool message)."""
        out: List[Dict[str, Any]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        i = 0
        n = len(items)
        while i < n:
            it = items[i]
            t = it.get("type")
            if t == "function_call":
                calls: List[Dict[str, Any]] = []
                while i < n and items[i].get("type") == "function_call":
                    c = items[i]
                    calls.append({
                        "id": str(c.get("call_id") or c.get("id")
                                  or _synth_call_id("oai", i)),
                        "type": "function",
                        "function": {
                            "name": str(c.get("name", "")),
                            "arguments": str(c.get("arguments", "{}")),
                        },
                    })
                    i += 1
                out.append({"role": "assistant", "content": None,
                            "tool_calls": calls})
                continue
            if t == "function_call_output":
                out.append({
                    "role": "tool",
                    "tool_call_id": str(it.get("call_id", "")),
                    "name": "",
                    "content": str(it.get("output", "")),
                })
                i += 1
                continue
            content = it.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    str(b.get("text", "")) for b in content
                    if isinstance(b, dict) and b.get("text"))
            out.append({"role": str(it.get("role", "user")),
                        "content": str(content or "")})
            i += 1
        return out

    # ------------------------------------------------------------------
    # Real-AutoGen continuation backend: the downstream is driven by the
    # ACTUAL agent source that produced the traces
    # (appworld_agents.code.autogen.agent -- the AutoGen AssistantAgent with
    # the agent's own tool functions), not by the simplified loop above.
    # ------------------------------------------------------------------

    @staticmethod
    def _recorded_to_autogen(messages: List[Dict[str, str]]) -> List[Any]:
        """Convert the restored conversation into autogen_core LLMMessages.

        Assistant turns serialized as [FunctionCall(id='...', ...)] text
        become AssistantMessage(FunctionCall list); the matching
        [FunctionExecutionResult(content='...')] turns become
        FunctionExecutionResult messages with the SAME call id (the OpenAI
        tool protocol requires the id linkage).  ToolCallSummaryMessage is
        UI-only and must never enter the model context (guard B). Unpaired
        tool_calls in the records (several calls but one recorded result)
        get a neutral failure receipt -- else assistant tool_calls lack a tool reply -> 400."""
        from autogen_agentchat.messages import (
            FunctionCall, FunctionExecutionResult, ToolCallSummaryMessage,
        )
        from autogen_core.models import (
            AssistantMessage, FunctionExecutionResultMessage,
            SystemMessage, UserMessage,
        )
        out: List[Any] = []
        pending_ids: List[str] = []
        pending_names: List[str] = []
        pending_calls: List[Any] = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "system":
                out.append(SystemMessage(content=content))
                continue
            if role == "assistant":
                calls = None
                if isinstance(m.get("tool_calls"), list):
                    # native tool_calls already spliced by the adapter (simplified session)
                    calls = [FunctionCall(
                        id=str(tc.get("id") or tc.get("call_id")
                               or "call_%d" % i),
                        arguments=str(tc.get("function", {}).get(
                            "arguments", "{}")),
                        name=str(tc.get("function", {}).get("name", "")),
                    ) for i, tc in enumerate(m["tool_calls"])]
                elif "[FunctionCall(" in content:
                    calls = []
                    for block in _FUNC_CALL_BLOCK_RE.findall(content):
                        id_m = re.search(r"id='([^']*)'", block)
                        name_m = re.search(r"name='([^']*)'", block)
                        args_m = re.search(r"arguments='((?:[^'\\]|\\.)*)'", block)
                        calls.append(FunctionCall(
                            id=id_m.group(1) if id_m else "call_%d" % len(calls),
                            arguments=args_m.group(1).replace("\\'", "'") if args_m else "{}",
                            name=name_m.group(1) if name_m else "",
                        ))
                elif content.strip().startswith("["):
                    try:
                        raw = json.loads(content)
                        calls = [FunctionCall(
                            id="call_%d" % i,
                            arguments=json.dumps(c.get("arguments") or {}, ensure_ascii=False),
                            name=str(c.get("name", "")),
                        ) for i, c in enumerate(raw)]
                    except ValueError:
                        calls = None
                if calls:
                    pending_ids = [c.id for c in calls]
                    pending_calls = calls
                    out.append(AssistantMessage(content=calls, source="assistant"))
                else:
                    pending_ids, pending_calls = [], []
                    out.append(AssistantMessage(content=content, source="assistant"))
                continue
            # user / tool turns: execution results link back to pending calls
            # one message may hold several results (exporter merged records):
            # parse all, pair them in order with pending calls -- only transcribe
            # what the original trace has, never add. Note content may be followed
            # by name=/call_id= fields, so do not require an immediate '\)', or records with fields never match
            results = re.findall(
                r"FunctionExecutionResult\(content='((?:[^'\\]|\\.)*)'", content)
            if pending_ids and results:
                consumed = 0
                for i, result_text in enumerate(results):
                    if i >= len(pending_calls):
                        break
                    try:
                        result_text = json.loads(result_text.replace("\\'", "'"))
                    except ValueError:
                        pass
                    call = pending_calls[i]
                    # tool results in the model context must be FunctionExecutionResultMessage
                    # (the container for a FunctionExecutionResult list) -- this is what the
                    # real agent's internal loop does in autogen 0.7.5 (_assistant_agent.py:1240
                    # add_message(FunctionExecutionResultMessage(content=...))).
                    # a bare FunctionExecutionResult is rejected by the model client
                    # (Unknown message type), and ToolCallSummaryMessage as a UI summary is even worse
                    out.append(FunctionExecutionResultMessage(
                        content=[FunctionExecutionResult(
                            content=str(result_text),
                            name=call.name,
                            call_id=call.id,
                        )]))
                    consumed += 1
                pending_ids = pending_ids[consumed:]
                pending_calls = pending_calls[consumed:]
            elif pending_ids and role == "tool":
                result_text = content
                call = pending_calls[0]
                out.append(FunctionExecutionResultMessage(
                    content=[FunctionExecutionResult(
                        content=str(result_text),
                        name=call.name,
                        call_id=call.id,
                    )]))
                pending_ids = pending_ids[1:]
                pending_calls = pending_calls[1:]
            else:
                pending_ids, pending_calls = [], []
                out.append(UserMessage(content=content, source="user"))
        return out

    def _autogen_tools(self, session: AppWorldSession) -> List[Any]:
        """Continuation tool set: content (tool names + schema) always comes from the trace
        recorded native request (recorded_tools = gen_ai.request.tools), verbatim identical
        to what the native model saw; execution wrapped as a harness-side BaseTool (via
        bridge.execute_tool straight to the same API functions of world.apis). Fall back
        to api_docs + allowed filtering when the trace recorded no tools.

        Do not call appworld_agents' get_autogen_tools/_build_tool_function: the local version's
        code generator has a bug (it writes argument values as literal repr(v) when generating
        the world.execute code string, so parameterized calls hit NameError -- observed:
        venmo__login and supervisor__complete_task all report "name 'v' is not defined"). The
        agent source is not modifiable (hard rule); tool-set content stays unchanged, only the
        execution wrapper is swapped: bridge.execute_tool calls world.apis[app][api](**kwargs)
        directly, semantically equal to world.execute's print(...) (same API functions),
        complete_task likewise writes the answer into the world.
        """
        from typing import Optional  # noqa: F401
        import pydantic
        from autogen_core.tools import BaseTool
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        if not hasattr(session, "autogen_tool_times"):
            session.autogen_tool_times = []
        result_model = pydantic.create_model("ToolResult", result=(str, ""))
        tools: List[Any] = []
        # prediction span (tool_selection): the injected API list decides the tool
        # set by the collection side's mechanism (run_benchmark_autogen.py: predicted
        # list filters api_docs + supervisor fully allowed, access_token out of LLM schema).
        # When T_span intervenes on the prediction, the SLM's new list thus truly drives
        # the downstream tool surface; I-arm injection = the recorded list, equivalent to the recorded next-span tools.
        prediction_docs: List[Dict[str, Any]] = []
        if span.span_descriptor.node_pattern == "tool_selection":
            _predicted = self._predicted_apis_from_text(bridge, session.injected)
            if _predicted:
                for _doc in bridge.world.task.api_docs.function_calling():
                    _fn = _doc["function"]
                    _name = _fn["name"]
                    if _name not in _predicted \
                            and not _name.startswith("supervisor__"):
                        continue
                    _props = {k: v for k, v in
                              (_fn.get("parameters") or {}).get("properties", {}).items()
                              if k != "access_token"}
                    _required = {k for k in
                                 (_fn.get("parameters") or {}).get("required", [])
                                 if k != "access_token"}
                    _fn2 = dict(_fn)
                    _fn2["parameters"] = {"type": "object",
                                          "properties": _props,
                                          "required": sorted(_required)}
                    prediction_docs.append(_fn2)
        # tool-set content and schema always come from the trace-recorded native
        # request (recorded_tools = gen_ai.request.tools): api_docs-rebuilt schemas
        # differ from native ones (sets/descriptions/defaults); in thinking mode the
        # model gives empty replies and no tool calls on rebuilt schemas but produces
        # tool calls normally on recorded schemas (diag: prompt 1663 vs 1366 tokens,
        # V1 recorded schema all correct). Only fall back to api_docs + allowed filter when the trace recorded no tools.
        recorded = [] if prediction_docs else (
            ((session.trace.artifacts.get("recorded_tools") or {})
             .get("tools:%d" % span.invocation_index)) or [])
        if not recorded and not prediction_docs:
            # prediction span (tool_selection) is a plain-text API prediction request with
            # no recorded tools (tools:0 = None); its continuation task-loop tool set =
            # the next span's native request tools (same alignment principle as _autogen_seed
            # seeding from the next span's session window)
            for _s in session.trace.spans:
                if _s.invocation_index > span.invocation_index:
                    _nxt = (session.trace.artifacts.get("recorded_tools")
                            or {}).get("tools:%d" % _s.invocation_index)
                    if _nxt:
                        recorded = _nxt
                        break
        fn_docs = list(prediction_docs)
        if recorded:
            for f in recorded:
                if isinstance(f, dict) and f.get("type") == "function" \
                        and isinstance(f.get("function"), dict):
                    fn_docs.append(f["function"])
        if not fn_docs:
            # allowed_tools is in dotted form (supervisor.complete_task),
            # api docs use double underscores: must normalize
            allowed = {t.replace(".", "__")
                       for t in (span.output_contract_runtime.allowed_tools
                                 or [])}
            allowed.add("supervisor__complete_task")
            fn_docs = [d["function"]
                       for d in bridge.world.task.api_docs.function_calling()
                       if d["function"]["name"] in allowed]
        names = {f.get("name") for f in fn_docs}
        if "supervisor__complete_task" not in names:
            for doc in bridge.world.task.api_docs.function_calling():
                if doc["function"]["name"] == "supervisor__complete_task":
                    fn_docs.append(doc["function"])
                    break
        rec_names = set()
        if recorded:
            rec_names = {f["function"]["name"] for f in recorded
                         if isinstance(f, dict)
                         and isinstance(f.get("function"), dict)}
        for fn in fn_docs:
            name = fn["name"]
            props = fn["parameters"].get("properties", {})
            required = set(fn["parameters"].get("required", []))
            fields = {}
            for pname, pinfo in props.items():
                ptype = {"string": str, "integer": int, "number": float,
                         "boolean": bool, "array": list, "object": dict}.get(
                             pinfo.get("type"), str)
                if "default" in pinfo:
                    fields[pname] = (ptype, pinfo["default"])
                elif pname in required:
                    fields[pname] = (ptype, ...)
                else:
                    fields[pname] = (Optional[ptype], None)
            args_model = pydantic.create_model(
                "Args_" + name.replace("__", "_"), **fields)
            desc = fn.get("description", "")
            # recorded schema goes out verbatim: a pydantic rebuild drops
            # per-property description etc. (observed: 4/10 tools differ
            # from the records, prompt 2061 vs 1663, model still replies
            # empty); the schema property returns the recorded dict directly so outbound tools are byte-identical to the native request
            rec_schema = fn if name in rec_names else None

            class _T(BaseTool[args_model, result_model]):  # type: ignore
                def __init__(self, _rec_schema):
                    super().__init__(args_model, result_model, name, desc)
                    self._rec_schema = _rec_schema

                @property
                def schema(self):
                    if self._rec_schema is not None:
                        return self._rec_schema
                    return super().schema

                async def run(self, args, cancellation_token,
                              _name=name):  # default arg binds the current loop value
                    data = AppWorldAgentRuntime._maybe_inject_token(
                        bridge, _name, args.model_dump())
                    t_tool0 = monotonic_ms()
                    res = bridge.execute_tool(_name, data)
                    session.autogen_tool_times.append(
                        (_name, monotonic_ms() - t_tool0))
                    AppWorldAgentRuntime._maybe_cache_token(bridge, _name, res)
                    text = (str(res["result"]) if res["ok"]
                            else "Error: %s" % res["error"])
                    return result_model(result=self._sub_obs(
                        session, _name, text))
            tools.append(_T(rec_schema))
        return tools

    def _autogen_inject_span(self, session: AppWorldSession) -> Tuple[List[Any], bool]:
        """Execute the intervened span's injected output through the REAL
        tool functions and return the autogen messages to append.
        Returns (messages, completed)."""
        import asyncio
        from autogen_agentchat.messages import (
            FunctionCall, FunctionExecutionResult, ToolCallSummaryMessage,
        )
        from autogen_core.models import (
            AssistantMessage, FunctionExecutionResultMessage,
        )
        from appworld_agents.code.autogen.agent import _current_world
        bridge = session.bridge
        assert bridge is not None
        parsed = session.injected
        if not isinstance(parsed, dict):
            from autogen_core.models import UserMessage
            return [UserMessage(content="Error: no usable output injected "
                                          "for this span.", source="user")], False
        if not parsed.get("tool_calls"):
            from autogen_core.models import AssistantMessage, UserMessage
            if "__span_failure__" in parsed:
                # fallback=none: the SLM output was rejected and no rescue is
                # configured -- surface the failure honestly so the agent's
                # own recovery path takes over (strict-failure accounting)
                self._trace_step(session, {
                    "span_type": "llm.call", "step": 0, "source": "slm",
                    "output": {"text": "Error: the offloaded span produced no "
                                "usable output: %s" % str(
                                    parsed.get("__span_failure__"))[:TRACE_RESULT_MAX_LEN]}})
                return [UserMessage(
                    content="Error: the offloaded span produced no usable "
                            "output: %s" % parsed.get("__span_failure__"),
                    source="user")], False
            # freeform spans (tool_selection api prediction, etc.): the
            # injected answer is the span's own output -- seed it as the
            # assistant turn and continue (the tool set is fixed by the
            # span's allowed_tools, exactly as in the recorded run)
            text = parsed.get("text") or json.dumps(parsed, ensure_ascii=False)
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"text": str(text)[:TRACE_RESULT_MAX_LEN]}})
            return [AssistantMessage(content=str(text), source="assistant")], False
        calls = [FunctionCall(
            id=_synth_call_id("inj", i),
            arguments=json.dumps(c.get("arguments") or {}, ensure_ascii=False),
            name=str(c.get("name", "")),
        ) for i, c in enumerate(parsed["tool_calls"])]
        # injected execution goes through bridge.execute_tool (straight to
        # the same API functions of world.apis, semantically equivalent to
        # world.execute); bypass appworld_agents' tool functions (its code generator has the repr(v) bug, parameterized calls hit NameError)
        results = []
        for call in calls:
            nm = call.name.replace(".", "__")
            try:
                kwargs = json.loads(call.arguments) if call.arguments else {}
                if not isinstance(kwargs, dict):
                    kwargs = {}
            except ValueError:
                kwargs = {}
            kwargs = self._maybe_inject_token(bridge, nm, kwargs)
            res = bridge.execute_tool(nm, kwargs)
            self._maybe_cache_token(bridge, nm, res)
            results.append(self._sub_obs(
                session, nm,
                str(res["result"]) if res["ok"] else "Error: %s" % res["error"]))
        out: List[Any] = [AssistantMessage(content=calls, source="assistant")]
        completed = False
        # structured trace: the injected turn's calls and real execution results (span structure aligned with the original trace)
        self._trace_step(session, {
            "span_type": "llm.call", "step": 0, "source": "slm",
            "output": {"tool_calls": [{"name": c.name,
                                       "arguments": _json_args_or_empty(c.arguments)}
                                      for c in calls]}})
        for call, result in zip(calls, results):
            if call.name.replace(".", "__").endswith("complete_task"):
                completed = True
            result_txt = str(result)
            self._trace_step(session, {
                "span_type": "tool.execution", "step": 0, "source": "slm",
                "tool": call.name.replace(".", "__"),
                "arguments": _json_args_or_empty(call.arguments),
                "ok": not result_txt.startswith("Error"),
                "result": result_txt[:TRACE_RESULT_MAX_LEN]})
            # tool results in the model context must be FunctionExecutionResultMessage
            # (what the real agent's internal loop does); bare FunctionExecutionResult and
            # ToolCallSummaryMessage are both rejected by the model client
            out.append(FunctionExecutionResultMessage(
                content=[FunctionExecutionResult(
                    content=str(result),
                    name=call.name,
                    call_id=call.id,
                )]))
        return out, completed

    # ------------------------------------------------------------------
    # Simplified-agent continuation backend: the AppWorld built-in
    # SimplifiedFunctionCallingAgent loop, whose termination is TASK-LEVEL
    # (world.task_completed()) rather than model-text based.
    # ------------------------------------------------------------------

    def _recorded_to_dict_messages(self, messages):
        """Convert the restored conversation into the simplified agent's
        OpenAI-native dict format (assistant turns carry tool_calls; tool
        turns carry tool_call_id linkage)."""
        out = []
        pending_calls = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            if role == "system":
                out.append({"role": "system", "content": content})
                continue
            if role == "assistant":
                tcs = None
                if isinstance(m.get("tool_calls"), list):
                    # native tool_calls already spliced by the adapter (e.g. simplified session)
                    tcs = []
                    for i, tc in enumerate(m["tool_calls"]):
                        fn = tc.get("function") or {}
                        cid = tc.get("id") or tc.get("call_id") or "call_%d" % i
                        tcs.append({"id": cid, "call_id": cid,
                                    "type": "function", "function": {
                            "name": str(fn.get("name", "")),
                            "arguments": str(fn.get("arguments", "{}")),
                        }})
                elif "[FunctionCall(" in content:
                    tcs = []
                    for block in _FUNC_CALL_BLOCK_RE.findall(content):
                        id_m = re.search(r"id='([^']*)'", block)
                        name_m = re.search(r"name='([^']*)'", block)
                        args_m = re.search(r"arguments='((?:[^'\\]|\\.)*)'", block)
                        cid = id_m.group(1) if id_m else "call_%d" % len(tcs)
                        tcs.append({"id": cid, "call_id": cid,
                                    "type": "function", "function": {
                            "name": name_m.group(1) if name_m else "",
                            "arguments": (args_m.group(1).replace("\\'", "'")
                                          if args_m else "{}"),
                        }})
                elif content.strip().startswith("["):
                    try:
                        raw = json.loads(content)
                        tcs = [{"id": "call_%d" % i, "call_id": "call_%d" % i,
                                "type": "function",
                                "function": {"name": str(c.get("name", "")),
                                             "arguments": json.dumps(
                                                 c.get("arguments") or {})}}
                               for i, c in enumerate(raw)]
                    except ValueError:
                        tcs = None
                if tcs:
                    pending_calls = list(tcs)
                    # assistant messages with tool_calls keep their narration
                    # text (the adapter already spliced tool_calls back into
                    # the narration turn). reasoning_content: the trace does
                    # not record thinking content, so pad it with the narration or fallback text, keeping structure consistent and satisfying DeepSeek validation
                    content_text = (m.get("content") or "").strip() or None
                    out.append({"role": "assistant",
                                "content": content_text,
                                "tool_calls": tcs,
                                "reasoning_content":
                                    (content_text or RC_PLACEHOLDER)})
                else:
                    pending_calls = []
                    out.append({"role": "assistant", "content": content})
                continue
            result_m = re.search(
                r"FunctionExecutionResult\(content='((?:[^'\\]|\\.)*)'", content)
            if pending_calls and result_m:
                try:
                    result_text = json.loads(result_m.group(1).replace("\\'", "'"))
                except ValueError:
                    result_text = result_m.group(1)
                call = pending_calls[0]
                out.append({"role": "tool",
                            "tool_call_id": call["call_id"],
                            "name": call["function"]["name"],
                            "content": str(result_text)})
                pending_calls = pending_calls[1:]
                continue
            if role == "tool" and pending_calls:
                # simplified sessions' tool messages carry only bare content;
                # add tool_call_id and name so the assistant tool_calls get matching tool replies
                call = pending_calls[0]
                out.append({"role": "tool",
                            "tool_call_id": call["call_id"],
                            "name": call["function"]["name"],
                            "content": content})
                pending_calls = pending_calls[1:]
                continue
            if role == "tool" and m.get("tool_call_id"):
                # adapter already paired the tool message (with tool_call_id): pass it through
                out.append({"role": "tool",
                            "tool_call_id": m["tool_call_id"],
                            "name": m.get("name", ""),
                            "content": content})
                continue
            pending_calls = []
            out.append({"role": "user", "content": content})
        return out

    @staticmethod
    def _span_narration(session: AppWorldSession, span: SpanRecord) -> str:
        """Extract from the next span's session window the baseline model's narration
        text for that span's turn (last assistant message with non-empty content and
        no tool_calls), to fill the injected message's reasoning_content (DeepSeek thinking mode requires non-empty)."""
        trace = session.trace
        if trace is None:
            return ""
        conv = (trace.artifacts.get("messages") or {}).get(
            "msg:%d" % (span.invocation_index + 1), [])
        # search only after the last task instruction (exclude tutorial turns like 'Sounds good!')
        start = 0
        for i, m in enumerate(conv):
            if m.get("role") == "user" and (
                    "# Task Instruction" in str(m.get("content", ""))
                    or "# Real Task Instruction" in str(m.get("content", ""))):
                start = i
        for m in reversed(conv[start + 1:]):
            if m.get("role") != "assistant":
                continue
            # the narration turn may already have tool_calls spliced in; that does not affect extracting its narration text
            text = str(m.get("content") or "").strip()
            if text and text != "None":
                return text
        return ""

    @staticmethod
    def _repair_skipped_tool_calls(agent: Any, inputs: List[Any],
                                   tool_names: List[str]) -> List[Dict[str, Any]]:
        """Repair the protocol gap from the agent skipping ill-formatted tool names.

        The agent's second_onwards skips tool names whose separator count is not 1 and produces no
        ExecutionIO, so in the next request those tool_calls have no tool reply -> 400
        "insufficient tool messages following tool_calls message". For each skipped call, pad one
        tool message; content uses a **neutral execution-failure receipt** (same format as the
        NameError the execution engine gives for undefined function names) -- only pad the
        protocol, no correction/guidance, and do not change the amount of information the
        downstream model sees. Returns the list of repaired calls (for the trace record)."""
        repaired: List[Dict[str, Any]] = []
        if not getattr(agent, "messages", None):
            return repaired
        last = agent.messages[-1]
        if last.get("role") != "assistant" or not last.get("tool_calls"):
            return repaired
        executed_ids = {str(i.metadata.get("id")) for i in inputs
                        if getattr(i, "metadata", None)}
        for tc in last["tool_calls"]:
            cid = str(tc.get("id") or tc.get("call_id") or "")
            if cid in executed_ids:
                continue
            name = str((tc.get("function") or {}).get("name", ""))
            agent.messages.append({
                "role": "tool", "tool_call_id": cid, "name": name,
                "content": ("Execution failed. Traceback:\n"
                            '  File "<python-input>", line 1, in <module>\n'
                            "NameError: name '%s' is not defined" % name),
            })
            repaired.append({"name": name, "id": cid})
        return repaired

    @staticmethod
    def _trace_step(session: AppWorldSession, entry: Dict[str, Any]) -> None:
        """Append one span to the session's structured downstream trace (shared by injection / continuation).

        The entry structure aligns with the original trace's spans: {"span_type": "llm.call"|"tool.execution", "step": n, "source": "slm"|"baseline",
        ...}

        Timestamps (monotonic clock, freezegun-immune): each entry is automatically
        stamped with ts_start_ms / ts_end_ms -- all spans within one session land on one
        time axis, giving both a single span's runtime and the gaps between spans (next
        ts_start minus previous ts_end). ts_start is derived from ts_end - elapsed_ms
        (the elapsed start measured at the call site is the real start, the append moment
        is the real end; in between is only entry-construction overhead,
        sub-millisecond)"""
        now_ms = monotonic_ms()
        elapsed = float(entry.get("elapsed_ms") or 0.0)
        entry.setdefault("elapsed_ms", elapsed)
        entry["ts_end_ms"] = now_ms
        entry["ts_start_ms"] = now_ms - elapsed
        # timing fields isomorphic to the collection-side trace (cf. trace_collector._add_span:
        # elapsed measured at the call site, backfilled round 3; gap = previous record point
        # -> this record point, equivalent under the hook sequence to "previous end -> this start", first entry 0.0, round 3)
        entry["duration_ms"] = round(elapsed, 3)
        if session.downstream_trace is None:
            session.downstream_trace = {"meta": {}, "prefix": {},
                                        "spans": [], "terminal": {}}
        spans_list = session.downstream_trace["spans"]
        prev_end = (spans_list[-1].get("ts_end_ms")
                    if spans_list else entry["ts_start_ms"])
        entry["gap_from_prev_ms"] = round(
            max(0.0, entry["ts_start_ms"] - prev_end), 3)
        spans_list.append(entry)

    @staticmethod
    def _trace_terminal(session: AppWorldSession, entry: Dict[str, Any]) -> None:
        """Write the downstream trace's terminal state (status / world_completed / backend_info, etc.)."""
        if session.downstream_trace is None:
            session.downstream_trace = {"meta": {}, "prefix": {},
                                        "spans": [], "terminal": {}}
        info = getattr(session, "backend_info", None) or []
        if info:
            entry.setdefault("backend_info", info)
        session.downstream_trace["terminal"] = entry

    def _inject_dict_span(self, session):
        """Execute the intervened span's output through the world's
        batch_execute (the simplified agent's execution path) and return
        (assistant message, ExecutionIO outputs, completed).

        The injected tool calls are executed for REAL against the world, and
        the ExecutionIO contents carry the real results (not the command
        echo), so both the world state and the tool messages the downstream
        model sees reflect the span's effects -- aligned with the autogen
        backend's _autogen_inject_span."""
        from appworld_agents.code.simplified.agent import ExecutionIO
        bridge = session.bridge
        assert bridge is not None
        parsed = session.injected
        if not isinstance(parsed, dict) or not parsed.get("tool_calls"):
            text = ""
            if isinstance(parsed, dict):
                text = str(parsed.get("text") or parsed.get("__span_failure__", ""))
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"text": (text or "no usable output")[:TRACE_RESULT_MAX_LEN]}})
            return [{"role": "user", "content": text or "no usable output"}], [], False
        tcs = []
        commands = []
        completed = False
        for i, c in enumerate(parsed["tool_calls"]):
            name = str(c.get("name", "")).replace(".", "__")
            args = c.get("arguments")
            if not isinstance(args, dict):
                args = {}
            args_str = json.dumps(args, ensure_ascii=False)
            cid = _synth_call_id("inj", i)
            tcs.append({"id": cid, "call_id": cid,
                        "type": "function",
                        "function": {"name": name, "arguments": args_str}})
            # consistent with the agent's own execution path: hand
            # print(app__api(**args)) to world.batch_execute
            commands.append("print(%s(**%s))" % (name, args_str))
            if name.endswith("complete_task"):
                completed = True
        # really execute the injected tool calls: world state advances past span j
        try:
            outputs_raw = bridge.world.batch_execute(commands)
        except Exception as exc:
            outputs_raw = ["Error: %s: %s" % (type(exc).__name__, exc)
                           for _ in commands]
        if not isinstance(outputs_raw, list) or len(outputs_raw) < len(commands):
            outputs_raw = (list(outputs_raw) if isinstance(outputs_raw, list)
                           else []) + ["(no output)"] * len(commands)
        outputs_raw = list(outputs_raw)[: len(commands)]
        eios = [ExecutionIO(
                    content=str(outputs_raw[i]),
                    metadata={"id": tcs[i]["id"],
                              "function_name": tcs[i]["function"]["name"]})
                for i in range(len(commands))]
        # structured trace: the injected turn's calls and real execution results (span structure aligned with the original trace)
        self._trace_step(session, {
            "span_type": "llm.call", "step": 0, "source": "slm",
            "output": {"tool_calls": [{"name": tcs[i]["function"]["name"],
                                       "arguments": _json_args_or_empty(
                                           tcs[i]["function"]["arguments"])}
                                      for i in range(len(tcs))]}})
        for i in range(len(eios)):
            content = str(eios[i].content)
            self._trace_step(session, {
                "span_type": "tool.execution", "step": 0, "source": "slm",
                "tool": tcs[i]["function"]["name"],
                "arguments": _json_args_or_empty(tcs[i]["function"]["arguments"]),
                "ok": "Execution failed" not in content
                      and not content.startswith("Error"),
                "result": content[:TRACE_RESULT_MAX_LEN]})
        # align with the original run's assistant message structure:
        #   content           = the turn's narration text (in the original
        #                       run the narration sits in content; extracted
        #                       from the next session window when the baseline has one, else None)
        #   reasoning_content = thinking text. The trace records no thinking content; fill order:
        #                       1. the SLM's own output text (truly generated content);
        #                       2. deterministic fallback text (non-empty in the original run, DeepSeek thinking mode requires the field non-empty)
        narration = self._span_narration(session, session.span)
        slm_text = str(getattr(session, "slm_raw_text", "") or "").strip()[:2000]
        rc = slm_text if slm_text else RC_PLACEHOLDER
        return [{"role": "assistant",
                 "content": narration or None,
                 "tool_calls": tcs,
                 "reasoning_content": rc}], eios, completed

    def _predicted_apis_from_text(self, bridge, injected: Any) -> List[str]:
        """Parse the predicted API list from the injected api-prediction free text.

        Rules same as appworld's APIPredictor.predicted_output_to_apis: check line
        by line, keep only names existing in the task's real API docs (matched after
        dot/double-underscore normalization), dedupe keeping order. Return an empty
        list when no valid API can be parsed (the caller falls back to the allowed_tools superset)."""
        text = ""
        if isinstance(injected, dict):
            text = str(injected.get("text") or "")
        api_names = {
            doc["function"]["name"].lower()
            for doc in bridge.world.task.api_docs.function_calling()
        }
        names: List[str] = []
        for line in text.strip().splitlines():
            name = line.strip().lower().replace(".", "__")
            if name in api_names and name not in names:
                names.append(name)
        return names

    def _continue_with_simplified(self, session, max_steps):
        """Drive the downstream with the REAL simplified agent loop (the
        agent's own next_execution_inputs / batch_execute machinery with the
        task-level termination condition)."""
        # downstream usage capture: the agent's own next_execution_inputs_usage_and_status
        # returns Usage(tokens) per round, previously discarded -> downstream tokens all 0.
        # Now accumulated into LMResponse for the engine to settle (B vs T_span downstream cost basis)
        usage_out: List[LMResponse] = []
        import appworld as _aw
        from appworld_agents.code.simplified.agent import ExecutionIO
        from appworld_agents.code.simplified.function_calling_agent import (
            SimplifiedFunctionCallingAgent,
        )
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        bc = session.baseline_client
        import inspect as _inspect
        from appworld_agents.code.simplified.api_predictor import             APIPredictor as _SimpleAPIPredictor
        from appworld_agents.code.simplified.function_calling_agent import (
            SimplifiedFunctionCallingAgent as _SFC,
        )
        from appworld_agents.code.simplified.language_model import (
            LanguageModel as _LM,
        )

        # signature-driven construction: pass args per the installed version's constructor
        # signature, immune to version differences. Environment fallback: the constructor chain unconditionally reads these two env vars.
        os.environ.setdefault("MODEL_SERVER_URL", bc.base_url or "http://localhost:8000")
        os.environ.setdefault("OPENAI_API_KEY", bc.api_key or "placeholder")
        import appworld as _aw
        repo_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(_aw.__file__))))

        _lm_params = set(_inspect.signature(_LM.__init__).parameters)
        model_config = {"name": bc.model}
        # client path switch: openai (default) | litellm
        _lm_client = os.environ.get("SPAN_OFFLOAD_CONT_LM_CLIENT", "openai")
        for _k, _v in {
            "client_name": _lm_client,
            "api_type": "chat_completions",
            "use_cache": False,
            "max_retries": 3,
        }.items():
            if _k in _lm_params:
                model_config[_k] = _v
        # thinking mode: enabled by default for all agents (server default,
        # same basis as collection); inject no switch to turn it off
        if "cost_per_token" in _lm_params:
            model_config["cost_per_token"] = {
                "input_cache_miss": 0.0,
                "input_cache_hit": 0.0,
                "input_cache_write": 0.0,
                "output": 0.0,
            }
        # base_url/api_key/temperature/max_tokens passed through via **generation_kwargs
        model_config.update({
            "base_url": bc.base_url,
            "api_key": bc.api_key or "placeholder",
            "temperature": bc.temperature,
            "max_tokens": bc.max_output_tokens,
            # consistent with the trace collection config: a fixed seed improves determinism (accepted on the openai path)
            "seed": 100,
        })

        _ap_params = set(_inspect.signature(_SimpleAPIPredictor.__init__).parameters)
        api_pred_cfg = {"model_config": dict(model_config)}
        if "prompt_file_path" in _ap_params:
            api_pred_cfg["prompt_file_path"] = os.path.join(
                repo_root, "experiments", "prompts", "api_predictor.txt")
        if "demo_task_ids" in _ap_params:
            api_pred_cfg["demo_task_ids"] = []
        if "mode" in _ap_params:
            api_pred_cfg["mode"] = "all"
        # note: app_api_separator is passed by the agent constructor itself, not here

        agent = _SFC(
            prompt_file_path=os.path.join(
                repo_root, "experiments", "prompts",
                "function_calling_agent", "instructions.txt"),
            api_predictor_config=api_pred_cfg,
            model_config=model_config,
            max_steps=max_steps,
        )
        if session.simplified_agent is None:
            agent.initialize(bridge.world)
            is_prediction = span.span_descriptor.node_pattern == "tool_selection"
            if is_prediction:
                # api prediction turn: the predictor session does not enter the task
                # loop -- the original agent discards the predictor session right after
                # predicting and the task loop starts from a fresh session (same as
                # _continue_loop's is_prediction branch). Instead seed with the next
                # span's session window (the task-loop start), avoiding the downstream model spinning without tool calls in the predictor-session format
                conv = self._next_span_conversation(session, span)
                if not conv:
                    conv = session.conversation
                agent.messages = self._recorded_to_dict_messages(conv)
            else:
                agent.messages = self._recorded_to_dict_messages(session.conversation)
            allowed = set(span.output_contract_runtime.allowed_tools or [])
            allowed.add("supervisor__complete_task")
            if is_prediction:
                # the real effect channel of the predicted output = tool-set filtering
                # (original agent semantics: predicted APIs decide the task loop's available
                # functions). Parse the SLM's prediction text (keep only lines appearing in
                # the task's real API docs, rules same as APIPredictor.predicted_output_to_apis);
                # when nothing parses, fall back to the allowed_tools superset
                predicted = self._predicted_apis_from_text(bridge, session.injected)
                if predicted:
                    allowed = set(predicted) | {"supervisor__complete_task"}
            agent.functions = [
                doc for doc in bridge.world.task.api_docs.function_calling()
                if doc["function"]["name"] in allowed]
            agent.step_number = 2  # skip the api-prediction branch

            inj_msgs, inj_outputs, completed = self._inject_dict_span(session)
            if not is_prediction:
                # the prediction text does not enter the task loop (the original agent only uses it to filter tools)
                agent.messages.extend(inj_msgs)
            session.simplified_agent = agent
            session.simplified_outputs = list(inj_outputs)
            session.simplified_completed = completed
        agent = session.simplified_agent
        execution_outputs = session.simplified_outputs
        if session.simplified_completed:
            return usage_out, True
        if session.simplified_trajectory is None:
            session.simplified_trajectory = []
        traj = session.simplified_trajectory
        for round_i in range(max_steps):
            t_llm0 = monotonic_ms()
            inputs, usage, status = agent.next_execution_inputs_usage_and_status(
                execution_outputs)
            round_llm_ms = monotonic_ms() - t_llm0
            try:
                toks = getattr(usage, "tokens", None)
                p = int(getattr(toks, "input_cache_miss", 0) or 0) \
                    + int(getattr(toks, "input_cache_hit", 0) or 0)
                c = int(getattr(toks, "output", 0) or 0)
                if p or c:
                    usage_out.append(LMResponse(prompt_tokens=p,
                                                completion_tokens=c))
            except Exception:
                pass  # when usage cannot be parsed, honestly drop this round's count, do not block the continuation
            names = [i.metadata.get("function_name", "?") for i in inputs]
            traj.append("round %d: calls=%s failed=%s" % (round_i, names, status.failed))
            # structured trace: the tool calls the model issued this round (span structure aligned with the original trace)
            args_by_id = {i.metadata.get("id"): _parse_command_args(str(i.content))
                          for i in inputs}
            # LLM thinking: after generation the agent unconditionally appends
            # this round's message to messages (the error path returns early
            # without appending); take the last assistant message's reasoning_content ("" if absent)
            reasoning = ""
            if not status.failed:
                _msgs = getattr(agent, "messages", None) or []
                if _msgs and isinstance(_msgs[-1], dict) \
                        and _msgs[-1].get("role") == "assistant":
                    reasoning = str(_msgs[-1].get("reasoning_content") or "")
            self._trace_step(session, {
                "span_type": "llm.call", "step": round_i, "source": "baseline",
                "elapsed_ms": round_llm_ms,
                "output": {"tool_calls": [{
                    "name": i.metadata.get("function_name", ""),
                    "arguments": args_by_id.get(i.metadata.get("id"), {})}
                    for i in inputs],
                    "reasoning": reasoning}})
            if status.failed:
                traj.append("round %d: status=%s" % (round_i, status.message[:150]))
                self._trace_terminal(session, {
                    "status": "failed: %s" % str(status.message)[:TRACE_RESULT_MAX_LEN],
                    "world_completed": bool(bridge.world.task_completed())})
                # when run_to_completion re-enters the loop, do not re-append the
                # injected turn's tool messages (else orphan tool messages -> another 400)
                session.simplified_outputs = []
                break
            # repair: the agent skips ill-formatted tool names (separator count != 1)
            # and produces no ExecutionIO, so the next request has tool_calls with no
            # tool-message reply -> 400. Pad tool messages for the skipped calls (content =
            # corrective feedback listing the exact available tools): keeps the protocol intact and shrinks the later hallucination space
            repaired = self._repair_skipped_tool_calls(
                agent, inputs,
                [d["function"]["name"] for d in (agent.functions or [])])
            if repaired:
                traj.append("round %d: repaired %d invalid tool calls" % (
                    round_i, len(repaired)))
                for rep in repaired:
                    self._trace_step(session, {
                        "span_type": "tool.execution", "step": round_i,
                        "source": "repair",
                        "tool": rep["name"], "arguments": {},
                        "ok": False, "elapsed_ms": 0.0,
                        "result": "invalid tool name; repaired with corrective tool message"})
            t_tool0 = monotonic_ms()
            outputs_raw = bridge.world.batch_execute(
                [i.content for i in inputs])
            batch_ms = monotonic_ms() - t_tool0
            execution_outputs = [
                ExecutionIO(content=o, metadata=i.metadata)
                for i, o in zip(inputs, outputs_raw)]
            # structured trace: execution results (environment observations);
            # the batch's total elapsed time is recorded on each entry (whole-batch wall-clock basis, no per-item breakdown)
            for eio in execution_outputs:
                content = str(eio.content)
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": round_i,
                    "source": "baseline",
                    "tool": eio.metadata.get("function_name", ""),
                    "arguments": args_by_id.get(eio.metadata.get("id"), {}),
                    "ok": "Execution failed" not in content
                          and not content.startswith("Error"),
                    "elapsed_ms": batch_ms,
                    "result": content[:TRACE_RESULT_MAX_LEN]})
            session.simplified_outputs = execution_outputs
            if bridge.world.task_completed():
                session.simplified_completed = True
                self._trace_terminal(session, {
                    "status": "completed", "world_completed": True})
                break
        return usage_out, session.simplified_completed

    # ------------------------------------------------------------------
    # Legacy backend: the REAL legacy FunctionCallingAgent machinery --
    # its OpenAILanguageModel plus the exact generate_second_step_text
    # loop body (generate(messages, functions, "required") ->
    # world.execute -> tool messages -> task_completed check).  The agent's
    # monolithic solve() cannot resume mid-loop, so the harness replicates
    # the loop body verbatim over the restored (spliced) message list --
    # same model, same functions protocol, same prompts.
    # ------------------------------------------------------------------

    @staticmethod
    def _legacy_functions(bridge: "AppWorldBridge",
                          allowed: List[str]) -> List[Dict[str, Any]]:
        """The same functions construction as generate_second_step_text:
        prepare_api_docs(app, format="function_calling") filtered by
        allowed (dotted form); supervisor__complete_task is always added
        (the agent calls it to finish the task)."""
        from appworld.api_docs import prepare_api_docs
        allowed_set = {t.replace("__", ".") for t in allowed}
        allowed_set.add("supervisor.complete_task")
        apps = sorted({t.replace("__", ".").split(".")[0] for t in allowed_set}
                      | {"supervisor"})
        functions: List[Dict[str, Any]] = []
        for app_name in apps:
            try:
                app_functions = prepare_api_docs(app_name,
                                                 format="function_calling")
            except Exception:
                continue
            for f in app_functions:
                _, api_name = f["function"]["name"].split("__", 1)
                if "%s.%s" % (app_name, api_name) in allowed_set:
                    functions.append(f)
        return functions

    @staticmethod
    def _legacy_tool_result(bridge: "AppWorldBridge", function_name: str,
                            arguments_str: str) -> str:
        """Execute a single tool call, consistent with the agent loop body:
        print(apis.<app>.<api>(**<args>)) -> world.execute; the output is the result text."""
        if function_name.count("__") != 1:
            return None  # ill-formatted tool name: the agent skips it and appends no tool message
        app_name, api_name = function_name.split("__", 1)
        try:
            arguments_str = str(json.loads(arguments_str))
        except json.JSONDecodeError:
            arguments_str = ""
        api_code = "print(apis.%s.%s(**%s))" % (app_name, api_name, arguments_str)
        return str(bridge.world.execute(api_code))

    def _continue_with_legacy(self, session: AppWorldSession,
                              stop_after_span: bool,
                              max_steps: int = MAX_LOOP_STEPS
                              ) -> Tuple[List[LMResponse], bool]:
        self._ensure_recorded_tokens(session)
        """Continuation via the REAL legacy FunctionCallingAgent machinery:
        OpenAILanguageModel + the generate_second_step_text loop body (see the
        class comment). The injected span's output is appended as native tool_calls
        and really executed; later turns all go through the baseline big model (same model/protocol/prompts as the original agent)."""
        from appworld.common.random import get_unique_id
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        usage: List[LMResponse] = []
        if session.legacy_messages is None:
            from appworld_agents.code.legacy.plain.language_models import (
                LanguageModel as _LegacyLM,
            )
            bc = session.baseline_client
            os.environ.setdefault("OPENAI_API_KEY", bc.api_key or "placeholder")
            if bc.base_url:
                os.environ["OPENAI_BASE_URL"] = bc.base_url
            session.legacy_lm = _LegacyLM.from_dict({
                "type": "openai_lm",
                "model": bc.model,
                "temperature": bc.temperature,
                "seed": 100,
            })
            messages = [dict(m) for m in session.conversation]
            # the injected span's output: native tool_calls assistant message + real execution
            parsed = session.injected
            if not isinstance(parsed, dict) or not parsed.get("tool_calls"):
                text = ""
                if isinstance(parsed, dict):
                    text = str(parsed.get("text")
                               or parsed.get("__span_failure__", ""))
                self._trace_step(session, {
                    "span_type": "llm.call", "step": 0, "source": "slm",
                    "output": {"text": (text or "no usable output")
                               [:TRACE_RESULT_MAX_LEN]}})
                messages.append({"role": "user", "content":
                                 text or "no usable output"})
                session.legacy_messages = messages
                session.legacy_completed = False
                return usage, False
            tcs = []
            for i, c in enumerate(parsed["tool_calls"]):
                args = c.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                tcs.append({"id": _synth_call_id("legacy", i),
                            "type": "function",
                            "function": {
                                "name": str(c.get("name", "")),
                                "arguments": json.dumps(args, ensure_ascii=False),
                            }})
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": tcs})
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"tool_calls": [{
                    "name": tc["function"]["name"],
                    "arguments": _json_args_or_empty(tc["function"]["arguments"])}
                    for tc in tcs]}})
            completed = False
            for tc in tcs:
                fn = tc["function"]
                output = self._legacy_tool_result(
                    bridge, fn["name"], fn["arguments"])
                if output is None:
                    self._trace_step(session, {
                        "span_type": "tool.execution", "step": 0,
                        "source": "slm", "tool": fn["name"], "arguments": {},
                        "ok": False,
                        "result": "invalid tool name; skipped (agent loop semantics)"})
                    continue  # the agent loop body likewise skips ill-formatted names
                messages.append({"tool_call_id": tc["id"], "role": "tool",
                                 "name": fn["name"], "content": output})
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": 0, "source": "slm",
                    "tool": fn["name"],
                    "arguments": _json_args_or_empty(fn["arguments"]),
                    "ok": "Execution failed" not in output,
                    "result": output[:TRACE_RESULT_MAX_LEN]})
                if bridge.world.task_completed():
                    completed = True
            session.legacy_messages = messages
            session.legacy_completed = completed
        messages = session.legacy_messages
        completed = session.legacy_completed
        if stop_after_span or completed:
            return usage, completed
        lm = session.legacy_lm
        functions = self._legacy_functions(
            bridge, span.output_contract_runtime.allowed_tools or [])
        for round_i in range(max_steps):
            try:
                t_llm0 = monotonic_ms()
                _, message_ = lm.generate(messages, functions, "required")
                round_llm_ms = monotonic_ms() - t_llm0
            except Exception as exc:
                self._trace_terminal(session, {
                    "status": "failed: %s: %s" % (type(exc).__name__, exc),
                    "world_completed": bool(bridge.world.task_completed())})
                break
            calls = message_.get("tool_calls") or []
            # consistent with the agent loop: pad deterministic call ids, then append the assistant message
            for tool_call in calls:
                tool_call["id"] = "call_" + get_unique_id(24)
            messages.append(message_)
            self._trace_step(session, {
                "span_type": "llm.call", "step": round_i, "source": "baseline",
                "elapsed_ms": round_llm_ms,
                "output": {"tool_calls": [{
                    "name": str(tc.get("function", {}).get("name", "")),
                    "arguments": _json_args_or_empty(
                        tc.get("function", {}).get("arguments", ""))}
                    for tc in calls]}})
            if not calls:
                # plain-text turn: the legacy loop body continues (the next
                # generate takes this text as context); when the model stops calling tools, treat it as stagnation
                break
            for tool_call in calls:
                fn = tool_call.get("function") or {}
                function_name = str(fn.get("name", ""))
                t_tool0 = monotonic_ms()
                output = self._legacy_tool_result(
                    bridge, function_name, str(fn.get("arguments", "{}")))
                tool_ms = monotonic_ms() - t_tool0
                if output is None:
                    self._trace_step(session, {
                        "span_type": "tool.execution", "step": round_i,
                        "source": "baseline", "tool": function_name,
                        "arguments": {}, "ok": False, "elapsed_ms": 0.0,
                        "result": "invalid tool name; skipped (agent loop semantics)"})
                    continue
                messages.append({"tool_call_id": tool_call.get("id", ""),
                                 "role": "tool", "name": function_name,
                                 "content": output})
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": round_i,
                    "source": "baseline", "tool": function_name,
                    "arguments": _json_args_or_empty(fn.get("arguments", "{}")),
                    "ok": "Execution failed" not in output,
                    "elapsed_ms": tool_ms,
                    "result": output[:TRACE_RESULT_MAX_LEN]})
            if bridge.world.task_completed():
                completed = True
                self._trace_terminal(session, {
                    "status": "completed", "world_completed": True})
                break
        if not completed:
            self._trace_terminal(session, {
                "status": "stopped: budget",
                "world_completed": bool(bridge.world.task_completed())})
        session.legacy_messages = messages
        session.legacy_completed = completed
        return usage, completed

    # ------------------------------------------------------------------
    # Smolagents backend: the REAL smolagents ToolCallingAgent protocol --
    # text history ("Calling tools:" / "Observation:") + tools +
    # tool_choice="required", driven by the agent's own model class.
    # Requires the smolagents package (unavailable -> generic loop fallback,
    # which follows the same text protocol via session.text_protocol).
    # ------------------------------------------------------------------

    def _continue_with_smolagents(self, session: AppWorldSession,
                                  stop_after_span: bool,
                                  max_steps: int = MAX_LOOP_STEPS
                                  ) -> Tuple[List[LMResponse], bool]:
        self._ensure_recorded_tokens(session)
        from smolagents.models import ChatMessage, MessageRole
        from smolagents.models import OpenAIServerModel
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        bc = session.baseline_client
        usage: List[LMResponse] = []
        if session.loop_messages is None:
            # local deterministic baseline support: pass seed signature-driven
            # (immune to version differences, same pattern as the simplified backend)
            import inspect as _inspect_sm
            _sm_kw = dict(model_id=bc.model, api_base=bc.base_url,
                          api_key=bc.api_key or "placeholder",
                          temperature=bc.temperature,
                          max_tokens=bc.max_output_tokens)
            if "seed" in _inspect_sm.signature(
                    OpenAIServerModel.__init__).parameters:
                _sm_kw["seed"] = 100
            model = OpenAIServerModel(**_sm_kw)
            session._smol_model = model
            messages = [dict(m) for m in session.conversation]
            session.loop_messages = messages
            session.loop_index = span.invocation_index
        model = session._smol_model
        messages = session.loop_messages
        completed = session.loop_completed
        tools = self._smolagents_tools(session)
        is_prediction = span.span_descriptor.node_pattern == "tool_selection"
        if is_prediction:
            # prediction turn: the injected API list only filters the tool set (consistent with the original agent semantics)
            predicted = self._predicted_apis_from_text(bridge, session.injected)
            if predicted:
                tools = [t for t in tools if t.name in predicted]
                session._smol_tools = tools
            return usage, completed
        for round_i in range(max_steps):
            if session.loop_index == span.invocation_index and not completed \
                    and session.injected is not None:
                parsed = session.injected
                if not isinstance(parsed, dict) or not parsed.get("tool_calls"):
                    break
                # injected span: appended in TOOL_CALL / TOOL_RESPONSE text
                # form (the original agent's recorded form), calls really executed
                tcs = parsed["tool_calls"]
                calls_repr = json.dumps([{
                    "id": _synth_call_id("smol", i), "type": "function",
                    "function": {"name": str(c.get("name", "")),
                                 "arguments": c.get("arguments") or {}}}
                    for i, c in enumerate(tcs)], ensure_ascii=False)
                messages.append({"role": "assistant", "content":
                                 "Calling tools:\n" + calls_repr})
                observations = []
                for c in tcs:
                    name = str(c.get("name", "")).replace("__", ".")
                    res = bridge.execute_tool(
                        name, c.get("arguments") or {})
                    observations.append(self._sub_obs(
                        session, name,
                        str(res["result"]) if res["ok"]
                        else "Error: %s" % res["error"]))
                    self._trace_step(session, {
                        "span_type": "tool.execution", "step": 0,
                        "source": "slm", "tool": str(c.get("name", "")),
                        "arguments": c.get("arguments") or {},
                        "ok": bool(res["ok"]),
                        "result": (str(res["result"]) if res["ok"]
                                   else str(res["error"]))[:TRACE_RESULT_MAX_LEN]})
                messages.append({"role": "user", "content":
                                 "Observation:\n" + "\n".join(observations)})
                session.loop_index += 1
                self._trace_step(session, {
                    "span_type": "llm.call", "step": 0, "source": "slm",
                    "output": {"tool_calls": [{
                        "name": str(c.get("name", "")),
                        "arguments": c.get("arguments")} for c in tcs]}})
                if stop_after_span:
                    break
                continue
            # real step: ChatMessage history -> model.generate (the original agent's call)
            chat_messages = [ChatMessage(
                role={"system": MessageRole.SYSTEM, "user": MessageRole.USER,
                      "assistant": MessageRole.ASSISTANT}.get(
                    m.get("role"), MessageRole.USER),
                content=m.get("content") or "") for m in messages]
            t_llm0 = monotonic_ms()
            out = model.generate(chat_messages, tools_to_call_from=tools,
                                 stop_sequences=["Observation:", "Calling tools:"])
            round_llm_ms = monotonic_ms() - t_llm0
            calls = []
            for tc in (out.tool_calls or []):
                fn = tc.function
                args = fn.arguments
                calls.append({"name": str(fn.name),
                              "arguments": args if isinstance(args, dict)
                              else json.loads(str(args))})
            if not calls:
                # plain-text turn = final answer: go through the real
                # TaskCompleter flow (answer-synthesis LLM call +
                # supervisor.complete_task), exactly like the collection script run_benchmark_builtin_smolagents
                answer = str(out.content or "")
                messages.append({"role": "assistant", "content": answer})
                self._trace_step(session, {
                    "span_type": "llm.call", "step": round_i,
                    "source": "baseline",
                    "elapsed_ms": round_llm_ms,
                    "output": {"text": answer[:TRACE_RESULT_MAX_LEN]}})
                try:
                    self._smolagents_complete(session, answer)
                    completed = bool(bridge.world.task_completed())
                except Exception as exc:
                    self._trace_terminal(session, {
                        "status": "stopped: task_completer failed: %s: %s"
                                  % (type(exc).__name__, exc),
                        "world_completed": False})
                if completed:
                    self._trace_terminal(session, {
                        "status": "completed", "world_completed": True})
                break
            messages.append(self._assistant_calls_msg(
                calls, text_protocol=True, tag="smol"))
            self._trace_step(session, {
                "span_type": "llm.call", "step": round_i, "source": "baseline",
                "elapsed_ms": round_llm_ms,
                "output": {"tool_calls": calls}})
            observations = []
            call_times = []
            for c in calls:
                name = str(c.get("name", "")).replace("__", ".")
                if name.endswith("complete_task"):
                    t_tool0 = monotonic_ms()
                    bridge.complete_task(c.get("arguments", {}).get("answer"),
                                         c.get("arguments", {}).get("status", "success"))
                    call_times.append(monotonic_ms() - t_tool0)
                    completed = True
                    observations.append("Marked the active task complete.")
                    break
                t_tool0 = monotonic_ms()
                res = bridge.execute_tool(name, c.get("arguments") or {})
                call_times.append(monotonic_ms() - t_tool0)
                observations.append(self._sub_obs(
                    session, name,
                    str(res["result"]) if res["ok"]
                    else "Error: %s" % res["error"]))
            messages.append({"role": "user", "content":
                             "Observation:\n" + "\n".join(observations)})
            # record the real ok status per call (previously the aggregated
            # Observation text was used with ok always True, swallowing single-failure signals -> downstream hall/failure-rate metrics distorted)
            for c, obs, cms in zip(calls, observations, call_times):
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": round_i,
                    "source": "baseline", "tool": str(c.get("name", "")),
                    "arguments": c.get("arguments") or {},
                    "ok": not obs.startswith("Error"),
                    "elapsed_ms": cms,
                    "result": obs[:TRACE_RESULT_MAX_LEN]})
            if completed:
                self._trace_terminal(session, {
                    "status": "completed", "world_completed": True})
                break
        session.loop_messages = messages
        session.loop_completed = completed
        return usage, completed

    @staticmethod
    def _smolagents_complete(session: AppWorldSession, answer: str) -> None:
        """The real TaskCompleter flow (same as the collection script): answer synthesis + completion marking."""
        from appworld_agents.code.smolagents.models import Model as _SMolModel
        from appworld_agents.code.smolagents.task_completer import (
            TaskCompleter,
        )
        import appworld_agents.code.smolagents as _smod
        bridge = session.bridge
        bc = session.baseline_client
        assert bridge is not None
        os.environ.setdefault("OPENAI_API_KEY", bc.api_key or "placeholder")
        smol_model = _SMolModel.from_dict({
            "type": "openai", "base_url": bc.base_url,
            "api_key_env_name": "OPENAI_API_KEY",
            "model_id": bc.model, "temperature": bc.temperature,
            "seed": 100, "use_cache": False,
        })
        code_root = os.path.dirname(os.path.abspath(_smod.__file__))
        exp_root = os.path.dirname(os.path.dirname(code_root))
        completer = TaskCompleter(
            model=smol_model,
            prompt_file_path=os.path.join(
                exp_root, "prompts", "smolagents",
                "task_completer_instructions.txt"))
        completer.complete(bridge.world, answer)

    def _smolagents_tools(self, session: AppWorldSession) -> List[Any]:
        """The real smolagents Tool list (smolagents.Tool wrapping bridge execution)."""
        from smolagents import Tool
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        allowed = {t.replace("__", ".") for t in
                   (span.output_contract_runtime.allowed_tools or [])}
        tools: List[Any] = []
        for doc in bridge.tool_docs():
            name = doc["name"]
            if name.replace("__", ".") not in allowed:
                continue
            app, api = name.split("__", 1)

            def _call(app=app, api=api, **kwargs):
                return self._sub_obs(
                    session, app + "__" + api,
                    bridge.execute_tool("%s__%s" % (app, api), kwargs)["result"])

            tools.append(Tool(
                name=name, description=doc.get("description", ""),
                inputs=doc.get("inputs") or {},
                output_type=doc.get("output_type", "string"),
                function=_call))
        return tools

    # ------------------------------------------------------------------
    # tau2 backend: the REAL tau2-bench LLMAgent.  Its get_init_state()
    # accepts a message history, so mid-run resume is the agent's own
    # mechanism; the driver loop replicates the collection script
    # (run_benchmark_tau2_llmagent.py).  Requires the tau2 package.
    # ------------------------------------------------------------------

    def _continue_with_tau2(self, session: AppWorldSession,
                            stop_after_span: bool,
                            max_steps: int = MAX_LOOP_STEPS
                            ) -> Tuple[List[LMResponse], bool]:
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        bc = session.baseline_client
        usage: List[LMResponse] = []
        # the tau2 chain was preloaded in restore before world creation (_prewarm_tau2,
        # which completes litellm's price-table fetch fallback under the real clock) -- this
        # import hits the module cache, no network. When not preloaded (e.g. called directly), fall back to a thawed import.
        try:
            from tau2.agent.llm_agent import LLMAgent
        except ImportError:
            raise RuntimeError(
                "tau2 backend unavailable: the downstream continuation "
                "must run the REAL agent source; generic approximation "
                "is not an acceptable fallback")
        from tau2.data_model.message import (
            AssistantMessage, MultiToolMessage, ToolCall, ToolMessage,
            UserMessage,
        )
        from tau2.environment.tool import as_tool
        if session.tau2_agent is None:
            # align the observation stream's random values: login tokens are
            # randomly generated by the world, replace them with recorded values (only in the model-visible observation stream; world state keeps real executed values)
            session.recorded_tokens = self._recorded_login_tokens(session.trace)
            os.environ.setdefault("OPENAI_API_KEY", bc.api_key or "placeholder")
            if bc.base_url:
                os.environ["OPENAI_BASE_URL"] = bc.base_url
            # tool surface: prediction spans (tool_selection) are driven by the injected
            # predicted list (aligned with _autogen_tools' prediction_docs: T arm = the SLM's
            # new list truly drives the downstream tool surface; I-arm injection = the recorded
            # list, equivalent to the recorded next-span tools); when no valid list parses, fall
            # back to the recorded allowed_tools superset. The span's allowed_tools is in dotted form.
            allowed = {t.replace(".", "__") for t in
                       (span.output_contract_runtime.allowed_tools or [])}
            if span.span_descriptor.node_pattern == "tool_selection":
                # I-arm injection = recorded output: keep the recorded allowed_tools as-is
                # (observed: the list parsed from the recorded prediction text has one more
                # name than the recorded next-span tools; direct replacement would deviate the
                # I-arm tool surface from the records). T-arm injection = SLM's new prediction: drive the tool surface with the parsed list.
                _inj = session.injected
                _rec = span.baseline_response.parsed
                _is_identity = (isinstance(_inj, dict) and isinstance(_rec, dict)
                                and (_inj.get("text") or "") == (_rec.get("text") or ""))
                if not _is_identity:
                    _predicted = self._predicted_apis_from_text(
                        bridge, _inj)
                    if _predicted:
                        allowed = set(_predicted)
            allowed.add("supervisor__complete_task")
            tools, tool_map = self._build_tau2_tools(bridge, allowed)
            from appworld.common.io import dump_yaml
            app_descs = dict(bridge.world.task.app_descriptions)
            app_descs.pop("api_docs", None)
            desc_yaml = dump_yaml(app_descs)
            domain_policy = (
                "Available apps:\n%s\n\nTask: %s\n\n"
                "Call supervisor__complete_task when the task is done."
                % (desc_yaml, bridge.world.task.instruction))
            agent = LLMAgent(
                tools=tools, domain_policy=domain_policy,
                llm="openai/%s" % bc.model,
                llm_args={"temperature": bc.temperature,
                          # same basis as the collection side (run_benchmark_tau2_llmagent.py
                          # also changed to 8192): thinking mode + 4096 lets thinking tokens
                          # eat the whole budget and ends with empty output
                          "max_tokens": 8192,
                          "seed": 100,
                          # thinking switch same basis as the collection side (model_config.py):
                          # Qwen (vLLM) must explicitly enable_thinking (default off, diverging
                          # from the collection behavior -> observed I-arm answers mismatch);
                          # deepseek server-side default on, not injected
                          **({"extra_body": {"enable_thinking": True}}
                             if str(bc.model).lower().startswith("qwen")
                             else {})},
            )
            # rebuild the tau2 message history from the spliced session (the agent's
            # native mechanism). Prediction span (tool_selection) special case: aligned
            # with _autogen_seed -- the native agent discards the predictor session right
            # after predicting and the task loop starts from the next span's session
            # window. Seeding with the predictor session leaves the history ending in an assistant message, the loop's first round terminates with "no tool calls" (smoke test 12/12 failed)
            seed_conv = session.conversation
            if span.span_descriptor.node_pattern == "tool_selection":
                seed_conv = self._next_span_conversation(session, span) \
                    or session.conversation
            history = []
            for m in seed_conv:
                role = m.get("role")
                if role == "system":
                    continue  # the system prompt is generated by the agent itself
                if role == "user":
                    history.append(UserMessage(role="user",
                                               content=str(m.get("content", ""))))
                elif role == "assistant":
                    tcs = None
                    if m.get("tool_calls"):
                        tcs = [ToolCall(
                            id=str(tc.get("id", "")),
                            name=str((tc.get("function") or {}).get("name", "")),
                            arguments=_json_args_or_empty(
                                (tc.get("function") or {}).get("arguments", "{}")))
                            for tc in m["tool_calls"]]
                    history.append(AssistantMessage(
                        role="assistant",
                        content=(str(m.get("content"))
                                 if m.get("content") else None),
                        tool_calls=tcs))
                elif role == "tool":
                    history.append(ToolMessage(
                        role="tool", id=str(m.get("tool_call_id", "")),
                        content=str(m.get("content", "")),
                        error="Execution failed" in str(m.get("content", ""))
                        or str(m.get("content", "")).startswith("Error")))
            state = agent.get_init_state(message_history=history)
            session.tau2_agent = agent
            session.tau2_tool_map = tool_map
            session.tau2_state = state
            session.tau2_completed = False
            # the injected span's output becomes the next round's assistant message (the history does not contain it)
            parsed = session.injected
            if isinstance(parsed, dict) and parsed.get("tool_calls"):
                inj = parsed["tool_calls"]
                inj_tcs = [ToolCall(
                    id=_synth_call_id("tau2", i),
                    name=str(c.get("name", "")),
                    arguments=c.get("arguments") or {}) for i, c in enumerate(inj)]
                assistant_msg = AssistantMessage(
                    role="assistant", content=None, tool_calls=inj_tcs)
                state.messages.append(assistant_msg)
                results = []
                for tc in inj_tcs:
                    tool = tool_map.get(tc.name)
                    if tool is None:
                        results.append(ToolMessage(
                            role="tool", id=tc.id,
                            content="Error: tool %s not found" % tc.name,
                            error=True))
                        continue
                    try:
                        out = tool(**tc.arguments)
                        out = self._substitute_recorded_token(
                            str(out), str(tc.name).split("__")[0].split(".")[0],
                            getattr(session, "recorded_tokens", {}))
                        results.append(ToolMessage(
                            role="tool", id=tc.id, content=str(out),
                            error=str(out).startswith("Execution failed")))
                    except Exception as exc:
                        results.append(ToolMessage(
                            role="tool", id=tc.id,
                            content="Error: %s" % exc, error=True))
                    if tc.name == "supervisor__complete_task":
                        session.tau2_completed = True
                state.messages.append(MultiToolMessage(
                    role="tool", tool_messages=results))
                self._trace_step(session, {
                    "span_type": "llm.call", "step": 0, "source": "slm",
                    "output": {"tool_calls": [{
                        "name": str(c.get("name", "")),
                        "arguments": c.get("arguments")} for c in inj]}})
                for _tc, _r in zip(inj_tcs, results):
                    self._trace_step(session, {
                        "span_type": "tool.execution", "step": 0,
                        "source": "slm", "tool": str(_tc.name),
                        "arguments": _tc.arguments,
                        "ok": not _r.error,
                        "result": str(_r.content)[:TRACE_RESULT_MAX_LEN]})
        agent = session.tau2_agent
        tool_map = session.tau2_tool_map
        state = session.tau2_state
        completed = session.tau2_completed
        if stop_after_span or completed:
            return usage, completed
        # downstream usage capture: tau2's llm_utils has a _TRACE_TRACER
        # sentinel (same protocol as the collection side, log_llm_response
        # carries input/output tokens); attach a shim to accumulate each round's tokens, restore after the loop
        from tau2.utils import llm_utils as _tau2_llm
        _usage_calls: List[Tuple[int, int]] = []

        class _UsageShim:
            def log_llm_call(self, model, messages, tools=None):
                return {"model": model, "messages": messages, "tools": tools}

            def log_llm_response(self, completion, model="", input_tokens=0,
                                 output_tokens=0, latency_ms=0.0,
                                 llm_call_span=None, **kw):
                _usage_calls.append((int(input_tokens or 0),
                                     int(output_tokens or 0)))

        _orig_tracer = getattr(_tau2_llm, "_TRACE_TRACER", None)
        _tau2_llm._TRACE_TRACER = _UsageShim()
        try:
            self._tau2_loop(session, agent, tool_map, state, bridge,
                            max_steps)
        finally:
            _tau2_llm._TRACE_TRACER = _orig_tracer
        usage = [LMResponse(prompt_tokens=p, completion_tokens=c)
                 for p, c in _usage_calls]
        return usage, session.tau2_completed

    @staticmethod
    def _recorded_login_tokens(trace: Optional[TraceRun]) -> Dict[str, str]:
        """Extract each app's access_token from the recorded login tool results.

        The world's fake API randomly generates a JWT at login (authentication.py:
        expires_in=random.randrange(10min,30min)) -- replay necessarily draws different
        values, and tokens entering the model context would fork the output. The
        observation stream must substitute the recorded values (the JWT signing key is a
        code constant and exp is after the world's frozen time, so recorded tokens remain valid in the replayed world).
        """
        import re as _re
        tokens: Dict[str, str] = {}
        if trace is None:
            return tokens
        for sp in trace.spans:
            te = sp.tool_execution
            if te is None:
                continue
            name = str(te.tool_name or "")
            if not (name.endswith("__login") or name.endswith(".login")):
                continue
            m = _re.search(
                r"access_token['\"]?\s*[:=]\s*['\"]([^'\"]+)",
                str(te.result or ""))
            if m:
                tokens[name.split("__")[0].split(".")[0]] = m.group(1)
        # tool_execution only carries each span's first tool; in spans with
        # parallel calls (autogen etc.) login may not be first -- also scan
        # the artifact's complete record list (span_tool_results, collected by the adapter per requesting span in order)
        results_by_span = (trace.artifacts.get("span_tool_results") or {})
        for _key, entries in results_by_span.items():
            for e in entries or []:
                name = str(e.get("tool") or "")
                if not (name.endswith("__login") or name.endswith(".login")):
                    continue
                m = _re.search(
                    r"access_token['\"]?\s*[:=]\s*['\"]([^'\"]+)",
                    str(e.get("result") or ""))
                if m:
                    tokens.setdefault(
                        name.split("__")[0].split(".")[0], m.group(1))
        return tokens

    @staticmethod
    def _substitute_recorded_token(text: str, app: str,
                                   recorded_tokens: Dict[str, str]) -> str:
        """Replace the replay-generated token in a tool result with the
        recorded token (observation stream only; the world state keeps the really executed values)."""
        rec = recorded_tokens.get(app)
        if not rec:
            return text
        import re as _re
        m = _re.search(r"access_token['\"]?\s*[:=]\s*['\"]([^'\"]+)", text)
        if m and m.group(1) != rec:
            return text.replace(m.group(1), rec, 1)
        return text

    def _ensure_recorded_tokens(self, session: "AppWorldSession") -> None:
        """Ensure the session carries the recorded token table (shared entry for all continuation backends)."""
        if not hasattr(session, "recorded_tokens"):
            session.recorded_tokens = self._recorded_login_tokens(session.trace)

    def _sub_obs(self, session: "AppWorldSession", tool_name: Any,
                 text: Any) -> str:
        """Convenience wrapper for observation-stream token substitution (shared by all continuation backends)."""
        return self._substitute_recorded_token(
            str(text),
            str(tool_name or "").replace(".", "__").split("__")[0],
            getattr(session, "recorded_tokens", {}))

    def _tau2_loop(self, session, agent, tool_map, state, bridge,
                   max_steps) -> None:
        """The tau2 driver loop body (same as the collection script; a
        standalone function so the outer layer can attach/detach the usage shim)."""
        from tau2.data_model.message import (
            AssistantMessage, MultiToolMessage, ToolMessage, UserMessage)
        usage: List[LMResponse] = []
        completed = bool(getattr(session, "tau2_completed", False))
        for round_i in range(max_steps):
            # driver loop (same as the collection script): the previous round's
            # MultiToolMessage / ToolMessage / UserMessage is this round's input directly;
            # an assistant message (only when the history ends with it, e.g. a plain-text turn) is replaced by a continuation trigger message
            if not state.messages:
                msg: Any = UserMessage(role="user",
                                       content=bridge.world.task.instruction)
            else:
                msg = state.messages[-1]
            if isinstance(msg, AssistantMessage):
                # same as the collection side: a plain-text turn ends directly
                # (the collection loop breaks), inject no trigger text (adding a synthetic message would change the agent's prompt)
                self._trace_terminal(session, {
                    "status": "stopped: no tool calls",
                    "world_completed": bool(bridge.world.task_completed())})
                break
            try:
                t_llm0 = monotonic_ms()
                assistant_msg, state = agent.generate_next_message(msg, state)
                round_llm_ms = monotonic_ms() - t_llm0
            except Exception as exc:
                self._trace_terminal(session, {
                    "status": "failed: %s: %s" % (type(exc).__name__, exc),
                    "world_completed": bool(bridge.world.task_completed())})
                break
            tcs = assistant_msg.tool_calls
            # LLM thinking: raw_data is the full response.to_dict() persisted by
            # llm_utils; the message's reasoning_content/reasoning passes through
            # as-is ("" when there is no thinking)
            _raw_msg = (((getattr(assistant_msg, "raw_data", None) or {})
                         .get("choices") or [{}])[0].get("message") or {})
            reasoning = ""
            if isinstance(_raw_msg, dict):
                reasoning = str(_raw_msg.get("reasoning_content")
                                or _raw_msg.get("reasoning") or "")
            # step uses round_i (previously always 0, which broke positional alignment and would invalidate dynamics analysis)
            self._trace_step(session, {
                "span_type": "llm.call", "step": round_i, "source": "baseline",
                "elapsed_ms": round_llm_ms,
                "output": {"tool_calls": [
                    {"name": str(tc.name), "arguments": tc.arguments}
                    for tc in (tcs or [])],
                    "reasoning": reasoning}})
            if not tcs:
                self._trace_terminal(session, {
                    "status": "stopped: no tool calls",
                    "world_completed": bool(bridge.world.task_completed())})
                break
            results = []
            for tc in tcs:
                tool = tool_map.get(tc.name)
                t_tool0 = monotonic_ms()
                try:
                    if tool is not None:
                        out = tool(**tc.arguments)
                        out = self._substitute_recorded_token(
                            str(out), str(tc.name).split("__")[0].split(".")[0],
                            getattr(session, "recorded_tokens", {}))
                    else:
                        out = "Tool %s not found" % tc.name
                    results.append(ToolMessage(role="tool", content=str(out),
                                               id=tc.id))
                except Exception as exc:
                    results.append(ToolMessage(
                        role="tool", content="Error: %s" % exc,
                        id=tc.id, error=True))
                tool_ms = monotonic_ms() - t_tool0
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": 0,
                    "source": "baseline", "tool": str(tc.name),
                    "arguments": tc.arguments,
                    "ok": not (str(getattr(results[-1], "content", "")).startswith("Error")
                               or str(getattr(results[-1], "content", "")).startswith("Execution failed")),
                    "elapsed_ms": tool_ms,
                    "result": str(getattr(results[-1], "content", ""))[:TRACE_RESULT_MAX_LEN]})
                if tc.name == "supervisor__complete_task":
                    completed = True
                    break
            state.messages.append(MultiToolMessage(
                role="tool", tool_messages=results))
            if completed or bridge.world.task_completed():
                if bridge.world.task_completed():
                    completed = True
                self._trace_terminal(session, {
                    "status": "completed" if bridge.world.task_completed()
                              else "stopped: complete_task called",
                    "world_completed": bool(bridge.world.task_completed())})
                break
        session.tau2_state = state
        session.tau2_completed = completed

    @staticmethod
    def _build_tau2_tools(bridge: "AppWorldBridge",
                          allowed: set) -> Tuple[List[Any], Dict[str, Any]]:
        """Generate the tau2 Tool list from the task's real API docs
        (as_tool-wrapped, executed via world.execute + print wrapping) -- same as the collection script's _build_tau2_tools."""
        from tau2.environment.tool import as_tool
        tools = []
        tool_map = {}
        for doc in bridge.world.task.api_docs.function_calling():
            fn = doc["function"]
            name = fn["name"]
            if name not in allowed and not name.startswith("supervisor__"):
                continue
            desc = fn["description"]
            props = fn["parameters"].get("properties", {})
            required = set(fn["parameters"].get("required", []))
            req_items = [(k, v) for k, v in props.items() if k in required]
            opt_items = [(k, v) for k, v in props.items() if k not in required]
            arg_parts = []
            for pname, pinfo in req_items + opt_items:
                pt = {"string": "str", "integer": "int", "number": "float",
                      "boolean": "bool"}.get(pinfo.get("type", "string"), "str")
                default = pinfo.get("default")
                if default is not None:
                    arg_parts.append("%s: %s = %r" % (pname, pt, default))
                elif pname not in required:
                    arg_parts.append("%s: %s = None" % (pname, pt))
                else:
                    arg_parts.append("%s: %s" % (pname, pt))
            args_str = ", ".join(arg_parts)
            kw_lines = "\n    ".join("kw['%s'] = %s" % (p, p)
                                     for p, _ in req_items + opt_items)
            src = ("def %s(%s):\n"
                   "    \"\"\"%s\"\"\"\n"
                   "    kw = {}\n    %s\n"
                   "    return executor('%s', '%s', kw)\n"
                   % (name, args_str, desc, kw_lines,
                      name.split("__", 1)[0], name.split("__", 1)[1]))
            app, api = name.split("__", 1)

            def _executor(app=app, api=api, kwargs=None):
                # same basis as the collection side (run_benchmark_tau2_llmagent.py):
                # executed via world.execute + print wrapping; failures are returned
                # **verbatim** as "Execution failed"-prefixed strings (world.execute
                # raises no exception and the collection side adds no prefix) -- adding
                # an "Error: " prefix would fork the failure observation stream from the records (one of the measured root causes of I-arm final-turn answer mismatches)
                code = "print(apis.%s.%s(**%r))" % (app, api, kwargs or {})
                return bridge.world.execute(code)

            ns = {"executor": _executor}
            exec(src, ns)
            fn_obj = ns[name]
            fn_obj.__doc__ = desc
            tool = as_tool(fn_obj)
            tools.append(tool)
            tool_map[name] = tool
        return tools, tool_map

    # ------------------------------------------------------------------
    # OpenAI-SDK backend: the REAL OpenAI Agents SDK agent.  The collection
    # script drove it over MCP servers; the harness runs the same SDK Agent
    # with direct function tools executing against the in-process world
    # (the SDK's function_call / function_call_output item protocol is
    # unchanged; only the tool transport differs, noted honestly).
    # Requires the `agents` package.
    # ------------------------------------------------------------------

    def _continue_with_openai_sdk(self, session: AppWorldSession,
                                  stop_after_span: bool,
                                  max_steps: int = MAX_LOOP_STEPS
                                  ) -> Tuple[List[LMResponse], bool]:
        self._ensure_recorded_tokens(session)
        import asyncio
        from agents import Agent as SDKAgent, Runner
        from agents.agent import StopAtTools
        from agents.model_settings import ModelSettings
        from agents.run import RunConfig
        from openai import AsyncOpenAI
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        bc = session.baseline_client
        usage: List[LMResponse] = []
        if session.openai_items is None:
            raise RuntimeError(
                "openai-SDK prompt reconstruction unavailable: the "
                "downstream continuation must run the REAL agent source; "
                "generic approximation is not an acceptable fallback")
        items = list(session.openai_items)
        # the injected span's output: function_call items + outputs after real execution
        parsed = session.injected
        if isinstance(parsed, dict) and parsed.get("tool_calls"):
            for i, c in enumerate(parsed["tool_calls"]):
                cid = _synth_call_id("oaiinj", i)
                items.append({
                    "type": "function_call",
                    "name": str(c.get("name", "")),
                    "arguments": json.dumps(c.get("arguments") or {},
                                            ensure_ascii=False),
                    "call_id": cid, "id": "fc_" + cid,
                    "status": "completed",
                })
                args = c.get("arguments") or {}
                name = str(c.get("name", "")).replace("__", ".")
                res = bridge.execute_tool(name, args)
                items.append({
                    "type": "function_call_output",
                    "call_id": cid,
                    "output": (json.dumps(res["result"], ensure_ascii=False)
                               if res["ok"] else "Error: %s" % res["error"]),
                })
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": 0, "source": "slm",
                    "tool": str(c.get("name", "")), "arguments": args,
                    "ok": bool(res["ok"]),
                    "result": (str(res["result"]) if res["ok"]
                               else str(res["error"]))[:TRACE_RESULT_MAX_LEN]})
        if stop_after_span:
            return usage, False
        # the same SDK config as the collection script (tool_choice auto / parallel tool calls)
        client = AsyncOpenAI(base_url=bc.base_url, api_key=bc.api_key)
        sdk_tools = self._openai_sdk_tools(session)
        stop_at = StopAtTools(stop_at_tool_names=["supervisor__complete_task"])
        agent = SDKAgent(
            name="Assistant", model=bc.model,
            model_settings=ModelSettings(
                tool_choice="auto", parallel_tool_calls=True,
                temperature=bc.temperature),
            instructions=session.openai_messages[0]["content"]
            if session.openai_messages else "",
            tools=sdk_tools,
            tool_use_behavior=stop_at,
            reset_tool_choice=False,
        )

        async def _run():
            streamer = Runner.run_streamed(
                starting_agent=agent, input=items,
                max_turns=max_steps,
                run_config=RunConfig(tracing_disabled=True))
            out_items = []
            async for event in streamer.stream_events():
                if event.type == "run_item_stream_event":
                    it = event.item
                    if it.type in ("tool_call_item", "tool_call_output_item",
                                   "message_output_item"):
                        out_items.append(it)
            return streamer, out_items

        t_run0 = monotonic_ms()
        streamer, out_items = asyncio.run(_run())
        run_ms = monotonic_ms() - t_run0
        tool_ms_total = sum(ms for _n, ms in getattr(
            session, "openai_tool_times", []) or [])
        n_llm_items = sum(1 for it in out_items if it.type == "tool_call_item")
        llm_share = max(0.0, run_ms - tool_ms_total) / max(n_llm_items, 1)
        completed = bool(bridge.world.task_completed())
        # step uses the event index (previously always 0, which broke positional alignment)
        for idx, it in enumerate(out_items):
            if it.type == "tool_call_item":
                raw = it.raw_item.model_dump()
                self._trace_step(session, {
                    "span_type": "llm.call", "step": idx, "source": "baseline",
                    # the SDK exposes no per-turn wall clock: the whole run
                    # minus tool time is amortized over the llm.call entries (synthetic basis; note for analysis)
                    "elapsed_ms": llm_share,
                    "output": {"tool_calls": [{
                        "name": raw.get("name", ""),
                        "arguments": json.loads(
                            raw.get("arguments") or "{}")}]}})
            elif it.type == "tool_call_output_item":
                tool_ms = 0.0
                if getattr(session, "openai_tool_times", None):
                    _tn, tool_ms = session.openai_tool_times.pop(0)
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": idx,
                    "source": "baseline", "tool": "", "arguments": {},
                    "ok": True, "elapsed_ms": tool_ms,
                    "result": str(it.raw_item.get("output", ""))[:TRACE_RESULT_MAX_LEN]})
        self._trace_terminal(session, {
            "status": "completed" if completed else "stopped: %s"
                      % str(streamer.last_response.status if streamer.last_response else "?"),
            "world_completed": completed})
        return usage, completed

    @staticmethod
    def _openai_sdk_tools(session: AppWorldSession) -> List[Any]:
        """SDK function_tool wrappers (executed via bridge.execute_tool; the
        tool catalog comes from the task's real API docs + complete_task) -- same protocol semantics as the MCP tools."""
        from agents import function_tool
        bridge = session.bridge
        span = session.span
        assert bridge is not None and span is not None
        allowed = {t.replace("__", ".") for t in
                   (span.output_contract_runtime.allowed_tools or [])}
        tools: List[Any] = []
        for doc in bridge.world.task.api_docs.function_calling():
            fn = doc["function"]
            name = fn["name"]
            if name.replace("__", ".") not in allowed:
                continue
            app, api = name.split("__", 1)
            params = fn["parameters"]

            def _call(app=app, api=api, **kwargs):
                t_tool0 = monotonic_ms()
                res = bridge.execute_tool("%s__%s" % (app, api), kwargs or {})
                if not hasattr(session, "openai_tool_times"):
                    session.openai_tool_times = []
                session.openai_tool_times.append(
                    (name, monotonic_ms() - t_tool0))
                if not res["ok"]:
                    return self._sub_obs(session, name,
                                         "Error: %s" % res["error"])
                return self._sub_obs(session, name, res["result"])

            tools.append(function_tool(
                _call,  # type: ignore[arg-type]
                name_override=name,
                description_override=fn.get("description", ""),
            ))
        # complete_task always included (the termination path of every agent)

        def _complete(answer=None, status="success", **kwargs):
            t_tool0 = monotonic_ms()
            bridge.complete_task(answer, status or "success")
            if not hasattr(session, "openai_tool_times"):
                session.openai_tool_times = []
            session.openai_tool_times.append(
                ("supervisor__complete_task",
                 monotonic_ms() - t_tool0))
            return "Marked the active task complete."

        tools.append(function_tool(
            _complete,  # type: ignore[arg-type]
            name_override="supervisor__complete_task",
            description_override="Mark the active task as complete with the final answer.",
        ))
        return tools

    @staticmethod
    def _autogen_model_inputs(messages: List[Any]) -> List[Any]:
        """Guard B: filter/convert autogen messages into the "model input" whitelist.

        Whitelist (may go directly into BufferedChatCompletionContext):
          - autogen_core.models.{UserMessage, AssistantMessage, SystemMessage}
          - autogen_core.models.FunctionExecutionResultMessage (the tool result container,
            exactly what the real agent's internal loop uses)
        Conversions:
          - ToolCallSummaryMessage (UI summary) -> wrap its embedded results into a
            FunctionExecutionResultMessage
          - a bare FunctionExecutionResult (old adapters may append it directly) -> wrap into the container
        All other types (TextMessage etc.) are skipped outright -- better to drop UI-layer
        messages than to feed illegal types to the model client and crash the whole turn.

        This function is the sole gate for autogen messages entering the model context;
        every adapter's output (_recorded_to_autogen / _autogen_inject_span etc.) must pass
        through it; an adapter wrongly wrapping a UI summary / bare result can no longer poison the context.
        """
        from autogen_core.models import (  # noqa: F401
            AssistantMessage, FunctionExecutionResultMessage,
            SystemMessage, UserMessage,
        )
        from autogen_agentchat.messages import (  # noqa: F401
            FunctionExecutionResult, ToolCallSummaryMessage,
        )
        out: List[Any] = []
        for msg in messages:
            if isinstance(msg, (UserMessage, AssistantMessage, SystemMessage,
                                FunctionExecutionResultMessage)):
                out.append(msg)
            elif isinstance(msg, ToolCallSummaryMessage):
                results = [r for r in (msg.results or [])
                           if isinstance(r, FunctionExecutionResult)]
                if results:
                    out.append(FunctionExecutionResultMessage(content=results))
            elif isinstance(msg, FunctionExecutionResult):
                out.append(FunctionExecutionResultMessage(content=[msg]))
            # the rest (TextMessage/StructuredMessage etc. UI-layer messages) are skipped
        return out

    def _autogen_seed(self, session: AppWorldSession) -> bool:
        self._ensure_recorded_tokens(session)
        """Build the autogen message history ONCE (restored conversation +
        the intervened span executed through the real tools) and store it on
        the session.  Returns whether the injected span itself completed the
        task.

        prediction span (tool_selection) special case: the native agent discards the
        predictor session right after predicting, the task loop starts from the next
        span's session window. Seeding with the predictor session makes the downstream
        model only continue text, never call tools (root cause of autogen inv0 all
        failing: 2 rounds TextMessage -> stop=None -> not_evaluated). Aligned with the simplified backend here: seed with the next span's session window instead, prediction text does not enter the task loop."""
        if session.autogen_messages is not None:
            return session.loop_completed
        span = session.span
        is_prediction = (span is not None
                         and span.span_descriptor.node_pattern == "tool_selection")
        if is_prediction:
            conv = self._next_span_conversation(session, span)
            if not conv:
                conv = session.conversation
            msgs = self._recorded_to_autogen(conv)
            injected_msgs: List[Any] = []
            completed = False
            # prediction text does not enter the task loop; only honestly record the injected turn
            parsed = session.injected
            self._trace_step(session, {
                "span_type": "llm.call", "step": 0, "source": "slm",
                "output": {"text": str(
                    (parsed or {}).get("text", "") if isinstance(parsed, dict)
                    else parsed or "")[:TRACE_RESULT_MAX_LEN]}})
        else:
            msgs = self._recorded_to_autogen(session.conversation)
            injected_msgs, completed = self._autogen_inject_span(session)
        session.autogen_messages = msgs + injected_msgs
        session.loop_completed = session.loop_completed or completed
        return completed

    def _autogen_continuation(self, session: AppWorldSession,
                              stop_after_span: bool) -> Tuple[List[LMResponse], bool]:
        """Drive the downstream with the REAL AutoGen agent: seed an
        AssistantAgent's model context with the restored conversation plus
        the injected span (executed through the real tools), then let the
        agent continue with the baseline model until the task completes."""
        import asyncio
        from autogen_agentchat.agents import AssistantAgent
        from autogen_core.model_context import BufferedChatCompletionContext
        from autogen_core import CancellationToken
        from autogen_core.models import ModelFamily, ModelInfo
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from appworld_agents.code.autogen.agent import _current_world

        bridge = session.bridge
        assert bridge is not None
        completed = self._autogen_seed(session)
        if stop_after_span or completed:
            return [], completed

        bc = session.baseline_client
        # local deterministic baseline support: the OpenAIChatCompletionClient
        # constructor has no seed, so a harness-side subclass injects
        # extra_create_args.seed on every create() (request args only; no agent logic touched)
        class _SeededOpenAIClient(OpenAIChatCompletionClient):
            def __init__(self, seed, **kwargs):
                super().__init__(**kwargs)
                self._seed = seed
                # kept for diagnosis: record the latest create() raw result
                # (finish_reason / thought=reasoning_content / content); on
                # empty replies in thinking mode it distinguishes "reasoning truncated (length)" from "the model chose not to speak (stop)"
                self._last_diag: Dict[str, Any] = {}

            def _process_create_args(self, messages, tools, tool_choice,
                                     json_output, extra_create_args):
                # CreateParams is a plain class (attribute access, not a dict)
                params = super()._process_create_args(
                    messages, tools, tool_choice, json_output,
                    extra_create_args)
                # align with the native / verified request: ① autogen adds
                # name="user" to user messages (the diag bare request lacks
                # this field; vLLM thinking templates are name-sensitive);
                # ② when tool_choice is None the SDK explicitly sends "tool_choice": null (the diag request did not send this field)
                for _m in params.messages:
                    if isinstance(_m, dict) and _m.get("role") == "user":
                        _m.pop("name", None)
                if params.create_args.get("tool_choice") is None:
                    params.create_args.pop("tool_choice", None)
                return params

            async def create(self, messages, *, tools=None, tool_choice=None,
                             json_output=None, extra_create_args=None,
                             cancellation_token=None):
                extra = dict(extra_create_args or {})
                extra.setdefault("seed", self._seed)
                # outbound payload kept for diagnosis: _process_create_args
                # is what the SDK will send (OpenAI-format messages/tools);
                # when empty replies remain unsolved, use it to compare field by field with the diag bare request
                try:
                    _p = super()._process_create_args(
                        messages, tools, tool_choice, json_output,
                        extra_create_args or {})
                    self._outbound = {
                        "msgs": json.dumps(list(_p.messages),
                                           ensure_ascii=False, default=str)[:900],
                        "tools": json.dumps(list(_p.tools or []),
                                            ensure_ascii=False, default=str)[:900],
                        "args": json.dumps(
                            {k: v for k, v in _p.create_args.items()},
                            ensure_ascii=False, default=str)[:300],
                    }
                except Exception:
                    self._outbound = {}
                t_create0 = monotonic_ms()
                result = await super().create(
                    messages, tools=tools, tool_choice=tool_choice,
                    json_output=json_output, extra_create_args=extra,
                    cancellation_token=cancellation_token)
                session.autogen_llm_times.append(
                    monotonic_ms() - t_create0)
                # LLM thinking: CreateResult.thought = the API's
                # reasoning_content (the vendored autogen already parsed
                # it); persisted in call order with the same discipline as llm_times, the event loop pops one by one to pair them
                session.autogen_thoughts.append(
                    str(getattr(result, "thought", "") or ""))
                self._last_diag = {
                    "finish_reason": str(getattr(result, "finish_reason", None)),
                    "content": str(getattr(result, "content", ""))[:800],
                    "thought": str(getattr(result, "thought", "") or "")[:800],
                    "n_tools_requested": len(tools or []),
                    "outbound": self._outbound if hasattr(self, "_outbound") else {},
                    "usage": {"prompt": getattr(result.usage, "prompt_tokens", None),
                              "completion": getattr(result.usage, "completion_tokens", None)}
                    if getattr(result, "usage", None) else None,
                }
                return result

        model_client = _SeededOpenAIClient(
            seed=bc.seed if getattr(bc, "seed", None) is not None else 100,
            model=bc.model,
            api_key=bc.api_key or "placeholder",
            base_url=bc.base_url,
            model_info=ModelInfo(vision=False, function_calling=True,
                                 json_output=True, structured_output=False,
                                 family=ModelFamily.UNKNOWN),
            temperature=bc.temperature,
            # unified convention (2026-09-20.2): all model calls explicitly
            # use max output 8192 (reverting .31/.20's "no cap / server
            # default" convention; 16.12 already verified 8192 does not truncate thinking mode)
            max_tokens=8192,
            # same convention as the collection side: _build_client explicitly uses timeout=600.0 / max_retries=3
            timeout=600.0,
            max_retries=3,
        )
        context = BufferedChatCompletionContext(buffer_size=512)

        # BufferedChatCompletionContext.add_message is async: must await,
        # otherwise the rebuilt session history is silently dropped and the continuation model gets no task context
        async def _seed_context() -> None:
            for msg in self._autogen_model_inputs(session.autogen_messages):
                await context.add_message(msg)
        asyncio.run(_seed_context())
        tools = self._autogen_tools(session)

        # system_message: the real agent's system prompt (exactly the
        # collection config: function_calling_agent/instructions.txt
        # template + task instruction + app descriptions + max_steps=50, using the same rendering functions the agent itself uses)
        import appworld_agents.code.autogen as _autogen_mod
        from appworld.common.io import dump_yaml, read_file
        from appworld.common.prompts import load_prompt_to_chat_messages
        from appworld.common.text import render_template
        _code_root = os.path.dirname(os.path.abspath(_autogen_mod.__file__))
        _exp_root = os.path.dirname(os.path.dirname(_code_root))
        _template = read_file(os.path.join(
            _exp_root, "prompts", "function_calling_agent", "instructions.txt"))
        _app_descs = dict(bridge.world.task.app_descriptions)
        _app_descs.pop("api_docs", None)
        _content = render_template(
            _template,
            instruction=bridge.world.task.instruction,
            app_descriptions=dump_yaml(_app_descs).rstrip(),
            main_user=bridge.world.task.supervisor,
            max_steps=50)
        _header = load_prompt_to_chat_messages(
            _content, skip_system_message=False, only_header=True)
        _system_text = "\n".join(str(m.get("content", "")) for m in _header
                                 if m.get("role") == "system")
        if not _system_text:
            _system_text = _content

        # align with the native request: when the seeded session already
        # contains the recorded system message, do not stack the rendered
        # system on top (else the model sees two systems + a duplicated task instruction; observed in thinking mode: degrades to plain-text reply / API list, no tool calls)
        _seeded = session.autogen_messages or []
        _has_system = bool(_seeded) and type(_seeded[0]).__name__ == "SystemMessage"
        agent = AssistantAgent(
            name="assistant",
            model_client=model_client,
            tools=tools,
            system_message=(None if _has_system else _system_text),
            # same convention as the collection side (run_benchmark_autogen.py):
            # reflect after tool execution before acting -- the source of
            # the "5-21 llm.call" counts, and the convergence opportunity before the final answer (bare-loop False means over-collection / verbose answers with no correction point)
            reflect_on_tool_use=True,
            # same convention as the collection side (run_benchmark_autogen.py):
            # max_tool_iterations=self.max_steps = agent_config.max_steps = 50
            # (the previous 30 was a leftover from the old collection version, 20 rounds fewer than the collection side; downstream would stop early)
            max_tool_iterations=50,
            model_context=context,
        )
        # task=None: inject no trigger text. 0.7.5's run(task=None) adds no message
        # and drives the model loop directly with the seeded session -- verbatim
        # identical to the recorded native request (observed: with run(task=None),
        # create() receives exactly the seeded [SystemMessage, UserMessage]). Both
        # previous versions failed: task=task instruction was answered as a new
        # question in plain text; task=continue_prompt is synthetic input (violates input fidelity, empty replies in thinking mode).
        if not hasattr(session, "autogen_llm_times"):
            session.autogen_llm_times = []
        if not hasattr(session, "autogen_tool_times"):
            session.autogen_tool_times = []
        if not hasattr(session, "autogen_thoughts"):
            session.autogen_thoughts = []
        token = _current_world.set(bridge.world)
        try:
            result = asyncio.run(agent.run(
                task=None, cancellation_token=CancellationToken()))
        finally:
            _current_world.reset(token)
        def _dump_msg(m: Any) -> Dict[str, str]:
            d = {"type": type(m).__name__}
            # put content first with enough length (str(m) puts content
            # last; the old [:200] truncation never showed the model's
            # actual reply); the reasoning field (thinking mode) is kept too
            for attr in ("content", "reasoning", "thought", "source"):
                if hasattr(m, attr):
                    d[attr] = str(getattr(m, attr))[:600].replace("\n", " ")
            return d

        session.autogen_run_info = {
            "stop_reason": str(result.stop_reason),
            "n_messages": len(result.messages),
            "last_message_types": [type(m).__name__ for m in result.messages[-3:]],
            "last_contents": [str(m)[:200].replace("\n", " ")
                              for m in result.messages[-3:]],
            # full record kept for diagnosis: raw content/reasoning of the last 3 messages
            "last_messages": [_dump_msg(m) for m in result.messages[-3:]],
            # the model's raw response (finish_reason/thought/content/usage)
            "model_diag": dict(getattr(model_client, "_last_diag", {}) or {}),
        }
        # structured trace: all events produced by the run (span structure aligned with the original trace)
        step = 1
        pending_args: Dict[str, Dict[str, Any]] = {}
        for msg in result.messages:
            mtype = type(msg).__name__
            if mtype == "ToolCallRequestEvent":
                calls = [{"name": c.name,
                          "arguments": _json_args_or_empty(c.arguments)}
                         for c in (msg.content or [])]
                pending_args = {c.id: _json_args_or_empty(c.arguments)
                                for c in (msg.content or [])}
                self._trace_step(session, {
                    "span_type": "llm.call", "step": step,
                    "source": "baseline",
                    "elapsed_ms": (session.autogen_llm_times.pop(0)
                                   if session.autogen_llm_times else 0.0),
                    "output": {"tool_calls": calls,
                               "reasoning": (session.autogen_thoughts.pop(0)
                                             if session.autogen_thoughts
                                             else "")}})
            elif mtype == "ToolCallExecutionEvent":
                for res in (msg.content or []):
                    txt = str(getattr(res, "content", ""))
                    tname = str(getattr(res, "name", "")).replace(".", "__")
                    tool_ms = 0.0
                    if session.autogen_tool_times:
                        _tn, tool_ms = session.autogen_tool_times.pop(0)
                    self._trace_step(session, {
                        "span_type": "tool.execution", "step": step,
                        "source": "baseline",
                        "tool": tname,
                        "arguments": pending_args.get(
                            getattr(res, "call_id", ""), {}),
                        "ok": not txt.startswith("Error"),
                        "elapsed_ms": tool_ms,
                        "result": txt[:TRACE_RESULT_MAX_LEN]})
            elif mtype in ("TextMessage", "TextMessageEvent",
                           "StructuredMessage"):
                # only record model-side text turns: the user side (echo of
                # the continuation trigger) has no corresponding llm.call
                # event in the recorded trace; recording it would add an R to the signature sequence and pollute the structural comparison with the recorded trajectory (d_t / edit distance)
                if str(getattr(msg, "source", "")) == "user":
                    continue
                txt = str(getattr(msg, "content", ""))
                if txt.strip():
                    self._trace_step(session, {
                        "span_type": "llm.call", "step": step,
                        "source": "baseline",
                        "elapsed_ms": (session.autogen_llm_times.pop(0)
                                       if session.autogen_llm_times else 0.0),
                        "output": {"text": txt[:TRACE_RESULT_MAX_LEN],
                                   "reasoning": (session.autogen_thoughts.pop(0)
                                                 if session.autogen_thoughts
                                                 else "")}})
            step += 1
        world_done = bool(bridge.world.task_completed())
        self._trace_terminal(session, {
            "status": "completed" if world_done
                      else "stopped: %s" % str(result.stop_reason),
            "world_completed": world_done})
        usage_total = model_client.total_usage()
        usage = [LMResponse(
            prompt_tokens=usage_total.prompt_tokens,
            completion_tokens=usage_total.completion_tokens,
            latency_ms=0.0,
        )]
        try:
            asyncio.run(model_client.close())
        except Exception:
            pass
        # completion is determined by the environment facts (consistent
        # with the simplified backend's task-level determination): once the
        # agent called complete_task inside run, world.task_completed() is
        # true; no harness-side continuation loop is needed. Previously this wrongly returned the seed stage's completed (always False), causing evaluator not_evaluated and strict all 0.
        return usage, world_done

    # -- 14.4 steps 7-8 (C0 frozen continuation) ---------------------------
    def _continue_frozen(self, session: AppWorldSession,
                         stop_after_span: bool) -> Tuple[List[LMResponse], bool]:
        """C0 frozen continuation (plan §4): the target span injects the recorded output
        y_j^0, all downstream LM spans inject the trace-recorded output, only tools/state
        transitions are re-executed, and each span's execution results are compared with
        the recorded observations -- no LM calls at all throughout.

        The key difference from live continuation: what drives this is not the agent loop
        but the recorded sequence itself -- C0 answers replay fidelity (are checkpoint / tools / evaluator reproducible), not model behavior. When a downstream recorded
        output cannot be parsed or has no tool calls, stop honestly and record it into c0_fidelity."""
        bridge = session.bridge
        span = session.span
        trace = session.trace
        assert bridge is not None and span is not None and trace is not None
        if session.frozen_plan is None:
            session.frozen_plan = sorted(
                (sp for sp in trace.spans
                 if sp.invocation_index > span.invocation_index),
                key=lambda sp: sp.invocation_index)
            session.frozen_index = 0
            session.c0_fidelity = []
        fidelity = session.c0_fidelity
        completed = bool(session.frozen_completed
                         or bridge.world.task_completed())

        def _exec_calls(calls: List[Dict[str, Any]], step: int,
                        source: str) -> Tuple[bool, List[Dict[str, Any]]]:
            """Execute a group of tool calls; complete_task is excluded from
            the comparison (the evaluator comparison already covers the final state). Returns (completed, execution observations excluding complete_task)."""
            executed: List[Dict[str, Any]] = []
            for c in calls:
                name = str(c.get("name", "")).replace("__", ".")
                args = c.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                if name.endswith("complete_task"):
                    t_tool0 = monotonic_ms()
                    bridge.complete_task(args.get("answer"),
                                         args.get("status", "success"))
                    tool_ms = monotonic_ms() - t_tool0
                    self._trace_step(session, {
                        "span_type": "tool.execution", "step": step,
                        "source": source,
                        "tool": name.replace(".", "__"),
                        "arguments": args, "ok": True,
                        "elapsed_ms": tool_ms,
                        "result": "Marked the active task complete."})
                    return True, executed
                t_tool0 = monotonic_ms()
                res = bridge.execute_tool(name, args)
                tool_ms = monotonic_ms() - t_tool0
                executed.append({
                    "tool": name,
                    "ok": bool(res["ok"]),
                    "result": (str(res["result"]) if res["ok"]
                               else str(res["error"]))[:TRACE_RESULT_MAX_LEN]})
                self._trace_step(session, {
                    "span_type": "tool.execution", "step": step,
                    "source": source,
                    "tool": name.replace(".", "__"), "arguments": args,
                    "ok": bool(res["ok"]),
                    "elapsed_ms": tool_ms,
                    "result": (str(res["result"]) if res["ok"]
                               else str(res["error"]))[:TRACE_RESULT_MAX_LEN]})
            return False, executed

        def _with_recorded_args(calls: List[Dict[str, Any]],
                                rec_res: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Frozen walk-through replays with the recorded execution args:
            agent frameworks add schema defaults and login tokens between
            the model output and execution (observed venmo call: output 5
            args, executed 9 args; replay without the token must 401), so
            replaying by the model output necessarily distorts. Pair with the recorded executions in order, replace the output args with the recorded arguments; keep the original args when unpaired."""
            from ..local_equivalence import _normalize_tool_name
            out: List[Dict[str, Any]] = []
            used: set = set()
            for c in calls:
                nm = _normalize_tool_name(c.get("name"))
                if nm.endswith("complete_task"):
                    # complete_task does not join the tool comparison; its
                    # recorded execution args may lack answer (recording
                    # basis), and the final-answer determination needs the
                    # full args (simplified observed: after replacement answer=None -> evaluation failed). Keep the model output's (recorded output's) original args.
                    out.append(c)
                    continue
                hit = None
                for i, r in enumerate(rec_res):
                    if i in used:
                        continue
                    if _normalize_tool_name(r.get("tool")) == nm:
                        hit = (i, r)
                        break
                if hit is not None:
                    used.add(hit[0])
                    args = hit[1].get("arguments")
                    if isinstance(args, dict):
                        c = dict(c)
                        # the tau2 collector's record format: the whole args
                        # dict is wrapped as a repr string under the "raw"
                        # key (the result of record_appworld_tool parsing
                        # the tau2 code string); parse out the real args.
                        # There are also raw=False args injected by the as_tool wrapper; the wrapper layer consumes them, they are not passed to the API -- strip them too.
                        _rawv = args.get("raw")
                        if isinstance(_rawv, str):
                            try:
                                import ast as _ast
                                _parsed = _ast.literal_eval(_rawv)
                                if isinstance(_parsed, dict):
                                    args = _parsed
                            except Exception:
                                pass
                        args = {k: v for k, v in args.items() if k != "raw"}
                        c["arguments"] = args
                out.append(c)
            return out

        def _compare(inv: int, executed: List[Dict[str, Any]],
                     recorded: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Two-layer comparison (plan §4.4): - tool_match (hard): tool names/counts/ok flags match
            (complete_task excluded on both sides -- its final effect is covered by the evaluator comparison)
            - value_match (soft): result text semantically consistent. Login tokens, timestamps and other
            ephemeral fields are naturally unreproducible, reported separately and not judged a hard
            failure"""
            from ..local_equivalence import compare_values, _normalize_tool_name
            rec = [r for r in recorded
                   if not str(r.get("tool", "")).endswith("complete_task")]
            exec_names = [_normalize_tool_name(e.get("tool")) for e in executed]
            rec_names = [_normalize_tool_name(r.get("tool")) for r in rec]
            names_match = sorted(exec_names) == sorted(rec_names)
            reqs = [{
                "requirement": "assert executed tool names match (ignore order, complete_task excluded).",
                "passed": names_match,
                "details": "exec=%s vs recorded=%s" % (exec_names, rec_names)
                if not names_match else "",
            }]
            ok_match = True
            value_match = True
            details = []
            if len(executed) != len(rec):
                ok_match = False
                details.append("execution count mismatch: exec=%d recorded=%d"
                               % (len(executed), len(rec)))
            else:
                rec_by_name = {_normalize_tool_name(r.get("tool")): r
                               for r in rec}
                for e in executed:
                    key = _normalize_tool_name(e.get("tool"))
                    r = rec_by_name.get(key)
                    if r is None:
                        ok_match = False
                        continue
                    if bool(e.get("ok")) != bool(r.get("ok")):
                        ok_match = False
                        details.append("%s: ok flag %r != %r" % (
                            key, e.get("ok"), r.get("ok")))
                        continue
                    ok_v, det = compare_values(
                        str(e.get("result", "")), str(r.get("result", "")))
                    if not ok_v:
                        value_match = False
                        details.append("%s: value diff %s" % (
                            key, str(det)[:120]))
            reqs.append({
                "requirement": "assert tool execution ok flags match (complete_task excluded).",
                "passed": ok_match,
                "details": "; ".join(details)[:500]})
            reqs.append({
                "requirement": "assert tool execution result values match (ephemeral fields may differ).",
                "passed": value_match,
                "details": "" if value_match
                else "value-level diffs (ephemeral-sensitive)"})
            return {
                "invocation_index": inv,
                "tool_match": bool(names_match and ok_match),
                "value_match": bool(value_match),
                "requirements": reqs,
                "executed": executed,
            }

        recorded_results = trace.artifacts.get("span_tool_results") or {}
        step = 0
        # 1) the target span (the injected y_j^0): run_to_node consumes it
        #    once and executes its tools; run_to_completion skips it on re-entry (frozen_target_done)
        if not completed and not getattr(session, "frozen_target_done", False):
            session.frozen_target_done = True
            parsed = session.injected
            if isinstance(parsed, dict) and parsed.get("tool_calls"):
                calls = parsed["tool_calls"]
                rec_res = recorded_results.get(
                    "span:%d" % span.invocation_index, [])
                calls = _with_recorded_args(calls, rec_res)
                self._trace_step(session, {
                    "span_type": "llm.call", "step": step, "source": "frozen",
                    "elapsed_ms": 0.0, "output": {"tool_calls": [
                        {"name": c.get("name", ""),
                         "arguments": c.get("arguments")} for c in calls]}})
                completed, executed = _exec_calls(calls, step, "frozen")
                fidelity.append(
                    _compare(span.invocation_index, executed, rec_res))
            else:
                # freeform recorded output (e.g. api prediction): no tools to execute, record honestly
                self._trace_step(session, {
                    "span_type": "llm.call", "step": step, "source": "frozen",
                    "elapsed_ms": 0.0, "output": {"text": str(parsed or "")[:TRACE_RESULT_MAX_LEN]}})
                fidelity.append({
                    "invocation_index": span.invocation_index,
                    "tool_match": None,
                    "note": "recorded target output has no tool calls"})
        # 2) the downstream span sequence: inject each recorded output and
        #    re-execute its tools. Recorded spans with no tool calls (text
        #    turns, e.g. freeform reasoning/answer) have no state effect:
        #    record and keep walking, do not stop -- the terminal complete_task often sits in the next span (383cbac_3 observed: inv9 text turn, inv10 submits)
        if not stop_after_span and not completed:
            for k in range(session.frozen_index, len(session.frozen_plan)):
                rec = session.frozen_plan[k]
                session.frozen_index = k + 1
                parsed = rec.baseline_response.parsed
                if not isinstance(parsed, dict) or not parsed.get("tool_calls"):
                    step += 1
                    self._trace_step(session, {
                        "span_type": "llm.call", "step": step,
                        "source": "frozen",
                        "elapsed_ms": 0.0, "output": {"text": str(parsed or "")[:TRACE_RESULT_MAX_LEN]}})
                    fidelity.append({
                        "invocation_index": rec.invocation_index,
                        "node_pattern": rec.span_descriptor.node_pattern,
                        "tool_match": None,
                        "note": "recorded downstream span has no tool calls (text-only, no state effect); walk continues"})
                    continue
                calls = parsed["tool_calls"]
                rec_res = recorded_results.get(
                    "span:%d" % rec.invocation_index, [])
                calls = _with_recorded_args(calls, rec_res)
                step += 1
                self._trace_step(session, {
                    "span_type": "llm.call", "step": step,
                    "source": "frozen",
                    "elapsed_ms": 0.0, "output": {"tool_calls": [
                        {"name": c.get("name", ""),
                         "arguments": c.get("arguments")} for c in calls]}})
                completed, executed = _exec_calls(calls, step, "frozen")
                fidelity.append(
                    _compare(rec.invocation_index, executed, rec_res))
                if completed:
                    break
        if completed and not session.frozen_completed:
            session.frozen_completed = True
            self._trace_terminal(session, {
                "status": "completed", "world_completed": True})
        elif not completed and not stop_after_span:
            self._trace_terminal(session, {
                "status": "frozen replay exhausted without completion",
                "world_completed": bool(bridge.world.task_completed())})
        return [], completed

    def run_to_node(self, session: AppWorldSession, span: SpanRecord,
                    client: Any, max_steps: int, timeout_s: float,
                    budget_usd: float) -> Tuple[DownstreamOutcome, List[LMResponse]]:
        outcome = DownstreamOutcome()
        n_before = len(session.loop_messages) if session.loop_messages is not None else 0
        backend = getattr(session, "backend", "") or "generic"
        try:
            if getattr(session, "frozen_downstream", False):
                usage, completed = self._continue_frozen(
                    session, stop_after_span=True)
                injected_consumed = True
            elif backend == "simplified":
                usage, completed = self._continue_with_simplified(
                    session, max_steps=max_steps)
                injected_consumed = True
            elif backend == "autogen":
                usage, completed = self._autogen_continuation(
                    session, stop_after_span=True)
                injected_consumed = session.autogen_messages is not None
            elif backend == "legacy":
                usage, completed = self._continue_with_legacy(
                    session, stop_after_span=True, max_steps=max_steps)
                injected_consumed = session.legacy_messages is not None
            elif backend == "smolagents":
                usage, completed = self._continue_with_smolagents(
                    session, stop_after_span=True, max_steps=max_steps)
                injected_consumed = (session.loop_messages is not None
                                     and len(session.loop_messages) > n_before)
            elif backend == "tau2":
                usage, completed = self._continue_with_tau2(
                    session, stop_after_span=True, max_steps=max_steps)
                injected_consumed = session.tau2_state is not None
            elif backend == "openai":
                usage, completed = self._continue_with_openai_sdk(
                    session, stop_after_span=True, max_steps=max_steps)
                injected_consumed = session.openai_items is not None
            else:  # unknown backend: report honestly per the hard rule, never degrade to an approximation
                raise RuntimeError(
                    "unknown continuation backend %r: the downstream "
                    "continuation must run the REAL agent source; generic "
                    "approximation is not an acceptable fallback" % backend)
                injected_consumed = (session.loop_messages is not None
                                     and len(session.loop_messages) > n_before + 1)
            outcome.next_node_reached = True  # loop advanced past span j
            outcome.next_tool_executed = injected_consumed
            outcome.downstream_execution_valid = injected_consumed
            outcome.downstream_checkpoint_reached = completed or bool(usage)
        except Exception:
            outcome.unhandled_exception = True
            outcome.downstream_execution_valid = False
            outcome.downstream_checkpoint_reached = False
            outcome.next_node_reached = False
            usage = []
        return outcome, usage

    def run_to_completion(self, session: AppWorldSession, span: SpanRecord,
                          client: Any, max_steps: int, timeout_s: float,
                          budget_usd: float) -> Tuple[FinalOutcome, List[LMResponse]]:
        final = FinalOutcome()
        bridge = session.bridge
        backend = getattr(session, "backend", "") or "generic"
        try:
            if getattr(session, "frozen_downstream", False):
                usage, completed = self._continue_frozen(
                    session, stop_after_span=False)
            elif backend == "simplified":
                usage, completed = self._continue_with_simplified(
                    session, max_steps=max_steps)
            elif backend == "autogen":
                usage, completed = self._autogen_continuation(
                    session, stop_after_span=False)
            elif backend == "legacy":
                usage, completed = self._continue_with_legacy(
                    session, stop_after_span=False, max_steps=max_steps)
            elif backend == "smolagents":
                usage, completed = self._continue_with_smolagents(
                    session, stop_after_span=False, max_steps=max_steps)
            elif backend == "tau2":
                usage, completed = self._continue_with_tau2(
                    session, stop_after_span=False, max_steps=max_steps)
            elif backend == "openai":
                usage, completed = self._continue_with_openai_sdk(
                    session, stop_after_span=False, max_steps=max_steps)
            else:  # unknown backend: report honestly per the hard rule, never degrade to an approximation
                raise RuntimeError(
                    "unknown continuation backend %r: the downstream "
                    "continuation must run the REAL agent source; generic "
                    "approximation is not an acceptable fallback" % backend)
            # guard A (backend-agnostic completion determination): whether
            # the world completed is an environment fact; trust no backend's
            # returned flag. Previously the autogen backend returned a
            # constant False due to a wiring error, causing evaluator not_evaluated; henceforth even if a backend under-reports, evaluate normally once the environment completed. Read-only environment state; touch no agent logic.
            if not completed:
                completed = bool(bridge.world.task_completed())
            if completed:
                eval_result = bridge.evaluate()
                final.goal_state_passed = eval_result["goal_state_passed"]
                final.final_task_success = eval_result["final_task_success"]
                final.collateral_damage = eval_result["collateral_damage"]
                final.evaluator_details = eval_result["evaluator_details"]
            else:
                traj = getattr(session, "simplified_trajectory", None)
                if traj:
                    final.evaluator_details.append(
                        "not_evaluated: trajectory(last 8)=%s" % traj[-8:])
                else:
                    info = getattr(session, "autogen_run_info", None)
                    if info:
                        final.evaluator_details.append(
                            "not_evaluated: stop=%s rounds=%d last_types=%s last_msgs=%s diag=%s" % (
                                info["stop_reason"], info["n_messages"],
                                info["last_message_types"],
                                json.dumps(info.get("last_messages", []),
                                           ensure_ascii=False)[:1200],
                                json.dumps(info.get("model_diag", {}),
                                           ensure_ascii=False)[:2200]))
                    else:
                        terminal = ((session.downstream_trace or {}).get("terminal") or {})
                        backend_info = getattr(session, "backend_info", None) or []
                        final.evaluator_details.append(
                            "not_evaluated: backend=%s terminal=%s%s" % (
                                backend,
                                str(terminal.get("status", "?"))[:200],
                                " %s" % backend_info[:2] if backend_info else ""))
        except Exception as exc:
            final.evaluator_details.append("exception: %s: %s" % (type(exc).__name__, exc))
            usage = []
        return final, usage

    def rollback(self, session: AppWorldSession) -> bool:
        # pre-commit verifiers only: nothing committed by the SLM output
        # before injection, so rollback is trivially successful.
        return True
