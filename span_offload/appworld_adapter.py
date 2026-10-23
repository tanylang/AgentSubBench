"""Adapter for the real AppWorld trace format (OTEL-style, e.g. 0d8a4ee_1.json).

Current exporter shape (smolagents-based agent):
  top level: trace_id, episode_id, agent_name, task, total_steps,
             total_llm_calls, total_tool_calls, input_tokens, output_tokens,
             spans
  SERVER span (invoke_agent): benchmark_task_id, model, agent.outcome
  CLIENT spans (llm.call): gen_ai.input.messages (JSON string; roles may be
             'MessageRole.SYSTEM' enums or plain; content may be text or a
             list of {'type':'text','text':...} blocks), gen_ai.output.messages
             (JSON tool-call list, answer JSON, or freeform text),
             usage tokens, event_type in {planning, replanning, action,
             reasoning}
  INTERNAL spans (tool.execution): tool name/arguments/result,
             event_type in {observation, action, repair, terminal}

Classification (source event_type + content, recorded transparently in
semantic_constraints.source_event_type):
  planning + freeform API-list output      -> tool_selection (api_prediction)
  reasoning (answer phase)                 -> final_synthesis (freeform)
  output = {"answer": ..., "status": ...}  -> final_synthesis (json)
  tool-call list output:
      last message starts with "Error:"    -> tool_argument_generation,
                                              control_role repair_controller,
                                              after_failure = True
      otherwise                            -> tool_argument_generation,
                                              local_executor

REPLAY CAPABILITY (honest): two exporter formats are supported.

1. Legacy format (no checkpoints): the recorded input.messages are the exact
   prompt the baseline model saw (possibly a truncated tail window), so the
   LOCAL replay layer (same prompt -> SLM -> original parser -> local
   verifier, 14.4 steps 3-5) is measurable today.  DOWNSTREAM/FINAL layers
   additionally require the framework's full state and a live/checkpointed
   AppWorld environment -> spans are flagged ineligible with
   `environment_state_not_checkpointed` and
   `agent_state_not_checkpointed: conversation tail window only`.  Use
   replay --mode trace-local to measure the local layer on these spans.

2. Checkpoint format (check_point_trace/*.json, AutoGen exporter): each LLM
   span carries a span-level check_point with the FULL conversation
   (has_full_conversation=true), an AppWorld "changes"-format env delta
   (env_changes, empty == unchanged since the previous checkpoint; the
   baseline at span 0), and an application_state_hash.  These spans pass the
   14.1 state-recoverability checks and unlock live replay once an
   AppWorld-backed AgentRuntime reconstructs the environment by replaying
   baseline + deltas.  Caveat: the current exporter writes the sha256 of the
   empty string as a placeholder hash; it is carried through and flagged
   (application_state_hash_is_placeholder) so replay records the mismatch
   honestly until the exporter computes real hashes.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    AgentCondition, BaselineResponse, Continuation, GraphContext, LMRequest,
    MechanismTags, OutputContractRuntime, RunMeta, SpanDescriptor, SpanRecord,
    StateSnapshot, TaskCondition, TaskManifest, ToolExecution, TraceRun,
    TRACE_SCHEMA_VERSION, state_hash,
)
from .trace_adapter import TraceAdapter

# Best-effort AppWorld tool effect table.  Unknown tools default to the
# conservative "external_state_mutation" so the 14.8 read_only allowlist rule
# never admits a potentially-mutating span.
#
# Apps added in newer AppWorld versions (amazon / gmail / splitwise / spotify /
# simple_note / file_system / api_docs / admin) follow the same convention:
# only clearly-read patterns (search_* / show_* / get_*) are marked read_only;
# everything else stays unknown -> conservative mutation.
APWORLD_TOOL_EFFECTS: Dict[str, str] = {
    "supervisor.show_active_task": "read_only",
    "supervisor.show_profile": "read_only",
    "supervisor.show_account_passwords": "read_only",
    "supervisor.show_addresses": "read_only",
    "supervisor.show_phone": "read_only",
    "supervisor.show_apis": "read_only",
    "supervisor.complete_task": "external_state_mutation",
    "phone.show_account": "read_only",
    "phone.show_contact_relationships": "read_only",
    "phone.show_contacts": "read_only",
    "phone.show_customers": "read_only",
    "phone.search_contacts": "read_only",
    "phone.send_text_message": "external_state_mutation",
    "phone.set_contact_phone": "external_state_mutation",
    "phone.update_contact_phone": "external_state_mutation",
    "phone.login": "external_state_mutation",
    "phone.get_current_date_and_time": "read_only",
    "venmo.search_users": "read_only",
    "venmo.search_friends": "read_only",
    "venmo.show_transactions": "read_only",
    "venmo.show_incoming_transactions": "read_only",
    "venmo.show_friend_relationships": "read_only",
    "venmo.send_money": "external_state_mutation",
    "venmo.request_money": "external_state_mutation",
    "venmo.login": "external_state_mutation",
    "calendar.show_calendar_events": "read_only",
    "calendar.search_calendar_events": "read_only",
    "calendar.create_calendar_event": "external_state_mutation",
    "email.search_emails": "read_only",
    "email.show_email": "read_only",
    "email.send_email": "external_state_mutation",
    "docs.search": "read_only",
    "docs.show_doc": "read_only",
    "docs.create_doc": "external_state_mutation",
    "settings.show": "read_only",
    "settings.update": "external_state_mutation",
    "met.send_money": "external_state_mutation",
    "met.show_balance": "read_only",
    # newer apps (check_point_trace era): read-only patterns only
    "amazon.search_products": "read_only",
    "amazon.show_product": "read_only",
    "amazon.show_product_details": "read_only",
    "amazon.show_purchase_history": "read_only",
    "gmail.search_emails": "read_only",
    "gmail.show_email": "read_only",
    "gmail.show_contacts": "read_only",
    "splitwise.search_friends": "read_only",
    "splitwise.show_transactions": "read_only",
    "splitwise.show_balances": "read_only",
    "splitwise.show_friend_relationships": "read_only",
    "spotify.search_songs": "read_only",
    "spotify.show_playlist": "read_only",
    "spotify.show_playlists": "read_only",
    "spotify.show_song_details": "read_only",
    "spotify.show_library": "read_only",
    "simple_note.search_notes": "read_only",
    "simple_note.show_note": "read_only",
    "file_system.search_files": "read_only",
    "file_system.show_file_content": "read_only",
    "file_system.show_directory": "read_only",
    "api_docs.search_api_docs": "read_only",
}

TOOL_CALL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["tool_calls"],
    "properties": {
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "arguments"],
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
            },
        },
    },
}

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["status"],
    "properties": {
        "answer": {},
        "status": {"type": "string"},
    },
}


def _reconstruct_system_prompt(system: str, known_tools: List[str]) -> str:
    """The exporter truncates the recorded system prompt (~500 chars), so the
    tool list and output-format instruction the baseline model actually saw are
    missing.  Rebuild them from the trace's own tool-execution records when the
    system message demonstrably lacks the tool names; the appended block is
    clearly marked so downstream records stay honest about the reconstruction."""
    missing = [t.replace("__", ".") for t in known_tools if t.replace("__", ".") not in system]
    if not missing:
        return system
    suffix = (
        "\n\n[NOTE: the recorded prompt was truncated by the trace exporter and "
        "did not include the tool list; span_offload appended the tools recorded "
        "in this trace so the candidate model sees the same capabilities.]\n"
        "You have access to these tools:\n" +
        "\n".join("- %s" % t for t in missing) +
        "\n\nFor each step, output EXACTLY a JSON list of tool calls in the form: "
        '[{"name": "<tool_name>", "arguments": {<json object>}}].'
    )
    return system + suffix


def _with_reconstructed_tools(messages: List[Dict[str, str]],
                              known_tools: List[str]) -> List[Dict[str, str]]:
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system":
            m["content"] = _reconstruct_system_prompt(m.get("content", ""), known_tools)
            break
    return out


# sha256 of the empty string: what the AutoGen exporter writes when it has not
# computed a real application_state_hash yet.  Detected and flagged, never
# silently trusted.
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _read_check_point(span: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the exporter's span-level check_point object
    (check_point_trace/*.json):

        {
          "task_id": ...,                 # AppWorld task id
          "application_state_hash": ...,  # sha256 hex; may be EMPTY_SHA256
          "env_changes": {                # AppWorld "changes"-format delta
              "<app>": [[sql, params, committed], ...], ...
          },
          "env_changes_baseline": {...},  # initial state, recorded on span 0
          "conversation": {
              "has_full_conversation": true,
              "messages": [{"role": ..., "content": ...}, ...],  # FULL history
              "output_text": ...
          },
        }

    Semantics: env_changes is the delta SINCE THE PREVIOUS span checkpoint, so
    restoring span j means replaying span 0's baseline plus the env_changes of
    every span before j.  An empty env_changes therefore means "state unchanged
    since the previous checkpoint" -- a valid checkpoint in a complete chain.
    """
    cp = span.get("check_point")
    return cp if isinstance(cp, dict) else {}


def _unwrap_text_blocks(content: str) -> str:
    """The smolagents collector str()'s the content-block list directly into
    "[{'type': 'text', 'text': ...}]". Restore it to plain text; only unwrap when
    ALL entries are {'type': 'text'} blocks (avoid breaking real JSON content that is itself list-shaped)."""
    stripped = (content or "").strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return content or ""
    import ast as _ast
    try:
        blocks = _ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return content
    if not isinstance(blocks, list) or not blocks:
        return content
    parts = []
    for b in blocks:
        if not isinstance(b, dict) or set(b.keys()) != {"type", "text"}:
            return content
        parts.append(str(b.get("text", "")))
    return "\n".join(parts)


def _checkpoint_conversation_messages(cp: Dict[str, Any],
                                      keep_raw_role: bool = False
                                      ) -> List[Dict[str, str]]:
    """Normalize the check_point conversation into OpenAI-style messages.
    The AutoGen exporter serializes every message with role "unk", so the true
    role is inferred from the content when needed:
      [FunctionCall(...)]           -> assistant (tool calls)
      [FunctionExecutionResult(...)] -> user (tool observation)
      anything else                 -> system (agent prompt / tool docs)
    Returns [] when the check_point carries no conversation.

    With keep_raw_role=True each message also keeps "_raw_role" (the raw,
    unnormalized role), used by the per-agent conversation splicing (smolagents'
    MessageRole.TOOL_CALL / TOOL_RESPONSE must be distinguished by raw role); the
    private key is stripped after splicing and never enters downstream artifacts / prompt."""
    conv = cp.get("conversation")
    if not isinstance(conv, dict):
        return []
    out: List[Dict[str, str]] = []
    for m in conv.get("messages") or []:
        if not isinstance(m, dict):
            continue
        # Don't lowercase up front: normalize_role's case-sensitive replacement (the
        # "MessageRole." prefix) depends on the original casing
        raw_role = str(m.get("role", "user")).strip()
        content = _unwrap_text_blocks(normalize_content(m.get("content", "")))
        role = normalize_role(raw_role)
        if raw_role in ("unk",):
            stripped = content.lstrip()
            if stripped.startswith("[FunctionCall("):
                role = "assistant"
            elif stripped.startswith("[FunctionExecutionResult("):
                role = "user"
        d: Dict[str, str] = {"role": role, "content": content}
        if keep_raw_role:
            d["_raw_role"] = raw_role.lower()
        out.append(d)
    return out


def _splice_tool_calls(messages: List[Dict[str, str]],
                         outputs: List[Any]) -> List[Dict[str, Any]]:
    """simplified-agent format: assistant messages carry no tool-call structure (tool calls are
    recorded only in each span's gen_ai.output.messages; the assistant message is either an empty
    placeholder or just narration). For the k-th assistant message after "# Real Task Instruction /
    # Task Instruction", splice in the k-th recorded output's tool_calls (native dict form), and
    attach tool_call_id / name to the following tool result messages — OpenAI requires every
    tool_call to be answered by a tool message, otherwise the downstream continuation (DeepSeek /
    vLLM strict validation) 400s directly. AutoGen-format assistant messages already contain
    FunctionCall text and outputs is empty, so this function is a no-op for them."""
    out: List[Dict[str, Any]] = []
    k = 0
    active = False
    for m in messages:
        mm = dict(m)
        role = mm.get("role", "")
        if role == "user":
            c = str(mm.get("content", ""))
            if "# Real Task Instruction" in c or "# Task Instruction" in c:
                active = True
            out.append(mm)
            continue
        if active and role == "assistant":
            # The k-th post-marker assistant corresponds to the k-th tool-call output; even if that
            # assistant message carries narration, its tool_calls must be spliced back (previously
            # only empty placeholders were patched, and narration turns lost their tool calls in
            # conversation rebuild). Messages whose content already carries tool-call text (AutoGen's
            # [FunctionCall(...)] / JSON list) are skipped, keeping the AutoGen path consistent with old behavior
            if ("[FunctionCall(" in str(mm.get("content", ""))
                    or str(mm.get("content", "")).strip().startswith("[")):
                k += 1
                out.append(mm)
                continue
            if k < len(outputs) and outputs[k] and outputs[k].get("tool_calls"):
                # OpenAI native tool_calls format (vLLM strict validation: id/type/function)
                mm["tool_calls"] = [{
                    "id": _synth_call_id("splice", k, i),
                    "type": "function",
                    "function": {
                        "name": str(c.get("name", "")),
                        "arguments": json.dumps(c.get("arguments") or {},
                                                ensure_ascii=False),
                    },
                } for i, c in enumerate(outputs[k]["tool_calls"])]
            k += 1
        out.append(mm)
    # Pairing: for the consecutive tool result messages after each spliced assistant
    # tool_calls, attach the corresponding call's tool_call_id and name in order
    for pos, mm in enumerate(out):
        tcs = mm.get("tool_calls") if mm.get("role") == "assistant" else None
        if not tcs:
            continue
        j = 0
        for nxt in out[pos + 1:]:
            if nxt.get("role") != "tool" or nxt.get("tool_call_id"):
                break
            if j >= len(tcs):
                break
            nxt["tool_call_id"] = tcs[j]["id"]
            nxt["name"] = tcs[j]["function"]["name"]
            j += 1
    # Remaining unpaired tool messages (e.g. a tutorial task's bare results with no
    # corresponding assistant tool_calls before them) are demoted to user, avoiding
    # sending the API tool messages without tool_call_id
    for mm in out:
        if mm.get("role") == "tool" and not mm.get("tool_call_id"):
            mm["role"] = "user"
    return out


def _strip_private(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip the private keys (_raw_role etc.) from conversation messages — they are
    for the splicer only and never enter artifacts / prompt."""
    return [{k: v for k, v in (m or {}).items() if not str(k).startswith("_")}
            for m in messages]


def _native_tool_calls(calls: List[Dict[str, Any]],
                       id_prefix: str = "splice") -> List[Dict[str, Any]]:
    """Structured calls -> OpenAI native tool_calls format (vLLM strict validation:
    id/type/function.name/function.arguments are all strings). arguments may be a
    dict or a JSON string; serialize uniformly to a JSON string."""
    out = []
    for i, c in enumerate(calls):
        args = c.get("arguments")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        elif args is None:
            args = "{}"
        out.append({
            "id": str(c.get("id") or _synth_call_id(id_prefix, i)),
            "type": "function",
            "function": {
                "name": str(c.get("name", "")),
                "arguments": str(args),
            },
        })
    return out


def _link_tool_messages(messages: List[Dict[str, Any]],
                        start: int, calls: List[Dict[str, Any]]) -> None:
    """Attach tool_call_id / name in order to the consecutive tool messages after
    start (the OpenAI protocol requires every tool_call to be answered by a tool message)."""
    j = 0
    for mm in messages[start:]:
        if mm.get("role") != "tool" or mm.get("tool_call_id"):
            break
        if j >= len(calls):
            break
        mm["tool_call_id"] = calls[j]["id"]
        mm["name"] = calls[j]["function"]["name"]
        j += 1


def _splice_output_calls(messages: List[Dict[str, Any]],
                         outputs_so_far: List[Any],
                         marker_gate: bool) -> List[Dict[str, Any]]:
    """legacy / tau2 splicing: the conversation is in OpenAI format, but the assistant
    turns' tool_calls structure was not recorded (legacy assistant content is 'None'; tau2
    assistant has only empty text). Backfill historical turns in order from each span's
    parsed output (outputs_so_far, structured tool_calls): "consecutive tool messages right after that
    assistant = that turn's call count" guarantees turn alignment; when the output and tool-message segment counts differ, take the smaller one (protocol first).

    With marker_gate=True (legacy), backfill only the turns after the LAST task-instruction
    marker — the legacy conversation head contains tutorial demo turns (assistant+tool
    exchanges) with no corresponding span outputs, which do not participate. The following
    tool messages must be a consecutive segment (stop at user/assistant), so narration turns are never wrongly merged in."""
    out = [dict(m) for m in messages]
    start = 0
    if marker_gate:
        for i, m in enumerate(out):
            if m.get("role") == "user" and any(
                    mk in str(m.get("content", ""))
                    for mk in ("# Real Task Instruction", "# Task Instruction")):
                start = i + 1  # the last marker wins (tutorial markers come first)
    k = 0
    for pos in range(start, len(out)):
        mm = out[pos]
        if mm.get("role") != "assistant" or mm.get("tool_calls"):
            continue
        n_tools = 0
        for nxt in out[pos + 1:]:
            if nxt.get("role") != "tool" or nxt.get("tool_call_id"):
                break
            n_tools += 1
        if n_tools == 0:
            continue
        if k >= len(outputs_so_far):
            break
        calls = (outputs_so_far[k] or {}).get("tool_calls", []) \
            if isinstance(outputs_so_far[k], dict) else []
        k += 1
        if not calls:
            continue
        take = min(n_tools, len(calls))
        mm["tool_calls"] = _native_tool_calls(calls[:take], "splice")
        if str(mm.get("content") or "").strip() in ("", "None"):
            mm["content"] = None
        _link_tool_messages(out, pos + 1, mm["tool_calls"])
    return out


def _splice_agent_conversation(agent_name: str, messages: List[Dict[str, Any]],
                               cp: Dict[str, Any],
                               outputs_so_far: List[Any]) -> List[Dict[str, Any]]:
    """Main per-agent conversation splicing entry: turn each agent's recorded
    conversation into the message list of its real request form.

    - simplified / autogen: reuse _splice_tool_calls (assistant placeholder +
      outputs_so_far; AutoGen's [FunctionCall(...)] text is a no-op)
    - legacy: the conversation contains tutorial demo turns; only the turns after
      the last task instruction are backfilled from outputs_so_far (tutorial turns have no span outputs)
    - tau2: the conversation is the full task-turn order; outputs_so_far is backfilled directly in order
    - smolagents: **keep the text tool protocol** — the original agent's real request form is
      TOOL_CALL -> assistant text ("Calling tools: ..."), TOOL_RESPONSE -> user text
      ("Observation: ..."), plus tools + tool_choice="required" (smolagents 1.19
      tool_role_conversions, no native tool_calls). normalize_role already does this
      mapping; no native structure is spliced here — the runtime follows the same protocol
      via the session.text_protocol flag
    - openai_agents: the conversation itself is not recorded (the collector passes empty
      messages); the caller rebuilds the transcript from output + tool execution records; kept as-is here
    Private keys are stripped after splicing."""
    if agent_name in ("AppWorld-Smolagents",):
        return _strip_private(messages)
    if agent_name in ("AppWorld-OpenAI-Agents"):
        # Conversation not recorded (except the span 0 "API prediction" placeholder
        # message); the transcript is rebuilt by the caller — here only strip private keys
        return _strip_private(messages)
    if agent_name in ("AppWorld-Legacy", "Tau2-LLMAgent"):
        return _strip_private(_splice_output_calls(
            messages, outputs_so_far,
            marker_gate=agent_name == "AppWorld-Legacy"))
    return _splice_tool_calls(messages, outputs_so_far)


def _openai_transcript(outputs_so_far: List[Any],
                       attached: List[List[ToolExecution]],
                       j: int) -> List[Dict[str, Any]]:
    """Conversation rebuild for the OpenAI SDK agent (transcript form): this agent's collector
    does not record the conversation (the streaming hook cannot see the request messages), so
    rebuild OpenAI-protocol messages from the span 0..j-1 parsed outputs (assistant tool calls)
    and tool execution records (tool results).

    The system/user head is a placeholder marked as such — the real system prompt needs task context
    and is re-rendered by the runtime with AppWorld during live replay (same template as the collection
    script); assistant calls are paired with tool results via deterministic synthetic ids; calls without execution records get a neutral placeholder receipt, keeping the OpenAI-protocol tool_call pairing complete."""
    msgs: List[Dict[str, Any]] = [{
        "role": "system",
        "content": ("[span_offload note: the OpenAI-SDK trace does not record "
                    "the conversation; this transcript was rebuilt from the "
                    "recorded tool calls and results. The live runtime "
                    "re-renders the real system prompt from the task context.]"),
    }, {
        "role": "user",
        "content": ("[span_offload note: the task instruction was not recorded "
                    "by the OpenAI-SDK collector; the live runtime restores it "
                    "from the task.]"),
    }]
    for k in range(1, j):
        # span k's output = outputs_so_far[k-1] (span 0 is the freeform API prediction and
        # does not enter outputs_so_far); attached is taken by span index
        out = outputs_so_far[k - 1] if k - 1 < len(outputs_so_far) else None
        calls = (out or {}).get("tool_calls", []) if isinstance(out, dict) else []
        if not calls:
            continue
        tcs = _native_tool_calls(calls, "oai")
        msgs.append({"role": "assistant", "content": None, "tool_calls": tcs})
        execs = attached[k] if k < len(attached) else []
        for i in range(len(tcs)):
            t = execs[i] if i < len(execs) else None
            msgs.append({
                "role": "tool",
                "tool_call_id": tcs[i]["id"],
                "name": tcs[i]["function"]["name"],
                "content": (str(t.result) if t is not None and t.success
                            else (str(t.error) if t is not None
                                  else "Error: no recorded execution")),
            })
    return msgs


def normalize_role(role: str) -> str:
    """'MessageRole.SYSTEM' / 'MessageRole.TOOL_RESPONSE' -> OpenAI-ish roles.
    TOOL_RESPONSE content already carries the 'Observation:' text, so mapping
    it to 'user' preserves semantics for any chat-template server.  'tool'
    stays 'tool' so _splice_tool_calls can pair the results with the spliced
    assistant tool_calls (any unpaired leftovers are demoted back to 'user')."""
    r = str(role).replace("MessageRole.", "").lower().strip()
    if r in ("system", "user", "assistant", "tool"):
        return r
    if r == "tool_call":
        return "assistant"
    if r == "tool_response":
        return "user"
    if r == "unk":
        # AutoGen's OTEL exporter records the system prompt with the "unk"
        # role; its content is the agent's system message.
        return "system"
    return "user"


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and "text" in b:
                parts.append(str(b["text"]))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def parse_messages(raw: str) -> List[Dict[str, str]]:
    """Parse gen_ai.input.messages into normalized OpenAI-style messages."""
    try:
        parsed = json.loads(raw or "[]")
    except ValueError:
        return [{"role": "user", "content": (raw or "")[:500]}]
    if not isinstance(parsed, list):
        return [{"role": "user", "content": str(parsed)[:500]}]
    out = []
    for m in parsed:
        if not isinstance(m, dict):
            continue
        out.append({
            "role": normalize_role(m.get("role", "user")),
            "content": normalize_content(m.get("content", "")),
        })
    return out


def _normalize_llm_output(text: str) -> str:
    """Strip reasoning-model artifacts before contract classification.

    MiniMax M-series (and similar reasoning SLMs) emit <think>...</think>
    blocks and may wrap JSON in markdown fences.  Recorded baseline outputs
    never contain these, so this is a no-op for them; the actual answer is
    whatever follows the last </think>."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    return text


def parse_llm_output(raw: str) -> Tuple[Any, str]:
    """Parse a recorded LLM output: tool-call list / answer JSON / freeform text.
    Returns (parsed, contract_kind) with contract_kind in
    {'tool_call_list', 'answer_json', 'freeform_text'}."""
    text = _normalize_llm_output(raw or "")
    if text.startswith("[{'name'"):
        # the simplified agent's output serialization (Python single-quoted dict list, arguments are strings)
        import ast as _ast
        try:
            raw_calls = _ast.literal_eval(text)
            normalized = []
            for c in raw_calls:
                args = c.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"_raw": args}
                normalized.append({"name": c.get("name", ""), "arguments": args or {}})
            if normalized:
                return ({"tool_calls": normalized}, "tool_call_list")
        except (ValueError, SyntaxError):
            pass
    if text.startswith("["):
        try:
            calls = json.loads(text)
            normalized = []
            for c in calls:
                args = c.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"_raw": args}
                normalized.append({"name": c.get("name", ""), "arguments": args or {}})
            return ({"tool_calls": normalized}, "tool_call_list")
        except ValueError:
            pass
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and ("answer" in obj or "status" in obj):
                return (obj, "answer_json")
        except ValueError:
            pass
    return ({"text": text}, "freeform_text")


def classify(event_type: str, output: str, messages: List[Dict[str, str]]) -> Tuple[str, str, str, str, List[str], List[str], List[str], bool]:
    """(node_pattern, control_role, phase, subtask, output_tags, verifier,
    context_sources, after_failure) -- canonical mapping of the smolagents
    span taxonomy (see module docstring)."""
    last = str(messages[-1].get("content") or "").strip() if messages else ""
    all_text = "\n".join(str(m.get("content") or "") for m in messages)
    if "List all APIs that may be needed" in all_text:
        return ("tool_selection", "global_planner", "initial", "api_prediction",
                ["freeform_text"], ["none"], ["tool_schema"], False)
    if event_type == "reasoning" or "please provide an answer" in last.lower():
        return ("final_synthesis", "terminal", "terminal", "final_answer_synthesis",
                ["freeform_text"], ["none"], ["conversation_history", "previous_tool_results"], False)
    parsed, kind = parse_llm_output(output)
    if kind == "answer_json":
        return ("final_synthesis", "terminal", "terminal", "final_answer_synthesis",
                ["json_schema"], ["schema_check"], ["conversation_history"], False)
    if kind == "tool_call_list":
        names = [str(c.get("name", "")) for c in parsed.get("tool_calls", [])]
        if any(n.endswith("complete_task") for n in names):
            # the agent's final commit step: AutoGen-style agents call
            # supervisor__complete_task as a regular tool call
            # ({"answer": ..., "status": ...})
            return ("final_synthesis", "terminal", "terminal",
                    "final_answer_synthesis",
                    ["json_schema"], ["schema_check"],
                    ["conversation_history", "previous_tool_results"], False)
        after_failure = last.startswith("Error:")
        if after_failure:
            return ("tool_argument_generation", "repair_controller", "recovery",
                    "error_recovery_retry",
                    ["tool_name", "tool_arguments", "json_schema"],
                    ["schema_check", "tool_exec_success"],
                    ["conversation_history", "tool_schema", "previous_tool_results"],
                    True)
        return ("tool_argument_generation", "local_executor", "execution",
                "tool_parameter_extraction",
                ["tool_name", "tool_arguments", "json_schema"],
                ["schema_check", "tool_exec_success"],
                ["conversation_history", "tool_schema", "previous_tool_results"],
                False)
    return ("other", "worker", "execution", event_type or "other",
            ["freeform_text"], ["none"], ["conversation_history"], False)


#: Derive clustering and architecture annotations from the agent name the trace
#: self-reports (gen_ai.agent.name). Agents that actually appear in the collection corpus:
#:   AppWorld-Simplified      AppWorld built-in SimplifiedFunctionCallingAgent
#:   AppWorld-Legacy          AppWorld built-in legacy FunctionCallingAgent
#:   AppWorld-OpenAI-Agents   OpenAI Agents SDK + MCP official flow
#:   AppWorld-Smolagents      smolagents ToolCallingAgent official flow
#:   AutoGen-AssistantAgent   AutoGen framework's AssistantAgent
#:   Tau2-LLMAgent            tau2-bench LLMAgent (driver loop in the adapter layer)
#: Unlisted names fall back honestly (slug + original name), never misclassified into known clusters.
_AGENT_ANNOTATIONS: Dict[str, Tuple[str, str]] = {
    "AppWorld-Simplified": (
        "appworld-simplified",
        "AppWorld built-in SimplifiedFunctionCallingAgent"),
    "AppWorld-Legacy": (
        "appworld-legacy",
        "AppWorld built-in legacy FunctionCallingAgent"),
    "AppWorld-OpenAI-Agents": (
        "appworld-openai-agents",
        "OpenAI Agents SDK agent (function calling over MCP)"),
    "AppWorld-Smolagents": (
        "appworld-smolagents",
        "smolagents ToolCallingAgent (official AppWorld flow)"),
    "AutoGen-AssistantAgent": (
        "appworld-autogen",
        "AutoGen AssistantAgent (autogen framework)"),
    "Tau2-LLMAgent": (
        "tau2-llmagent",
        "tau2-bench LLMAgent (driver loop in the benchmark harness)"),
}


def _agent_annotation(agent_name: str) -> Tuple[str, str]:
    if agent_name in _AGENT_ANNOTATIONS:
        return _AGENT_ANNOTATIONS[agent_name]
    name = str(agent_name).lower().strip()
    if name.startswith("appworld-"):
        name = name[len("appworld-"):]
    slug = re.sub(r"[^a-z0-9]+", "-", name).strip("-")
    return ("appworld-" + (slug or "unknown"),
            "unknown agent (%s)" % agent_name)


def _synth_call_id(*parts: Any) -> str:
    """Synthesize a server-style tool_call id (call_<24 hex digits>).

    The trace does not record the real server-generated call id, so the rebuilt
    context can only use a synthetic id; generate it with a deterministic hash so
    the same splice point is reproducible across runs, and its shape matches the
    real DeepSeek/vLLM id format (call_<alnum>), minimizing the text difference between the rebuilt context and the original request."""
    import hashlib
    h = hashlib.md5(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return "call_" + h[:24]


class AppWorldTraceAdapter(TraceAdapter):
    """Maps one native AppWorld trace file to the canonical TraceRun."""

    name = "appworld"

    def __init__(self, manifest: Optional[TaskManifest] = None):
        self.manifest = manifest

    # ------------------------------------------------------------------
    def load(self, path: str) -> TraceRun:
        # Windows-extracted spans files carry backslashes in trace_path and cannot be
        # opened on Linux; normalize to forward slashes (Windows accepts both, behavior unchanged)
        path = str(path).replace("\\", "/")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        server = next(s for s in data["spans"] if s.get("kind") == "SERVER")
        attrs: Dict[str, Any] = server.get("attributes", {})
        task_id = attrs.get("benchmark_task_id", "unknown_task")
        model = attrs.get("model", "unknown-model")
        agent_name = attrs.get("gen_ai.agent.name") or data.get("agent_name", "unknown-agent")
        # Baseline success follows the AppWorld built-in verifier's pass field (agent.outcome is
        # the agent's self-report; measured 59/65 claimed success but only 7/65 actually passed).
        # Old-format traces without a pass field fall back to agent.outcome.
        top_pass = data.get("pass")
        if top_pass is not None:
            outcome_ok = str(top_pass) == "success"
        else:
            outcome_ok = attrs.get("agent.outcome", "") == "success"

        llm_spans = [s for s in data["spans"]
                     if s.get("kind") == "CLIENT" and s.get("type") == "llm.call"]
        tool_spans = [s for s in data["spans"]
                      if s.get("kind") == "INTERNAL" and s.get("type") == "tool.execution"]

        known_tools: List[str] = []
        for s in data["spans"]:
            name = (s.get("attributes") or {}).get("gen_ai.tool.name", "")
            if name and name not in known_tools:
                known_tools.append(name)

        artifacts: Dict[str, Any] = {
            "state_snapshots": {}, "messages": {}, "system_prompts": {},
            "tool_schemas": {}, "tool_lists": {}, "previous_tool_results": {},
            "conversation_histories": {}, "plan_states": {}, "schemas": {},
            "response_formats": {}, "recorded_tools": {}, "span_tool_results": {},
        }

        node_ids: List[str] = []
        spans: List[SpanRecord] = []
        attached: List[List[ToolExecution]] = [[] for _ in llm_spans]
        # Write-delta accounting: the exporter attaches env_changes to tool.execution check_points
        # (LLM span check_points are always empty deltas). Attribute them to "the LLM span that
        # requested the tool" in order (tool execution order) — losing these deltas makes the recorded
        # hash and the delta chain diverge at the first write (measured on the 65/65 batch).
        tool_env_changes: List[List[Dict[str, Any]]] = [[] for _ in llm_spans]

        # walk in trace order: tool spans attach to the last-seen LLM span
        last_llm_index = -1
        for s in data["spans"]:
            if s in llm_spans:
                last_llm_index = llm_spans.index(s)
                continue
            if s in tool_spans and last_llm_index >= 0:
                ta = s.get("attributes", {})
                name = ta.get("gen_ai.tool.name", "").replace("__", ".")
                args_raw = ta.get("gen_ai.tool.call.arguments", "{}")
                try:
                    arguments = json.loads(args_raw)
                except ValueError:
                    arguments = {"_raw": args_raw}
                result_text = ta.get("gen_ai.tool.call.result", "")
                ok = "Execution failed" not in result_text and not result_text.startswith("Error")
                node_id = "tool:%d:%d" % (last_llm_index, len(attached[last_llm_index]))
                node_ids.append(node_id)
                attached[last_llm_index].append(ToolExecution(
                    tool_name=name, arguments=arguments, success=bool(ok),
                    result=result_text, error="" if ok else result_text[:500],
                    node_id=node_id,
                ))
                tcp = _read_check_point(s)
                tch = (tcp.get("env_changes") or {}) if isinstance(tcp, dict) else {}
                if tch:
                    tool_env_changes[last_llm_index].append(tch)
                continue

        # The AutoGen exporter drops the task instruction from the recorded
        # window of the first tool-loop span(s); extract it once from the
        # api-prediction span's conversation (same trace, verbatim) and append
        # it to any tool-loop span whose window lacks it.  The collection
        # harness may run several tasks back-to-back in one api-prediction
        # conversation, so the LAST "# Task Instruction" is the task this
        # trace belongs to.  Provenance is recorded in
        # semantic_constraints.instruction_reconstructed.
        outputs_so_far: List[Any] = []
        instruction_text = ""
        if llm_spans:
            cp0 = _read_check_point(llm_spans[0])
            for m in _checkpoint_conversation_messages(cp0):
                c = m.get("content", "")
                if "# Task Instruction" in c:
                    text = c[c.find("# Task Instruction"):]
                    # keep only the task itself: cut the api-prediction suffix
                    # ("List all APIs that may be needed...") that follows the
                    # task in the predictor's message
                    cut = text.find("\n\n\nList all APIs")
                    if cut != -1:
                        text = text[:cut]
                    instruction_text = text.strip()  # last wins

        for j, s in enumerate(llm_spans):
            a = s.get("attributes", {})
            event_type = a.get("event_type") or s.get("event_type") or ""
            raw_output = a.get("gen_ai.output.messages", "")
            parsed, contract_kind = parse_llm_output(raw_output)

            # checkpoint-aware input: the exporter's span-level check_point
            # carries the FULL conversation; the attribute copy is a truncated
            # tail window kept only as a legacy fallback.
            cp = _read_check_point(s)
            cp_conversation = cp.get("conversation")
            has_full_conversation = bool(
                cp_conversation.get("has_full_conversation")
                if isinstance(cp_conversation, dict) else False)

            # keep_raw_role: smolagents' MessageRole.TOOL_CALL/TOOL_RESPONSE must be
            # distinguished by raw role; normalization loses that info (see _splice_smolagents)
            cp_messages = _checkpoint_conversation_messages(cp, keep_raw_role=True)
            openai_reconstructed = False
            if cp_messages:
                input_messages = cp_messages
            elif agent_name == "AppWorld-OpenAI-Agents":
                # The OpenAI SDK agent's collector does not record the conversation (the
                # streaming hook cannot see request messages). Rebuild the transcript from the
                # parsed span outputs + tool execution records (assistant calls + tool results);
                # the real system prompt needs task context and is re-rendered by the runtime
                # with AppWorld during live replay (the transcript is only a local-layer fallback, marked as such).
                input_messages = _openai_transcript(outputs_so_far, attached, j)
                openai_reconstructed = True
            else:
                input_messages = parse_messages(a.get("gen_ai.input.messages", ""))
            last_message = input_messages[-1]["content"] if input_messages else ""

            if j == 0 and contract_kind == "freeform_text":
                # The first span's freeform text = API prediction (tau2 / openai prediction
                # conversations lack the "List all APIs" marker, so the generic rule applies):
                # in the whole collection corpus, span 0 is always api_prediction
                (node_pattern, control_role, phase, subtask, output_tags, verifier,
                 context_sources, after_failure) = (
                    "tool_selection", "global_planner", "initial",
                    "api_prediction", ["freeform_text"], ["none"],
                    ["tool_schema"], False)
            else:
                (node_pattern, control_role, phase, subtask, output_tags, verifier,
                 context_sources, after_failure) = classify(event_type, raw_output, input_messages)

            # prompt reconstruction only for legacy traces whose recorded
            # request is a truncated tail window; full conversations from
            # check_point are used as-is.
            replay_messages = input_messages
            if contract_kind == "tool_call_list" and not has_full_conversation \
                    and not openai_reconstructed:
                replay_messages = _with_reconstructed_tools(input_messages, known_tools)
            # per-agent conversation splicing: empty assistant placeholders -> native tool_calls
            # spliced in, tool messages get tool_call_id/name attached (OpenAI protocol requires pairing)
            replay_messages = _splice_agent_conversation(
                agent_name, replay_messages, cp, outputs_so_far)
            if contract_kind == "tool_call_list":
                outputs_so_far.append(parsed)

            # the recorded window of early tool-loop spans lacks the task
            # instruction (the baseline model demonstrably saw it); append the
            # verbatim instruction from the api-prediction conversation.
            # Note: in the simplified window the task instruction appears as
            # "# Real Task Instruction" in the template-rendered user message; both
            # markers count as "instruction already present", otherwise a duplicate message
            instruction_reconstructed = False
            if (contract_kind == "tool_call_list" and instruction_text and
                    not any(m.get("role") == "user"
                            and ("# Task Instruction" in m.get("content", "")
                                 or "# Real Task Instruction" in m.get("content", "")
                                 # smolagents task messages carry the full instruction themselves
                                 # ("New task:\n<instruction>"), no need to add it
                                 or "New task:" in m.get("content", "")
                                 # the new autogen collector embeds the task at the end of the app
                                 # description message ("\n\n\nTask: ..."), also already present; the
                                 # newline prefix avoids hitting "Subtask:"
                                 or "\nTask:" in m.get("content", "")
                                 or m.get("content", "").lstrip().startswith("Task:"))
                            for m in replay_messages)):
                replay_messages = [dict(m) for m in replay_messages]
                replay_messages.append({"role": "user", "content": instruction_text})
                instruction_reconstructed = True

            # state snapshot: checkpoint-aware.  env_changes is a delta since
            # the previous span checkpoint (empty == state unchanged); the
            # runtime reconstructs by replaying the baseline (span 0) plus the
            # deltas of every span before j.  A check_point object on the span
            # means the exporter promises such a chain.
            env_changes = cp.get("env_changes") or {}
            env_baseline = cp.get("env_changes_baseline") or {}
            cp_hash = str(cp.get("application_state_hash") or "")
            hash_is_placeholder = bool(cp_hash) and cp_hash == EMPTY_SHA256
            # Tools executed before span j = the tools attached to llm 0..j-1. Do not use
            # the full cumulative list: the first walk has already scanned the whole trace,
            # and the full list includes tools after span j (even complete_task); restore
            # replay would push the world ahead to task completion, distorting the replay
            prev_results = [
                {"tool": t.tool_name, "arguments": t.arguments,
                 "ok": t.success, "result": t.result}
                for k in range(j) for t in attached[k]
            ]
            env_snapshot: Dict[str, Any] = {
                "checkpointed": bool(cp),
                "checkpoint_payload_empty": bool(cp) and not (
                    env_changes or env_baseline or tool_env_changes[j]),
                "env_changes": env_changes,
                # Write deltas produced by the tools this span requested (order preserved,
                # same as collection-side accounting; runtime feeds them at the matching hash-chain position and applies them to the world)
                "tool_env_changes": list(tool_env_changes[j]),
                "env_changes_baseline": env_baseline,
                "application_state_hash_is_placeholder": hash_is_placeholder,
                "recorded_tool_results": prev_results,
            }
            agent_state: Dict[str, Any] = {
                "conversation_window": list(input_messages),
                "has_full_conversation": has_full_conversation,
            }

            ref = "env:%d" % j
            artifacts["state_snapshots"][ref] = env_snapshot
            artifacts["state_snapshots"]["agent:%d" % j] = agent_state
            artifacts["messages"]["msg:%d" % j] = replay_messages
            artifacts["system_prompts"]["sys:%d" % j] = next(
                (m["content"] for m in input_messages if m["role"] == "system"), "")
            artifacts["conversation_histories"]["conv:%d" % j] = input_messages
            artifacts["plan_states"]["plan:%d" % j] = {
                "window": not has_full_conversation,
                "full": has_full_conversation,
            }
            artifacts["previous_tool_results"]["prev:%d" % j] = prev_results
            # The tools span j itself called and their execution results (baseline observations), for
            # the local verifier's "environment observation after SLM execution vs baseline observation" comparison
            artifacts["span_tool_results"]["span:%d" % j] = [
                {"tool": t.tool_name, "arguments": t.arguments,
                 "ok": t.success, "result": t.result}
                for t in attached[j]
            ]
            artifacts["tool_lists"]["list:%d" % j] = list(known_tools)
            artifacts["tool_schemas"]["tools:%d" % j] = {
                name: {"type": "object"} for name in known_tools}
            artifacts["response_formats"]["fmt:%d" % j] = {"type": "json_object"}
            # the exporter now records the request's native function-calling
            # tool schemas (gen_ai.request.tools); keep them for faithful
            # continuation (the runtime prefers them over api-doc rebuilds)
            tools_attr = a.get("gen_ai.request.tools", "")
            if isinstance(tools_attr, str) and tools_attr.strip():
                try:
                    artifacts["recorded_tools"]["tools:%d" % j] = json.loads(tools_attr)
                except ValueError:
                    artifacts["recorded_tools"]["tools:%d" % j] = None
            if contract_kind == "tool_call_list":
                artifacts["schemas"]["schema:%d" % j] = TOOL_CALL_SCHEMA
            elif contract_kind == "answer_json":
                artifacts["schemas"]["schema:%d" % j] = ANSWER_SCHEMA

            tool_effects = [APWORLD_TOOL_EFFECTS.get(
                c["name"].replace("__", "."), "external_state_mutation")
                for c in parsed.get("tool_calls", [])] if contract_kind == "tool_call_list" else []
            state_effect = sorted(set(tool_effects)) if tool_effects else ["read_only"]

            descriptor = SpanDescriptor(
                node_pattern=node_pattern,
                control_role=control_role,
                span_subtask_type=subtask,
                mechanism_tags=MechanismTags(
                    interface=["api_tool"],
                    context_source=context_sources,
                    output_contract=output_tags,
                    state_effect=state_effect,
                ),
                verifier_type=verifier,
                graph_context=GraphContext(
                    inside_loop=node_pattern == "tool_argument_generation",
                    control_phase=phase,
                    predecessor_class=_predecessor(j, llm_spans),
                    after_failure=after_failure,
                    after_verification_failure=False,
                    loop_iteration_bucket="1",
                ),
            )

            prompt_tokens = int(a.get("gen_ai.usage.input_tokens", 0) or 0)
            completion_tokens = int(a.get("gen_ai.usage.output_tokens", 0) or 0)
            node_id = "lm:%d" % j
            node_ids.append(node_id)
            spans.append(SpanRecord(
                invocation_index=j,
                native_span_id=s.get("span_id", ""),
                node_id=node_id,
                span_descriptor=descriptor,
                request=LMRequest(
                    system_prompt_ref="sys:%d" % j,
                    messages_ref="msg:%d" % j,
                    previous_tool_results_ref="prev:%d" % j,
                    available_tools_ref="list:%d" % j,
                    tool_schemas_ref="tools:%d" % j,
                    response_format_ref="fmt:%d" % j,
                    temperature=0.0,
                    max_output_tokens=8192,
                ),
                state_snapshot=StateSnapshot(
                    agent_state_ref="agent:%d" % j,
                    environment_snapshot_ref=ref,
                    conversation_history_ref="conv:%d" % j,
                    plan_state_ref="plan:%d" % j,
                    # recorded hash wins; legacy traces fall back to a hash of
                    # the (checkpoint-less) env snapshot itself
                    application_state_hash=cp_hash or state_hash(env_snapshot),
                ),
                output_contract_runtime=OutputContractRuntime(
                    contract_type=contract_kind,
                    schema_ref="schema:%d" % j if contract_kind in ("tool_call_list", "answer_json") else "",
                    allowed_tools=list(known_tools),
                    semantic_constraints={
                        "source_event_type": event_type,
                        "instruction_reconstructed": instruction_reconstructed,
                        "openai_prompt_reconstructed": openai_reconstructed,
                    },
                ),
                verifier_runtime=span_verifier(verifier),
                continuation=Continuation(),  # finalized below
                baseline_response=BaselineResponse(
                    raw=raw_output, parsed=parsed, parse_success=True, parse_error="",
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    latency_ms=0.0, cost=0.0,
                ),
                tool_execution=attached[j][0] if attached[j] else None,
                meta_hint="%s|%s|%d" % (task_id, event_type or "other", j),
            ))

        # continuation points
        for span in spans:
            j = span.invocation_index
            pos = node_ids.index("lm:%d" % j)
            span.continuation = Continuation(
                resume_node_id=node_ids[pos + 1] if pos + 1 < len(node_ids)
                else "final_evaluator",
                later_lm_model=model,
                evaluator_ref="%s#appworld_final_evaluator" % task_id,
                max_remaining_steps=max(1, len(node_ids) - pos) + 10,
            )

        task = self.manifest.get(task_id) if self.manifest else None
        task_condition = TaskCondition(
            task_type=task.task_type if task else "appworld",
            involved_apps=list(task.involved_apps) if task else [],
            cross_app=bool(task.cross_app) if task else True,
            horizon=task.horizon if task else "long",
            task_state_effect=task.task_state_effect if task else "state_mutating",
            branching=task.branching if task else "none",
            evaluator_type=task.evaluator_type if task else "appworld_final_evaluator",
            collateral_risk=task.collateral_risk if task else "unknown",
        )

        return TraceRun(
            schema_version=TRACE_SCHEMA_VERSION,
            run_meta=RunMeta(
                agent_id=agent_name,
                task_id=task_id,
                run_id=data.get("episode_id") or data.get("trace_id", "run1"),
                baseline_model=model,
                created_at="",
                final_evaluator_passed=outcome_ok,
                goal_state_passed=outcome_ok,
                collateral_damage=False,  # unknown at trace level; evaluator decides
                total_cost=0.0,
                total_latency_ms=0.0,
                notes=json.dumps({
                    "trace_id": data.get("trace_id"),
                    "total_steps": data.get("total_steps"),
                    "input_tokens": data.get("input_tokens"),
                    "output_tokens": data.get("output_tokens"),
                    "collateral_damage_recorded": False,
                    "validator_pass": data.get("pass"),
                    "validator_pass_details": data.get("pass_details"),
                    "agent_self_reported_outcome": attrs.get("agent.outcome"),
                }),
            ),
            task_condition=task_condition,
            agent_condition=AgentCondition(
                representative_id=agent_name,
                source_cluster_id=_agent_annotation(agent_name)[0],
                architecture_annotation=_agent_annotation(agent_name)[1],
            ),
            artifacts=artifacts,
            node_ids=node_ids,
            spans=spans,
        )


def span_verifier(verifier: List[str]):
    from .schemas import VerifierRuntime
    return VerifierRuntime(
        verifier_type=verifier,
        verifier_location="pre_commit",
        pre_commit=True,
        rollback_available=False,
    )


def _predecessor(j: int, llm_spans: List[Dict[str, Any]]) -> str:
    if j == 0:
        return ""
    return llm_spans[j - 1].get("event_type", "") or ""
