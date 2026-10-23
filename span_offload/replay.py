"""Single-span counterfactual replay engine (spec 14.4-14.5).

One experiment intervenes on exactly ONE span:

    Prefix -> SLM_s -> Baseline LLM for later spans -> Final Evaluator

so outcome changes are attributable to the current span.  Multi-span joint
dynamic replacement is out of scope for Stage 10 (TraceOffload boundary, 14.9).

All SLM candidates are started from the same span-before state snapshot
(restore() is called fresh per replay) so there is no cross-replay state
pollution (14.3).
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

from .timing_util import monotonic_ms

from .config import Stage10Config
from .llm_client import ChatModelClient, LMResponse
from .runtime import AgentRuntime, ResolvedRequest, RestoredSession, StopReplay
from .schemas import (
    DownstreamOutcome, FallbackOutcome, FinalOutcome, LocalOutcome,
    ReplayRecord, ResourceOutcome, SpanInstance, SpanRecord, TraceRun,
    derive_strict_success, state_hash, to_dict,
)
from .verifiers import VerifierVerdict

FALLBACK_MODES = {"none", "recorded_baseline", "live_baseline"}

#: Engine code version (written into every replay_meta.engine_version; when engine
#: behavior changes, bump it in lockstep; analysis scripts tell data conventions apart by it)
#: .4 = simplified backend downstream usage capture (downstream tokens no longer always 0)
#: 2026-09-13.1 = autogen backend prediction span (tool_selection) continuation fix:
#:    seed from the next span's conversation window (the old logic used the predictor
#:    session -> only text output, no tool calls, inv0 all failed)
#: 2026-09-16.1 = environment restore chain includes tool-execution write deltas (the
#:    exporter attaches them to tool.execution check_point env_changes, adapter attributes
#:    them to the requesting span, restore feeds and applies them at the hash-chain position; fixes 65/65 trace restore hash mismatches)
#: 2026-09-16.2 = C0 frozen walkthrough fidelity: ① replay the pre-span login after restore
#:    to rebuild the in-memory session state (not carried by the delta chain, without it
#:    protected APIs all 401); ② walkthrough replays tools with the recorded execution args
#:    (agent framework fills schema defaults + login token between output and execution; replaying with output args is bound to distort)
#: 2026-09-16.3 = autogen live continuation aligned with the native request: ① when the
#:    seeded session already contains the recorded system, do not stack the rendered system
#:    (double system + duplicate instructions degrades to a plain-text reply in thinking
#:    mode); ② the adapter recognizes the autogen collector's embedded task marker ("Task:"),
#:    no duplicate instruction appended; ③ run_info archives the original message content/reasoning (visible for diagnosis)
#: 2026-09-16.4 = continuation drops the trigger phrase entirely: run(task=None) adds
#:    no message at all, the model's input is verbatim identical to the recorded native
#:    request (the trigger phrase was synthetic input; thinking mode measured empty replies)
#: 2026-09-16.5 = continuation toolset now uses the trace-recorded native schema
#:    (recorded_tools = gen_ai.request.tools): the api_docs-rebuilt schema differs from
#:    the native one, and in thinking mode the model replied empty to the rebuilt schema
#:    (diag evidence: the recorded-schema prompt of 1663 tokens produced tool calls normally)
#: 2026-09-16.6 = outgoing tool schema byte-level alignment: the pydantic rebuild drops
#:    per-property description (4/10 tools differ from the record, prompt 2061 vs 1663,
#:    model still replies empty); BaseTool.schema returns the recorded dict directly
#: 2026-09-16.7 = prediction span (tool_selection) continuation toolset takes the next
#:    span's natively recorded tools (tools:0 does not exist, prediction is a plain-text
#:    request); the previous api_docs-rebuild fallback (7-tool prompt 1366) still replied empty
#: 2026-09-16.8 = outgoing requests strip autogen serialization noise: user messages drop
#:    name="user", tool_choice=None is not sent (consistent with the bare diag request;
#:    the vLLM thinking template is sensitive to the name/tool_choice fields)
#: 2026-09-16.9 = reproduce the login-token injection: the native agent framework
#:    auto-injects access_token between model output and execution (the tool schema has no
#:    such parameter, the model cannot pass it itself); without the harness reproducing it,
#:    protected APIs all 401 and the model re-logs in in a death loop (measured venmo__login
#:    ×16); the bridge keeps a per-app token cache, login success populates the cache, later
#:    calls missing a token get it injected (inject/continuation/login-replay share one path)
#: 2026-09-16.10 = continuation aligned with the collection-side driver config
#:    (trace_collect's run_benchmark_autogen.py is the authority): reflect_on_tool_use=True,
#:    max_tool_iterations=30, max_tokens=4096, bridge world add_login_shortcut=True +
#:    prefetch the main user token (the same mechanism as collection, replacing the login
#:    result cache). temperature stays 0.0 (experiment convention unchanged).
#: 2026-09-16.11 = prediction-span intervention fidelity: the injected API list decides the
#:    downstream toolset via the same mechanism as collection (predicted list filters api_docs +
#:    supervisor admission + access_token kept out of the schema) — under T_span intervention on a
#:    prediction the SLM's new list truly drives the tool surface (before, recorded next-span tools won and the intervention was ignored)
#: 2026-09-16.12 = ① token injection adds the collection-side condition: inject only for
#:    APIs whose schema contains access_token (injecting into login caused Unexpected
#:    parameter errors and random retries); ② max_tokens 4096→8192 (thinking-mode inference
#:    burns tokens; 4096 measured truncation with finish=length and empty content)
#: 2026-09-16.13 = tau2/global continuation conventions aligned: ① thinking mode on by
#:    default for all agents (deleted the off switches for tau2 and simplified-litellm);
#:    ② tau2 gets max_tokens=4096 (same convention as collection); ③ tau2 plain-text turns
#:    end directly, the synthetic trigger phrase removed (adding messages changes the prompt)
#: 2026-09-16.14 = frozen walkthrough strips the tau2 tool wrapper parameter raw (as_tool adds
#:    raw to the signature, the wrapper layer consumes it and never passes it to the API; recorded
#:    execution args contain raw=False, direct API calls rejected with Unexpected parameter 'raw', C0 28/52 mismatch)
#: 2026-09-16.15 = fix tau2 recorded-argument parsing: the collector wraps the real arguments
#:    dict as a repr string under a "raw" key (same name as the wrapper parameter); parse the raw
#:    value into a dict first, then strip the raw key (16.14 stripped the key directly and lost all arguments -> 401)
#: 2026-09-16.16 = tau2 data-convention fixes: ① injected execution traces get the tool
#:    name filled in (previously empty; the I-arm equivalence check saw all-empty executed
#:    names and mismatched); ② attach the llm_utils._TRACE_TRACER shim to capture downstream
#:    usage (previously downstream tokens were always 0)
#: 2026-09-16.17 = value-comparison normalization fixes: ① every string is first tried as
#:    JSON (pretty JSON starts with '{' and contains null, which literal_eval cannot parse);
#:    ② structured parsing uses the raw text instead of normalize_text (normalization changes
#:    case and destroys None/null literals) — tau2 recorded values (pretty JSON) vs replay
#:    values (repr) were previously misjudged unequal, dragging down the I-arm eq
#: 2026-09-16.18 = JWT semantic exemption in value comparison: access_token's exp is decided
#:    by a random expiry offset at login (10-30 minutes, random state not reproducible), so
#:    compare the payload after stripping exp; every other value difference stays hard
#:    (passwords etc. are deterministically generated via random_seed=100, measured reproducible, no exemption)
#: 2026-09-19.19 = per-span runtime capture (purely additive convention, experiment logic
#:    unchanged): replay_meta gains restore_ms/downstream_ms; resource gains
#:    target_generation_ms (= slm_resp.latency_ms persisted); each downstream span gains
#:    elapsed_ms (entries with no call / no timing = 0.0). Old data parses missing keys as 0
#: 2026-09-19.21 = per-span timestamps: every downstream_trace span gains
#:    ts_start_ms/ts_end_ms (freezegun-immune monotonic clock, one timeline per session,
#:    giving inter-span gaps); replay_meta gains ts_round_start_ms (round-level anchor)
#: 2026-09-19.23 = downstream span timing fields match the collection-side trace: every
#:    span gains duration_ms (= elapsed_ms, naming aligned) and gap_from_prev_ms (the gap
#:    since the previous span's end, 0.0 for the first)
#: 2026-09-20.2 = every model call explicitly sets max output tokens = 8192 (reverting
#:    the .31/.20 "no cap / server default" convention): ① slurm config generator and local
#:    config max_output_tokens 4096→8192 (both baseline and slm sides, the downstream
#:    continuation LM follows via bc.max_output_tokens); ② target-span requests hardcoded 4096
#:    all changed to 8192 (locomo/swebench resolve_request and adapters, appworld_agent); ③
#:    former None sites changed to explicit 8192: the appworld adapter LMRequest and the autogen
#:    continuation client max_tokens (16.12 already verified 8192 does not truncate thinking
#:    mode); ④ tau2 llm_args max_tokens
#:    4096→8192
#: 2026-09-20.1 = reasoning persisted for downstream live baseline calls: every
#:    source=="baseline" llm.call span gains output.reasoning (same structure as
#:    slm_output.reasoning / the original trace's output_reasoning, "" when no thinking);
#:    autogen takes CreateResult.thought, tau2 takes AssistantMessage.raw_data, locomo/
#:    simplified take the message's reasoning_content, swebench takes the message dict;
#:    slm spans get no reasoning (small-model output is all in content); live_baseline
#:    fallback gains slm_output.fallback_reasoning. legacy/smolagents/openai_sdk message
#:    structures carry no reasoning, C0 frozen has no live calls —
#:    none of them record it
#: 2026-09-19.33 = large-model thinking becomes a separate output part: llm_client
#:    captures reasoning/reasoning_content into LMResponse.reasoning_content, replay persists
#:    slm_output.reasoning (same structure as the original trace's output_reasoning); not
#:    merged into raw_text, not part of the injected rc chain (reverts .32's think merge)
#: 2026-09-19.32 = large-model thinking merged back into content: when the response carries
#:    reasoning/reasoning_content, splice them back into the body with <think> tags (same
#:    shape as the 9B service that does not strip them; the parsing side strips at the last
#:    </think>, contract parsing unaffected)
#: 2026-09-19.31 = truncation removed everywhere: ① LMRequest.max_output_tokens set to
#:    None + llm_client omits the max_tokens field when max_tokens=None (server default,
#:    same as the collection side which has no such key) — fixes B-arm target-layer
#:    generation hard-truncated at 4096; ② slm_raw_text / narration / downstream span tool
#:    results and prefix messages are no longer truncated (TRACE_RESULT_MAX_LEN /
#:    TRACE_PREFIX_MSG_MAX = None)
#: 2026-09-19.30 = reasoning consumption chain reverted (.22-.29): recorded
#:    output_reasoning parsing, message-level reasoning_content, injected rc chain,
#:    slm_reasoning/injected_reasoning fields all rolled back, all output goes directly
#:    into content; injected rc returns to the placeholder text mode. Timing fields kept
#: 2026-09-19.29 = llm_client latency measurement switched to monotonic_ms (target-layer
#:    generation happens in the freeze leak region after restore, perf_counter always 0 -> target_gen
#:    always 0; after the freeze-immune monotonic clock, target_generation_ms recovers real values)
#: 2026-09-19.28 = llm_client thinking split now keyed on the </think> marker (without a
#:    reasoning parser configured on vLLM the whole generation stays in content although the
#:    model actually emitted <think>reasoning</think>; same convention as adapter
#:    _normalize_llm_output: take the content after the last </think>)
#: 2026-09-19.27 = llm_client text thinking split: when a non-thinking-channel model writes
#:    its thinking into the body ("Thinking Process:" prefix), the prefix segment goes into
#:    reasoning_content and the rest stays in the body (record structure aligned with real
#:    reasoning; not triggered when real reasoning is non-empty)
#: 2026-09-19.26 = llm_client thinking field tolerates the reasoning key name (newer vLLM
#:    Qwen3 returns reasoning instead of reasoning_content; DeepSeek/openai SDK uses
#:    reasoning_content; both are read, and the model_extra fallback is compatible too)
#: 2026-09-19.25 = fix the injected_reasoning read timing: injection actually happens inside
#:    _run_to_node, the read moved to after the continuation (.24 read too early right after
#:    the inject no-op, always empty)
#: 2026-09-19.24 = injected-message rc archived: runtime stashes it to session.injected_reasoning
#:    after assembly, replay persists it to slm_output.injected_reasoning (the rc the injected
#:    message actually used, including the fallback-chain result); the SLM thinking switch is
#:    controlled by the server-side config extra_body
#: 2026-09-19.22 = new-trace reasoning_content consumption: the adapter parses
#:    output_reasoning and attaches it in order to the spliced assistant messages; injected
#:    rc priority = SLM real thinking > recorded thinking > placeholder (old traces fall back
#:    to the placeholder); LMResponse/BaselineResponse gain a reasoning field (purely additive)
#: 2026-09-19.20 = autogen continuation finally aligned with the collection side (checked
#:    against trace_collect's run_benchmark_autogen.py + code/autogen/agent.py): ① max_tool_
#:    iterations 30→50 (= agent_config.max_steps, the old 30 was a leftover from an outdated
#:    collection version, terminating the downstream early); ② model client drops the
#:    max_tokens=8192 hard cap (collection-side model_config has no such key -> None, server
#:    default); ③ client gains timeout=600.0 / max_retries=3 (same as collection _build_client)
ENGINE_VERSION = "2026-09-20.2"


def record_contract_problems(record: "ReplayRecord") -> List[str]:
    """Data-convention self-check (2026-09-20.1/.2): returns the entries of this
    record violating the new conventions; empty list = compliant. Checks:
      - replay_meta.engine_version matches the current engine (old-engine data mixed in is reported);
      - slm_output.reasoning key exists (target-span call convention);
      - source=="baseline" live llm.call spans in downstream_trace must carry output.reasoning
        ("" when no thinking); source=="slm" spans must not carry it (small-model output
        all in content); frozen/C0, repair and other sources are not checked.
    Exemption: legacy/smolagents/openai_sdk backends carry no reasoning; their
    baseline spans skip the key check — only the backend self-reported in
    terminal.backend_info counts, all others strictly checked. Call each record
    after smoke/full runs; any violation should stop and check code sync state."""
    problems: List[str] = []
    rm = record.replay_meta or {}
    if rm.get("engine_version") != ENGINE_VERSION:
        problems.append("engine_version=%r != %r"
                        % (rm.get("engine_version"), ENGINE_VERSION))
    so = record.slm_output or {}
    if "reasoning" not in so:
        problems.append("slm_output missing reasoning key")
    dt = record.downstream_trace or {}
    binfo = " ".join(str(x) for x in (
        (dt.get("terminal") or {}).get("backend_info") or []))
    backend = ""
    _bm = binfo.split("backend=", 1)
    if len(_bm) > 1:
        backend = _bm[1].split()[0]
    skip_baseline_check = backend in ("legacy", "smolagents", "openai")
    for e in dt.get("spans", []):
        if e.get("span_type") != "llm.call":
            continue
        out = e.get("output") or {}
        src = e.get("source")
        if (src == "baseline" and not skip_baseline_check
                and "reasoning" not in out):
            problems.append("step=%s baseline llm.call missing output.reasoning"
                            % e.get("step"))
        elif src == "slm" and "reasoning" in out:
            problems.append("step=%s slm llm.call must not carry reasoning (output should be all in content)"
                            % e.get("step"))
    return problems


class SingleSpanReplayEngine:
    def __init__(self, cfg: Stage10Config, runtime: AgentRuntime,
                 baseline_client: ChatModelClient,
                 slm_clients: Dict[str, ChatModelClient],
                 fallback_mode: Optional[str] = None):
        self.cfg = cfg
        self.runtime = runtime
        self.baseline_client = baseline_client
        self.slm_clients = slm_clients
        self.fallback_mode = fallback_mode or cfg.replay.fallback_mode
        if self.fallback_mode not in FALLBACK_MODES:
            raise ValueError("fallback_mode must be one of %s" % sorted(FALLBACK_MODES))

    # ------------------------------------------------------------------
    def replay(self, span: SpanInstance, trace: TraceRun, model_id: str,
               force_injected: Optional[object] = None,
               arm: str = "T_span") -> ReplayRecord:
        """Single-span counterfactual replay.

        arm (plan §0.4 protocol):
          T_span  target span generated by the SLM, downstream live baseline LLM (main treatment group)
          B       target span live-resampled by the baseline LLM, downstream live (main control)
          I       inject the recorded baseline output, no model call generates the target output (background variance)
          C0      inject the recorded output + inject recorded outputs for all downstream, only re-execute
                  tools (replay fidelity gate, no causal estimation)
        force_injected not None: skip the target span's model call, inject the given output directly
        (identity control round); arm="C0" auto-injects the recorded baseline output when not given.
        """
        if arm not in ("T_span", "B", "I", "C0"):
            raise ValueError("arm must be T_span | B | I | C0, got %r" % arm)
        slm = self.slm_clients.get(model_id)
        if slm is None:
            raise KeyError("no SLM client registered for model %r" % model_id)

        t0 = time.perf_counter()
        # Monotonic-clock timeline anchor (same clock source as the downstream
        # spans' ts_start/ts_end), used to align the span timeline to round level (restore/target/downstream/eval)
        meta_ts_round_start = monotonic_ms()
        record = ReplayRecord(
            span_id=span.span_id, native_span_id=span.native_span_id,
            agent_id=span.agent_id, task_id=span.task_id,
            run_id=span.run_id, invocation_index=span.invocation_index,
            model=model_id,
            span_descriptor=span.span_descriptor,
            task_condition=span.task_condition,
            agent_condition=span.agent_condition,
            stratification_unit=span.stratification_unit,
            unit_key=span.unit_key,
        )
        # baseline accounting of this span (recorded during the native run)
        record.resource.baseline_span_cost = span.baseline_span_cost
        record.resource.baseline_span_latency = span.baseline_span_latency_ms
        record.resource.token_delta = span.baseline_tokens
        meta: Dict = {
            "arm": arm,
            # Engine version marker: verify from the data side which code version
            # ran server/locally, so an old engine cannot silently produce old-convention data
            "engine_version": ENGINE_VERSION,
            "trace_path": span.trace_path,
            "fallback_mode": self.fallback_mode,
            "baseline_tokens": span.baseline_tokens,
            "snapshot_hash_recorded": span.replay_input.state_snapshot.application_state_hash,
            "restore_ok": False,
            "restore_ms": 0.0,
            "downstream_ms": 0.0,
            "ts_round_start_ms": meta_ts_round_start,
            "error": None,
            # "live" = full replay (downstream+final measured);
            # "trace_local" = local layer only, excluded from O_strict profiles
            "outcome_source": getattr(self.runtime, "outcome_source", "live"),
        }
        span_rec = self._find_span_record(trace, span)
        if span_rec is None:
            meta["error"] = "span record not found in trace"
            record.replay_meta = meta
            return record

        try:
            # step 3: identical pre-span state
            t_restore0 = monotonic_ms()
            session = self.runtime.restore(trace, span_rec, self.baseline_client)
            meta["restore_ms"] = monotonic_ms() - t_restore0
            meta["restore_ok"] = True
            meta["snapshot_hash_restored"] = session.snapshot_hash
            # Fill in the downstream trace's meta: offloaded span and model info (upstream info lives in prefix)
            ds_trace = getattr(session, "downstream_trace", None)
            if ds_trace is not None:
                ds_trace["meta"].update({
                    "span_id": span.span_id,
                    "slm_model": model_id,
                    "fallback_mode": self.fallback_mode,
                })
            # State hash gate: raw-to-raw comparison in the same space (restore-chain bare hash
            # vs collection-side recorded hash). session.snapshot_hash carries env_delta:/tool_replay:
            # prefixes (restore-method annotations), not directly comparable to recorded hash (old impl false-reported all)
            raw = getattr(session, "restored_env_hash_raw", "") or ""
            if raw and meta["snapshot_hash_recorded"] \
                    and raw != meta["snapshot_hash_recorded"]:
                meta["error"] = "snapshot hash mismatch: recorded %s vs restored %s" % (
                    meta["snapshot_hash_recorded"], raw)

            # step 4: same prompt / tool schema / output contract -> target model
            req = self.runtime.resolve_request(trace, span_rec, session)
            from .runtime import SpanContext
            req_ctx = SpanContext(
                invocation_index=span_rec.invocation_index,
                node_pattern=span_rec.span_descriptor.node_pattern,
                contract_type=span_rec.output_contract_runtime.contract_type,
                meta_hint=req.meta_hint,
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
            )
            # C0 gate record entry: prompt/schema hash (explicit material for checking against the recorded convention)
            meta["prompt_hash"] = state_hash({
                "messages": req.messages, "tools": req.tools,
                "tool_choice": req.tool_choice})
            # C0: inject the recorded output + freeze the downstream (no model calls at all)
            if arm == "C0":
                if not hasattr(session, "frozen_downstream"):
                    meta["error"] = (meta["error"] or "") + "; c0_unsupported_by_runtime"
                else:
                    session.frozen_downstream = True
            # C0/I target output = recorded baseline output; explicit force_injected wins
            base_parsed = span_rec.baseline_response.parsed
            if force_injected is None and arm in ("I", "C0"):
                force_injected = base_parsed
            if force_injected is not None:
                # Identity control round: no SLM call; build an equivalent response directly from the
                # given output (recorded baseline output); parse/verify/inject/continuation flow is identical
                inj_calls = (force_injected.get("tool_calls", [])
                             if isinstance(force_injected, dict) else [])
                if inj_calls:
                    raw_text = json.dumps(force_injected, ensure_ascii=False)
                elif isinstance(force_injected, dict):
                    # freeform: take the text directly, avoid double json.dumps wrapping
                    # (otherwise the parsed text is '{"text": ...}' and the prediction API
                    # parses empty and falls back to the full toolset, distorting the control)
                    raw_text = str(force_injected.get("text", "") or "")
                else:
                    raw_text = str(force_injected)
                slm_resp = LMResponse(
                    raw_text=raw_text,
                    tool_calls=[{"id": "call_identity_%d" % i,
                                 "name": str(c.get("name", "")),
                                 "arguments": c.get("arguments") or {}}
                                for i, c in enumerate(inj_calls)],
                )
                meta["injection"] = "identity"
            else:
                # B arm: the target span is live-resampled by the baseline LLM (the only
                # difference from T_span is the generating model; prompt/contract/downstream identical)
                target_client = self.baseline_client if arm == "B" else slm
                slm_resp = target_client.complete(
                    messages=req.messages,
                    tools=req.tools,
                    response_format=req.response_format,
                    temperature=req.temperature,
                    max_tokens=req.max_output_tokens,
                    tool_choice=req.tool_choice,
                    ctx=req_ctx,
                )
                meta["injection"] = "baseline" if arm == "B" else "slm"
            if slm_resp.error:
                # the SLM call itself failed: treat like an unparseable output
                # so the configured fallback can rescue the span (14.5)
                record.local_outcome.parse_error = slm_resp.error
                meta["error"] = (meta["error"] or "") + "; slm_call_failed: " + slm_resp.error

            # Target span output vs baseline output comparison material (normalized to {name, arguments})
            record.slm_output["target_arm"] = arm
            record.slm_output["baseline_tool_calls"] = [
                {"name": c.get("name", ""), "arguments": c.get("arguments")}
                for c in base_parsed.get("tool_calls", [])
            ] if isinstance(base_parsed, dict) else []
            record.slm_output["slm_tool_calls"] = [
                {"name": c.get("name", ""), "arguments": c.get("arguments")}
                for c in slm_resp.tool_calls
            ]
            record.slm_output["slm_raw_text"] = slm_resp.raw_text
            # Large-model thinking: persisted as a separate part of the output (same
            # structure as the original trace's output_reasoning); not part of the injected rc chain
            record.slm_output["reasoning"] = slm_resp.reasoning_content
            record.slm_output["slm_error"] = slm_resp.error
            # Hand the SLM's raw text to the runtime: the injected message's reasoning_content
            # is filled from it (DeepSeek thinking mode requires the assistant message right
            # after tool results to carry a non-empty reasoning_content)
            session.slm_raw_text = slm_resp.raw_text or ""

            # step 5: original parser + local verifier
            parsed = self._step_parse_and_verify(record, session, span_rec, slm_resp)
            chosen: Optional[object] = parsed if record.local_outcome.local_output_valid else None

            # Local equivalence verifier (modeled on the AppWorld verifier): static comparison
            # of SLM output vs baseline output (tool names/arguments/answer/text); the tool
            # execution result (environment observation) comparison is added after injected execution
            from .local_equivalence import verify_local_equivalence
            slm_parsed_for_eq = (parsed if isinstance(parsed, dict)
                                 else {"text": slm_resp.raw_text or ""})
            record.local_outcome.equivalence = verify_local_equivalence(
                span_rec.output_contract_runtime.contract_type,
                slm_parsed_for_eq, span_rec.baseline_response.parsed)

            # fallback (14.5) when the SLM output is not allowed to commit
            if chosen is None and self.fallback_mode != "none":
                chosen, fb_cost = self._step_fallback(record, session, trace, span_rec, req, slm_resp)
                record.resource.fallback_cost = fb_cost

            if chosen is None:
                # no usable output: signal failure into the workflow so the
                # agent's own failure/recovery path takes over (honest strict
                # failure accounting)
                record.slm_output["injected_tool_calls"] = []
                self.runtime.inject(session, span_rec, {"__span_failure__": record.local_outcome.parse_error or "slm_output_rejected"})
            else:
                record.slm_output["injected_tool_calls"] = [
                    {"name": c.get("name", ""), "arguments": c.get("arguments")}
                    for c in chosen.get("tool_calls", [])
                ] if isinstance(chosen, dict) else []
                ok = self.runtime.inject(session, span_rec, chosen)
                if not ok:
                    meta["error"] = (meta["error"] or "") + "; inject_failed"
                    record.replay_meta = meta
                    record.replay_meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
                    record.strict_success = False
                    record.downstream_trace = getattr(session, "downstream_trace", None) or {}
                    return record

            # steps 7-8: later LM calls on baseline LLM, continue to final evaluator
            max_steps = span_rec.continuation.max_remaining_steps + self.cfg.replay.max_remaining_steps_margin
            t_ds0 = monotonic_ms()
            downstream, downstream_usage = self._run_to_node(record, session, span_rec, max_steps)
            final, final_usage = self._run_to_completion(record, session, span_rec, max_steps)
            meta["downstream_ms"] = monotonic_ms() - t_ds0

            # Tool execution result (environment observation) comparison: the injected round's real
            # execution results vs the recorded baseline tool execution results for this span (C0's
            # frozen path already compares per span, so no duplicate; the source tags differ too)
            if arm != "C0" \
                    and span_rec.output_contract_runtime.contract_type == "tool_call_list":
                from .local_equivalence import compare_execution_results
                slm_exec = [e for e in (getattr(session, "downstream_trace", None) or {}).get("spans", [])
                            if e.get("span_type") == "tool.execution"
                            and e.get("step") == 0 and e.get("source") == "slm"]
                base_exec = (trace.artifacts.get("span_tool_results") or {}).get(
                    "span:%d" % span.invocation_index, [])
                exec_out = compare_execution_results(slm_exec, base_exec)
                eq = record.local_outcome.equivalence
                eq.setdefault("requirements", [])
                eq["requirements"].extend(exec_out["requirements"])
                eq["passed"] = bool(eq.get("passed", True)) and bool(exec_out["passed"])

            record.downstream_outcome = downstream
            record.final_outcome = final
            record.strict_success = derive_strict_success(final)
            record.resource.total_counterfactual_latency = (time.perf_counter() - t0) * 1000.0
            # C0: persist the per-span frozen execution comparison (tool execution results vs recorded observations)
            if getattr(session, "c0_fidelity", None) is not None:
                meta["c0_fidelity"] = list(session.c0_fidelity)
            self._finalize_resources(record, slm, slm_resp,
                                     downstream_usage + final_usage, arm=arm)
            # Persist the structured full trace of injection + continuation
            record.downstream_trace = getattr(session, "downstream_trace", None) or {}
        except StopReplay as stop:
            record.final_outcome.budget_exhausted = stop.flag == "budget_exhausted"
            record.final_outcome.timeout = stop.flag == "timeout"
            meta["error"] = (meta["error"] or "") + "; stopped: " + stop.flag
        except Exception as exc:  # never let one span kill the batch
            record.downstream_outcome.unhandled_exception = True
            meta["error"] = (meta["error"] or "") + "; exception: %s: %s" % (type(exc).__name__, exc)
            record.final_outcome.evaluator_details.append("replay_exception: %s" % exc)

        # End of each replay round: deterministically close the AppWorld bridge (the world is an in-memory
        # DB, unclosed it relies on GC for delayed release) and trigger one GC -- long batch jobs (200+
        # rounds) keep accumulating memory and get killed by cgroup/OOM. mock/trace-local sessions have no bridge, skip safely.
        try:
            sess = locals().get("session")
            bridge = getattr(sess, "bridge", None)
            if bridge is not None:
                bridge.__exit__(None, None, None)
        except Exception:
            pass
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

        record.replay_meta = meta
        record.replay_meta["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
        record.strict_success = derive_strict_success(record.final_outcome)
        return record

    # ------------------------------------------------------------------
    def _find_span_record(self, trace: TraceRun, span: SpanInstance) -> Optional[SpanRecord]:
        for rec in trace.spans:
            if rec.invocation_index == span.invocation_index:
                return rec
        return None

    def _step_parse_and_verify(self, record: ReplayRecord, session: RestoredSession,
                               span_rec: SpanRecord, slm_resp: LMResponse) -> Optional[object]:
        """Original parser + schema + local verifiers on the SLM output.

        Native tool_calls (when the request carried the tools protocol) are
        the protocol-level answer and skip the text parser; text output falls
        back to the original parser."""
        lo = record.local_outcome
        if getattr(slm_resp, "tool_calls", None):
            parsed = {"tool_calls": [{
                "name": c["name"], "arguments": c["arguments"],
            } for c in slm_resp.tool_calls]}
            lo.parse_success = True
            lo.schema_valid = True
        else:
            parse = self.runtime.parse(span_rec, slm_resp.raw_text)
            lo.parse_success = parse.ok
            parsed = parse.parsed
            if not parse.ok:
                lo.parse_error = parse.error
                return None

        lo.schema_valid = self.runtime.schema_check(session, span_rec, parsed)
        if not lo.schema_valid:
            return parsed

        verdict = self.runtime.verify_local(session, span_rec, parsed)
        record.fallback.verifier_triggered = verdict.triggered
        record.fallback.verifier_detected_failure = verdict.detected_failure
        lo.local_verifier_passed = verdict.passed
        lo.verifier_errors = list(verdict.errors)
        lo.local_output_valid = verdict.passed
        return parsed

    def _step_fallback(self, record: ReplayRecord, session: RestoredSession,
                       trace: TraceRun, span_rec: SpanRecord,
                       req: ResolvedRequest, slm_resp: LMResponse) -> Tuple[Optional[object], float]:
        """Fallback when the SLM output cannot commit (14.5 fallback block)."""
        fb = record.fallback
        fb.fallback_triggered = True
        mode = self.fallback_mode
        fb.fallback_model = mode

        if mode == "recorded_baseline":
            # baseline's own parsed output is a known-good rollback target
            base = span_rec.baseline_response.parsed
            if base is not None:
                fb.rollback_succeeded = True  # nothing committed pre-injection
                fb.fallback_recovered = True
                return base, 0.0
            return None, 0.0

        if mode == "live_baseline":
            from .runtime import SpanContext
            req_ctx = SpanContext(
                invocation_index=span_rec.invocation_index,
                node_pattern=span_rec.span_descriptor.node_pattern,
                contract_type=span_rec.output_contract_runtime.contract_type,
                meta_hint=req.meta_hint,
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
            )
            resp = self.baseline_client.complete(
                messages=req.messages, tools=req.tools,
                response_format=req.response_format,
                temperature=req.temperature, max_tokens=req.max_output_tokens,
                ctx=req_ctx,
            )
            cost = self.baseline_client.cost_for(resp.prompt_tokens, resp.completion_tokens)
            if resp.error:
                fb.fallback_recovered = False
                return None, cost
            parse = self.runtime.parse(span_rec, resp.raw_text)
            if parse.ok:
                fb.rollback_succeeded = True
                fb.fallback_recovered = True
                # The fallback is a live large-model call; persist its reasoning with the
                # same structure (same convention as slm_output.reasoning)
                record.slm_output["fallback_reasoning"] = resp.reasoning_content
                return parse.parsed, cost
            fb.fallback_recovered = False
            return None, cost

        return None, 0.0

    def _run_to_node(self, record: ReplayRecord, session: RestoredSession,
                     span_rec: SpanRecord, max_steps: int) -> Tuple[DownstreamOutcome, List[LMResponse]]:
        outcome, usage = self.runtime.run_to_node(
            session, span_rec, self.baseline_client, max_steps,
            self.cfg.replay.timeout_s, self.cfg.replay.budget_usd,
        )
        return outcome, usage

    def _run_to_completion(self, record: ReplayRecord, session: RestoredSession,
                           span_rec: SpanRecord, max_steps: int) -> Tuple[FinalOutcome, List[LMResponse]]:
        outcome, usage = self.runtime.run_to_completion(
            session, span_rec, self.baseline_client, max_steps,
            self.cfg.replay.timeout_s, self.cfg.replay.budget_usd,
        )
        return outcome, usage

    def _finalize_resources(self, record: ReplayRecord, slm: ChatModelClient,
                            slm_resp: LMResponse, later_usage: List[LMResponse],
                            arm: str = "T_span") -> None:
        res = record.resource
        # Unified cost convention (plan §9.3): target and downstream itemized tokens + context load
        res.target_prompt_tokens = int(slm_resp.prompt_tokens or 0)
        res.target_completion_tokens = int(slm_resp.completion_tokens or 0)
        # Target-layer generation time: slm_resp.latency_ms (single successful attempt
        # convention, already filled by llm_client); identity/C0 constructed resp is 0.0 (no generation)
        res.target_generation_ms = float(slm_resp.latency_ms or 0.0)
        d_prompt = d_comp = 0
        for u in later_usage:
            d_prompt += int(u.prompt_tokens or 0)
            d_comp += int(u.completion_tokens or 0)
        res.downstream_prompt_tokens = d_prompt
        res.downstream_completion_tokens = d_comp
        res.context_load = res.target_prompt_tokens + d_prompt
        if arm == "B":
            # B arm: target generation cost = baseline LLM live resampling (the NetSaving
            # baseline term; analysis scripts compute the per-block B vs T_span difference)
            res.baseline_target_cost = self.baseline_client.cost_for(
                res.target_prompt_tokens, res.target_completion_tokens)
            res.verifier_cost = 0.0
            res.fallback_cost = 0.0
            downstream_cost = 0.0
            for u in later_usage:
                downstream_cost += self.baseline_client.cost_for(
                    u.prompt_tokens, u.completion_tokens)
            res.total_counterfactual_cost = res.baseline_target_cost + downstream_cost
            res.cost_saving = 0.0  # B is the baseline itself, no saving computed
            res.latency_saving = 0.0
            baseline_tokens = int(record.replay_meta.get("baseline_tokens", 0) or 0)
            res.token_delta = baseline_tokens - res.target_prompt_tokens - res.target_completion_tokens
            return
        res.slm_cost = slm.cost_for(slm_resp.prompt_tokens, slm_resp.completion_tokens)
        res.verifier_cost = 0.0  # local deterministic verifiers only (documented)
        downstream_cost = 0.0
        for u in later_usage:
            downstream_cost += self.baseline_client.cost_for(u.prompt_tokens, u.completion_tokens)
        res.total_counterfactual_cost = res.slm_cost + res.verifier_cost + res.fallback_cost + downstream_cost
        res.cost_saving = res.baseline_span_cost - (res.slm_cost + res.verifier_cost + res.fallback_cost)
        res.latency_saving = res.baseline_span_latency - slm_resp.latency_ms
        baseline_tokens = int(record.replay_meta.get("baseline_tokens", 0) or 0)
        res.token_delta = baseline_tokens - slm_resp.total_tokens


def replay_batch(engine: SingleSpanReplayEngine, spans: List[SpanInstance],
                 traces: Dict[str, TraceRun], model_ids: List[str],
                 limit: Optional[int] = None, progress=None) -> List[ReplayRecord]:
    """Replay every span in the (already-filtered) list x every candidate SLM
    model.  Callers decide the gate: full-live mode passes only spec-eligible
    spans (14.1); trace-local mode passes spans with a recorded request."""
    records: List[ReplayRecord] = []
    todo = list(spans)
    if limit is not None:
        todo = todo[:limit]
    for i, span in enumerate(todo):
        trace = traces.get(span.trace_path)
        if trace is None:
            continue
        for model_id in model_ids:
            record = engine.replay(span, trace, model_id)
            records.append(record)
            if progress:
                progress(i + 1, len(todo), span.span_id, model_id)
    return records


def save_replays(records: List[ReplayRecord], path: str) -> str:
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(to_dict(rec), ensure_ascii=False) + "\n")
    return path


def load_replays(path: str) -> List[dict]:
    import json
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
