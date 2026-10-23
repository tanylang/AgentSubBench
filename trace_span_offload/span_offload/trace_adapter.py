"""Native-trace ingestion contract for real AppWorld data.

Stage 10 consumes native traces as INPUT.  The harness generates its own
traces (canonical schema `span_offload-native-trace/v1`, see data_demo/traces/)
only for smoke testing.  A real pilot plugs in the successful baseline runs
produced by the frozen representative agent panel (s = (H, t, r, j), 14.1).

Two integration paths:

  A) CONVERT to the canonical format (recommended when you control the trace
     writer): produce JSON files matching `TraceRun` (trace_io.save_trace
     serializes it; validate the result with `validate_canonical_trace`).

  B) ADAPT at load time: keep your native format and subclass TraceAdapter;
     the adapter maps each native file to a `TraceRun`.  Wire the adapter into
     the CLI / pipeline by replacing `iter_traces` / `load_trace` calls.

What a trace MUST contain (this is exactly what extraction needs, spec
14.1 / 14.3):

  run_meta         agent_id, task_id, run_id, baseline_model,
                   final_evaluator_passed, goal_state_passed,
                   collateral_damage (eligibility condition 5, 14.1)
  task_condition   frozen-manifest task condition (14.2.3) -- read from the
                   TaskManifest, NOT from the trace itself if possible
  agent_condition  representative_id, source_cluster_id, architecture_annotation
  artifacts        shared bulky payloads keyed by ref:
                     state_snapshots:   {"agent:<i>": agent state (serializable),
                                         "env:<i>": environment snapshot}
                     messages:          {"msg:<i>": the exact LM request messages}
                     system_prompts / tool_schemas / tool_lists /
                     previous_tool_results / conversation_histories /
                     plan_states / schemas / response_formats
  node_ids         ordered node sequence of the run (incl. tool-exec nodes)
  spans            one SpanRecord per LM invocation j with:
                     request (refs + temperature + max_output_tokens)
                     state_snapshot (refs + application_state_hash)
                     span_descriptor / output_contract_runtime /
                     verifier_runtime
                     baseline_response (raw, parsed, parse_success, usage,
                                        latency, cost)
                     continuation (resume_node_id, later_lm_model,
                                   evaluator_ref, max_remaining_steps)

The exact field-by-field shape is defined in span_offload/schemas.py
(TraceRun / SpanRecord / ReplayInput dataclasses) and instantiated in
span_offload/harness/mock_runtime.py:TraceRecorder -- use that as the reference
producer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .schemas import TraceRun
from .trace_io import load_trace


class TraceAdapter(ABC):
    """Maps one native trace file to the canonical TraceRun."""

    name = "abstract"

    @abstractmethod
    def load(self, path: str) -> TraceRun:
        """Parse `path` into a TraceRun.  Raise ValueError on unreadable or
        incomplete files so the pipeline can skip/report them."""


class CanonicalTraceAdapter(TraceAdapter):
    """Default adapter: traces already in the canonical JSON format."""

    name = "canonical"

    def load(self, path: str) -> TraceRun:
        return load_trace(path)


def validate_canonical_trace(trace: TraceRun, verbose: bool = True) -> List[str]:
    """Self-check for trace converters: reports everything extraction needs.

    Returns a list of problems; empty list == trace is extraction-ready.
    """
    problems: List[str] = []

    def add(msg: str) -> None:
        problems.append(msg)
        if verbose:
            print("  [trace-check] " + msg)

    meta = trace.run_meta
    for field in ("agent_id", "task_id", "run_id", "baseline_model"):
        if not getattr(meta, field, ""):
            add("run_meta.%s missing" % field)

    artifacts = trace.artifacts or {}
    states = artifacts.get("state_snapshots") or {}
    messages = artifacts.get("messages") or {}

    if not trace.spans:
        add("no spans")
    for span in trace.spans:
        i = span.invocation_index
        # state recoverable (14.1 condition 1)
        if span.state_snapshot.agent_state_ref not in states:
            add("span %d: agent_state_ref %r not in state_snapshots"
                % (i, span.state_snapshot.agent_state_ref))
        if span.state_snapshot.environment_snapshot_ref not in states:
            add("span %d: environment_snapshot_ref %r not in state_snapshots"
                % (i, span.state_snapshot.environment_snapshot_ref))
        if not span.state_snapshot.application_state_hash:
            add("span %d: application_state_hash missing" % i)
        # request complete (14.1 condition 2)
        if span.request.messages_ref not in messages:
            add("span %d: messages_ref %r not in messages"
                % (i, span.request.messages_ref))
        if span.request.tool_schemas_ref and \
                span.request.tool_schemas_ref not in artifacts.get("tool_schemas", {}):
            add("span %d: tool_schemas_ref %r unresolved"
                % (i, span.request.tool_schemas_ref))
        # baseline parseable (14.1 condition 4)
        if not span.baseline_response.parse_success:
            add("span %d: baseline_response not parseable" % i)
        if span.baseline_response.parsed is None:
            add("span %d: baseline_response.parsed missing (not resumable)" % i)
        # source-node mapping (14.1 condition 6)
        if span.node_id not in (trace.node_ids or []):
            add("span %d: node_id %r not in trace.node_ids" % (i, span.node_id))
        # continuation (14.1 condition 3)
        cont = span.continuation
        if cont.resume_node_id not in (trace.node_ids or []) and \
                cont.resume_node_id != "final_evaluator":
            add("span %d: continuation.resume_node_id %r not locatable"
                % (i, cont.resume_node_id))
        if not cont.evaluator_ref:
            add("span %d: continuation.evaluator_ref missing" % i)
    return problems
