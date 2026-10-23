#!/usr/bin/env python
"""Single-span live counterfactual replay batch runner (real AppWorld env + real agent continuation).

Usage (from the project root, after the README "environment setup"):

    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
        .venv_appworld/Scripts/python.exe scripts/runners/run_live_replay.py \
        --traces check_point_traces/appworld_legacy_batch1 \
        --trace 0d8a4ee_1 --fallback none --out data/appworld/replays_legacy.jsonl

Arguments:
    --traces       trace directory. All six agent batches work directly:
                   appworld_simplified_batch1 / appworld_autogen_batch1 /
                   appworld_legacy_batch1 / appworld_openai_agents_batch1 /
                   appworld_smolagents_batch1 / appworld_tau2_batch1
    --trace        (repeatable) only replay traces whose filename contains the substring; default = all
    --manifest     task manifest (default data/appworld/task_manifest.json)
    --spans        existing spans.jsonl (default re-extract next to --out)
    --fallback     none | recorded_baseline | live_baseline
                   none               = SLM offloads by itself, failures recorded as strict=0
                   recorded_baseline  = fallback injects the baseline recorded output (identity
                                        ceiling / bridge fidelity upper bound)
                   live_baseline      = fall back to calling the baseline LLM live
    --model        SLM model id (default the first candidate in config.json)
    --limit        max number of spans to replay
    --out          output replays.jsonl path
    --backend      force the continuation backend (generic | simplified | autogen | legacy |
                   smolagents | tau2 | openai); default picks by the trace's
                   agent identity (same agent source as the trace producer),
                   honestly falling back to the generic loop when framework deps are missing

The continuation backend is auto-resolved from the agent identity the
trace reports (AppWorldAgentRuntime._backend_for); session splicing,
protocol flags (legacy/smolagents tool_choice required, smolagents text
protocol, OpenAI-SDK session rebuild) are handled inside span_offload; harness is generic across all six agents.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402
from span_offload.config import load_config  # noqa: E402
from span_offload.harness.appworld_runtime import AppWorldAgentRuntime  # noqa: E402
from span_offload.llm_client import ChatModelClient  # noqa: E402
from span_offload.replay import SingleSpanReplayEngine  # noqa: E402
from span_offload.schemas import SpanInstance, from_dict, to_dict  # noqa: E402
from span_offload.span_extraction import load_spans  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402


def make_client(section):
    c = section
    return ChatModelClient(
        base_url=c.base_url, api_key=c.resolved_api_key,
        model=c.candidate_models[0], temperature=c.temperature,
        max_output_tokens=c.max_output_tokens,
        price_per_1k_input=c.price_per_1k_input,
        price_per_1k_output=c.price_per_1k_output,
        timeout_s=c.timeout_s, max_retries=c.max_retries,
        extra_body=c.extra_body,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="check_point_traces/appworld_simplified_batch1")
    ap.add_argument("--trace", action="append", default=[],
                    help="trace filename substring (repeatable); default = all")
    ap.add_argument("--manifest", default="data/appworld/task_manifest.json")
    ap.add_argument("--spans", default="", help="existing spans.jsonl; default re-extract automatically")
    ap.add_argument("--fallback", default="none",
                    choices=["none", "recorded_baseline", "live_baseline"])
    ap.add_argument("--model", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/appworld/replays_live.jsonl")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--backend", default="",
                    choices=["", "generic", "simplified", "autogen", "legacy",
                             "smolagents", "tau2", "openai"],
                    help="force the continuation backend; default auto-select by trace agent identity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    model_id = args.model or slm.name
    if args.backend:
        os.environ["SPAN_OFFLOAD_CONT_BACKEND"] = args.backend
        from span_offload.harness.appworld_runtime import AppWorldAgentRuntime
        AppWorldAgentRuntime.continuation_backend = args.backend

    spans_path = args.spans
    if not spans_path:
        spans_path = os.path.join(os.path.dirname(args.out) or ".", "spans.jsonl")
        import subprocess
        cmd = [sys.executable, "-m", "span_offload.cli", "extract",
               "--traces", args.traces, "--manifest", args.manifest,
               "--out", spans_path, "--trace-format", "appworld"]
        print("extracting: %s" % " ".join(cmd))
        subprocess.run(cmd, check=True)

    spans_all = load_spans(spans_path)
    todo = [from_dict(SpanInstance, s) for s in spans_all
            if s.get("eligible")]
    if args.trace:
        todo = [s for s in todo
                if any(t in s.trace_path for t in args.trace)]
    if args.limit:
        todo = todo[:args.limit]
    print("eligible spans to replay: %d (fallback=%s, SLM=%s)" % (
        len(todo), args.fallback, model_id))

    manifest = load_manifest(args.manifest)
    adapter = AppWorldTraceAdapter(manifest)
    traces = {}
    for s in todo:
        if s.trace_path not in traces:
            traces[s.trace_path] = adapter.load(s.trace_path)

    runtime = AppWorldAgentRuntime()
    engine = SingleSpanReplayEngine(
        cfg, runtime, baseline_client=baseline,
        slm_clients={model_id: slm}, fallback_mode=args.fallback)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    n_strict = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, span in enumerate(todo):
            record = engine.replay(span, traces[span.trace_path], model_id)
            f.write(json.dumps(to_dict(record), ensure_ascii=False) + "\n")
            f.flush()
            n_strict += int(record.strict_success)
            print("[%d/%d] span %d [%s]: local(p=%s) fb=%s | final(task=%s) | "
                  "strict=%s | %.0fs" % (
                      i + 1, len(todo), span.invocation_index,
                      span.span_descriptor.node_pattern,
                      record.local_outcome.parse_success,
                      record.fallback.fallback_triggered,
                      record.final_outcome.final_task_success,
                      record.strict_success, time.time() - t0), flush=True)
    print("done: strict success %d/%d -> %s" % (n_strict, len(todo), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
