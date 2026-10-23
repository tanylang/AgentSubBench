"""Stage 10 command-line interface.

    python -m span_offload.cli demo     --config config.json [--offline] [--out data]
                                   # full harness pipeline (14.1-14.8)
    python -m span_offload.cli extract  --traces data/traces --manifest tasks.json
                                   --out data/spans.jsonl
    python -m span_offload.cli replay   --spans data/spans.jsonl --config config.json
                                   --out data/replays.jsonl [--models a,b] [--limit N]
    python -m span_offload.cli analyze  --replays data/replays.jsonl
                                   --out data/report [--min-samples 10]
    python -m span_offload.cli validate --config config.json --out data/report
                                   # multi-span allowlist sanity check (14.8)

For a real AppWorld pilot the usual flow is extract -> replay -> analyze
(+ validate on held-out tasks); `demo` additionally generates the native
traces with the harness agents and a scripted/mock offline mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .config import load_config
from .demo import (
    HARNESS_MANIFEST, collect_spans, generate_traces, make_clients,
    run_analysis, run_pipeline,
)
from .replay import SingleSpanReplayEngine, replay_batch, save_replays, load_replays
from .report import (
    agent_section, completeness_section, delta_task_section, profiles_section,
    save_json, task_conditioned_section, validation_section, write_report,
)
from .schemas import TaskManifest, from_dict, SpanInstance
from .span_extraction import census, save_spans, load_spans
from .trace_io import load_manifest, load_trace


def _as_instances(span_dicts: List[dict]) -> List[SpanInstance]:
    return [from_dict(SpanInstance, d) for d in span_dicts]


def _make_adapter(trace_format: str, manifest: TaskManifest):
    if trace_format == "appworld":
        from .appworld_adapter import AppWorldTraceAdapter
        return AppWorldTraceAdapter(manifest)
    if trace_format == "swebench":
        from .swebench_adapter import SwebenchTraceAdapter
        return SwebenchTraceAdapter(manifest)
    if trace_format == "locomo":
        from .locomo_adapter import LocomoTraceAdapter
        return LocomoTraceAdapter(manifest)
    if trace_format == "langgraph":
        from .langgraph_adapter import LangGraphTraceAdapter
        return LangGraphTraceAdapter(manifest)
    from .trace_adapter import CanonicalTraceAdapter
    return CanonicalTraceAdapter()


def _load_trace_any(path: str, adapter) -> Any:
    return adapter.load(path)


def cmd_demo(args) -> int:
    cfg = load_config(args.config)
    if args.offline:
        cfg.offline = True
    path = run_pipeline(cfg, out_dir=args.out, task_ids=args.tasks, replay_limit=args.limit)
    print("report written to %s" % path)
    return 0


def cmd_extract(args) -> int:
    manifest = load_manifest(args.manifest)
    adapter = _make_adapter(args.trace_format, manifest)
    spans = collect_spans(args.traces, manifest, args.out, adapter=adapter,
                          trace_filter=getattr(args, "trace_filter", "") or "")
    cens = census(_as_instances(spans))
    print(json.dumps(cens, ensure_ascii=False, indent=1))
    print("spans written to %s" % args.out)
    return 0


def cmd_replay(args) -> int:
    cfg = load_config(args.config)
    if args.offline:
        cfg.offline = True
    manifest = load_manifest(cfg.paths.manifest_path or HARNESS_MANIFEST)
    baseline_client, slm_clients = make_clients(cfg, cfg.offline)
    models = args.models.split(",") if args.models else sorted(slm_clients.keys())
    spans = _as_instances(load_spans(args.spans))
    adapter = _make_adapter(args.trace_format, manifest)

    if args.mode == "trace-local":
        from .trace_local_runtime import TraceLocalRuntime, RecordedEchoSLM, build_echo_outputs
        runtime = TraceLocalRuntime()
        # relaxed gate: local layer needs the recorded request + baseline success
        todo = [s for s in spans if not s.baseline_failed
                and bool(s.replay_input.lm_request.messages_ref)]
        traces = {}
        for s in todo:
            if s.trace_path not in traces:
                try:
                    traces[s.trace_path] = _load_trace_any(s.trace_path, adapter)
                except (ValueError, KeyError) as exc:
                    print("  skip trace %s: %s" % (s.trace_path, exc))
        if cfg.offline:
            # identity SLM (echoes the recorded baseline output): smoke-tests
            # the local measurement machinery before a real SLM is pointed at it
            outputs = build_echo_outputs(list(traces.values()))
            slm_clients = {m: RecordedEchoSLM(outputs, model=m) for m in models}
        print("trace-local mode: %d spans with recorded requests (%d eligible for "
              "full live replay)" % (len(todo),
                                     sum(1 for s in spans if s.eligible)))
    else:
        from .harness.mock_runtime import MockAppWorldRuntime
        from .harness.sample_agent import AGENTS
        runtime = MockAppWorldRuntime(AGENTS, manifest)
        todo = [s for s in spans if s.eligible]
        traces = {}
        for s in todo:
            if s.trace_path not in traces:
                try:
                    traces[s.trace_path] = _load_trace_any(s.trace_path, adapter)
                except (ValueError, KeyError) as exc:
                    print("  skip trace %s: %s" % (s.trace_path, exc))

    engine = SingleSpanReplayEngine(cfg, runtime, baseline_client, slm_clients)
    records = replay_batch(engine, todo, traces, models, limit=args.limit)
    save_replays(records, args.out)
    print("replay records: %d -> %s" % (len(records), args.out))
    return 0


def cmd_analyze(args) -> int:
    replays = load_replays(args.replays)
    manifest = load_manifest(args.manifest) if args.manifest else None

    if args.source == "trace_local":
        from . import analysis as an
        from .report import local_profiles_section
        models = sorted({r.get("model", "?") for r in replays})
        local = {m: an.local_profile_m(replays, m, args.min_samples) for m in models}
        save_json(os.path.join(args.out, "local_profiles.json"), local)
        path = write_report(
            os.path.join(args.out, "analysis_local.md"),
            {"Local-layer profiles (trace_local, NOT strict success)":
             local_profiles_section(local)},
            title="Stage 10 local-layer analysis")
        print("local-layer analysis report written to %s" % path)
        return 0

    payload = run_analysis(replays, manifest, args.min_samples, args.out)
    # write the markdown sections
    sections = {
        "Offloadability profiles (14.6)": profiles_section(payload["profiles"]),
        "Task-conditioned analysis (14.7.1)":
            task_conditioned_section(payload["task_tables"])
            + "\n\n" + delta_task_section(payload["delta_task"]),
        "Agent & architecture-conditioned (14.7.2-14.7.4)":
            agent_section(payload["within_agent"], payload["matched"],
                          payload["architecture"], payload["joint"]),
        "Descriptor completeness (14.7.5)": completeness_section(payload["completeness"]),
    }
    path = write_report(os.path.join(args.out, "analysis.md"), sections,
                        title="Stage 10 analysis")
    print("analysis report written to %s" % path)
    return 0


def cmd_validate_appworld(args) -> int:
    """Multi-span allowlist sanity check on REAL AppWorld tasks (14.8).

    Requires the appworld package fully installed (apps bundle unpacked) and
    task data downloaded.  Tool calls execute in the real AppWorld
    environment; model calls use the user-configured clients (config.json)."""
    cfg = load_config(args.config)
    if args.offline:
        cfg.offline = True
    from .demo import make_clients
    baseline_client, slm_clients = make_clients(cfg, cfg.offline)
    models = sorted(slm_clients.keys())
    slm_client = slm_clients[models[0]]
    task_ids = args.tasks or []

    allowlist = None
    if args.replays:
        from .allowlist import build_allowlist
        replays = load_replays(args.replays)
        allowlist = build_allowlist(replays, models[0], cfg.replay.min_cell_samples,
                                    cfg.allowlist.lcb_threshold, split="pilot")
    from .harness.appworld_agent import run_appworld_validation
    validation = run_appworld_validation(
        task_ids, baseline_client, slm_client, allowlist,
        on_local_failure=args.on_local_failure,
        progress=lambda t, c: print("  validation run %s / %s" % (t, c)))
    os.makedirs(args.out, exist_ok=True)
    save_json(os.path.join(args.out, "validation_appworld.json"), validation)
    from .report import validation_section, write_report
    write_report(os.path.join(args.out, "validation_appworld.md"),
                 {"Allowlist validation on real AppWorld (14.8)":
                  validation_section(validation, {})},  # no oracle bound here
                 title="Stage 10 AppWorld validation")
    print("validation report written to %s" %
          os.path.join(args.out, "validation_appworld.md"))
    return 0


def cmd_validate(args) -> int:
    cfg = load_config(args.config)
    if args.offline:
        cfg.offline = True
    manifest = load_manifest(cfg.paths.manifest_path or HARNESS_MANIFEST)
    from .allowlist import build_allowlist, oracle_bound, run_validation
    from .harness.mock_runtime import MockAppWorldRuntime
    from .harness.sample_agent import AGENTS
    baseline_client, slm_clients = make_clients(cfg, cfg.offline)
    models = sorted(slm_clients.keys())
    replays = load_replays(args.replays)
    allowlist = build_allowlist(replays, models[0], cfg.replay.min_cell_samples,
                                cfg.allowlist.lcb_threshold, split="pilot")
    heldout_tasks = [t for t in manifest.tasks if t.split == "heldout"]
    if args.tasks:
        heldout_tasks = [t for t in heldout_tasks if t.task_id in args.tasks]
    validation = run_validation(
        heldout_tasks, [AGENTS["repA"], AGENTS["repB"]],
        baseline_client, slm_clients[models[0]], allowlist)
    heldout_spans = [s for s in load_spans(args.spans)
                     if s.get("experiment_split") == "heldout"]
    oracle = oracle_bound(heldout_spans, replays, models[0],
                          cfg.replay.min_cell_samples, cfg.allowlist.lcb_threshold)
    save_json(os.path.join(args.out, "validation.json"), validation)
    save_json(os.path.join(args.out, "oracle.json"), oracle)
    write_report(os.path.join(args.out, "validation.md"),
                 {"Allowlist utility validation (14.8)":
                  validation_section(validation, oracle)},
                 title="Stage 10 allowlist validation")
    print("validation report written to %s" % os.path.join(args.out, "validation.md"))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="span_offload",
                                     description="Stage 10: Conditional Span Offloadability Pilot")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="full harness pipeline (traces -> replay -> analysis -> validation)")
    p.add_argument("--config", default="config.json")
    p.add_argument("--out", default="data")
    p.add_argument("--offline", action="store_true", help="deterministic scripted LLM")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("extract", help="extract spans from native traces")
    p.add_argument("--traces", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trace-filter", default="",
                   help="extract only traces whose filename contains this substring (e.g. conv-26)")
    p.add_argument("--trace-format", default="canonical",
                   choices=["canonical", "appworld", "swebench", "locomo",
                            "langgraph"],
                   help="native trace format: canonical JSON (default), "
                        "AutoGen+AppWorld OTEL-style, trace_record SWE-bench, "
                        "trace_record LocoMo, or LocoMo+LangGraph+LangMem")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("replay", help="single-span counterfactual SLM replay")
    p.add_argument("--spans", required=True)
    p.add_argument("--config", default="config.json")
    p.add_argument("--out", required=True)
    p.add_argument("--models", default=None, help="comma-separated candidate SLM models")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--mode", default="live", choices=["live", "trace-local"],
                   help="live = full replay (requires env checkpoints); "
                        "trace-local = local layer only (same prompt -> SLM -> "
                        "parser -> contract checks), for checkpoint-less traces")
    p.add_argument("--trace-format", default="canonical",
                   choices=["canonical", "appworld", "swebench", "locomo",
                            "langgraph"],
                   help="format of the trace files referenced by spans.jsonl")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("analyze", help="offloadability profiles + conditioned analyses")
    p.add_argument("--replays", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--source", default="live", choices=["live", "trace_local"],
                   help="which outcome source to analyze: live (strict-success "
                        "profiles, 14.6-14.7) or trace_local (local-layer only)")
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("validate", help="multi-span static-allowlist sanity check (14.8)")
    p.add_argument("--config", default="config.json")
    p.add_argument("--replays", required=True)
    p.add_argument("--spans", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--offline", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("validate-appworld",
                       help="multi-span allowlist sanity check on REAL AppWorld "
                            "tasks (requires appworld installed + task data)")
    p.add_argument("--config", default="config.json")
    p.add_argument("--tasks", required=True, nargs="+",
                   help="AppWorld task ids to run, e.g. 0d8a4ee 3ab5b8b")
    p.add_argument("--replays", default=None,
                   help="optional replays.jsonl to build the allowlist from")
    p.add_argument("--out", required=True)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--on-local-failure", default="fallback",
                   choices=["fallback", "skip"],
                   help="after a local-layer verification failure: fallback = the large model recomputes"
                        " the span then continues downstream; skip = record the failure without fallback, offload the next span directly")
    p.set_defaults(fn=cmd_validate_appworld)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
