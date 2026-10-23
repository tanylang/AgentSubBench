"""End-to-end pilot orchestration (the harness smoke test of 14.1-14.8).

Pipeline:
  1. native traces: frozen representative agents x frozen task manifest, run
     with the baseline LLM (offline: deterministic scripted LLM)
  2. span extraction -> spans.jsonl (eligibility per 14.1)
  3. single-span counterfactual replay of pilot-split spans with every
     candidate SLM -> replays.jsonl (14.4-14.5)
  4. conditional offloadability profiles + task/agent/architecture-conditioned
     analyses + descriptor completeness test (14.6-14.7)
  5. static allowlist frozen from pilot profiles, multi-span end-to-end sanity
     check on held-out tasks incl. the oracle opportunity bound (14.8)

For a real AppWorld pilot, steps 2-4 are reused as-is via the CLI
(extract / replay / analyze); only step 1 is replaced by the real trace
collection of Stage 9.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import analysis
from .allowlist import build_allowlist, oracle_bound, run_validation
from .config import Stage10Config
from .harness.mock_runtime import MockAppWorldRuntime, run_baseline_trace
from .harness.mock_world import MockWorld
from .harness.sample_agent import AGENTS
from .harness.scripted_llm import MockLMClient
from .llm_client import ChatModelClient
from .replay import SingleSpanReplayEngine, replay_batch, save_replays, load_replays
from .report import (
    agent_section, completeness_section, delta_task_section, profiles_section,
    save_json, task_conditioned_section, validation_section, write_report,
)
from .schemas import TaskManifest
from .span_extraction import (
    census, extract_all, extract_spans_from_trace, save_spans, load_spans,
)
from .trace_io import iter_traces, load_manifest, save_trace

HARNESS_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "harness", "tasks.json")


def make_clients(cfg: Stage10Config, offline: bool) -> Tuple[Any, Dict[str, Any]]:
    """Build baseline + candidate SLM clients.

    offline=True                  -> everything is the deterministic scripted LLM
    offline=False, baseline URL set   -> real baseline LLM + real SLMs
    offline=False, baseline URL empty -> SCRIPTED baseline (native traces are
                                    generated deterministically) + real SLMs:
                                    the recommended mode when only the SLM API
                                    is available; SLM quality is then measured
                                    against deterministic baseline traces."""
    if offline:
        baseline = MockLMClient(
            model=cfg.baseline_llm.model or "mock-baseline",
            price_per_1k_input=cfg.baseline_llm.price_per_1k_input,
            price_per_1k_output=cfg.baseline_llm.price_per_1k_output,
            latency_ms=60.0,
        )
        slms: Dict[str, Any] = {}
        for name in cfg.slm.candidate_models or ["mock-slm"]:
            slms[name] = MockLMClient(
                model=name,
                price_per_1k_input=cfg.slm.price_per_1k_input,
                price_per_1k_output=cfg.slm.price_per_1k_output,
                latency_ms=15.0,
            )
        return baseline, slms

    # baseline LLM: real API if configured, otherwise the scripted mock
    if cfg.baseline_llm.base_url:
        baseline = ChatModelClient(
            base_url=cfg.baseline_llm.base_url, api_key=cfg.baseline_llm.resolved_api_key,
            model=cfg.baseline_llm.model, temperature=cfg.baseline_llm.temperature,
            max_output_tokens=cfg.baseline_llm.max_output_tokens,
            price_per_1k_input=cfg.baseline_llm.price_per_1k_input,
            price_per_1k_output=cfg.baseline_llm.price_per_1k_output,
            timeout_s=cfg.baseline_llm.timeout_s, max_retries=cfg.baseline_llm.max_retries,
            extra_body=cfg.baseline_llm.extra_body,
        )
    else:
        print("  note: baseline_llm.base_url empty -> scripted baseline traces "
              "(real SLM will be measured against them)")
        baseline = MockLMClient(
            model=cfg.baseline_llm.model or "mock-baseline",
            price_per_1k_input=cfg.baseline_llm.price_per_1k_input,
            price_per_1k_output=cfg.baseline_llm.price_per_1k_output,
            latency_ms=60.0,
        )

    if not cfg.slm.candidate_models:
        raise ValueError("no candidate SLM models configured (slm.models)")
    if not cfg.slm.base_url:
        raise ValueError("slm.base_url is empty: point it at your deployed SLM "
                         "(OpenAI-compatible /v1 root), or run with --offline")
    slms = {}
    for name in cfg.slm.candidate_models:
        slms[name] = ChatModelClient(
            base_url=cfg.slm.base_url, api_key=cfg.slm.resolved_api_key,
            model=name, temperature=cfg.slm.temperature,
            max_output_tokens=cfg.slm.max_output_tokens,
            price_per_1k_input=cfg.slm.price_per_1k_input,
            price_per_1k_output=cfg.slm.price_per_1k_output,
            timeout_s=cfg.slm.timeout_s, max_retries=cfg.slm.max_retries,
            extra_body=cfg.slm.extra_body,
        )
    return baseline, slms


def generate_traces(manifest: TaskManifest, cfg: Stage10Config,
                    baseline_client: Any, traces_dir: str,
                    task_ids: Optional[List[str]] = None,
                    max_attempts: int = 3,
                    progress=None) -> List[Tuple[str, str]]:
    """Run the frozen agents on (a subset of) the frozen tasks with the
    baseline LLM and save native traces.

    Keeps the first evaluator-passing attempt per (agent, task); if every
    attempt fails, keeps the last attempt flagged baseline_failed so it can
    feed the failure analysis (14.1: it stays out of the main denominator)."""
    produced: List[Tuple[str, str]] = []
    for task in manifest.tasks:
        if task_ids and task.task_id not in task_ids:
            continue
        for agent_id, agent_cfg in AGENTS.items():
            kept_path = None
            kept_ok = False
            last_trace = None
            for attempt in range(1, max_attempts + 1):
                trace, _result, eval_res, _world = run_baseline_trace(
                    task, agent_cfg, baseline_client)
                trace.run_meta.run_id = "run%d" % attempt
                trace.run_meta.attempts = attempt
                last_trace = trace
                passed = (eval_res.get("goal_state_passed")
                          and not eval_res.get("collateral_damage"))
                if passed:
                    kept_path = os.path.join(traces_dir, "%s__%s__run1.json"
                                             % (agent_id, task.task_id))
                    save_trace(trace, kept_path)
                    kept_ok = True
                    break
            if kept_path is None and last_trace is not None:
                # keep the failed attempt for failure analysis only
                kept_path = os.path.join(traces_dir, "%s__%s__run%d.json"
                                         % (agent_id, task.task_id, last_trace.run_meta.attempts))
                save_trace(last_trace, kept_path)
            if kept_path:
                produced.append((kept_path, task.task_id))
            if progress:
                progress(task.task_id, agent_id, kept_ok)
    return produced


def collect_spans(traces_dir: str, manifest: TaskManifest,
                  spans_path: str, adapter=None,
                  trace_filter: str = "") -> List[dict]:
    import glob
    from .trace_io import load_trace
    paths = sorted(glob.glob(os.path.join(traces_dir, "*.json")))
    if trace_filter:
        # Filter by filename (e.g. TRACE_ID=conv-26): take the batch's real files directly,
        # no temp dirs/symlinks — the trace_path recorded in spans stays loadable long-term
        paths = [p for p in paths
                 if trace_filter in os.path.basename(p)]
    instances = []
    if adapter is not None and hasattr(adapter, "iter_units"):
        # Unit-based adapter (e.g. langgraph: one unit trace per QA)
        for path in paths:
            try:
                for _task_id, trace in adapter.iter_units(path):
                    instances.extend(
                        extract_spans_from_trace(trace, path, manifest))
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    else:
        loader = adapter.load if adapter is not None else load_trace
        for path in paths:
            try:
                trace = loader(path)
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
            instances.extend(extract_spans_from_trace(trace, path, manifest))
    save_spans(instances, spans_path)
    return load_spans(spans_path)


def run_analysis(replays: List[dict], manifest: TaskManifest,
                 min_samples: int, out_dir: str) -> Dict[str, Any]:
    """All 14.6-14.7 analyses over the replay records."""
    models = sorted({r.get("model", "?") for r in replays})
    payload: Dict[str, Any] = {"models": models}
    payload["profiles"] = {m: analysis.profile_m(replays, m, mode="unit",
                                                 min_samples=min_samples) for m in models}
    payload["profiles_full"] = {m: analysis.profile_m(replays, m, mode="full",
                                                      min_samples=min_samples) for m in models}
    payload["task_tables"] = analysis.task_dimension_tables(replays, min_samples)
    payload["delta_task"] = analysis.delta_task_table(replays, min_samples)
    payload["within_agent"] = analysis.within_agent_tables(replays, min_samples)
    payload["matched"] = analysis.matched_agent_comparison(replays, min_samples)
    payload["architecture"] = analysis.architecture_tables(replays, min_samples)
    payload["joint"] = analysis.joint_matched_tables(replays, min_samples)
    payload["completeness"] = analysis.completeness_test(replays, min_samples)
    save_json(os.path.join(out_dir, "analysis.json"), payload)
    return payload


def run_pipeline(cfg: Stage10Config, out_dir: Optional[str] = None,
                 task_ids: Optional[List[str]] = None,
                 replay_limit: Optional[int] = None) -> str:
    out = out_dir or "data"
    traces_dir = os.path.join(out, "traces")
    report_dir = os.path.join(out, "report")
    spans_path = os.path.join(out, "spans.jsonl")
    replays_path = os.path.join(out, "replays.jsonl")
    for d in (traces_dir, report_dir):
        os.makedirs(d, exist_ok=True)

    manifest = load_manifest(cfg.paths.manifest_path or HARNESS_MANIFEST)
    offline = cfg.offline
    baseline_client, slm_clients = make_clients(cfg, offline)
    models = sorted(slm_clients.keys())
    min_samples = cfg.replay.min_cell_samples

    print("[1/6] generating baseline native traces%s ..."
          % (" (offline scripted)" if offline else ""))
    generate_traces(manifest, cfg, baseline_client, traces_dir,
                    task_ids=task_ids,
                    progress=lambda t, a, ok: print("  trace %s/%s: %s"
                                                    % (a, t, "ok" if ok else "retrying")))

    print("[2/6] extracting spans ...")
    spans = collect_spans(traces_dir, manifest, spans_path)
    cens = census([_as_instance(s) for s in spans])
    save_json(os.path.join(report_dir, "census.json"), cens)
    print("  spans=%d eligible=%d ineligible=%d baseline_failed=%d"
          % (cens["total_spans"], cens["eligible_spans"],
             cens["ineligible_spans"], cens["baseline_failed_spans"]))

    print("[3/6] single-span counterfactual replay (models: %s) ..." % ", ".join(models))
    pilot = [_as_instance(s) for s in spans
             if s.get("experiment_split") == "pilot" and s.get("eligible")]
    heldout_spans = [s for s in spans if s.get("experiment_split") == "heldout"]
    runtime = MockAppWorldRuntime(AGENTS, manifest)
    from .trace_io import load_trace
    traces = {}
    for s in pilot:
        if s.trace_path not in traces:
            traces[s.trace_path] = load_trace(s.trace_path)
    engine = SingleSpanReplayEngine(cfg, runtime, baseline_client, slm_clients)
    records = replay_batch(engine, pilot, traces, models, limit=replay_limit,
                           progress=lambda i, n, sid, m: print(
                               "  [%d/%d] %s @ %s" % (i, n, sid, m)) if i % 10 == 0 or i == n else None)
    save_replays(records, replays_path)
    replays = load_replays(replays_path)
    print("  replay records: %d" % len(replays))

    print("[4/6] analysis (14.6-14.7) ...")
    payload = run_analysis(replays, manifest, min_samples, report_dir)

    print("[5/6] allowlist + validation (14.8) ...")
    allowlist = build_allowlist(replays, models[0], min_samples,
                                cfg.allowlist.lcb_threshold, split="pilot")
    save_json(os.path.join(report_dir, "allowlist.json"),
              {"model": allowlist.model, "lcb_threshold": allowlist.lcb_threshold,
               "n_allowed_cells": allowlist.size(),
               "entries": allowlist.entries})
    heldout_tasks = [t for t in manifest.tasks if t.split == "heldout"]
    if task_ids:
        heldout_tasks = [t for t in heldout_tasks if t.task_id in task_ids]
    validation = run_validation(
        heldout_tasks, [AGENTS["repA"], AGENTS["repB"]],
        baseline_client, slm_clients[models[0]], allowlist,
        progress=lambda t, a: print("  validation run %s / %s" % (t, a)))
    save_json(os.path.join(report_dir, "validation.json"), validation)
    oracle = oracle_bound(heldout_spans, replays, models[0], min_samples,
                          cfg.allowlist.lcb_threshold)
    save_json(os.path.join(report_dir, "oracle.json"), oracle)

    print("[6/6] writing report ...")
    setup = ("- baseline LLM: `%s`%s\n- candidate SLMs: %s\n- fallback mode: `%s`\n"
             "- allowlist: LCB >= %s (min cell samples %d)\n- offline mock: %s"
             % (baseline_client.name,
                " (prices $%.4f/$%.4f per 1k)" % (cfg.baseline_llm.price_per_1k_input,
                                                  cfg.baseline_llm.price_per_1k_output),
                ", ".join("`%s`" % m for m in models),
                cfg.replay.fallback_mode, cfg.allowlist.lcb_threshold,
                cfg.allowlist.min_cell_samples, offline))
    sections = {
        "Setup": setup,
        "Span census (14.1-14.2)": _census_markdown(cens),
        "Offloadability profiles (14.6)": profiles_section(payload["profiles"]),
        "Task-conditioned analysis (14.7.1)":
            task_conditioned_section(payload["task_tables"])
            + "\n\n" + delta_task_section(payload["delta_task"]),
        "Agent & architecture-conditioned (14.7.2-14.7.4)":
            agent_section(payload["within_agent"], payload["matched"],
                          payload["architecture"], payload["joint"]),
        "Descriptor completeness (14.7.5)": completeness_section(payload["completeness"]),
        "Allowlist utility validation (14.8)": validation_section(validation, oracle),
    }
    report_path = write_report(os.path.join(report_dir, "report.md"), sections)
    return report_path


def _as_instance(span_dict: dict) -> Any:
    from .schemas import from_dict, SpanInstance
    return from_dict(SpanInstance, span_dict)


def _census_markdown(cens: Dict[str, Any]) -> str:
    from .report import table
    lines: List[str] = []
    lines.append("- total spans: %d (eligible %d, ineligible %d, baseline-failed %d)"
                 % (cens["total_spans"], cens["eligible_spans"],
                    cens["ineligible_spans"], cens["baseline_failed_spans"]))
    lines.append("")
    lines.append("Ineligibility reasons (baseline-failed runs are excluded from the "
                 "main offloadability denominator, 14.1):")
    lines.append("")
    rows = [[k, str(v)] for k, v in sorted(cens.get("ineligibility_reason_counts", {}).items(),
                                           key=lambda kv: -kv[1])]
    lines.append(table(["reason", "count"], rows))
    lines.append("")
    lines.append("Descriptor census:")
    lines.append("")
    rows = [[k, str(cens["node_pattern_counts"].get(k, 0))]
            for k in ["intent_interpretation", "planning", "tool_argument_generation",
                      "observation_interpretation", "failure_diagnosis", "replanning",
                      "policy_checking", "final_synthesis"]]
    lines.append(table(["node_pattern", "count"], rows))
    lines.append("")
    rows = [[k, str(cens["control_role_counts"].get(k, 0))]
            for k in ["global_planner", "local_executor", "repair_controller",
                      "verifier", "terminal"]]
    lines.append(table(["control_role", "count"], rows))
    lines.append("")
    rows = [[k, str(cens["state_effect_counts"].get(k, 0))]
            for k in ["read_only", "external_state_mutation", "memory_write",
                      "artifact_generation", "permission_sensitive"]]
    lines.append(table(["state_effect", "count"], rows))
    lines.append("")
    rows = [[k, str(cens["verifier_type_counts"].get(k, 0))]
            for k in ["schema_check", "tool_exec_success", "policy_check",
                      "goal_state_match", "none"]]
    lines.append(table(["verifier_type", "count"], rows))
    return "\n".join(lines)
