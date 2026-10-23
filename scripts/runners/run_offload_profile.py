#!/usr/bin/env python
"""Quantifies span offloadability: multiple full replays per eligible span.

Each replay starts from a fresh restore (SingleSpanReplayEngine restores once
per round, no cross-round contamination), quantifying:
  - round-to-round stability of strict_success (variance of success rate)
  - pass rate of the local equivalence validator (equivalence)
  - downstream trace size (tool executions / LLM rounds) and resource cost

Usage (server, inside container):
  python scripts/runners/run_offload_profile.py \
      --config OUT_DIR/config.json --spans OUT_DIR/spans.jsonl \
      --rounds 3 --out OUT_DIR/offload_profile.jsonl \
      --summary OUT_DIR/offload_profile_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402
from span_offload.config import load_config  # noqa: E402
from span_offload.harness.appworld_runtime import AppWorldAgentRuntime  # noqa: E402
from span_offload.llm_client import ChatModelClient  # noqa: E402
from span_offload.replay import (  # noqa: E402
    SingleSpanReplayEngine, record_contract_problems,
)
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


def _trace_stats(record) -> tuple:
    dt = record.downstream_trace or {}
    spans = dt.get("spans", [])
    n_llm = sum(1 for e in spans if e.get("span_type") == "llm.call")
    n_tools = sum(1 for e in spans if e.get("span_type") == "tool.execution")
    n_fail = sum(1 for e in spans
                 if e.get("span_type") == "tool.execution" and not e.get("ok"))
    return n_llm, n_tools, n_fail


#: normalize arm label (back-fill from injection for old records without an arm field)
_ARM_ALIAS = {"identity": "I", "slm": "T_span"}
_ARM_FILES = ("T_span", "B", "I")

#: agent_id -> output subdirectory name (offload collections for different agents are stored in separate dirs)
_AGENT_DIR = {
    "AppWorld-Simplified": "simplified",
    "AutoGen-AssistantAgent": "autogen",
    "AppWorld-Legacy": "legacy",
    "AppWorld-OpenAI-Agents": "openai",
    "AppWorld-Smolagents": "smolagents",
    "Tau2-LLMAgent": "tau2",
}


def _record_arm(r: dict) -> str:
    m = r.get("replay_meta") or {}
    arm = m.get("arm") or _ARM_ALIAS.get(m.get("injection", ""),
                                         m.get("injection"))
    return arm if arm in _ARM_FILES else ""


class _TaskMultiplexer:
    """Splits output by task: <out_dir>/<stem>_<task>.jsonl, lazily opened."""

    def __init__(self, out_dir: str, stem: str, mode: str):
        self._dir = out_dir
        self._stem = stem
        self._mode = mode
        self._files = {}

    def _path(self, task: str) -> str:
        # task_id may contain colons/slashes (locomo:conv-26:q001): sanitize for filenames
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(task))
        return os.path.join(self._dir, "%s_%s.jsonl" % (self._stem, safe))

    def for_task(self, task: str):
        if task not in self._files:
            self._files[task] = open(self._path(task), self._mode,
                                     encoding="utf-8")
        return self._files[task]

    def write(self, task: str, text: str) -> None:
        self.for_task(task).write(text)

    def flush(self, task: str) -> None:
        self.for_task(task).flush()

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()


def _load_resume(path: str, resume_recs) -> int:
    """Reads one output file and merges stats of completed (arm, round) into resume_recs."""
    n = 0
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue  # half-written leftover line from an interrupted run
            arm = _record_arm(r)
            if not arm:
                continue
            rnd = (r.get("replay_meta") or {}).get("profile_round")
            if rnd is None:
                continue
            dt = r.get("downstream_trace") or {}
            sp = dt.get("spans") or []
            n_llm = sum(1 for e in sp if e.get("span_type") == "llm.call")
            n_tools = sum(1 for e in sp
                          if e.get("span_type") == "tool.execution")
            resume_recs[r.get("span_id")].append(
                (arm, rnd, bool(r.get("strict_success")),
                 bool((r.get("local_outcome") or {}).get("equivalence", {})
                      .get("passed")), n_llm, n_tools))
            n += 1
    return n


def _migrate_to_task_files(legacy_path: str, out_dir: str, stem: str) -> None:
    """Migrates the old single-file output on resume: distribute by task into
    <out_dir>/<stem>_<task>.jsonl (dedupe by (span,arm,round)), rename the
    original to <legacy>.legacy. Runs once: legacy file no longer exists after migration."""
    import glob as _glob
    seen = set()
    for p in _glob.glob(os.path.join(out_dir, "%s_*.jsonl" % stem)):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                seen.add((r.get("span_id"), _record_arm(r),
                          (r.get("replay_meta") or {}).get("profile_round")))
    buckets = defaultdict(list)
    n_skip = n_dup = 0
    with open(legacy_path, encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                n_skip += 1
                continue
            if not _record_arm(r):
                n_skip += 1
                continue
            key = (r.get("span_id"), _record_arm(r),
                   (r.get("replay_meta") or {}).get("profile_round"))
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            buckets[r.get("task_id")].append(line)
    for task, lines in buckets.items():
        with open(os.path.join(out_dir, "%s_%s.jsonl" % (stem, task)),
                  "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    os.rename(legacy_path, legacy_path + ".legacy")
    print("migrate: distributed %d lines from the old single file by task"
          " (skipped %d leftover/%d duplicate), renamed original to %s.legacy" % (
              sum(len(v) for v in buckets.values()), n_skip, n_dup,
              legacy_path), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--spans", required=True)
    ap.add_argument("--manifest", default="data/appworld/task_manifest.json")
    ap.add_argument("--trace-id", default="", help="trace filename substring")
    ap.add_argument("--n-spans", type=int, default=0, help="0 = all")
    ap.add_argument("--confirm-select", type=int, default=0,
                    help="confirm-set sampling: pick N spans deterministically"
                         " from the eligible spans in --spans by screening"
                         " effect (takes precedence over --n-spans); requires --screening-records")
    ap.add_argument("--screening-records", default="",
                    help="screening dedup records jsonl (profile data source for confirm-set sampling)")
    ap.add_argument("--quota", default="",
                    help="confirm-set node_pattern quotas (ts,fs,tag,other,"
                         "comma-separated; empty = scale by 6,6,4,2 ratio)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--identity-rounds", type=int, default=0,
                    help="identity control rounds: inject baseline recorded"
                         "output (no SLM calls), to separate harness rebuild bias from SLM bias")
    ap.add_argument("--baseline-rounds", type=int, default=0,
                    help="B arm rounds: target span resampled live by baseline LLM"
                         "(differs from T_span only in the generator model, primary comparison baseline)")
    ap.add_argument("--block-seed", type=int, default=0,
                    help="seed for randomizing I/B/T_span order within a block (plan §3.3)")
    ap.add_argument("--mode", default="both",
                    choices=["both", "slm", "identity", "baseline"],
                    help="both = SLM+identity controls; slm = SLM rounds only;"
                         "identity = identity rounds only; baseline = B arm"
                         "rounds only (collect control runs separately)")
    ap.add_argument("--fallback-mode", default="",
                    help="override fallback_mode from config")
    ap.add_argument("--stage", default="",
                    help="override the output stage subdirectory"
                         "(screening/confirm); empty = auto (confirm when confirm-select, else screening)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from checkpoint: skip completed (span,arm,round)"
                         " in output files and append; summary merges stats of"
                         " completed rounds. Requires the same"
                         " ROUNDS/IDENTITY_ROUNDS/BASELINE_ROUNDS/BLOCK_SEED config (block randomization reproduces by seed)")
    ap.add_argument("--out", default="out_span_offload/offload_profile.jsonl")
    ap.add_argument("--summary", default="out_span_offload/offload_profile_summary.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    model_id = slm.name
    fallback_mode = args.fallback_mode or cfg.replay.fallback_mode

    spans_all = load_spans(args.spans)
    todo = [from_dict(SpanInstance, s) for s in spans_all if s.get("eligible")]
    if args.trace_id:
        todo = [s for s in todo if args.trace_id in s.trace_path]
    # Output layout: offload/<stage> subdirectory, per-task files:
    #   <out dir>/offload/<stage>/<stem>_<task>.jsonl
    # (agent level is already expressed by slurm's out_span_offload/<agent>/<trace_id>;
    # stage = confirm(--confirm-select) | screening(others, incl. N_SPANS sampling))
    stage = args.stage or ("confirm" if args.confirm_select > 0 else "screening")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                           "offload", stage)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    # summary and sampling manifest follow into the offload directory
    args.summary = os.path.join(out_dir, os.path.basename(args.summary))
    print("output dir: %s (per-task files %s_<task>.jsonl)" % (
        out_dir, stem), flush=True)
    sampling_meta = None
    if args.confirm_select > 0:
        # confirm-set sampling (takes precedence over N_SPANS stratified
        # sampling): deterministically pick N spans by screening T−B effect, then run directly -- one command does "sampling + run"
        if not args.screening_records:
            raise SystemExit("--confirm-select requires --screening-records")
        from select_confirmation_spans import (
            per_span_profiles, select_confirmation,
        )
        profiles = per_span_profiles(args.screening_records)
        node_map = {}
        for s in todo:
            np_ = s.span_descriptor.node_pattern
            if np_ == "tool_argument_generation":
                se = s.span_descriptor.mechanism_tags.state_effect
                np_ += "_write" if "external_state_mutation" in se else "_read"
            node_map[(s.task_id, s.invocation_index)] = np_
        profiles = {k: v for k, v in profiles.items() if k in node_map}
        if args.quota:
            q = [int(x) for x in args.quota.split(",")]
            if len(q) != 4 or sum(q) != args.confirm_select:
                raise SystemExit("--quota must have 4 values and sum == --confirm-select")
            quotas = dict(zip(("tool_selection", "final_synthesis",
                               "tool_argument_generation", "other"), q))
        else:
            base = (6, 6, 4, 2)
            quotas = {k: max(1, round(args.confirm_select * v / 18))
                      for k, v in zip(("tool_selection", "final_synthesis",
                                       "tool_argument_generation", "other"),
                                      base)}
            # fine-tune after scaling so the sum == n
            diff = args.confirm_select - sum(quotas.values())
            for k in sorted(quotas, key=lambda x: -quotas[x]):
                if diff == 0:
                    break
                quotas[k] += 1 if diff > 0 else -1
                diff += -1 if diff > 0 else 1
        picked_keys = select_confirmation(
            profiles, node_map, args.confirm_select, quotas)
        todo = [s for s in todo if (s.task_id, s.invocation_index)
                in picked_keys]
        sampling_meta = {"method": "confirm_select_deterministic",
                         "n": args.confirm_select, "quota": quotas}
        # persist the confirm-set manifest (same dir as summary)
        manifest_path = args.summary.rsplit(".", 1)[0] + ".confirm_spans.json"
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)),
                    exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump([{
                "task_id": s.task_id, "span_id": s.span_id,
                "invocation_index": s.invocation_index,
                "node_pattern": s.span_descriptor.node_pattern,
                "control_role": s.span_descriptor.control_role,
                "trace_path": s.trace_path,
            } for s in sorted(todo, key=lambda s: (s.task_id,
                                                   s.invocation_index))],
                mf, ensure_ascii=False, indent=1)
        print("confirm-set sampling: %d spans (quota %s), manifest %s" % (
            len(todo), quotas, manifest_path), flush=True)
    elif args.n_spans > 0:
        # stratified deterministic sampling (zero randomness, same input
        # yields same output): no longer take the first N from the file
        # (task-biased), but sample by control role x task coverage per span_sampling.stratified_sample, keeping small samples representative
        from span_offload.span_sampling import stratified_sample
        picked_ids = {s["span_id"] for s in stratified_sample(
            [s for s in spans_all if s.get("eligible")
             and (args.trace_id in s.get("trace_path", "") or not args.trace_id)],
            n=args.n_spans)}
        todo = [s for s in todo if s.span_id in picked_ids]
        sampling_meta = {"method": "stratified_deterministic_v1",
                         "n": args.n_spans}
        # persist the sampling manifest (same dir as summary, same input = same content)
        manifest_path = args.summary.rsplit(".", 1)[0] + ".sampled_spans.json"
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)),
                    exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump([{
                "task_id": s.task_id, "span_id": s.span_id,
                "invocation_index": s.invocation_index,
                "node_pattern": s.span_descriptor.node_pattern,
                "control_role": s.span_descriptor.control_role,
                "trace_path": s.trace_path,
            } for s in sorted(todo, key=lambda s: (s.task_id,
                                                   s.invocation_index))],
                mf, ensure_ascii=False, indent=1)
        print("sampling manifest: %s (%d spans)" % (manifest_path, len(todo)),
              flush=True)
    n_slm_rounds = args.rounds if args.mode in ("both", "slm") else 0
    n_id_rounds = args.identity_rounds if args.mode in ("both", "identity") else 0
    n_b_rounds = args.baseline_rounds if args.mode in ("both", "baseline") else 0
    print("span count: %d%s x (%d slm rounds + %d baseline rounds + %d identity rounds,"
          "mode=%s,in-block randomization seed=%d),SLM=%s baseline=%s fallback=%s "
          "backend=%s" % (
              len(todo),
              " (N_SPANS=stratified deterministic sampling)" if sampling_meta else "",
              n_slm_rounds, n_b_rounds, n_id_rounds, args.mode,
              args.block_seed, model_id, baseline.name, fallback_mode,
              (AppWorldAgentRuntime.continuation_backend or "auto(by trace agent)")), flush=True)

    manifest = load_manifest(args.manifest)
    adapter = AppWorldTraceAdapter(manifest)
    traces = {}
    for s in todo:
        if s.trace_path not in traces:
            traces[s.trace_path] = adapter.load(s.trace_path)

    runtime = AppWorldAgentRuntime()
    engine = SingleSpanReplayEngine(
        cfg, runtime, baseline_client=baseline,
        slm_clients={model_id: slm}, fallback_mode=fallback_mode)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("summary: %s" % args.summary, flush=True)
    # resume: migrate the old single file into per-task files (once, renamed .legacy)
    if args.resume and os.path.exists(args.out):
        _migrate_to_task_files(args.out, out_dir, stem)
    # resume: load completed (span_id, arm, round) and stats from per-task files
    import glob as _glob
    resume_recs = defaultdict(list)
    if args.resume:
        n_resume = 0
        for p in _glob.glob(os.path.join(out_dir, "%s_*.jsonl" % stem)):
            n_resume += _load_resume(p, resume_recs)
        print("resume: loaded %d completed records (%d spans), will skip completed rounds" % (
            n_resume, len(resume_recs)), flush=True)
    per_span = []
    mode = "a" if args.resume else "w"
    f = _TaskMultiplexer(out_dir, stem, mode)
    # data-consistency self-check (2026-09-20.1/.2): count violations of
    # the reasoning-dump and engine_version contract; any violation means the code sync is incomplete
    contract_fails = 0
    for si, span in enumerate(todo):
            trace = traces[span.trace_path]
            span_rec = next(s for s in trace.spans
                            if s.invocation_index == span.invocation_index)
            base_parsed = span_rec.baseline_response.parsed
            groups = {"slm": {"strict": [], "eq": [], "llm": [], "tools": []},
                      "baseline": {"strict": [], "eq": [], "llm": [], "tools": []},
                      "identity": {"strict": [], "eq": [], "llm": [], "tools": []}}
            # in-block I/B/T_span order randomization (plan §3.3): each
            # span's round plan is shuffled independently, seed = block_seed
            # + span stable hash, reproducible across reruns
            plan = (["I"] * n_id_rounds + ["B"] * n_b_rounds
                    + ["T_span"] * n_slm_rounds)
            # stable seed: cannot use hash(str) (Python string hashing has a
            # per-process random salt, plan drift across processes/resume ->
            # duplicate rounds); md5 guarantees the same span gets the same seed so shuffles are identical across runs
            import hashlib
            stable = int(hashlib.md5(
                span.span_id.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
            rng = random.Random(args.block_seed + stable)
            rng.shuffle(plan)
            total_rounds = len(plan)
            # resume: merge stats of completed rounds (filtered by this plan
            # -- on config changes old rounds are neither counted nor skipped), and skip by (arm, rnd)
            plan_keys = set(zip(plan, range(total_rounds)))
            loaded = [(a, r, st, eq, nl, nt)
                      for a, r, st, eq, nl, nt
                      in resume_recs.get(span.span_id, [])
                      if (a, r) in plan_keys]
            done_keys = set()
            for a, r, st, eq, nl, nt in sorted(loaded, key=lambda x: x[1]):
                g = groups[{"T_span": "slm", "B": "baseline",
                            "I": "identity"}[a]]
                g["strict"].append(st)
                g["eq"].append(eq)
                g["llm"].append(nl)
                g["tools"].append(nt)
                done_keys.add((a, r))
            if done_keys:
                print("span %d/%d [%s inv=%d] resume: skipping %d completed rounds" % (
                    si + 1, len(todo), span.span_descriptor.node_pattern,
                    span.invocation_index, len(done_keys)), flush=True)
            for rnd, arm in enumerate(plan):
                if (arm, rnd) in done_keys:
                    continue
                if arm == "I":
                    record = engine.replay(span, trace, model_id,
                                           force_injected=base_parsed, arm="I")
                elif arm == "B":
                    record = engine.replay(span, trace, model_id, arm="B")
                else:
                    record = engine.replay(span, trace, model_id, arm="T_span")
                record.replay_meta["profile_round"] = rnd
                record.replay_meta["block_seed"] = args.block_seed
                if sampling_meta:
                    record.replay_meta["span_sampling"] = dict(sampling_meta)
                f.write(record.task_id,
                        json.dumps(to_dict(record), ensure_ascii=False) + "\n")
                f.flush(record.task_id)
                g = groups["slm" if arm == "T_span"
                           else ("baseline" if arm == "B" else "identity")]
                g["strict"].append(bool(record.strict_success))
                g["eq"].append(bool((record.local_outcome.equivalence or {}).get("passed")))
                n_llm, n_tools, _ = _trace_stats(record)
                g["llm"].append(n_llm)
                g["tools"].append(n_tools)
                print("span %d/%d [%s inv=%d] %-8s round %d/%d: strict=%s eq=%s "
                      "(llm=%d tools=%d)" % (
                          si + 1, len(todo), span.span_descriptor.node_pattern,
                          span.invocation_index, arm, rnd + 1, total_rounds,
                          g["strict"][-1], g["eq"][-1], n_llm, n_tools), flush=True)
                cprobs = record_contract_problems(record)
                if cprobs:
                    contract_fails += 1
                    print("  [contract FAIL] %s" % "; ".join(cprobs[:3]), flush=True)
            entry = {
                "task_id": span.task_id,
                "span_id": span.span_id,
                "invocation_index": span.invocation_index,
                "node_pattern": span.span_descriptor.node_pattern,
                "control_role": span.span_descriptor.control_role,
                "rounds": n_slm_rounds,
                "identity_rounds": n_id_rounds,
                "baseline_rounds": n_b_rounds,
                "block_seed": args.block_seed,
                "mode": args.mode,
            }
            if n_slm_rounds > 0:
                slm_g = groups["slm"]
                entry.update({
                    "strict_seq": slm_g["strict"],
                    "strict_rate": sum(slm_g["strict"]) / max(1, len(slm_g["strict"])),
                    "eq_seq": slm_g["eq"],
                    "eq_rate": sum(slm_g["eq"]) / max(1, len(slm_g["eq"])),
                    "stable": len(set(slm_g["strict"])) == 1,
                    "avg_llm_steps": sum(slm_g["llm"]) / max(1, len(slm_g["llm"])),
                    "avg_tool_executions": sum(slm_g["tools"]) / max(1, len(slm_g["tools"])),
                })
            if n_id_rounds > 0:
                id_g = groups["identity"]
                entry["identity"] = {
                    "strict_seq": id_g["strict"],
                    "strict_rate": sum(id_g["strict"]) / max(1, len(id_g["strict"])),
                    "eq_rate": sum(id_g["eq"]) / max(1, len(id_g["eq"])),
                    "avg_llm_steps": sum(id_g["llm"]) / max(1, len(id_g["llm"])),
                    "avg_tool_executions": sum(id_g["tools"]) / max(1, len(id_g["tools"])),
                }
                if n_slm_rounds > 0:
                    # offload downstream delta effect (Δ = slm group − identity group)
                    entry["delta"] = {
                        "strict_rate": entry["strict_rate"] - entry["identity"]["strict_rate"],
                        "eq_rate": entry["eq_rate"] - entry["identity"]["eq_rate"],
                        "avg_llm_steps": entry["avg_llm_steps"] - entry["identity"]["avg_llm_steps"],
                        "avg_tool_executions": entry["avg_tool_executions"] - entry["identity"]["avg_tool_executions"],
                    }
            if n_b_rounds > 0:
                b_g = groups["baseline"]
                entry["baseline"] = {
                    "strict_seq": b_g["strict"],
                    "strict_rate": sum(b_g["strict"]) / max(1, len(b_g["strict"])),
                    "eq_rate": sum(b_g["eq"]) / max(1, len(b_g["eq"])),
                    "avg_llm_steps": sum(b_g["llm"]) / max(1, len(b_g["llm"])),
                    "avg_tool_executions": sum(b_g["tools"]) / max(1, len(b_g["tools"])),
                }
                if n_slm_rounds > 0:
                    # primary comparison (plan §7.3): T_span vs B delta effect
                    entry["delta_vs_baseline"] = {
                        "strict_rate": entry["strict_rate"] - entry["baseline"]["strict_rate"],
                        "eq_rate": entry["eq_rate"] - entry["baseline"]["eq_rate"],
                        "avg_llm_steps": entry["avg_llm_steps"] - entry["baseline"]["avg_llm_steps"],
                        "avg_tool_executions": entry["avg_tool_executions"] - entry["baseline"]["avg_tool_executions"],
                    }
            per_span.append(entry)
    f.close()

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(per_span, f, ensure_ascii=False, indent=1)

    # ---- summary ----
    n = len(per_span)
    print()
    print("=== offloadability summary (%d spans x %d slm rounds + %d baseline rounds + %d "
          "identity rounds, mode=%s) ===" % (
              n, n_slm_rounds, n_b_rounds, n_id_rounds, args.mode))
    if n == 0:
        print("no matching spans (check TRACE_IDS or eligibility; runs whose"
              "baseline failed are excluded from the offloadability denominator)")
        return 0
    if n_slm_rounds > 0:
        all_flat = [v for p in per_span for v in p["strict_seq"]]
        eq_flat = [v for p in per_span for v in p["eq_seq"]]
        print("SLM group strict_success: %d/%d (%.0f%%)" % (
            sum(all_flat), len(all_flat), 100 * sum(all_flat) / max(1, len(all_flat))))
        print("SLM group equivalence:    %d/%d (%.0f%%)" % (
            sum(eq_flat), len(eq_flat), 100 * sum(eq_flat) / max(1, len(eq_flat))))
        stable = sum(1 for p in per_span if p["stable"])
        print("round-stable spans:       %d/%d (%.0f%%)" % (
            stable, n, 100 * stable / max(1, n)))
    if n_id_rounds > 0:
        id_strict = [v for p in per_span for v in p["identity"]["strict_seq"]]
        print("identity control strict pass rate: %d/%d (%.0f%%)" % (
            sum(id_strict), len(id_strict),
            100 * sum(id_strict) / max(1, len(id_strict))))
        id_llm = sum(p["identity"]["avg_llm_steps"] for p in per_span) / n
        id_tools = sum(p["identity"]["avg_tool_executions"] for p in per_span) / n
        print("identity control avg downstream: LLM steps %.1f, tool executions %.1f" % (id_llm, id_tools))
    if n_slm_rounds > 0 and n_id_rounds > 0:
        d_strict = sum(p["delta"]["strict_rate"] for p in per_span) / n
        d_tools = sum(p["delta"]["avg_tool_executions"] for p in per_span) / n
        d_llm = sum(p["delta"]["avg_llm_steps"] for p in per_span) / n
        print("offload delta effect (avg Δ vs identity): strict %+.0f%%, LLM steps %+.1f,"
              "tool executions %+.1f" % (100 * d_strict, d_llm, d_tools))
    if n_b_rounds > 0:
        b_strict = [v for p in per_span for v in p["baseline"]["strict_seq"]]
        print("B arm (baseline resampling) strict pass rate: %d/%d (%.0f%%)" % (
            sum(b_strict), len(b_strict),
            100 * sum(b_strict) / max(1, len(b_strict))))
        b_llm = sum(p["baseline"]["avg_llm_steps"] for p in per_span) / n
        b_tools = sum(p["baseline"]["avg_tool_executions"] for p in per_span) / n
        print("B arm avg downstream: LLM steps %.1f, tool executions %.1f" % (b_llm, b_tools))
    if n_slm_rounds > 0 and n_b_rounds > 0:
        d = sum(p["delta_vs_baseline"]["strict_rate"] for p in per_span) / n
        d_tools = sum(p["delta_vs_baseline"]["avg_tool_executions"]
                      for p in per_span) / n
        d_llm = sum(p["delta_vs_baseline"]["avg_llm_steps"] for p in per_span) / n
        print("primary comparison T_span vs B (avg Δ): strict %+.0f%%, LLM steps %+.1f,"
              "tool executions %+.1f" % (100 * d, d_llm, d_tools))
    if n_slm_rounds > 0:
        by_node = defaultdict(list)
        for p in per_span:
            by_node[p["node_pattern"]].append(p)
        for np_, ps in sorted(by_node.items()):
            s_flat = [v for p in ps for v in p["strict_seq"]]
            e_flat = [v for p in ps for v in p["eq_seq"]]
            print("  %-26s strict=%.0f%% eq=%.0f%% (n=%d)" % (
                np_, 100 * sum(s_flat) / len(s_flat),
                100 * sum(e_flat) / max(1, len(e_flat)), len(ps)))
        fully = sum(1 for p in per_span
                    if p["strict_rate"] == 1.0 and p["eq_rate"] == 1.0)
        never = sum(1 for p in per_span if p["strict_rate"] == 0.0)
        print("fully offloadable (strict=eq=100%%): %d/%d" % (fully, n))
        print("never offloadable (strict=0%%):     %d/%d" % (never, n))
    print("details: %s" % args.summary)
    if contract_fails:
        print()
        print("=== data-consistency self-check FAIL: %d records violate the "
              "reasoning/engine contract (incomplete code sync or output from an old process) ===" % contract_fails, flush=True)
        return 2
    print()
    print("=== data-consistency self-check PASS: all records satisfy the reasoning/engine contract ===",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
