#!/usr/bin/env python
"""LoCoMo + LangGraph ReAct + LangMem span offload runner: I/B/T_span three arms,
reusing all infrastructure from run_offload_profile.py.

Differences from the locomo version:
  - adapter = LangGraphTraceAdapter, runtime = LangGraphRuntime
  - offload target = QA-phase spans only (adapter already filters)
  - --relax-baseline-pass (on by default): the trace's pass is written by the
    capture-side finalize; batches never finalized (always null) all enter
    the denominator via the baseline-fail exemption, conditionally analyzed against the final evaluation after replay
  - env vars: TRACE_COLLECT (authoritative locomo_trace_core) / LOCOMO_DATA_FILE

Usage (server, inside the langgraph venv):
  python scripts/runners/run_langgraph_offload.py \
      --config OUT_DIR/config.json --spans OUT_DIR/spans_langgraph.jsonl \
      --manifest data/locomo/task_manifest.json \
      --rounds 1 --identity-rounds 1 --relax-baseline-pass \
      --out OUT_DIR/offload_profile_langgraph.jsonl \
      --summary OUT_DIR/offload_profile_summary_langgraph.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# reuse run_offload_profile's infrastructure
from run_offload_profile import (  # noqa: E402
    _load_resume, _migrate_to_task_files, _record_arm, _TaskMultiplexer,
    _trace_stats, make_client,
)

from span_offload.config import load_config  # noqa: E402
from span_offload.harness import langgraph_env  # noqa: E402
from span_offload.harness.langgraph_runtime import LangGraphRuntime  # noqa: E402
from span_offload.langgraph_adapter import LangGraphTraceAdapter  # noqa: E402
from span_offload.replay import SingleSpanReplayEngine  # noqa: E402
from span_offload.schemas import SpanInstance, from_dict, to_dict  # noqa: E402
from span_offload.span_extraction import load_spans  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402

#: relaxed threshold: spans ineligible only due to baseline final-evaluation failure (never-finalized batch convention)
RELAXABLE = {"baseline_run_failed_final_evaluator", "baseline_goal_state_failed"}
#: agent output directory name
AGENT_DIR = "langgraph"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--spans", required=True)
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    ap.add_argument("--trace-id", default="", help="trace filename substring")
    ap.add_argument("--n-spans", type=int, default=0, help="0 = all")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--identity-rounds", type=int, default=0)
    ap.add_argument("--baseline-rounds", type=int, default=0)
    ap.add_argument("--block-seed", type=int, default=0)
    ap.add_argument("--mode", default="both",
                    choices=["both", "slm", "identity", "baseline"])
    ap.add_argument("--fallback-mode", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--relax-baseline-pass", action="store_true",
                    help="relax the baseline evaluation threshold (on by default for never-finalized batches)")
    ap.add_argument("--out", default="out_span_offload/langgraph/offload_profile.jsonl")
    ap.add_argument("--summary",
                    default="out_span_offload/langgraph/offload_profile_summary.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # judge convention: the validator LLM judge uses the judge section of the
    # offload-side config (capture/offload sides are separate; the server cannot access $TRACE_COLLECT/config.json)
    langgraph_env.set_judge_config(cfg)
    langgraph_env.judge_selfcheck()
    slm = make_client(cfg.slm)
    baseline = make_client(cfg.baseline_llm)
    model_id = slm.name
    fallback_mode = args.fallback_mode or cfg.replay.fallback_mode

    manifest = load_manifest(args.manifest)
    spans_all = load_spans(args.spans)
    # filter by the raw dicts before deserializing (unitized batches can reach
    # millions of lines; a full from_dict costs tens of minutes + several GB of memory per run)
    selected = [s for s in spans_all
                if not args.trace_id or args.trace_id in s.get("trace_path", "")]
    sampling_meta = None
    if args.n_spans > 0:
        selected = selected[: args.n_spans]
        sampling_meta = {"method": "first_n", "n": args.n_spans}
    todo = [from_dict(SpanInstance, s) for s in selected if s.get("eligible")]
    if args.relax_baseline_pass:
        relaxed = [s for s in selected
                   if not s.get("eligible")
                   and set(s.get("ineligibility_reasons") or [])
                   .issubset(RELAXABLE)]
        todo = todo + [from_dict(SpanInstance, s) for s in relaxed]
        print("relax-baseline-pass: +%d relaxed spans (total %d)" % (
            len(relaxed), len(todo)), flush=True)
    n_slm_rounds = args.rounds if args.mode in ("both", "slm") else 0
    n_id_rounds = args.identity_rounds if args.mode in ("both", "identity") else 0
    n_b_rounds = args.baseline_rounds if args.mode in ("both", "baseline") else 0
    print("span count: %d x (%d slm rounds + %d baseline rounds + %d identity rounds,"
          "mode=%s,seed=%d),SLM=%s baseline=%s" % (
              len(todo), n_slm_rounds, n_b_rounds, n_id_rounds, args.mode,
              args.block_seed, model_id, baseline.name), flush=True)

    # output dir: <parent of --out>/offload/ (consistent with the other agents)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "offload")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    args.summary = os.path.join(out_dir, os.path.basename(args.summary))
    if args.resume and os.path.exists(args.out):
        _migrate_to_task_files(args.out, out_dir, stem)
    resume_recs = defaultdict(list)
    if args.resume:
        import glob as _glob
        n_resume = 0
        for p in _glob.glob(os.path.join(out_dir, "%s_*.jsonl" % stem)):
            n_resume += _load_resume(p, resume_recs)
        print("resume: loaded %d records (%d spans)" % (n_resume, len(resume_recs)),
              flush=True)

    adapter = LangGraphTraceAdapter(manifest)
    traces = {}
    for s in todo:
        key = (s.trace_path, s.task_id)
        if key not in traces:
            traces[key] = adapter.load_unit(s.trace_path, s.task_id)

    runtime = LangGraphRuntime(manifest)
    engine = SingleSpanReplayEngine(
        cfg, runtime, baseline_client=baseline,
        slm_clients={model_id: slm}, fallback_mode=fallback_mode)

    per_span = []
    mode = "a" if args.resume else "w"
    f = _TaskMultiplexer(out_dir, stem, mode)
    skipped_empty = 0
    for si, span in enumerate(todo):
        trace = traces[(span.trace_path, span.task_id)]
        # units whose baseline did not answer (empty answer, recursion cutoff)
        # stay out of the offload pool: with no valid baseline reference, "can it pass" is unknowable -> not a valid offload experiment unit
        lg = trace.artifacts.get("langgraph") or {}
        rec_ans = (lg.get("answers") or [{}])[0]
        if not str(rec_ans.get("predicted_answer", "") or "").strip():
            skipped_empty += 1
            continue
        span_rec = next(s for s in trace.spans
                        if s.invocation_index == span.invocation_index)
        base_parsed = span_rec.baseline_response.parsed
        groups = {"slm": {"strict": [], "eq": [], "llm": [], "tools": []},
                  "baseline": {"strict": [], "eq": [], "llm": [], "tools": []},
                  "identity": {"strict": [], "eq": [], "llm": [], "tools": []}}
        plan = (["I"] * n_id_rounds + ["B"] * n_b_rounds
                + ["T_span"] * n_slm_rounds)
        import hashlib
        stable = int(hashlib.md5(
            span.span_id.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
        rng = random.Random(args.block_seed + stable)
        rng.shuffle(plan)
        total_rounds = len(plan)
        plan_keys = set(zip(plan, range(total_rounds)))
        loaded = [(a, r, st, eq, nl, nt)
                  for a, r, st, eq, nl, nt
                  in resume_recs.get(span.span_id, [])
                  if (a, r) in plan_keys]
        done_keys = set()
        for a, r, st, eq, nl, nt in sorted(loaded, key=lambda x: x[1]):
            g = groups[{"T_span": "slm", "B": "baseline", "I": "identity"}[a]]
            g["strict"].append(st)
            g["eq"].append(eq)
            g["llm"].append(nl)
            g["tools"].append(nt)
            done_keys.add((a, r))
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
            record.replay_meta["eligibility_relaxed"] = (
                args.relax_baseline_pass
                and not any(s.get("span_id") == span.span_id and s.get("eligible")
                            for s in spans_all))
            f.write(record.task_id,
                    json.dumps(to_dict(record), ensure_ascii=False) + "\n")
            f.flush(record.task_id)
            g = groups["slm" if arm == "T_span"
                       else ("baseline" if arm == "B" else "identity")]
            g["strict"].append(bool(record.strict_success))
            g["eq"].append(bool((record.local_outcome.equivalence or {})
                                .get("passed")))
            n_llm, n_tools, _ = _trace_stats(record)
            g["llm"].append(n_llm)
            g["tools"].append(n_tools)
            print("span %d/%d [%s inv=%d] %-8s round %d/%d: strict=%s eq=%s "
                  "(llm=%d tools=%d)" % (
                      si + 1, len(todo), span.span_descriptor.node_pattern,
                      span.invocation_index, arm, rnd + 1, total_rounds,
                      g["strict"][-1], g["eq"][-1], n_llm, n_tools),
                  flush=True)
        entry = {
            "task_id": span.task_id, "span_id": span.span_id,
            "invocation_index": span.invocation_index,
            "node_pattern": span.span_descriptor.node_pattern,
            "control_role": span.span_descriptor.control_role,
            "rounds": n_slm_rounds, "identity_rounds": n_id_rounds,
            "baseline_rounds": n_b_rounds, "block_seed": args.block_seed,
            "mode": args.mode,
        }
        for key, gkey in (("", "slm"), ("identity", "identity"),
                          ("baseline", "baseline")):
            g = groups[gkey]
            if not g["strict"]:
                continue
            sub = {
                "strict_seq": g["strict"],
                "strict_rate": sum(g["strict"]) / max(1, len(g["strict"])),
                "eq_rate": sum(g["eq"]) / max(1, len(g["eq"])),
                "avg_llm_steps": sum(g["llm"]) / max(1, len(g["llm"])),
                "avg_tool_executions": sum(g["tools"]) / max(1, len(g["tools"])),
            }
            if key:
                entry[key] = sub
            else:
                entry.update(sub)
        per_span.append(entry)
    if skipped_empty:
        print("skipped empty-baseline units: %d" % skipped_empty, flush=True)
    f.close()
    with open(args.summary, "w", encoding="utf-8") as sf:
        json.dump(per_span, sf, ensure_ascii=False, indent=1)
    print()
    print("details: %s | summary: %s" % (out_dir, args.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
