#!/usr/bin/env python
"""Batch 3 confirm-set span selection (deterministic, zero randomness).

Picks --n representative spans from the completed screening data (48 fully
paired spans), covering the plan §12.3 dimensions:
  - node_pattern quotas (control roles): tool_selection /
    tool_argument_generation / final_synthesis / other
  - T_span−B effect direction: positive / negative / zero (within a group, sort by effect and take both extremes + uniform middle picks)
  - task diversity: at most 2 per task
Determinism guarantee: all sort keys are deterministic, same input = same output.

Output:
  - <out>  confirm-set spans file (for run_offload_profile --spans)
  - <out>.selection.json  selection report (effect/variance/quota basis per span)

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/utils/select_confirmation_spans.py \
      --spans out_span_offload/spans_simplified_qwen27b.jsonl \
      --records out_span_offload/offload_profile_simplified_qwen3_8-27b_Qwen3_8-27B_Qwen_Qwen3_5-9B_dedup.jsonl \
      --out out_span_offload/spans_simplified_qwen27b_confirm18.jsonl \
      --n 18
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def per_span_profiles(records_path: str) -> dict:
    """Per-span screening profile: effect (T−B strict), B-group tool-count coefficient of variation.

    records_path supports three forms (use the latter two when output is per-task files without a total file):
      single file path / glob pattern / directory (all *.jsonl under it)"""
    import glob as _glob
    if os.path.isdir(records_path):
        paths = sorted(_glob.glob(os.path.join(records_path, "*.jsonl")))
    elif any(ch in records_path for ch in "*?["):
        paths = sorted(_glob.glob(records_path))
    else:
        paths = [records_path]
    by = defaultdict(lambda: defaultdict(list))
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                r = json.loads(l)
                by[(r["task_id"], r["invocation_index"])][
                    r["replay_meta"].get("arm")].append(r)
    out = {}
    for (t, i), arms in by.items():
        st = {a: sum(bool(r["strict_success"]) for r in arms[a]) / 5
              for a in ("T_span", "B", "I")}
        b_tools = [sum(1 for e in r["downstream_trace"]["spans"]
                       if e.get("span_type") == "tool.execution")
                   for r in arms["B"]]
        cv = statistics.pstdev(b_tools) / max(1, statistics.mean(b_tools))
        out[(t, i)] = {
            "task": t, "inv": i, "eff": st["T_span"] - st["B"],
            "sT": st["T_span"], "sB": st["B"], "sI": st["I"],
            "cv_tools": cv,
        }
    return out


def pick_group(items: list, quota: int) -> list:
    """Deterministic in-group selection: sort by effect, take lowest, highest
    + uniform middle picks, then adjust by task diversity (at most 2 per task)."""
    items = sorted(items, key=lambda x: (x["eff"], x["task"], x["inv"]))
    if len(items) <= quota:
        return items
    picks = []
    idx = {0}
    step = (len(items) - 1) / max(1, quota - 1)
    for k in range(quota):
        idx.add(round(k * step))
    for j in sorted(idx):
        picks.append(items[j])
    # task diversity: when a task exceeds 2, move the extras to the nearest unpicked span
    task_count = Counter(p["task"] for p in picks)
    picked_tasks = {p["task"] for p in picks}
    taken = {(p["task"], p["inv"]) for p in picks}
    for j in range(len(picks)):
        p = picks[j]
        if task_count[p["task"]] > 2:
            alt = next((x for x in items
                        if (x["task"], x["inv"]) not in taken
                        and task_count[x["task"]] < 2), None)
            if alt is None:
                continue
            task_count[p["task"]] -= 1
            task_count[alt["task"]] += 1
            taken.discard((p["task"], p["inv"]))
            taken.add((alt["task"], alt["inv"]))
            picks[j] = alt
    return picks


def select_confirmation(profiles: dict, node_map: dict, n: int,
                        quotas: dict) -> list:
    """Core confirm-set sampling (deterministic, reused by CLI and run_offload_profile).

    profiles: {(task, inv): {eff, sT, sB, sI, cv_tools, task, inv}}
    node_map: {(task, inv): node_pattern}
    quotas:   {node_pattern: quota}, sum required == n
    Returns the picked key list sorted by (task, inv)."""
    by_node = defaultdict(list)
    for key, p in profiles.items():
        p = dict(p)
        p["node"] = node_map.get(key, "other")
        by_node[p["node"]].append(p)
    picked = []
    for np_, quota in quotas.items():
        if np_ == "tool_argument_generation" and quota >= 2:
            # split the execution-span quota between read and write (write
            # takes the larger share): fixes the historical selection bias (taking by inv first yields almost all read spans)
            wq = (quota + 1) // 2
            rq = quota - wq
            picked.extend(pick_group(by_node.get(np_ + "_write", []), wq))
            picked.extend(pick_group(by_node.get(np_ + "_read", []), rq))
        else:
            picked.extend(pick_group(by_node.get(np_, []), quota))
    return sorted((p["task"], p["inv"]) for p in picked)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", required=True, help="original spans jsonl")
    ap.add_argument("--records", required=True, help="screening dedup records jsonl")
    ap.add_argument("--out", required=True, help="confirm-set spans output")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--quota", default="6,6,4,2",
                    help="node_pattern quotas (ts,fs,tag,other), comma-separated")
    ap.add_argument("--split-into", type=int, default=0,
                    help="split the picked spans into N task-balanced parts (for parallel submission)")
    ap.add_argument("--split-prefix", default="",
                    help="split file prefix, outputs <prefix>.part1..N.jsonl")
    args = ap.parse_args()

    quotas = dict(zip(
        ("tool_selection", "final_synthesis",
         "tool_argument_generation", "other"),
        (int(x) for x in args.quota.split(","))))
    assert sum(quotas.values()) == args.n, "quota sum must equal --n"

    profiles = per_span_profiles(args.records)
    spans = [json.loads(l) for l in open(args.spans, encoding="utf-8")
             if l.strip()]
    spans_by_key = {(s["task_id"], s["invocation_index"]): s for s in spans}
    node_map = {}
    for k, s in spans_by_key.items():
        np_ = s["span_descriptor"]["node_pattern"]
        if np_ == "tool_argument_generation":
            se = s["span_descriptor"]["mechanism_tags"].get("state_effect", [])
            np_ += "_write" if "external_state_mutation" in se else "_read"
        node_map[k] = np_

    out_keys = select_confirmation(profiles, node_map, args.n, quotas)
    print("picked %d spans (%d tasks)" % (
        len(out_keys), len(set(t for t, _ in out_keys))))
    with open(args.out, "w", encoding="utf-8") as f:
        for t, i in sorted(out_keys):
            f.write(json.dumps(spans_by_key[(t, i)], ensure_ascii=False) + "\n")
    report = {"n": len(out_keys), "rule": "eff-extremes+uniform, task≤2",
              "quota": quotas,
              "spans": [profiles[k] for k in out_keys]}
    with open(args.out + ".selection.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("confirm-set spans: %s" % args.out)
    print("selection report: %s.selection.json" % args.out)
    # task-balanced split (for parallel submission): tasks sorted by span count desc, greedily packed into the least-loaded bucket
    if args.split_into > 0:
        if not args.split_prefix:
            raise SystemExit("--split-into requires --split-prefix")
        tasks = defaultdict(list)
        for t, i in out_keys:
            tasks[t].append(spans_by_key[(t, i)])
        task_list = sorted(tasks.items(),
                           key=lambda kv: -len(kv[1]))
        buckets = [[] for _ in range(args.split_into)]
        loads = [0] * args.split_into
        for t, ss in task_list:
            k = min(range(args.split_into), key=lambda x: loads[x])
            buckets[k].append(t)
            loads[k] += len(ss)
        for bi in range(args.split_into):
            if not buckets[bi]:
                continue
            p = "%s.part%d.jsonl" % (args.split_prefix, bi + 1)
            with open(p, "w", encoding="utf-8") as f:
                for t in sorted(buckets[bi]):
                    for s in tasks[t]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print("  part%d: %d spans (%d tasks) -> %s" % (
                bi + 1, loads[bi], len(buckets[bi]), p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
