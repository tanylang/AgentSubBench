#!/usr/bin/env python
"""Control-flow metric supplement: impact of span offloading on downstream control flow (53 paired spans).

Adds 6 control-flow variables (all computed from the tool execution sequence of downstream_trace):
  1. BranchWidth        branch width: number of distinct successful tool names
  2. ToolEntropy        normalized Shannon entropy of the tool name distribution (exploration diversity)
  3. RevisitRate        revisit rate: fraction of calls to "previously called tool names"
  4. SameToolPersistence consecutive same-tool retry rate (adjacent calls to the same tool)
  5. AbnormalExit       abnormal exit: the last tool execution of the trajectory is not complete_task
  6. FirstCompleteNormPos normalized position of the first complete_task (0=never committed to termination)

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/controlflow_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SLM_FILES = [
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl",
]
ID_FILES = [
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_21abae1_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_2.jsonl",
]


def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs.extend(json.loads(l) for l in f if l.strip())
    return recs


def cf_vars(record: dict, base_names=None) -> dict:
    """Control-flow variables for one record. base_names = baseline remaining tool name set (optional,
    used by NewBranchRate / MissingBranchRate)."""
    spans = (record.get("downstream_trace") or {}).get("spans", [])
    tools = [e for e in spans if e.get("span_type") == "tool.execution"]
    names = [str(e.get("tool", "")) for e in tools]
    ok_names = [str(e.get("tool", "")) for e in tools if e.get("ok")]
    n = len(names)

    # 1. BranchWidth: number of distinct successful tool names
    bw = len(set(ok_names))

    # 2. ToolEntropy: normalized entropy of the successful tool name distribution (0=single tool, 1=uniform)
    if ok_names:
        cnt = Counter(ok_names)
        H = -sum((c / len(ok_names)) * math.log(c / len(ok_names))
                 for c in cnt.values())
        ent = H / math.log(len(cnt)) if len(cnt) > 1 else 0.0
    else:
        ent = 0.0

    # 3. RevisitRate: fraction of calls to tool names called before (backtracking/looping)
    seen = set()
    rev = 0
    for nm in names:
        if nm in seen:
            rev += 1
        seen.add(nm)
    revisit = rev / max(n, 1)

    # 4. SameToolPersistence: fraction of adjacent call pairs to the same tool (consecutive retries)
    persist = sum(1 for i in range(n - 1) if names[i] == names[i + 1]) \
        / max(n - 1, 1)

    # 5. AbnormalExit: the last tool execution is not complete_task (1=abnormal exit)
    abnormal = 1 if not (names and names[-1].endswith("complete_task")) else 0

    # 6. FirstCompleteNormPos: position of first complete_task / total length (0=never appears)
    fcp = next((i + 1 for i, nm in enumerate(names)
                if nm.endswith("complete_task")), None)
    first_pos = fcp / n if fcp else 0.0

    # 7. NewBranchRate: fraction of successful calls whose tool name the baseline never called (new-branch exploration)
    # 8. MissingBranchRate: fraction of baseline remaining tool names not covered by successful calls
    if base_names:
        newb = sum(1 for nm in ok_names if nm not in base_names) / max(n, 1)
        missb = 1 - len(set(ok_names) & set(base_names)) / max(len(set(base_names)), 1)
    else:
        newb = float("nan")
        missb = float("nan")

    # 9. BatchDecisionRate: fraction of decisions with ≥2 tool calls (parallelism)
    nb = nbatch = 0
    for e in spans:
        if e.get("span_type") == "llm.call":
            calls = (e.get("output") or {}).get("tool_calls") or []
            if calls:
                nb += 1
                if len(calls) >= 2:
                    nbatch += 1
    batch = nbatch / max(nb, 1)

    # 10. DeadEndRate: P(next is a failure | current is a decision)
    evs = []
    for e in spans:
        if e.get("source") == "repair":
            evs.append("Rp")
        elif e.get("span_type") == "llm.call":
            evs.append("D" if (e.get("output") or {}).get("tool_calls") else "R")
        elif e.get("span_type") == "tool.execution":
            evs.append("F" if not e.get("ok") else "T")
    tr = Counter((evs[i], evs[i + 1]) for i in range(len(evs) - 1))
    dead = tr[("D", "F")] / max(evs.count("D"), 1)

    # 11. FirstWriteNormPos: normalized position of the first write operation (complete_task)
    fp = next((i + 1 for i, nm in enumerate(names)
               if nm.endswith("complete_task")), None)
    fw = fp / n if fp else 0.0

    # 12. LoopPatternCount: count of repeated subsequence patterns (X,Y,X,Y and X,Y,X)
    lp = sum(1 for i in range(len(names) - 3)
             if names[i] and names[i] == names[i + 2]
             and names[i + 1] == names[i + 3])
    tri = sum(1 for i in range(len(names) - 2)
               if names[i] and names[i] == names[i + 2])
    loops = lp + tri

    return {"BranchWidth": bw, "ToolEntropy": ent, "RevisitRate": revisit,
            "SameToolPersistence": persist, "AbnormalExit": abnormal,
            "FirstCompleteNormPos": first_pos,
            "NewBranchRate": newb, "MissingBranchRate": missb,
            "BatchDecisionRate": batch, "DeadEndRate": dead,
            "FirstWriteNormPos": fw, "LoopPatternCount": loops}


def main() -> int:
    random.seed(42)
    slm = defaultdict(list)
    idd = defaultdict(list)
    for r in load(SLM_FILES + ID_FILES):
        k = (r["task_id"], r["invocation_index"])
        (idd if r.get("replay_meta", {}).get("injection") == "identity"
         else slm)[k].append(r)
    keys = sorted(set(slm) & set(idd))
    print("paired span count: %d" % len(keys))

    # baseline remaining tool name sets (needed by NewBranch/MissingBranch)
    from span_offload.appworld_adapter import AppWorldTraceAdapter
    adapter = AppWorldTraceAdapter()
    base = {}
    import glob as _glob
    for f in _glob.glob("check_point_traces/appworld_simplified_batch1/*.json"):
        tr = adapter.load(f)
        tid = tr.run_meta.task_id
        for j, s in enumerate(tr.spans):
            base[(tid, s.invocation_index)] = set(
                q["name"] for x in tr.spans[j:]
                for q in x.baseline_response.parsed.get("tool_calls", []))

    VARS = ["BranchWidth", "ToolEntropy", "RevisitRate",
            "SameToolPersistence", "AbnormalExit", "FirstCompleteNormPos",
            "NewBranchRate", "MissingBranchRate", "BatchDecisionRate",
            "DeadEndRate", "FirstWriteNormPos", "LoopPatternCount"]
    span_means = {}
    for k in keys:
        sv = {v: [cf_vars(r, base[k])[v] for r in slm[k]] for v in VARS}
        iv = {v: [cf_vars(r, base[k])[v] for r in idd[k]] for v in VARS}
        span_means[k] = (sv, iv)

    print("\n=== control-flow variable discriminability (SLM vs identity, paired bootstrap) ===")
    print("%-22s %9s %9s %+9s  %s" % ("variable", "μ_slm", "μ_id", "Δ", "95%CI"))
    for v in VARS:
        diffs = [statistics.mean(span_means[k][0][v])
                 - statistics.mean(span_means[k][1][v]) for k in keys]
        mu = statistics.mean(diffs)
        boots = sorted(statistics.mean(random.choices(diffs, k=len(diffs)))
                       for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        sig = "significant" if (lo > 0 or hi < 0) else "n.s."
        print("%-22s %9.3f %9.3f %+9.3f  [%+.3f, %+.3f] %s" % (
            v, statistics.mean(statistics.mean(span_means[k][0][v]) for k in keys),
            statistics.mean(statistics.mean(span_means[k][1][v]) for k in keys),
            mu, lo, hi, sig))

    print("\n=== per-span standardized effect size d (fraction with |d|>1) ===")
    for v in VARS:
        vals = []
        for k in keys:
            iv = span_means[k][1][v]
            sd = statistics.pstdev(iv) if len(iv) > 1 else 0.0
            d = (statistics.mean(span_means[k][0][v]) - statistics.mean(iv)) / sd \
                if sd > 0 else 0.0
            if sd > 0:
                vals.append(d)
        n_exc = sum(1 for d in vals if abs(d) > 1)
        print("  %-22s |d|>1: %2d/%-2d (%.0f%%)" % (v, n_exc, len(vals),
                                                    100 * n_exc / max(len(vals), 1)))

    # association with response classes: per-class means
    classes = {r["task"] + "__" + str(r["inv"]): r["response_class"]
               for r in json.load(open("results/appworld_53span/response_analysis_clean53.json",
                                       encoding="utf-8"))["per_span"]}
    print("\n=== control-flow variables × response class (SLM group means) ===")
    by_cls = defaultdict(list)
    for k in keys:
        by_cls[classes.get(k[0] + "__" + str(k[1]), "?")].append(k)
    for cls in sorted(by_cls):
        g = by_cls[cls]
        row = []
        for v in ["RevisitRate", "SameToolPersistence", "AbnormalExit",
                  "FirstCompleteNormPos"]:
            row.append("%.2f" % statistics.mean(
                statistics.mean(span_means[k][0][v]) for k in g))
        print("  %-24s n=%2d | Revisit %s | Persist %s | AbnExit %s | 1stComplete %s"
              % (cls, len(g), *row))

    out = {"per_span": [
        {"task": k[0], "inv": k[1],
         "class": classes.get(k[0] + "__" + str(k[1]), "?"),
         **{("s_" + v): statistics.mean(span_means[k][0][v]) for v in VARS},
         **{("i_" + v): statistics.mean(span_means[k][1][v]) for v in VARS}}
        for k in keys]}
    with open("results/appworld_53span/controlflow_53.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/appworld_53span/controlflow_53.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
