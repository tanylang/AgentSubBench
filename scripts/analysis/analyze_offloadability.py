#!/usr/bin/env python
"""Downstream response variable analysis for span offloading (22 variables + group discriminability test).

Computes the downstream behavior variable vector for each span from
offload_profile*.jsonl (replay records of both the SLM arm and the identity arm), and tests
each variable's discriminability for the offload intervention by comparing arms (SLM vs identity).

Event mapping (downstream_trace.spans compressed into symbol strings):
  D decision = llm.call with tool_calls
  R thinking = llm.call with text only (idle round)
  O observe  = tool.execution ok=True and tool is read_only
  A action   = tool.execution ok=True and it is a write operation (including complete_task)
  F failure  = tool.execution ok=False (hallucinated execution)
  Rp repair  = source == "repair"

Usage (server, inside container):
  python scripts/analyze_offloadability.py \
      --inputs out_span_offload/offload_profile*.jsonl \
      --traces-dir check_point_traces/appworld_simplified_batch1 \
      --out out_span_offload/offloadability_variables.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from span_offload.appworld_adapter import (  # noqa: E402
    APWORLD_TOOL_EFFECTS, AppWorldTraceAdapter,
)
from span_offload.local_equivalence import compare_values  # noqa: E402


def _effect_of(name: str) -> str:
    key = str(name or "").replace("__", ".")
    return APWORLD_TOOL_EFFECTS.get(key, "external_state_mutation")


def _events(record: dict) -> list:
    """Compress the downstream trajectory of one record into an event symbol string.

    Note: harness repair events (completing a correction receipt for an invalid tool name) also
    carry span_type=="tool.execution" with ok=False; source must be checked before the tool
    branch, otherwise they would be misrecorded as F (hallucinated failure).
    """
    evs = []
    for e in (record.get("downstream_trace") or {}).get("spans", []):
        if e.get("source") == "repair":
            evs.append("Rp")
        elif e.get("span_type") == "llm.call":
            out = e.get("output") or {}
            evs.append("D" if out.get("tool_calls") else "R")
        elif e.get("span_type") == "tool.execution":
            if not e.get("ok"):
                evs.append("F")
            elif _effect_of(e.get("tool", "")) == "read_only":
                evs.append("O")
            else:
                evs.append("A")
    return evs


def _transitions(evs: list) -> Counter:
    return Counter((evs[i], evs[i + 1]) for i in range(len(evs) - 1))


def _max_fail_run(evs: list) -> int:
    best = cur = 0
    for e in evs:
        cur = cur + 1 if e == "F" else 0
        best = max(best, cur)
    return best


def round_variables(record: dict, base: dict) -> dict:
    """Downstream variables for a single round (one record); base is the baseline remaining info for the span."""
    evs = _events(record)
    n = Counter(evs)
    tr = _transitions(evs)
    nD, nR, nO, nA, nF, nRp = (n[x] for x in ("D", "R", "O", "A", "F", "Rp"))
    n_llm = nD + nR
    n_tools = nO + nA + nF

    # duplicate call rate: tool calls whose (name, args) appear >=2 times
    call_keys = []
    for e in (record.get("downstream_trace") or {}).get("spans", []):
        if e.get("span_type") != "tool.execution":
            continue
        call_keys.append((str(e.get("tool")), json.dumps(
            e.get("arguments") or {}, sort_keys=True)))
    dup = sum(1 for k, c in Counter(call_keys).items() if c > 1)

    # structural similarity (needs baseline remaining info)
    cf_names = [k[0] for k in call_keys]
    base_names = base.get("remaining_tools", [])
    base_mult = Counter(base_names)
    cf_mult = Counter(cf_names)
    inter = sum((cf_mult & base_mult).values())
    union = sum((cf_mult | base_mult).values())
    jaccard = inter / union if union else 1.0
    steps_ratio = n_llm / max(base.get("remaining_steps", 1), 1)

    # argument match rate: pair with baseline remaining calls by name and compare arguments
    base_calls = base.get("remaining_calls", [])
    unmatched = list(base_calls)
    matched = 0
    denom = 0
    for e in (record.get("downstream_trace") or {}).get("spans", []):
        if e.get("span_type") != "tool.execution" or not e.get("ok"):
            continue
        name = str(e.get("tool", ""))
        if name not in base_mult:
            continue
        denom += 1
        args = e.get("arguments") or {}
        for bi, bc in enumerate(unmatched):
            if str(bc.get("name", "")) == name and compare_values(
                    args, bc.get("arguments"))[0]:
                matched += 1
                unmatched.pop(bi)
                break
    param_match = matched / max(denom, 1)

    # key chain coverage: LCS ratio between the baseline key chain (first call name per step) and the counterfactual name sequence
    chain = base.get("key_chain", [])
    if chain:
        dp = [[0] * (len(cf_names) + 1) for _ in range(len(chain) + 1)]
        for i in range(1, len(chain) + 1):
            for j in range(1, len(cf_names) + 1):
                dp[i][j] = (dp[i - 1][j - 1] + 1 if chain[i - 1] == cf_names[j - 1]
                            else max(dp[i - 1][j], dp[i][j - 1]))
        key_chain = dp[len(chain)][len(cf_names)] / len(chain)
    else:
        key_chain = 1.0

    # novel info rate: fraction of calls whose result text (hash of first 120 chars) appears for the first time
    seen = set()
    novel = 0
    n_obs = 0
    for e in (record.get("downstream_trace") or {}).get("spans", []):
        if e.get("span_type") != "tool.execution" or not e.get("ok"):
            continue
        n_obs += 1
        h = hash(str(e.get("result", ""))[:120])
        if h not in seen:
            seen.add(h)
            novel += 1
    novel_rate = novel / max(n_obs, 1)

    # control flow deviation: whether the injection used a terminating tool (complete_task) and differs from the baseline
    inj_calls = [str(c.get("name", "")) for c in
                 (record.get("slm_output") or {}).get("injected_tool_calls", [])]
    base_calls = [str(c.get("name", "")) for c in
                  (record.get("slm_output") or {}).get("baseline_tool_calls", [])]
    inj_term = any(n.endswith("complete_task") for n in inj_calls)
    base_term = any(n.endswith("complete_task") for n in base_calls)
    res = record.get("resource") or {}
    return {
        "tools": n_tools,
        "llm": n_llm,
        "hall": nF / max(n_tools, 1),
        "strict": int(bool(record.get("strict_success"))),
        # scale/granularity
        "ActionsPerDecision": (nA + nF) / max(nD, 1),
        "ObservePerDecision": nO / max(nD, 1),
        "WriteRatio": nA / max(nA + nO, 1),
        "ReasonOnlyRate": nR / max(n_llm, 1),
        # sequence rhythm
        "DirectActionRate": tr[("D", "A")] / max(nD, 1),
        "ThinkThenObserve": tr[("D", "O")] / max(nD, 1),
        "ObserveChaining": tr[("O", "O")] / max(nO, 1),
        "ObsBeforeAction": tr[("O", "A")] / max(nA, 1),
        "FailChaining": tr[("F", "F")] / max(nF, 1),
        "FailRetryRate": tr[("F", "D")] / max(nF, 1),
        "MaxFailRun": _max_fail_run(evs),
        # repair/redundancy/semantics
        "RepairRate": nRp / max(n_tools, 1),
        "DupCallRate": dup / max(n_tools, 1),
        "ParamMatchRate": param_match,
        "KeyChainCoverage": key_chain,
        "NovelInfoRate": novel_rate,
        # structural similarity
        "ToolSeqJaccard": jaccard,
        "StepsRatio": steps_ratio,
        # cost
        "TokenDelta": res.get("token_delta", 0),
        "ControlFlowDeviation": int(inj_term != base_term),
    }


#: Reduced variable set (binned by function). Only these are output by default; --all outputs all 22.
#: Pruning criteria: no discriminability (group CI crosses 0 and Δ≈0), no variance (constant within group),
#: redundant with retained variables.
KEPT_VARIABLES = {
    # core response variables: carry the detectable signal of "offload impact on downstream" (dependent variables of H2/H3)
    "ParamMatchRate": "core_response",
    "ControlFlowDeviation": "core_response",
    "KeyChainCoverage": "core_response",
    "DirectActionRate": "core_response",
    "WriteRatio": "core_response",
    # coarse-grained baseline control: not used for testing, used for reported comparison and sanity check
    "tools": "baseline",
    "llm": "baseline",
    "hall": "baseline",
    "strict": "baseline",
    # cost
    "TokenDelta": "cost",
    # background descriptive variables: describe the spiral phenomenon itself (also present in the identity arm), not used as offload response variables
    "FailChaining": "background",
    "MaxFailRun": "background",
    "FailRetryRate": "background",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="result files (glob supported), grouped automatically by the injection field")
    ap.add_argument("--traces-dir", default="check_point_traces/appworld_simplified_batch1")
    ap.add_argument("--out", default="out_span_offload/offloadability_variables.json")
    ap.add_argument("--all", action="store_true",
                    help="output all 22 variables (default outputs the 12-variable reduced set)")
    args = ap.parse_args()

    files = []
    for pat in args.inputs:
        files.extend(glob.glob(pat))
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            records.extend(json.loads(l) for l in fh if l.strip())

    S = defaultdict(list)
    I = defaultdict(list)
    for r in records:
        k = (r["task_id"], r["invocation_index"])
        (I if r["replay_meta"].get("injection") == "identity" else S)[k].append(r)

    # baseline remaining info (per span: remaining steps / tool multiset / call details / key chain)
    adapter = AppWorldTraceAdapter()
    base = {}
    for f in glob.glob(os.path.join(args.traces_dir, "*.json")):
        tr = adapter.load(f)
        tid = tr.run_meta.task_id
        for j, s in enumerate(tr.spans):
            chain = []
            for x in tr.spans[j:]:
                cs = x.baseline_response.parsed.get("tool_calls", [])
                if cs:
                    chain.append(cs[0]["name"])
            all_calls = [{"name": q["name"], "arguments": q.get("arguments")}
                         for x in tr.spans[j:]
                         for q in x.baseline_response.parsed.get("tool_calls", [])]
            base[(tid, s.invocation_index)] = {
                "remaining_steps": len(tr.spans) - j,
                "remaining_tools": [c["name"] for c in all_calls],
                "remaining_calls": all_calls,
                "key_chain": chain,
            }

    common = sorted(set(S) & set(I) & set(base))
    print("paired span count: %d (SLM arm %d rounds / identity arm %d rounds)" % (
        len(common), sum(len(S[k]) for k in common), sum(len(I[k]) for k in common)))

    # variable vectors of the two arms per span
    var_names = None
    span_rows = []
    for k in common:
        sv = [round_variables(r, base[k]) for r in S[k]]
        iv = [round_variables(r, base[k]) for r in I[k]]
        if var_names is None:
            all_names = sorted(sv[0].keys())
            var_names = all_names if args.all else [v for v in all_names
                                                    if v in KEPT_VARIABLES]
        row = {"task": k[0], "inv": k[1]}
        for v in var_names:
            row["s_" + v] = statistics.mean(x[v] for x in sv)
            row["i_" + v] = statistics.mean(x[v] for x in iv)
        row["sigma_tools"] = statistics.pstdev(x["tools"] for x in sv) if len(sv) > 1 else 0.0
        span_rows.append(row)

    # group discriminability: paired Δ per variable + bootstrap CI
    random.seed(42)
    print("\n=== variable discriminability (SLM vs identity, paired bootstrap) ===")
    print(f"{'variable':<22} {'role':<14} {'μ_slm':>8} {'μ_id':>8} {'Δ':>8} {'95%CI':>18} {'sig':<6}")
    per_var = {}
    for v in var_names:
        diffs = [row["s_" + v] - row["i_" + v] for row in span_rows]
        mu = statistics.mean(diffs)
        boots = sorted(statistics.mean(random.choices(diffs, k=len(diffs)))
                       for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        sig = "significant" if (lo > 0 or hi < 0) else "n.s."
        role = KEPT_VARIABLES.get(v, "pruned")
        per_var[v] = {"role": role,
                      "mu_slm": statistics.mean(row["s_" + v] for row in span_rows),
                      "mu_id": statistics.mean(row["i_" + v] for row in span_rows),
                      "delta": mu, "ci": [lo, hi], "significant": sig}
        print(f"{v:<22} {role:<14} {per_var[v]['mu_slm']:>8.3f} {per_var[v]['mu_id']:>8.3f} "
              f"{mu:>+8.3f} [{lo:>+8.3f},{hi:>+8.3f}] {sig:<6}")

    # high-variance span ranking
    hi_var = sorted(span_rows, key=lambda r: -r["sigma_tools"])[:8]
    print("\n=== top 8 spans by σ_tools (suggested for extra rounds) ===")
    for r in hi_var:
        print("  %s inv=%d σ=%.1f(SLM tools μ=%.1f)" % (
            r["task"], r["inv"], r["sigma_tools"], r["s_tools"]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"per_span": span_rows, "per_variable": per_var}, f,
                  ensure_ascii=False, indent=1)
    print("\nDetails: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
