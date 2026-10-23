"""Canonical implementation of the control-flow metrics master table (the single
code source for CONTROL_FLOW_METRICS.md).

Layer structure (spec doc L0-L5 + baseline/semantic/cost):
  L0  event encoding: downstream trace compressed event-by-event into a symbol string (D/R/O/A/F/Rp)
  L1  path and branch structure (which paths were taken)
  L2  decision rhythm (how decisions are made)
  L3  loops/backtracking/retries (going over things again)
  L4  failure spirals (failure dynamics)
  L5  termination/exit/state writes (how it ends)
  baseline  scale comparison (tools/llm/hall/strict)
  semantic  semantics (ParamMatchRate/NovelInfoRate)
  cost      cost (TokenDelta)

scripts/analyze_offloadability.py (22 variables) and scripts/controlflow_analysis.py
(12 control-flow-specific) are historical implementations; new analysis goes through this module.
Convention-difference fixes: controlflow_analysis's DeadEndRate event encoding (D/R/F/T) aligned to the L0 spec;
NovelInfoRate uses md5 not hash() (reproducible across processes); redundancy-analysis conclusions baked into METRICS tiers (dropped=deleted, derived=derived quantity).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .appworld_adapter import APWORLD_TOOL_EFFECTS, AppWorldTraceAdapter
from .local_equivalence import compare_values

# ---------------------------------------------------------------------------
# Metric registry: name -> (layer, tier, definition)
# tier: core=core response variable; boundary=boundary-significant; background=background description;
#       descriptor=mechanism descriptor; derived=derived quantity; dropped=deleted by redundancy analysis;
#       baseline=coarse-grained comparison; cost=cost
# ---------------------------------------------------------------------------
METRICS: Dict[str, Tuple[str, str, str]] = {
    # ---- L1 path and branch structure ----
    "KeyChainCoverage": ("L1", "core",
                         "LCS coverage of the baseline key chain (first tool name of each step) in the counterfactual sequence"),
    "ToolSeqJaccard": ("L1", "background",
                       "tool-name multiset Jaccard (counterfactual vs baseline remaining)"),
    "BranchWidth": ("L1", "background", "|distinct successful tool names| (fan-out)"),
    "MissingBranchRate": ("L1", "core",
                          "proportion of baseline remaining tool names not covered by successful calls"),
    "NewBranchRate": ("L1", "background",
                      "share of successful calls whose tool names the baseline never called"),
    "StepsRatio": ("L1", "background", "downstream LLM steps / baseline remaining steps"),
    # ---- L2 decision rhythm ----
    "DirectActionRate": ("L2", "core", "P(A|D): direct action after a decision"),
    "BatchDecisionRate": ("L2", "background", "share of decisions with ≥2 tool calls"),
    "ActionsPerDecision": ("L2", "background", "(nA+nF)/nD"),
    "ObservePerDecision": ("L2", "background", "nO/nD"),
    "ReasonOnlyRate": ("L2", "core", "nR/n_llm (share of idle turns)"),
    "ThinkThenObserve": ("L2", "background", "P(O|D)"),
    "ObserveChaining": ("L2", "background", "P(O|O)"),
    "ObsBeforeAction": ("L2", "background", "P(O|A)"),
    "ToolEntropy": ("L2", "core", "normalized Shannon entropy of the successful tool-name distribution (exploration diversity)"),
    # ---- L3 loops/backtracking/retries ----
    "RevisitRate": ("L3", "descriptor",
                    "share of calls to \"tool names already called before\" (strong by-category gradient, n.s. globally)"),
    "SameToolPersistence": ("L3", "descriptor",
                            "P(name_t+1 = name_t) (failure-class signature)"),
    "DupCallRate": ("L3", "core", "duplicate-call rate for identical (tool, arguments)"),
    "LoopPatternCount": ("L3", "dropped",
                         "count of repeated subsequence patterns (X,Y,X,Y and X,Y,X) — deleted by redundancy analysis"),
    "FailRetryRate": ("L3", "background", "P(D|F)"),
    "DeadEndRate": ("L3", "derived", "P(F|D)≈1−DirectActionRate (derived quantity)"),
    # ---- L4 failure spirals ----
    "hall": ("L4", "background", "share of failed executions (nF/n_tools)"),
    "FailChaining": ("L4", "background", "P(F|F)"),
    "MaxFailRun": ("L4", "background", "longest consecutive failure run"),
    "RepairRate": ("L4", "descriptor", "share of protocol repair events (4-6× on C-class spans)"),
    # ---- L5 termination/exit/state writes ----
    "ControlFlowDeviation": ("L5", "core",
                             "whether terminal-tool usage diverges from the baseline (0/1)"),
    "AbnormalExit": ("L5", "boundary", "last event is not complete_task (1=abnormal termination)"),
    "FirstCompleteNormPos": ("L5", "boundary",
                             "first complete_task position / total length (0=never committed to termination)"),
    "FirstWriteNormPos": ("L5", "dropped",
                          "first write position / total length — deleted by redundancy analysis (≡FirstComplete)"),
    "WriteRatio": ("L5", "boundary", "share of writes among (writes + observations)"),
    # ---- coarse-grained baseline comparison ----
    "tools": ("baseline", "baseline", "downstream tool execution count"),
    "llm": ("baseline", "baseline", "downstream LLM turn count"),
    "strict": ("baseline", "baseline", "strict_success(0/1)"),
    # ---- semantics ----
    "ParamMatchRate": ("semantic", "core",
                       "argument match rate among calls paired by name with baseline remaining calls"),
    "NovelInfoRate": ("semantic", "background",
                      "share of calls whose result text (first 120 chars, md5) appears for the first time"),
    # ---- cost ----
    "TokenDelta": ("cost", "dropped",
                   "legacy convention (mixes collection-side/live usage, target layer only) — "
                   "disabled for analysis; use NetSaving_tokens (paired per-item total token difference)"),
    # ---- utility ----
    "TotalTokens": ("utility", "cost",
                    "total tokens per round = target (prompt+completion) + downstream"
                    "(prompt+completion), itemized with the same convention"),
    "NetSavingTokens": ("utility", "dropped",
                        "per-span B−T paired difference of total tokens (positive = offload saves tokens);"
                        "computed by main_comparison at the analysis layer, not inside cf_metrics"),
}

#: Core table metrics (the retained set after redundancy analysis; see CONTROL_FLOW_METRICS.md §redundancy analysis)
CORE_TABLE = [m for m, (_, tier, _) in METRICS.items() if tier != "dropped"]


def _effect_of(name: str) -> str:
    key = str(name or "").replace("__", ".")
    return APWORLD_TOOL_EFFECTS.get(key, "external_state_mutation")


# ---------------------------------------------------------------------------
# L0 event encoding
# ---------------------------------------------------------------------------
def encode_events(record: Dict[str, Any]) -> List[str]:
    """Compress one record's downstream trace into an event symbol string (spec L0).

    Note: harness repair events (corrective receipts for illegal tool names) also
    carry span_type=="tool.execution" and ok=False; source must be checked before
    the tool branch, otherwise they are mislabeled as F (hallucinated failure)."""
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


def _transitions(evs: List[str]) -> Counter:
    return Counter((evs[i], evs[i + 1]) for i in range(len(evs) - 1))


def _max_fail_run(evs: List[str]) -> int:
    best = cur = 0
    for e in evs:
        cur = cur + 1 if e == "F" else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# Baseline remaining information (reference for L1 structural similarity metrics)
# ---------------------------------------------------------------------------
def base_info(trace: Any, j: int) -> Dict[str, Any]:
    """Baseline remaining information for span j: remaining steps / tool multiset / call details / key chain."""
    all_calls = [{"name": q["name"], "arguments": q.get("arguments")}
                 for x in trace.spans[j:]
                 for q in x.baseline_response.parsed.get("tool_calls", [])]
    chain = []
    for x in trace.spans[j:]:
        cs = x.baseline_response.parsed.get("tool_calls", [])
        if cs:
            chain.append(cs[0]["name"])
    return {
        "remaining_steps": len(trace.spans) - j,
        "remaining_tools": [c["name"] for c in all_calls],
        "remaining_calls": all_calls,
        "key_chain": chain,
    }


def build_base_map(traces_dir: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Build the baseline remaining-information map keyed by (task_id, invocation_index)."""
    adapter = AppWorldTraceAdapter()
    base = {}
    import glob as _glob
    for f in _glob.glob(os.path.join(traces_dir, "*.json")):
        tr = adapter.load(f)
        tid = tr.run_meta.task_id
        for j, s in enumerate(tr.spans):
            base[(tid, s.invocation_index)] = base_info(tr, j)
    return base


# ---------------------------------------------------------------------------
# All metrics for a single record (L1-L5 + baseline + semantic + cost)
# ---------------------------------------------------------------------------
def cf_metrics(record: Dict[str, Any],
               base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The full control-flow metric set for one record. base = base_info (baseline
    remaining information, required by L1's KeyChainCoverage/ToolSeqJaccard/
    StepsRatio/ParamMatchRate; these metrics are NaN when it is missing)."""
    spans = (record.get("downstream_trace") or {}).get("spans", [])
    evs = encode_events(record)
    n = Counter(evs)
    tr = _transitions(evs)
    nD, nR, nO, nA, nF, nRp = (n[x] for x in ("D", "R", "O", "A", "F", "Rp"))
    n_llm = nD + nR
    n_tools = nO + nA + nF

    # Tool execution sequence (names / successful names / call keys)
    tool_events = [e for e in spans if e.get("span_type") == "tool.execution"]
    names = [str(e.get("tool", "")) for e in tool_events]
    ok_names = [str(e.get("tool", "")) for e in tool_events if e.get("ok")]
    call_keys = [(str(e.get("tool", "")),
                  json.dumps(e.get("arguments") or {}, sort_keys=True))
                 for e in tool_events]

    # ---- L1 path and branch structure ----
    base_names = (base or {}).get("remaining_tools", [])
    base_mult = Counter(base_names)
    cf_mult = Counter(names)
    inter = sum((cf_mult & base_mult).values())
    union = sum((cf_mult | base_mult).values())
    jaccard = inter / union if union else 1.0
    steps_ratio = n_llm / max((base or {}).get("remaining_steps", 1), 1)
    chain = (base or {}).get("key_chain", [])
    if chain:
        dp = [[0] * (len(names) + 1) for _ in range(len(chain) + 1)]
        for i in range(1, len(chain) + 1):
            for j in range(1, len(names) + 1):
                dp[i][j] = (dp[i - 1][j - 1] + 1 if chain[i - 1] == names[j - 1]
                            else max(dp[i - 1][j], dp[i][j - 1]))
        key_chain_cov = dp[len(chain)][len(names)] / len(chain)
    else:
        key_chain_cov = 1.0
    branch_width = len(set(ok_names))
    if base_names:
        new_branch = sum(1 for nm in ok_names
                         if nm not in set(base_names)) / max(len(names), 1)
        missing_branch = 1 - len(set(ok_names) & set(base_names)) \
            / max(len(set(base_names)), 1)
    else:
        new_branch = missing_branch = float("nan")

    # ---- L2 decision rhythm ----
    batch = 0
    nb = 0
    for e in spans:
        if e.get("span_type") == "llm.call":
            calls = (e.get("output") or {}).get("tool_calls") or []
            if calls:
                nb += 1
                if len(calls) >= 2:
                    batch += 1
    batch_rate = batch / max(nb, 1)
    if ok_names:
        cnt = Counter(ok_names)
        H = -sum((c / len(ok_names)) * math.log(c / len(ok_names))
                 for c in cnt.values())
        entropy = H / math.log(len(cnt)) if len(cnt) > 1 else 0.0
    else:
        entropy = 0.0

    # ---- L3 loops/backtracking/retries ----
    seen = set()
    rev = 0
    for nm in names:
        if nm in seen:
            rev += 1
        seen.add(nm)
    revisit = rev / max(len(names), 1)
    persist = sum(1 for i in range(len(names) - 1)
                  if names[i] == names[i + 1]) / max(len(names) - 1, 1)
    dup = sum(1 for k, c in Counter(call_keys).items() if c > 1)
    loops = sum(1 for i in range(len(names) - 3)
                if names[i] and names[i] == names[i + 2]
                and names[i + 1] == names[i + 3]) \
        + sum(1 for i in range(len(names) - 2)
              if names[i] and names[i] == names[i + 2])

    # ---- L5 termination/exit/state writes ----
    abnormal = 1 if not (names and names[-1].endswith("complete_task")) else 0
    fcp = next((i + 1 for i, nm in enumerate(names)
                if nm.endswith("complete_task")), None)
    first_pos = fcp / len(names) if fcp else 0.0

    # ---- semantic: argument match rate / novel info rate ----
    base_calls = (base or {}).get("remaining_calls", [])
    unmatched = list(base_calls)
    matched = 0
    denom = 0
    for e in tool_events:
        if not e.get("ok"):
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
    param_match = matched / max(denom, 1) if base_calls else float("nan")
    seen_h = set()
    novel = 0
    n_obs = 0
    for e in tool_events:
        if not e.get("ok"):
            continue
        n_obs += 1
        h = hashlib.md5(str(e.get("result", ""))[:120].encode()).hexdigest()
        if h not in seen_h:
            seen_h.add(h)
            novel += 1
    novel_rate = novel / max(n_obs, 1)

    # ---- cost: control-flow deviation (whether terminal-tool usage diverges from the baseline) ----
    inj_calls = [str(c.get("name", "")) for c in
                 (record.get("slm_output") or {}).get("injected_tool_calls", [])]
    base_inj = [str(c.get("name", "")) for c in
                (record.get("slm_output") or {}).get("baseline_tool_calls", [])]
    inj_term = any(x.endswith("complete_task") for x in inj_calls)
    base_term = any(x.endswith("complete_task") for x in base_inj)
    res = record.get("resource") or {}

    return {
        # L1
        "KeyChainCoverage": key_chain_cov,
        "ToolSeqJaccard": jaccard,
        "BranchWidth": branch_width,
        "MissingBranchRate": missing_branch,
        "NewBranchRate": new_branch,
        "StepsRatio": steps_ratio,
        # L2
        "DirectActionRate": tr[("D", "A")] / max(nD, 1),
        "BatchDecisionRate": batch_rate,
        "ActionsPerDecision": (nA + nF) / max(nD, 1),
        "ObservePerDecision": nO / max(nD, 1),
        "ReasonOnlyRate": nR / max(n_llm, 1),
        "ThinkThenObserve": tr[("D", "O")] / max(nD, 1),
        "ObserveChaining": tr[("O", "O")] / max(nO, 1),
        "ObsBeforeAction": tr[("O", "A")] / max(nA, 1),
        "ToolEntropy": entropy,
        # L3
        "RevisitRate": revisit,
        "SameToolPersistence": persist,
        "DupCallRate": dup / max(n_tools, 1),
        "LoopPatternCount": loops,
        "FailRetryRate": tr[("F", "D")] / max(nF, 1),
        "DeadEndRate": tr[("D", "F")] / max(nD, 1),
        # L4
        "hall": nF / max(n_tools, 1),
        "FailChaining": tr[("F", "F")] / max(nF, 1),
        "MaxFailRun": _max_fail_run(evs),
        "RepairRate": nRp / max(n_tools, 1),
        # L5
        "ControlFlowDeviation": int(inj_term != base_term),
        "AbnormalExit": abnormal,
        "FirstCompleteNormPos": first_pos,
        "FirstWriteNormPos": first_pos,  # same definition and value (redundancy-analysis conclusion: keep the former)
        "WriteRatio": nA / max(nA + nO, 1),
        # baseline
        "tools": n_tools,
        "llm": n_llm,
        "strict": int(bool(record.get("strict_success"))),
        # semantic
        "ParamMatchRate": param_match,
        "NovelInfoRate": novel_rate,
        # cost
        "TokenDelta": res.get("token_delta", 0),
        # utility: total tokens per round (target + downstream itemized, same convention)
        "TotalTokens": ((res.get("target_prompt_tokens", 0) or 0)
                        + (res.get("target_completion_tokens", 0) or 0)
                        + (res.get("downstream_prompt_tokens", 0) or 0)
                        + (res.get("downstream_completion_tokens", 0) or 0)),
    }


# ---------------------------------------------------------------------------
# Paired master-table generation (SLM vs identity, span-level means + paired Δ + bootstrap CI)
# ---------------------------------------------------------------------------
def paired_span_means(records: List[Dict[str, Any]],
                      base_map: Dict[Tuple[str, int], Dict[str, Any]],
                      slm_pred=lambda r: True) -> Tuple[
        List[Tuple[str, int]], Dict[Tuple[str, int], Dict[str, List[float]]]]:
    """Group records by (task, inv) × group (SLM/identity), return
    (keys, {key: {"s": {var: [..]}, "i": {var: [..]}}})."""
    S: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    I: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        k = (r["task_id"], r["invocation_index"])
        (I if r.get("replay_meta", {}).get("injection") == "identity"
         else S)[k].append(r)
    keys = sorted(set(S) & set(I))
    out = {}
    for k in keys:
        out[k] = {
            "s": {v: [cf_metrics(r, base_map.get(k))[v] for r in S[k]]
                  for v in CORE_TABLE},
            "i": {v: [cf_metrics(r, base_map.get(k))[v] for r in I[k]]
                  for v in CORE_TABLE},
        }
    return keys, out


def paired_table(keys, span_means, variables=None, n_boot=2000,
                 seed: int = 42) -> List[Dict[str, Any]]:
    """Paired Δ + bootstrap CI + significance, returns the canonical master-table rows."""
    random.seed(seed)
    variables = variables or CORE_TABLE
    rows = []
    for v in variables:
        layer, tier, definition = METRICS[v]
        diffs = [statistics.mean(span_means[k]["s"][v])
                 - statistics.mean(span_means[k]["i"][v]) for k in keys]
        mu = statistics.mean(diffs)
        boots = sorted(statistics.mean(random.choices(diffs, k=len(diffs)))
                       for _ in range(n_boot))
        lo, hi = boots[n_boot // 40], boots[-n_boot // 40 - 1]
        rows.append({
            "metric": v, "layer": layer, "tier": tier,
            "definition": definition,
            "mu_slm": statistics.mean(
                statistics.mean(span_means[k]["s"][v]) for k in keys),
            "mu_id": statistics.mean(
                statistics.mean(span_means[k]["i"][v]) for k in keys),
            "delta": mu, "ci_lo": lo, "ci_hi": hi,
            "significant": bool(lo > 0 or hi < 0),
        })
    return rows


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    """Render the canonical master table (same format as CONTROL_FLOW_METRICS.md)."""
    lines = ["| Metric | Definition | μ_slm | μ_id | Δ [95%CI] | Verdict |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        sig = "**significant**" if r["significant"] else "n.s."
        if r["tier"] in ("boundary", "derived"):
            sig = "boundary" if r["significant"] else sig
        definition = r["definition"].replace("|", "\\|")
        lines.append("| %s | %s | %.3f | %.3f | %+.3f [%+.3f,%+.3f] | %s |" % (
            r["metric"], definition, r["mu_slm"], r["mu_id"],
            r["delta"], r["ci_lo"], r["ci_hi"], sig))
    return "\n".join(lines)
