#!/usr/bin/env python
"""Cost decomposition (plan v0.5 §9): cost-vector decomposition of B (27B) vs identity (deepseek).

Metrics (§9.3-9.5):
  - Direct Saving     = C_target^B − C_target^SLM (hypothetical, converted via SLM price list)
  - Cost_comp         = C_down^B − C_down^I (downstream compensation cost, same-caliber tokens)
  - StepAppetite      = E[N_llm_calls] (number of downstream LLM rounds)
  - ContextLoad       = target prompt + downstream prompt (cumulative prefill area)
  - tool behavior     = execution count / failure count / repair count
  - rho_comp          = Cost_comp / Saving_direct (>1 = negative offload economics)

Caliber notes:
  - B (27B) downstream tokens: real usage (383cbac_1/3, 4ec8de5_1) + offline calibration
    reconstruction (the other 4 tasks, marked via replay_meta.token_reconstruction)
  - identity downstream tokens: reconstructed the same way (cl100k raw estimate; deepseek has
    no real usage calibration source, used as a same-caliber yardstick, marked est)
  - price list (conversion illustration only, not a deployment price): baseline=0.00022/0.00066,
    SLM=0.0003/0.0012 (27B has no local price list)

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/cost_decomposition.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from reconstruct_downstream_tokens import estimate  # noqa: E402

B_FILES = [
    "out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_21abae1_1.jsonl",
    "out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_23cf851_1.jsonl",
    "out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_23cf851_2.jsonl",
    "out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_23cf851_3.jsonl",
    "out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_383cbac_1.jsonl",
    "out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_383cbac_3.jsonl",
    "out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl",
]
# identity (deepseek downstream) clean53 I rounds
I_FILES = [
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_21abae1_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_2.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl",
]
# Public price list (USD/1M tokens, queried 2026-09-13, see comments for sources; rate =
# 7.14 CNY per 1 USD):
#   deepseek-v4-flash: DeepSeek official API price effective 2026-09-10 (off-peak)
#     ¥1 input (cache miss) / ¥0.02 input (cache hit) / ¥4 output
#   Qwen3.8-27B: Alibaba Cloud Model Studio official (Singapore USD page)
#     $0.424 input / $0.085 implicit cache hit / $1.696 output
#   Qwen3.5-9B: Alibaba Cloud Model Studio official (USD page, official endpoint price recorded by token-calculator)
#     $0.15 input / $0.6 output
PRICING = {
    "deepseek-v4-flash": {
        "source": "DeepSeek official API, effective 2026-09-10 (off-peak)",
        "in_miss": 0.14, "in_hit": 0.003, "out": 0.56,
    },
    "Qwen3.8-27B": {
        "source": "Alibaba Model Studio official (Singapore USD)",
        "in_miss": 0.424, "in_hit": 0.085, "out": 1.696,
    },
    "Qwen3.5-9B": {
        "source": "Alibaba Model Studio official (USD)",
        "in_miss": 0.15, "in_hit": 0.15, "out": 0.6,
    },
}
# Deployment note: the 27B in this B batch is self-hosted on local vLLM; the public price list is used as a rental-equivalent reference
# (opportunity cost); local electricity/depreciation is computed separately and not mixed in.



def rec_metrics(r: dict) -> dict:
    dt = r.get("downstream_trace") or {}
    spans = dt.get("spans") or []
    n_llm = sum(1 for e in spans if e.get("span_type") == "llm.call")
    n_tool = sum(1 for e in spans if e.get("span_type") == "tool.execution")
    n_fail = sum(1 for e in spans
                 if e.get("span_type") == "tool.execution" and not e.get("ok"))
    n_repair = sum(1 for e in spans if e.get("source") == "repair")
    res = r.get("resource") or {}
    return {
        "task": r.get("task_id"), "inv": r.get("invocation_index"),
        "node": (r.get("span_descriptor") or {}).get("node_pattern"),
        "strict": bool(r.get("strict_success")),
        "n_llm": n_llm, "n_tool": n_tool, "n_fail": n_fail,
        "n_repair": n_repair,
        "target_p": res.get("target_prompt_tokens", 0) or 0,
        "target_c": res.get("target_completion_tokens", 0) or 0,
        "down_p": res.get("downstream_prompt_tokens", 0) or 0,
        "down_c": res.get("downstream_completion_tokens", 0) or 0,
        "context_load": res.get("context_load", 0) or 0,
    }


def load(arm: str):
    files = B_FILES if arm == "B" else I_FILES
    out = []
    for p in files:
        with open(p, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                r = json.loads(l)
                if arm == "I" and r.get("replay_meta", {}).get("injection") != "identity":
                    continue
                m = rec_metrics(r)
                if arm == "I":
                    # identity: no target generation; downstream tokens reconstructed the same way (cl100k est)
                    ep, ec, _n = estimate(r)
                    m["down_p"], m["down_c"] = ep, ec
                    m["target_p"], m["target_c"] = 0, 0
                    m["context_load"] = ep
                else:
                    # B: B vs I comparison uses the same ruler (cl100k raw estimate); real /
                    # calibrated tokens saved separately, absolute caliber reported separately
                    ep, ec, _n = estimate(r)
                    m["est_down_p"], m["est_down_c"] = ep, ec
                m["arm"] = arm
                out.append(m)
    return out


def stats(vals):
    if not vals:
        return dict(mean=0, median=0, p95=0, max=0)
    s = sorted(vals)
    return dict(mean=statistics.mean(vals), median=statistics.median(vals),
                p95=s[int(len(s) * 0.95) - 1], max=s[-1])


def usd(p_tok, c_tok, model, cache_aware=False):
    """Convert to USD per round using the public price list. With cache_aware=True, input is
    charged at the cache-hit price (the Agent loop resends the full session every round, most continued-round inputs hit cache)"""
    pr = PRICING[model]
    p_in = pr["in_hit"] if cache_aware else pr["in_miss"]
    return p_tok / 1e6 * p_in + c_tok / 1e6 * pr["out"]


def main() -> int:
    B = load("B")
    I = load("I")
    print("record count: B=%d I=%d" % (len(B), len(I)))

    def agg(recs):
        return {
            "n": len(recs),
            "strict": sum(r["strict"] for r in recs) / max(1, len(recs)),
            "n_llm": stats([r["n_llm"] for r in recs]),
            "n_tool": stats([r["n_tool"] for r in recs]),
            "n_fail": stats([r["n_fail"] for r in recs]),
            "target_p": stats([r["target_p"] for r in recs]),
            "target_c": stats([r["target_c"] for r in recs]),
            "down_p": stats([r["down_p"] for r in recs]),
            "down_c": stats([r["down_c"] for r in recs]),
            "context_load": stats([r["context_load"] for r in recs]),
        }

    aB, aI = agg(B), agg(I)
    print()
    print("=" * 74)
    print("Cost decomposition: B (27B) vs identity (deepseek downstream) — 53 spans × 5 rounds")
    print("=" * 74)
    print("%-22s %18s %18s" % ("", "B(27B)", "identity(deepseek)"))
    rows = [
        ("strict success rate", aB["strict"], aI["strict"], "%"),
        ("StepAppetite (llm rounds) mean", aB["n_llm"]["mean"], aI["n_llm"]["mean"], ""),
        ("  p95", aB["n_llm"]["p95"], aI["n_llm"]["p95"], ""),
        ("tool execution mean", aB["n_tool"]["mean"], aI["n_tool"]["mean"], ""),
        ("tool failure mean", aB["n_fail"]["mean"], aI["n_fail"]["mean"], ""),
        ("target prompt mean", aB["target_p"]["mean"], aI["target_p"]["mean"], "tok"),
        ("target completion mean", aB["target_c"]["mean"], aI["target_c"]["mean"], "tok"),
        ("downstream prompt mean", aB["down_p"]["mean"], aI["down_p"]["mean"], "tok"),
        ("downstream completion mean", aB["down_c"]["mean"], aI["down_c"]["mean"], "tok"),
        ("ContextLoad mean", aB["context_load"]["mean"], aI["context_load"]["mean"], "tok"),
        ("ContextLoad p95", aB["context_load"]["p95"], aI["context_load"]["p95"], "tok"),
        ("ContextLoad max", aB["context_load"]["max"], aI["context_load"]["max"], "tok"),
    ]
    for name, b, i, u in rows:
        if u == "%":
            print("%-22s %17.0f%% %17.0f%%" % (name, 100 * b, 100 * i))
        elif u == "tok":
            print("%-22s %17.0f %17.0f" % (name, b, i))
        else:
            print("%-22s %17.1f %17.1f" % (name, b, i))

    # ---- economic accounting (§9.4-9.5, public price list 2026-09-13 edition) ----
    print()
    print("---- economic accounting (public price list 2026-09-13 edition, USD/round) ----")
    c_t_b = usd(aB["target_p"]["mean"], aB["target_c"]["mean"], "Qwen3.8-27B")
    c_t_slm = usd(aB["target_p"]["mean"], aB["target_c"]["mean"], "Qwen3.5-9B")
    saving = c_t_b - c_t_slm
    # same caliber: both B and I use cl100k raw estimates
    est_b_p = statistics.mean(r.get("est_down_p", 0) for r in B)
    est_b_c = statistics.mean(r.get("est_down_c", 0) for r in B)
    est_i_p = statistics.mean(r["down_p"] for r in I)
    est_i_c = statistics.mean(r["down_c"] for r in I)
    c_d_b_real = usd(aB["down_p"]["mean"], aB["down_c"]["mean"], "Qwen3.8-27B")
    c_d_b_hit = usd(aB["down_p"]["mean"], aB["down_c"]["mean"], "Qwen3.8-27B",
                    cache_aware=True)
    c_d_i_miss = usd(est_i_p, est_i_c, "deepseek-v4-flash")
    c_d_i_hit = usd(est_i_p, est_i_c, "deepseek-v4-flash", cache_aware=True)
    cost_comp_miss = c_d_b_real - c_d_i_miss
    cost_comp_hit = c_d_b_hit - c_d_i_hit
    print("C_target^B   = $%.4f/round (Qwen3.8-27B public price)" % c_t_b)
    print("C_target^SLM = $%.4f/round (Qwen3.5-9B public price)" % c_t_slm)
    print("Saving_direct = $%.4f/round" % saving)
    print("C_down^B(27B)= $%.4f/round cache-miss | $%.4f/round cache-aware" % (
        c_d_b_real, c_d_b_hit))
    print("C_down^I(ds) = $%.4f/round cache-miss | $%.4f/round cache-aware" % (
        c_d_i_miss, c_d_i_hit))
    print("Cost_comp(B−I)= $%.4f/round miss | $%.4f/round cache-aware" % (
        cost_comp_miss, cost_comp_hit))
    print("rho_comp     = %.1f (miss) | %.1f (cache-aware)  >1 = negative offload economics"
          % (cost_comp_miss / saving if saving > 0 else float("inf"),
             cost_comp_hit / saving if saving > 0 else float("inf")))

    # ---- StepAppetite and ContextLoad by node_pattern / task ----
    print()
    print("---- by node_pattern (B) ----")
    by = defaultdict(list)
    for r in B:
        by[r["node"]].append(r)
    for k in sorted(by):
        g = agg(by[k])
        print("  %-26s n=%2d strict=%.0f%% llm mean=%4.1f ContextLoad mean=%7.0f" % (
            k, g["n"], 100 * g["strict"], g["n_llm"]["mean"],
            g["context_load"]["mean"]))
    print("---- by task (B) ----")
    by = defaultdict(list)
    for r in B:
        by[r["task"]].append(r)
    for k in sorted(by):
        g = agg(by[k])
        print("  %-10s n=%2d strict=%.0f%% llm mean=%4.1f ContextLoad mean=%7.0f p95=%7.0f" % (
            k, g["n"], 100 * g["strict"], g["n_llm"]["mean"],
            g["context_load"]["mean"], g["context_load"]["p95"]))

    # ---- span-level pairing: downstream token amplification ratio B vs I ----
    print()
    print("---- span level: downstream token amplification ratio (B/I, paired by task+inv, est caliber) ----")
    pairs = defaultdict(lambda: {"B": [], "I": []})
    for r in B:
        pairs[(r["task"], r["inv"])]["B"].append(
            (r.get("est_down_p", 0) or 0) + (r.get("est_down_c", 0) or 0))
    for r in I:
        pairs[(r["task"], r["inv"])]["I"].append(r["down_p"] + r["down_c"])
    ratios = []
    for (t, i), v in sorted(pairs.items()):
        if not v["B"] or not v["I"]:
            continue
        ratio = statistics.mean(v["B"]) / max(1, statistics.mean(v["I"]))
        ratios.append(ratio)
        print("  %-10s inv=%2d  B=%7.0f  I=%7.0f  amplification ×%.1f" % (
            t, i, statistics.mean(v["B"]), statistics.mean(v["I"]), ratio))
    if ratios:
        print("  median amplification: ×%.1f | max: ×%.1f" % (
            statistics.median(ratios), max(ratios)))

    out = {"B_agg": aB, "I_agg": aI, "economic": {
        "c_target_B": c_t_b, "c_target_SLM": c_t_slm,
        "saving_direct": saving,
        "c_down_B_miss": c_d_b_real, "c_down_B_hit": c_d_b_hit,
        "c_down_I_miss": c_d_i_miss, "c_down_I_hit": c_d_i_hit,
        "cost_comp_miss": cost_comp_miss, "cost_comp_hit": cost_comp_hit,
        "rho_comp_miss": cost_comp_miss / saving if saving > 0 else None,
        "rho_comp_hit": cost_comp_hit / saving if saving > 0 else None,
        "pricing": PRICING,
        "price_note": "公开价目表 2026-09-13 版,USD/1M tokens;"
                      "deepseek 按 1USD=7.14CNY 换算 off-peak;"
                      "27B 本批为本地 vLLM 自托管,公开价作为租赁等价参照"},
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "cost_decomposition_b27b.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/appworld_53span/cost_decomposition_b27b.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
