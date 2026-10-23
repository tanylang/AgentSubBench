#!/usr/bin/env python
"""Response analysis: extract intervention response conclusions from existing offload_profile data.

Computed from 43 spans with clean identity baselines (SLM×5 vs identity×5), per the research
plan (H1-H4, A-F response classes, §26 open questions):

  S1 data inventory
  S2 global paired effect (clean 43 spans)
  S3 per-span standardized effect size d distribution
  S4 H1 stability (σ_slm vs σ_id, strict across-round stability)
  S5 H2 structure (node_pattern / control_role stratification + permutation test)
  S6 A-F response classification + structure crosstab
  S7 variance decomposition (natural stochasticity vs intervention effect)
  S8 verifier / repair mechanism activity
  S9 sensitivity: clean vs contaminated baseline
  S10 derived Offloadability view

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/response_analysis.py
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import analyze_offloadability as AO  # noqa: E402  (reuse round_variables)

CLEAN_ID_FILES = [
    # 2026-09-07 remote rerun (identity, MODE=identity, 6 spans × 5 rounds)
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_21abae1_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_1.jsonl",
    # 23cf851_2 was not rerun; keep using the fixed clean identity
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_2.jsonl",
]
SLM_BATCH1 = ["results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl"]
NEW_TASK_FILES = [
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl",
]
CONTAMINATED = "results/offload_profile_id_contaminated_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl"
TRACES_DIR = "check_point_traces/appworld_simplified_batch1"


def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs.extend(json.loads(l) for l in f if l.strip())
    return recs


def main() -> int:
    random.seed(42)

    # ---------- S1 data inventory ----------
    slm = defaultdict(list)
    id_clean = defaultdict(list)
    id_cont = defaultdict(list)
    for r in load(SLM_BATCH1):            # batch-1 SLM (old format, no injection field)
        slm[(r["task_id"], r["invocation_index"])].append(r)
    for r in load(NEW_TASK_FILES):
        k = (r["task_id"], r["invocation_index"])
        (slm if r["replay_meta"].get("injection") != "identity" else id_clean)[k].append(r)
    for r in load(CLEAN_ID_FILES):
        id_clean[(r["task_id"], r["invocation_index"])].append(r)
    for r in load([CONTAMINATED]):
        id_cont[(r["task_id"], r["invocation_index"])].append(r)

    clean_keys = sorted(k for k in slm if k in id_clean)
    cont_only = sorted(k for k in slm if k not in id_clean)
    print("S1 inventory: SLM covers %d spans; clean identity paired %d spans; contaminated-baseline-only %d spans"
          % (len(slm), len(clean_keys), len(cont_only)))

    # ---------- baseline remaining info (raw trace remaining per span) ----------
    from span_offload.appworld_adapter import AppWorldTraceAdapter  # noqa: E402
    adapter = AppWorldTraceAdapter()
    base = {}
    import glob
    for f in glob.glob(os.path.join(TRACES_DIR, "*.json")):
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
    missing = [k for k in clean_keys if k not in base]
    if missing:
        print("Warning: %d spans lack baseline remaining info and will be skipped: %s" % (len(missing), missing))
    clean_keys = [k for k in clean_keys if k in base]

    # ---------- per-span variables ----------
    CORE = ["tools", "llm", "hall", "strict", "ParamMatchRate", "KeyChainCoverage",
            "ControlFlowDeviation", "DirectActionRate", "WriteRatio", "RepairRate",
            "TokenDelta", "StepsRatio"]
    span_vars = {}   # k -> {"s": {var:[...]}, "i": {var:[...]}}
    verifier = {}    # k -> {"s": [...], "i": [...]}
    for k in clean_keys:
        sv = defaultdict(list)
        iv = defaultdict(list)
        vs, vi = [], []
        for r in slm[k]:
            v = AO.round_variables(r, base[k])
            for name, val in v.items():
                sv[name].append(val)
            vs.append(int(bool((r.get("fallback") or {}).get("verifier_triggered"))))
        for r in id_clean[k]:
            v = AO.round_variables(r, base[k])
            for name, val in v.items():
                iv[name].append(val)
            vi.append(int(bool((r.get("fallback") or {}).get("verifier_triggered"))))
        span_vars[k] = {"s": sv, "i": iv}
        verifier[k] = {"s": vs, "i": vi}

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    def pvar(xs):
        return statistics.pvariance(xs) if len(xs) > 1 else 0.0

    # ---------- S2 global paired effects ----------
    print("\nS2 global paired effects (clean %d spans, paired at span-level means)" % len(clean_keys))
    print("%-22s %9s %9s %+9s  %s" % ("variable", "μ_slm", "μ_id", "Δ", "95%CI(bootstrap)"))
    for v in CORE:
        diffs = [mean(span_vars[k]["s"][v]) - mean(span_vars[k]["i"][v]) for k in clean_keys]
        mu = mean(diffs)
        boots = sorted(mean(random.choices(diffs, k=len(diffs))) for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        sig = "significant" if (lo > 0 or hi < 0) else "n.s."
        print("%-22s %9.3f %9.3f %+9.3f  [%+.3f, %+.3f] %s"
              % (v, mean(mean(span_vars[k]["s"][v]) for k in clean_keys),
                 mean(mean(span_vars[k]["i"][v]) for k in clean_keys), mu, lo, hi, sig))

    # ---------- S3 per-span standardized effect size d ----------
    print("\nS3 per-span standardized effect size d = (μ_slm−μ_id)/σ_id (clean %d spans)" % len(clean_keys))
    D_VARS = ["tools", "llm", "hall", "strict", "ParamMatchRate", "KeyChainCoverage",
              "DirectActionRate", "WriteRatio"]
    d_map = {}
    for v in D_VARS:
        vals = []
        for k in clean_keys:
            s = span_vars[k]["s"][v]
            i = span_vars[k]["i"][v]
            sd = pvar(i) ** 0.5
            if sd == 0:
                d = 0.0 if mean(s) == mean(i) else float("inf")
            else:
                d = (mean(s) - mean(i)) / sd
            vals.append(d)
        d_map[v] = vals
        fin = [d for d in vals if d != float("inf")]
        n_exc = sum(1 for d in fin if abs(d) > 1)
        print("  %-22s |d|>1: %2d/%-2d (%.0f%%),  d median %+.2f"
              % (v, n_exc, len(fin), 100 * n_exc / max(len(fin), 1), statistics.median(fin)))
    per_span_d = {k: {"tools": mean(span_vars[k]["s"]["tools"]) - mean(span_vars[k]["i"]["tools"]),
                      "llm": mean(span_vars[k]["s"]["llm"]) - mean(span_vars[k]["i"]["llm"]),
                      "strict": mean(span_vars[k]["s"]["strict"]) - mean(span_vars[k]["i"]["strict"])}
                  for k in clean_keys}

    # ---------- S4 H1 stability ----------
    print("\nS4 H1 stability (SLM between-round variance vs identity between-round variance)")
    ratios = []
    for k in clean_keys:
        ss = pvar(span_vars[k]["s"]["tools"])
        si = pvar(span_vars[k]["i"]["tools"])
        if si > 0:
            ratios.append((ss / si) ** 0.5)
    print("  σ_slm/σ_id (tools) median: %.2f (n=%d), spans with >2: %d"
          % (statistics.median(ratios), len(ratios), sum(1 for r in ratios if r > 2)))
    # strict across-round stability
    n_stable = 0
    for k in clean_keys:
        seq = span_vars[k]["s"]["strict"]
        n_stable += int(len(set(seq)) == 1)
    print("  spans with SLM strict identical across all 5 rounds: %d/%d" % (n_stable, len(clean_keys)))

    # ---------- S5 H2 structure ----------
    print("\nS5 H2 structure (stratified by node_pattern / control_role)")
    from collections import defaultdict as dd
    def desc(k):
        r = slm[k][0]
        return (r["span_descriptor"]["node_pattern"],
                r["span_descriptor"]["control_role"])
    by_pattern = dd(list)
    by_role = dd(list)
    for k in clean_keys:
        np_, cr = desc(k)
        by_pattern[np_].append(per_span_d[k]["tools"])
        by_role[cr].append(per_span_d[k]["tools"])
    for g, d in sorted(by_pattern.items()):
        print("  %-28s n=%2d  Δtools mean %+8.2f median %+8.2f" % (g, len(d), mean(d), statistics.median(d)))
    for g, d in sorted(by_role.items()):
        print("  %-28s n=%2d  Δtools mean %+8.2f median %+8.2f" % (g, len(d), mean(d), statistics.median(d)))
    # Permutation test: whether Δtools has the same distribution across node_pattern groups (permutation version of Kruskal-Wallis)
    groups = list(by_pattern.values())
    obs = _kw_stat(groups)
    allv = [x for g in groups for x in g]
    cnt = 0
    for _ in range(5000):
        random.shuffle(allv)
        i = 0
        gs = []
        for g in groups:
            gs.append(allv[i:i + len(g)])
            i += len(g)
        if _kw_stat(gs) >= obs:
            cnt += 1
    print("  node_pattern permutation test: H=%.3f, p=%.4f" % (obs, cnt / 5000))
    groups = list(by_role.values())
    obs = _kw_stat(groups)
    allv = [x for g in groups for x in g]
    cnt = 0
    for _ in range(5000):
        random.shuffle(allv)
        i = 0
        gs = []
        for g in groups:
            gs.append(allv[i:i + len(g)])
            i += len(g)
        if _kw_stat(gs) >= obs:
            cnt += 1
    print("  control_role permutation test: H=%.3f, p=%.4f" % (obs, cnt / 5000))

    # ---------- S6 A-F response classification ----------
    print("\nS6 A-F response classification (operationalized per research plan §6, clean %d spans)" % len(clean_keys))
    def classify(k):
        sv, iv = span_vars[k]["s"], span_vars[k]["i"]
        strict_s = mean(sv["strict"])
        strict_i = mean(iv["strict"])
        eq_s = mean(sv["ParamMatchRate"])  # proxy for semantic/argument consistency
        eq_low = eq_s < 0.6
        sd = pvar(iv["tools"]) ** 0.5
        if sd > 0:
            d_tools = (mean(sv["tools"]) - mean(iv["tools"])) / sd
        else:
            # zero-variance baseline + unequal means = strongest deviation signal, consistent with S3, treated as over threshold
            d_tools = float("inf") if mean(sv["tools"]) != mean(iv["tools"]) else 0.0
        sd = pvar(iv["llm"]) ** 0.5
        if sd > 0:
            d_llm = (mean(sv["llm"]) - mean(iv["llm"])) / sd
        else:
            d_llm = float("inf") if mean(sv["llm"]) != mean(iv["llm"]) else 0.0
        v_s, v_i = mean(verifier[k]["s"]), mean(verifier[k]["i"])
        r_s, r_i = mean(sv["RepairRate"]), mean(iv["RepairRate"])
        if strict_s < strict_i:
            return "F 放大失败"
        if (v_s > v_i + 0.01) or (r_s > r_i + 0.02):
            return "C 纠正性响应"
        if strict_s == 1.0 and (d_tools > 1 or d_llm > 1):
            return "D 代价适应(成功)"
        if strict_s == 1.0 and abs(d_tools) <= 1 and abs(d_llm) <= 1 and not eq_low:
            return "A 透明响应"
        if strict_s == 1.0:
            return "B 良性分歧(语义变化、结构不变)"
        return "E 基线不稳(不可归因)"
    classes = Counter(classify(k) for k in clean_keys)
    for c, n in classes.most_common():
        print("  %-28s %d" % (c, n))
    print("  --- crosstab: response class × node_pattern ---")
    tab = dd(Counter)
    for k in clean_keys:
        np_, _ = desc(k)
        tab[np_][classify(k)] += 1
    for np_, cnt in sorted(tab.items()):
        print("  %-28s %s" % (np_, dict(cnt)))
    # Permutation test: whether class F (amplified failure) concentrates in tool_argument_generation
    obs_f = sum(1 for k in clean_keys if classify(k).startswith("F"))
    obs_f_local = sum(1 for k in clean_keys
                      if classify(k).startswith("F") and desc(k)[0] == "tool_argument_generation")
    n_local = sum(1 for k in clean_keys if desc(k)[0] == "tool_argument_generation")
    # Fisher exact (one-sided): >= obs_f_local F's drawn among n_local local spans
    import math
    def nCk(n, k):
        return math.comb(n, k) if 0 <= k <= n else 0
    p_val = 0.0
    N, K, n = len(clean_keys), obs_f, n_local
    for x in range(obs_f_local, min(K, n) + 1):
        p_val += nCk(n, x) * nCk(N - n, K - x) / nCk(N, K)
    print("  Fisher (class F concentrated in tool_argument_generation): %d/%d F's all in that class, p=%.4f"
          % (obs_f_local, obs_f, p_val))

    # ---------- S7 variance decomposition ----------
    print("\nS7 variance decomposition (tools variable)")
    within = mean([(pvar(span_vars[k]["s"]["tools"]) + pvar(span_vars[k]["i"]["tools"])) / 2
                   for k in clean_keys])
    span_means = [mean(span_vars[k]["s"]["tools"]) for k in clean_keys] + \
                 [mean(span_vars[k]["i"]["tools"]) for k in clean_keys]
    deltas = [mean(span_vars[k]["s"]["tools"]) - mean(span_vars[k]["i"]["tools"]) for k in clean_keys]
    v_interv = pvar(deltas)
    print("  within-span (natural noise) σ² = %.1f → σ = %.1f" % (within, within ** 0.5))
    print("  intervention effect (Δ across spans) σ² = %.1f → σ = %.1f" % (v_interv, v_interv ** 0.5))
    print("  intervention effect SD / natural noise SD = %.2f" % ((v_interv ** 0.5) / (within ** 0.5)))
    grand = mean(deltas)
    se = (v_interv / len(deltas)) ** 0.5
    print("  global Δtools = %+.1f ± %.1f (SE)" % (grand, se))

    # ---------- S8 verifier / repair ----------
    print("\nS8 repair mechanism activity (SLM vs identity)")
    for name, get in [("verifier_triggered", lambda k: verifier[k]),
                      ("repair", lambda k: None)]:
        if get(clean_keys[0]) is None:
            pass
    v_s = mean([mean(verifier[k]["s"]) for k in clean_keys])
    v_i = mean([mean(verifier[k]["i"]) for k in clean_keys])
    r_s = mean([mean(span_vars[k]["s"]["RepairRate"]) for k in clean_keys])
    r_i = mean([mean(span_vars[k]["i"]["RepairRate"]) for k in clean_keys])
    print("  verifier trigger rate: SLM %.3f vs identity %.3f" % (v_s, v_i))
    print("  repair rate:      SLM %.3f vs identity %.3f" % (r_s, r_i))

    # ---------- S9 sensitivity: contaminated baseline version ----------
    print("\nS9 sensitivity (old contaminated baseline vs new clean baseline, global Δ)")
    for v in ["tools", "llm", "strict"]:
        diffs_cont = []
        for k in sorted(slm.keys()):
            # Contaminated version: spans with contaminated records prefer the contaminated baseline (the mixed state before the rerun)
            src = id_cont.get(k) or id_clean.get(k)
            if not src or k not in base:
                continue
            s_m = mean(AO.round_variables(r, base[k])[v] for r in slm[k])
            i_m = mean(AO.round_variables(r, base[k])[v] for r in src)
            diffs_cont.append(s_m - i_m)
        if diffs_cont:
            print("  Δ%s (old contaminated baseline, n=%d): %+8.2f" % (v, len(diffs_cont), mean(diffs_cont)))
        diffs_clean = [mean(span_vars[k]["s"][v]) - mean(span_vars[k]["i"][v]) for k in clean_keys]
        print("  Δ%s (new clean baseline, n=%d): %+8.2f" % (v, len(diffs_clean), mean(diffs_clean)))

    # ---------- S10 derived offloadability ----------
    print("\nS10 derived offloadability (strict not reduced + downstream delta within natural fluctuation → offloadable candidates)")
    ok = [k for k in clean_keys
          if mean(span_vars[k]["s"]["strict"]) >= mean(span_vars[k]["i"]["strict"])
          and abs(d_map["tools"][clean_keys.index(k)]) <= 1
          and abs(d_map["llm"][clean_keys.index(k)]) <= 1]
    print("  offloadable candidates: %d/%d (%.0f%%)" % (len(ok), len(clean_keys), 100 * len(ok) / len(clean_keys)))
    for k in ok:
        print("    %s inv=%d [%s]" % (k[0], k[1], desc(k)[0]))

    out = {
        "clean_span_count": len(clean_keys),
        "per_span": [{"task": k[0], "inv": k[1],
                      "node_pattern": desc(k)[0], "control_role": desc(k)[1],
                      "d_tools": per_span_d[k]["tools"], "d_llm": per_span_d[k]["llm"],
                      "d_strict": per_span_d[k]["strict"],
                      "response_class": classify(k)} for k in clean_keys],
        "response_classes": dict(classes),
        "variance": {"within": within, "intervention": v_interv},
    }
    with open("results/appworld_53span/response_analysis_clean53.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/appworld_53span/response_analysis_clean53.json")
    return 0


def _kw_stat(groups):
    """Kruskal-Wallis H (ties use mean-rank approximation)."""
    allv = [x for g in groups for x in g]
    n = len(allv)
    order = sorted(range(n), key=lambda i: allv[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        r = (i + 1 + j + 1) / 2.0
        for t in range(i, j + 1):
            ranks[order[t]] = r
        i = j + 1
    H = 0.0
    idx = 0
    for g in groups:
        m = len(g)
        if m == 0:
            continue
        rg = sum(ranks[idx:idx + m]) / m
        H += m * (rg - (n + 1) / 2.0) ** 2
        idx += m
    return H * 12.0 / (n * (n + 1))


if __name__ == "__main__":
    raise SystemExit(main())
