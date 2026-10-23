#!/usr/bin/env python
"""tau2 batch span offload data analysis (three-in-one):
  1) metrics summary table: full CORE_TABLE three-arm means + paired Δ (T:B / B:I / T:I) + bootstrap CI
  2) success group vs failure group: variable profile grouped by strict (6 groups)
  3) offloadability metrics: pre-offload feature stratification × outcome gate (noise correction: two-layer
     bootstrap / Wilson UCB / MW distribution test / d_delta denoised zero-detour)
Data preparation: drop truncated rounds (missing downstream token accounting), ≥5 valid rounds per span per arm.

Known prior findings (data audit):
  - all 23 tau2 batch spans are tool_argument_generation (inside the loop) — node_pattern has no
    variance; stratification uses position / after_failure / continuous features instead
  - baseline_span_cost=0 → token caliber only
  - 4ec8de5_1 inv9 truncated-round contamination → that span is dropped after filtering (22 spans remain)
Output: results/tau2/tau2_analysis.json
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from trace_edit_distance import edit_distance, signature  # noqa: E402
from span_offload.controlflow_metrics import (  # noqa: E402
    CORE_TABLE, build_base_map, cf_metrics,
)

MIN_ROUNDS = 5
ZERO_DETOUR_DNORM = 0.1
BOOT = 2000
SEED = 42
Z = 1.644854

RECORD_DIR = "out_span_offload/tau2/screening"
TRACES_DIR = "check_point_traces/appworld_tau2_qwen3_8-27b_batch1"
SPANS_PATH = "out_span_offload/tau2/spans_tau2_qwen27b.jsonl"
OUT = "results/tau2/tau2_analysis.json"


def _total(r):
    res = r.get("resource") or {}
    return ((res.get("target_prompt_tokens", 0) or 0)
            + (res.get("target_completion_tokens", 0) or 0)
            + (res.get("downstream_prompt_tokens", 0) or 0)
            + (res.get("downstream_completion_tokens", 0) or 0))


def _target(r):
    res = r.get("resource") or {}
    return ((res.get("target_prompt_tokens", 0) or 0)
            + (res.get("target_completion_tokens", 0) or 0))


def _down(r):
    res = r.get("resource") or {}
    return ((res.get("downstream_prompt_tokens", 0) or 0)
            + (res.get("downstream_completion_tokens", 0) or 0))


def _valid(r):
    """Truncated round: downstream token accounting missing (resume residue)."""
    return _down(r) > 0


def _d_cross(recs_a, recs_b):
    sa = [signature(r) for r in recs_a]
    sb = [signature(r) for r in recs_b]
    return statistics.mean(
        edit_distance(x, y) / max(len(x), len(y), 1)
        for x in sa for y in sb)


def _d_self(recs):
    ss = [signature(r) for r in recs]
    ds = [edit_distance(ss[i], ss[j]) / max(len(ss[i]), len(ss[j]), 1)
          for i in range(len(ss)) for j in range(i + 1, len(ss))]
    return statistics.mean(ds) if ds else 0.0


def wilson_ucb(x, n, z=Z):
    if n == 0:
        return 1.0
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return min(1.0, center + half)


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mw_test(a, b):
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    if not a or not b:
        return None, None
    vals = [(v, 0) for v in a] + [(v, 1) for v in b]
    vals.sort(key=lambda x: x[0])
    n = len(vals)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[j + 1][0] == vals[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n1 = len(a)
    r1 = sum(ranks[k] for k in range(n) if vals[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    cnt = Counter(v[0] for v in vals)
    t = sum(c * (c * c - 1) for c in cnt.values())
    var = n1 * (n - n1) / 12 * ((n + 1) - t / (n * (n - 1))) if n > 1 else 0.0
    zs = (u1 - n1 * (n - n1) / 2) / (var ** 0.5) if var > 0 else 0.0
    return u1, 2 * (1 - _norm_cdf(abs(zs)))


def _spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = ((sum((a - mx) ** 2 for a in rx)
            * sum((b - my) ** 2 for b in ry)) ** 0.5)
    return num / den if den else 0.0


def _boot2(g, key_a, key_b, nboot=BOOT, seed=SEED):
    rng = random.Random(seed)
    draws = []
    for _ in range(nboot):
        sample = rng.choices(g, k=len(g))
        s = 0.0
        for sp in sample:
            a = statistics.mean(rng.choices(sp[key_a], k=MIN_ROUNDS))
            b = statistics.mean(rng.choices(sp[key_b], k=MIN_ROUNDS))
            s += a - b
        draws.append(s / len(sample))
    draws.sort()
    return draws[int(0.025 * nboot)], draws[int(0.975 * nboot)]


def main() -> int:
    records = []
    for p in sorted(glob.glob(os.path.join(RECORD_DIR, "*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if _valid(r):
                records.append(r)
    arm = lambda r: (r.get("replay_meta") or {}).get("arm", "")  # noqa: E731
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        if arm(r) in ("T_span", "B", "I"):
            by[(r["task_id"], r["invocation_index"])][arm(r)].append(r)
    complete = {k: v for k, v in by.items()
                if all(len(v.get(a, [])) >= MIN_ROUNDS for a in ("T_span", "B", "I"))}
    print("valid records %d | complete spans %d (after dropping truncated rounds)" % (len(records), len(complete)))

    spans_map = {}
    for l in open(SPANS_PATH, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        s = json.loads(l)
        spans_map[(s["task_id"], s["invocation_index"])] = s
    max_inv = defaultdict(int)
    for (t, _), s in spans_map.items():
        max_inv[t] = max(max_inv[t], s["invocation_index"])
    base_map = build_base_map(TRACES_DIR)

    # ---- per-span summary ----
    rows = []
    for (t, i), arms in sorted(complete.items()):
        recT, recB, recI = (arms[a] for a in ("T_span", "B", "I"))
        sd = recT[0].get("span_descriptor") or {}
        np_ = sd.get("node_pattern", "other")
        gc = sd.get("graph_context") or {}
        af = bool(gc.get("after_failure", False))
        pos = ("early" if i / max(max_inv[t], 1) <= 0.34 else
               ("late" if i / max(max_inv[t], 1) >= 0.67 else "middle"))
        sp = spans_map.get((t, i), {})
        bt = (recT[0].get("replay_meta") or {}).get("baseline_tokens")
        if bt is None:
            bt = sp.get("baseline_tokens")
        cont = ((sp.get("replay_input") or {}).get("continuation") or {})
        mrs = cont.get("max_remaining_steps")
        na = len(((sp.get("replay_input") or {}).get("output_contract_runtime")
                  or {}).get("allowed_tools") or [])
        rt_T = [int(bool(r["strict_success"])) for r in recT]
        rt_B = [int(bool(r["strict_success"])) for r in recB]
        rt_I = [int(bool(r["strict_success"])) for r in recI]
        rt_Ttot = [_total(r) for r in recT]
        rt_Btot = [_total(r) for r in recB]
        rt_Ttgt = [_target(r) for r in recT]
        rt_Btgt = [_target(r) for r in recB]
        rt_Tdwn = [_down(r) for r in recT]
        rt_Bdwn = [_down(r) for r in recB]
        rt_colT = [int((r.get("final_outcome") or {}).get("collateral_damage", False))
                   for r in recT]
        dcross = _d_cross(recT, recB)
        ddelta = max(0.0, dcross - min(_d_self(recT), _d_self(recB)))
        rows.append({
            "task": t, "inv": i, "node": np_, "after_failure": af,
            "position": pos, "baseline_tokens": bt,
            "max_remaining_steps": mrs, "n_allowed_tools": na,
            "strict_T": statistics.mean(rt_T), "strict_B": statistics.mean(rt_B),
            "strict_I": statistics.mean(rt_I),
            "d_success": statistics.mean(rt_T) - statistics.mean(rt_B),
            "d_IB": statistics.mean(rt_I) - statistics.mean(rt_B),
            "total_T": statistics.mean(rt_Ttot), "total_B": statistics.mean(rt_Btot),
            "total_I": statistics.mean(_total(r) for r in recI),
            "net_saving": statistics.mean(rt_Btot) - statistics.mean(rt_Ttot),
            "direct_saving": statistics.mean(rt_Btgt) - statistics.mean(rt_Ttgt),
            "downstream_comp": statistics.mean(rt_Tdwn) - statistics.mean(rt_Bdwn),
            "rho_comp": (statistics.mean(rt_Tdwn) - statistics.mean(rt_Bdwn))
                        / max(statistics.mean(rt_Btgt) - statistics.mean(rt_Ttgt), 1e-9),
            "d_delta": ddelta, "zero_detour_delta": ddelta < ZERO_DETOUR_DNORM,
            "rt_T": rt_T, "rt_B": rt_B, "rt_I": rt_I,
            "rt_Ttot": rt_Ttot, "rt_Btot": rt_Btot,
            "rt_Ttgt": rt_Ttgt, "rt_Btgt": rt_Btgt,
            "rt_Tdwn": rt_Tdwn, "rt_Bdwn": rt_Bdwn,
            "rt_colT": rt_colT,
        })
    print("node_pattern distribution:", dict(Counter(s["node"] for s in rows)))
    print("after_failure distribution:", dict(Counter(s["after_failure"] for s in rows)))
    print("position distribution:", dict(Counter(s["position"] for s in rows)))

    # ============ 1) metrics summary table ============
    print("\n" + "=" * 78)
    print("1. metrics summary table (per-span three-arm means + paired Δ, span-level bootstrap CI)")
    print("=" * 78)
    span_means = {}
    for s in rows:
        k = (s["task"], s["inv"])
        span_means[k] = {}
        for arm_key, recs in (("T_span", complete[k]["T_span"]),
                              ("B", complete[k]["B"]), ("I", complete[k]["I"])):
            span_means[k][arm_key] = {}
            for v in CORE_TABLE:
                xs = [x for x in (cf_metrics(r, base_map.get(k))[v] for r in recs)
                      if x == x]
                span_means[k][arm_key][v] = statistics.mean(xs) if xs else float("nan")
    total_rows = []
    for pair in ("T_span:B", "T_span:I", "B:I"):
        a, b = pair.split(":")
        print("\n--- %s vs %s ---" % (a, b))
        print("%-24s %10s %10s %+9s %s" % ("metric", "mu_" + a, "mu_" + b, "Δ", "95%CI"))
        for v in CORE_TABLE:
            diffs = []
            for s in rows:
                k = (s["task"], s["inv"])
                va, vb = span_means[k][a][v], span_means[k][b][v]
                if va == va and vb == vb:
                    diffs.append(va - vb)
            if len(diffs) < 3:
                continue
            mu = statistics.mean(diffs)
            rng = random.Random(SEED)
            boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                           for _ in range(BOOT))
            lo, hi = boots[50], boots[1949]
            sig = "★" if (lo > 0 or hi < 0) else ""
            total_rows.append({"metric": v, "pair": pair, "delta": mu,
                               "mu_a": statistics.mean(span_means[(s["task"], s["inv"])][a][v] for s in rows),
                               "mu_b": statistics.mean(span_means[(s["task"], s["inv"])][b][v] for s in rows),
                               "ci_lo": lo, "ci_hi": hi, "significant": bool(lo > 0 or hi < 0)})
            print("%-24s %10.3f %10.3f %+9.3f [%+.3f,%+.3f] %s" % (
                v, total_rows[-1]["mu_a"], total_rows[-1]["mu_b"], mu, lo, hi, sig))
        # token caliber: TotalTokens for all pairs; NetSaving/direct/comp only for T:B
        tk = {"T_span": "total_T", "B": "total_B", "I": "total_I"}
        diffs = [s[tk[a]] - s[tk[b]] for s in rows]
        mu = statistics.mean(diffs)
        rng = random.Random(SEED)
        boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                       for _ in range(BOOT))
        lo, hi = boots[50], boots[1949]
        print("%-24s %10s %10s %+9.0f [%+.0f,%+.0f] %s" % (
            "TotalTokens(a-b)", "-", "-", mu, lo, hi,
            "★" if (lo > 0 or hi < 0) else ""))
        total_rows.append({"metric": "TotalTokens", "pair": pair, "delta": mu,
                           "ci_lo": lo, "ci_hi": hi,
                           "significant": bool(lo > 0 or hi < 0)})
        if pair == "T_span:B":
            for name in ("net_saving", "direct_saving", "downstream_comp"):
                diffs = [s[name] for s in rows]
                mu = statistics.mean(diffs)
                rng = random.Random(SEED)
                boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                               for _ in range(BOOT))
                lo, hi = boots[50], boots[1949]
                print("%-24s %10s %10s %+9.0f [%+.0f,%+.0f] %s" % (
                    name + "(token)", "-", "-", mu, lo, hi,
                    "★" if (lo > 0 or hi < 0) else ""))
                total_rows.append({"metric": name, "pair": pair, "delta": mu,
                                   "ci_lo": lo, "ci_hi": hi,
                                   "significant": bool(lo > 0 or hi < 0)})

    # ============ 2) success group vs failure group ============
    print("\n" + "=" * 78)
    print("2. success group vs failure group (pooled at round level, 6-group variable profile)")
    print("=" * 78)
    groups = defaultdict(list)
    for r in records:
        a = arm(r)
        if a not in ("T_span", "B", "I"):
            continue
        g = ("%s%s" % (a, "成" if r["strict_success"] else "败"))
        groups[g].append(r)
    per_var = []
    for v in CORE_TABLE:
        row = {"var": v}
        for gname in ("T_span成", "T_span败", "B成", "B败", "I成", "I败"):
            rs = groups.get(gname, [])
            k0 = next(iter(rs), None)
            if k0 is None:
                row[gname] = None
                continue
            base = base_map.get((k0["task_id"], k0["invocation_index"]))
            row[gname] = statistics.mean(cf_metrics(r, base)[v] for r in rs)
        per_var.append(row)
    print("%-24s %10s %10s %10s %10s %10s %10s" % (
        "variable", "Tsucc", "Tfail", "Bsucc", "Bfail", "Isucc", "Ifail"))
    for row in per_var:
        print("%-24s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (
            row["var"], row["T_span成"], row["T_span败"], row["B成"],
            row["B败"], row["I成"], row["I败"]))
    print("(rounds per group:", {k: len(v) for k, v in groups.items()}, ")")

    # ============ 3) offloadability metrics ============
    print("\n" + "=" * 78)
    print("3. offloadability metrics (pre-offload feature stratification × outcome gate, noise-corrected)")
    print("=" * 78)
    strata = []
    for key, label in (("position", "position"), ("after_failure", "after_failure")):
        grp = defaultdict(list)
        for s in rows:
            grp[str(s[key])].append(s)
        for name, g in sorted(grp.items()):
            lcb_d, ucb_d = _boot2(g, "rt_T", "rt_B")
            lcb_n, ucb_n = _boot2(g, "rt_Ttot", "rt_Btot")
            rho = [s["rho_comp"] for s in g if abs(s["rho_comp"]) < 1e4]
            coll_n = sum(len(s["rt_colT"]) for s in g)
            coll_x = sum(sum(s["rt_colT"]) for s in g)
            _, p_st = mw_test([x for s in g for x in s["rt_T"]],
                              [x for s in g for x in s["rt_B"]])
            _, p_tk = mw_test([x for s in g for x in s["rt_Ttot"]],
                              [x for s in g for x in s["rt_Btot"]])
            strata.append({
                "key": label, "stratum": name, "n": len(g),
                "strict_T": statistics.mean(s["strict_T"] for s in g),
                "strict_B": statistics.mean(s["strict_B"] for s in g),
                "strict_I": statistics.mean(s["strict_I"] for s in g),
                "d_success": statistics.mean(s["d_success"] for s in g),
                "d_success_lcb": lcb_d, "d_success_ucb": ucb_d,
                "net_saving": statistics.mean(s["net_saving"] for s in g),
                "net_saving_lcb": lcb_n, "net_saving_ucb": ucb_n,
                "rho_comp_median": statistics.median(rho) if rho else None,
                "zero_detour_delta_ratio": sum(s["zero_detour_delta"] for s in g) / len(g),
                "collateral_wilson_ucb": wilson_ucb(coll_x, coll_n),
                "mw_strict_p": p_st, "mw_tokens_p": p_tk,
            })
        print("\n== %s ==" % label)
        print("%-16s %2s %5s %5s %5s %8s %16s %9s %16s %6s %5s %6s %8s %8s" % (
            "stratum", "n", "stT", "stB", "stI", "ΔSucc", "[LCB,UCB](2-layer)",
            "NetSav", "[LCB,UCB](2-layer)", "ρ_med", "zeroDet%", "collUCB",
            "MWp(strict)", "MWp(tok)"))
        for s_ in strata[-len(grp):]:
            print("%-16s %2d %5.2f %5.2f %5.2f %+8.3f [%+.3f,%+.3f] %+9.0f [%+.0f,%+.0f] %6.1f %5.0f%% %6.3f %8.3f %8.3f" % (
                s_["stratum"], s_["n"], s_["strict_T"], s_["strict_B"],
                s_["strict_I"], s_["d_success"], s_["d_success_lcb"],
                s_["d_success_ucb"], s_["net_saving"], s_["net_saving_lcb"],
                s_["net_saving_ucb"], s_["rho_comp_median"] or 0,
                100 * s_["zero_detour_delta_ratio"], s_["collateral_wilson_ucb"],
                s_["mw_strict_p"], s_["mw_tokens_p"]))

    # continuous feature correlation (two-layer bootstrap)
    print("\n== continuous features × outcomes (Spearman, two-layer bootstrap CI) ==")
    # tuple = (ak, bk), ys = mean(bk) − mean(ak)
    targets = {"net_saving": ("rt_Ttot", "rt_Btot"),        # B − T
               "direct_saving": ("rt_Ttgt", "rt_Btgt"),     # B − T
               "downstream_comp": ("rt_Bdwn", "rt_Tdwn"),   # T − B
               "d_success": ("rt_B", "rt_T")}               # T − B
    corr = []
    rng = random.Random(SEED)
    for f in ("baseline_tokens", "max_remaining_steps", "n_allowed_tools"):
        for tg, (ak, bk) in targets.items():
            pairs = [(s[f], s) for s in rows if s[f] is not None]
            if len(pairs) < 8 or len({p[0] for p in pairs}) < 3:
                continue
            xs = [p[0] for p in pairs]
            ys = [statistics.mean(p[1][bk]) - statistics.mean(p[1][ak])
                  for p in pairs]
            rho = _spearman(xs, ys)
            draws = []
            for _ in range(BOOT):
                sample = rng.choices(pairs, k=len(pairs))
                xs_b = [p[0] for p in sample]
                ys_b = [statistics.mean(rng.choices(p[1][bk], k=MIN_ROUNDS))
                        - statistics.mean(rng.choices(p[1][ak], k=MIN_ROUNDS))
                        for p in sample]
                draws.append(_spearman(xs_b, ys_b))
            draws.sort()
            corr.append({"feature": f, "target": tg, "n": len(pairs),
                         "rho": rho, "ci_lo": draws[int(0.025 * BOOT)],
                         "ci_hi": draws[int(0.975 * BOOT)],
                         "significant": bool(draws[int(0.025 * BOOT)] > 0
                                             or draws[int(0.975 * BOOT)] < 0)})
    for f in ("baseline_tokens", "max_remaining_steps", "n_allowed_tools"):
        pairs = [(s[f], s["d_delta"]) for s in rows if s[f] is not None]
        if len(pairs) < 8:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        rho = _spearman(xs, ys)
        draws = []
        for _ in range(BOOT):
            sample = rng.choices(pairs, k=len(pairs))
            draws.append(_spearman([p[0] for p in sample], [p[1] for p in sample]))
        draws.sort()
        corr.append({"feature": f, "target": "d_delta", "n": len(pairs),
                     "rho": rho, "ci_lo": draws[int(0.025 * BOOT)],
                     "ci_hi": draws[int(0.975 * BOOT)],
                     "significant": bool(draws[int(0.025 * BOOT)] > 0
                                         or draws[int(0.975 * BOOT)] < 0)})
    print("%-20s %-18s %3s %+7.3f [%+.3f,%+.3f] %s" % ("feature", "outcome", "n", 0, 0, 0, ""))
    for c in corr:
        print("%-20s %-18s %3d %+7.3f [%+.3f,%+.3f] %s" % (
            c["feature"], c["target"], c["n"], c["rho"], c["ci_lo"], c["ci_hi"],
            "★" if c["significant"] else ""))

    out = {"n_spans": len(rows), "per_span": rows, "total_table": total_rows,
           "success_failure": per_var, "strata": strata, "corr": corr}
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
