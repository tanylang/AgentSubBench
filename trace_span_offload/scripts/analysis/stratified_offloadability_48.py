#!/usr/bin/env python
"""48-span pre-offload (ex-ante) offloadability metric validation (background noise-corrected v2).

Adds five noise treatments on top of v1 (structural stratification + continuous feature correlation):
  1) zero_detour switches to the denoised d_delta = d_cross − min(d_self_T, d_self_B)
  2) stratified ΔSucc / NetSaving / DiD use two-layer bootstrap (span layer × round layer),
     within-round noise propagates into the CI
  3) collateral reported as Wilson one-sided 95% UCB (T-arm rounds pooled per stratum)
  4) stratified DiD: (T−B) − (I−B) = net effect "beyond natural fluctuation (I injection)"
  5) key strata T vs B two-sample Mann-Whitney test (strict / total tokens)
The correlation analysis likewise switches to two-layer bootstrap CIs (for targets that are
round-level arrays); d_norm has no round-level array, keeps span-layer bootstrap and is noted.

Same caliber as main_comparison.py: first 5 rounds in file order per (task, inv) × arm,
keeping only spans with all three arms (n=48).

Output: results/stratification/stratified_offloadability_48.json (per_span + strata + corr)
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from trace_edit_distance import edit_distance, signature  # noqa: E402
from span_offload.controlflow_metrics import (  # noqa: E402
    build_base_map, cf_metrics,
)

MIN_ROUNDS = 5
ZERO_DETOUR_DNORM = 0.1
BOOT = 2000
SEED = 42
Z = 1.644854  # one-sided 95% normal quantile (Wilson UCB)

SPANS_PATH = "out_span_offload/spans/spans_simplified_qwen27b_50.jsonl"
TRACES_DIR = "check_point_traces/appworld_simplified_qwen3_8-27b_batch1"
OUT = "results/stratification/stratified_offloadability_48.json"


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
    """Wilson one-sided 95% upper bound for binomial rate p=x/n (gives a nonzero bound even at 0 count)."""
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
    """Mann-Whitney U (group a vs group b), average ranks for ties, normal-approximation two-tailed p."""
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


def _boot2_stat(g, key, nboot=BOOT, seed=SEED):
    """Two-layer bootstrap: span resampling + per-span round resampling, returns (lo, hi)."""
    rng = random.Random(seed)
    draws = []
    for _ in range(nboot):
        sample = rng.choices(g, k=len(g))
        s = 0.0
        for sp in sample:
            a = statistics.mean(rng.choices(sp["rt_%s" % key[0]], k=MIN_ROUNDS))
            b = statistics.mean(rng.choices(sp["rt_%s" % key[1]], k=MIN_ROUNDS))
            s += a - b
        draws.append(s / len(sample))
    draws.sort()
    return draws[int(0.025 * nboot)], draws[int(0.975 * nboot)]


def _boot2_did(g, nboot=BOOT, seed=SEED):
    """Two-layer bootstrap: DiD = (T−B) − (I−B) per span, returns (lo, hi)."""
    rng = random.Random(seed)
    draws = []
    for _ in range(nboot):
        sample = rng.choices(g, k=len(g))
        s = 0.0
        for sp in sample:
            t = statistics.mean(rng.choices(sp["rt_T"], k=MIN_ROUNDS))
            b = statistics.mean(rng.choices(sp["rt_B"], k=MIN_ROUNDS))
            i = statistics.mean(rng.choices(sp["rt_I"], k=MIN_ROUNDS))
            s += (t - b) - (i - b)
        draws.append(s / len(sample))
    draws.sort()
    return draws[int(0.025 * nboot)], draws[int(0.975 * nboot)]


def pick_files():
    """Pick one record file per task: prefer the new naming (lowercase qwen3_8-27b prefix)."""
    files = sorted(glob.glob(
        "out_span_offload/simplified/screening/offload_profile_simplified_*.jsonl"))
    chosen = {}
    for p in files:
        m = re.search(r"([0-9a-f]{6,8}_\d)\.jsonl$", os.path.basename(p))
        if not m:
            continue
        task = m.group(1)
        is_new = "qwen3_8-27b_Qwen" in os.path.basename(p)
        if task not in chosen or is_new:
            chosen[task] = p
    return sorted(chosen.values())


def load_spans(path):
    spans = {}
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        s = json.loads(l)
        spans[(s["task_id"], s["invocation_index"])] = s
    return spans


def main() -> int:
    records = []
    for p in pick_files():
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                records.append(json.loads(l))
            except ValueError:
                continue
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        arm = (r.get("replay_meta") or {}).get("arm", "")
        if arm in ("T_span", "B", "I"):
            by[(r["task_id"], r["invocation_index"])][arm].append(r)
    complete = {k: v for k, v in by.items()
                if all(len(v.get(a, [])) >= MIN_ROUNDS for a in ("T_span", "B", "I"))}
    print("complete paired spans: %d (each arm ≥%d rounds)" % (len(complete), MIN_ROUNDS))

    spans = load_spans(SPANS_PATH)
    max_inv = defaultdict(int)
    for (t, _), s in spans.items():
        max_inv[t] = max(max_inv[t], s["invocation_index"])
    base_map = build_base_map(TRACES_DIR)

    rows = []
    for (t, i), arms in sorted(complete.items()):
        recT, recB, recI = (arms[a][:MIN_ROUNDS] for a in ("T_span", "B", "I"))
        # ---- pre-offload features ----
        sd = recT[0].get("span_descriptor") or {}
        np_ = sd.get("node_pattern", "other")
        se = (tuple(sorted(sd.get("mechanism_tags", {}).get("state_effect", [])))
              [0] if sd.get("mechanism_tags", {}).get("state_effect")
              else "read_only")
        pos = ("early" if i / max(max_inv[t], 1) <= 0.34 else
               ("late" if i / max(max_inv[t], 1) >= 0.67 else "middle"))
        sp = spans.get((t, i), {})
        gc = ((sp.get("span_descriptor") or {}).get("graph_context") or {})
        oc = "+".join(sorted(sd.get("mechanism_tags", {}).get("output_contract", []))) or "none"
        cs = "+".join(sorted(sd.get("mechanism_tags", {}).get("context_source", []))) or "none"
        vt = "+".join(sorted(sd.get("verifier_type", []) or [])) or "none"
        tc = (sp.get("task_condition") or {}).get("cross_app", False)
        base_tok = (recT[0].get("replay_meta") or {}).get("baseline_tokens")
        if base_tok is None:
            base_tok = sp.get("baseline_tokens")
        n_allowed = len(((sp.get("replay_input") or {})
                         .get("output_contract_runtime") or {}).get("allowed_tools") or [])
        max_rem = ((sp.get("replay_input") or {}).get("continuation") or {}) \
            .get("max_remaining_steps")
        # ---- round-level arrays (strict / total / target / down) ----
        rt_T = [int(bool(r["strict_success"])) for r in recT]
        rt_B = [int(bool(r["strict_success"])) for r in recB]
        rt_I = [int(bool(r["strict_success"])) for r in recI]
        rt_Ttot = [_total(r) for r in recT]
        rt_Btot = [_total(r) for r in recB]
        rt_Ttgt = [_target(r) for r in recT]
        rt_Btgt = [_target(r) for r in recB]
        rt_Tdwn = [_down(r) for r in recT]
        rt_Bdwn = [_down(r) for r in recB]
        # ---- denoised divergence: d_delta = d_cross − min(d_self) ----
        dcross = _d_cross(recT, recB)
        ddelta = max(0.0, dcross - min(_d_self(recT), _d_self(recB)))
        # ---- outcomes (5-round means) ----
        stT, stB, stI = (statistics.mean(x) for x in (rt_T, rt_B, rt_I))
        totT, totB = statistics.mean(rt_Ttot), statistics.mean(rt_Btot)
        tgtT, tgtB = statistics.mean(rt_Ttgt), statistics.mean(rt_Btgt)
        dwnT, dwnB = statistics.mean(rt_Tdwn), statistics.mean(rt_Bdwn)
        direct = tgtB - tgtT
        comp = dwnT - dwnB
        dm = {}
        for arm, recs_ in (("T", recT), ("B", recB)):
            for v in ("llm", "tools", "hall", "MaxFailRun", "RevisitRate",
                      "NewBranchRate", "DirectActionRate"):
                dm["%s_%s" % (v, arm)] = statistics.mean(
                    cf_metrics(r, base_map.get((t, i)))[v] for r in recs_)
        rows.append({
            "task": t, "inv": i,
            "node": np_, "state_effect": se, "position": pos,
            "control_phase": gc.get("control_phase", ""),
            "inside_loop": bool(gc.get("inside_loop", False)),
            "output_contract": oc, "context_source": cs,
            "verifier_type": vt, "cross_app": bool(tc),
            "baseline_tokens": base_tok,
            "n_allowed_tools": n_allowed,
            "max_remaining_steps": max_rem,
            "cls": "%s_%s" % (np_, se),
            "strict_T": stT, "strict_B": stB, "strict_I": stI,
            "d_success": stT - stB, "d_success_IB": stI - stB,
            "total_T": totT, "total_B": totB,
            "net_saving": totB - totT,
            "direct_saving": direct, "downstream_comp": comp,
            "rho_comp": comp / max(direct, 1e-9),
            "d_cross": dcross, "d_delta": ddelta,
            "zero_detour": dcross < ZERO_DETOUR_DNORM,
            "zero_detour_delta": ddelta < ZERO_DETOUR_DNORM,
            "collateral_T": max(int((r.get("final_outcome") or {}).get(
                "collateral_damage", False)) for r in recT),
            "rt_T": rt_T, "rt_B": rt_B, "rt_I": rt_I,
            "rt_colT": [int((r.get("final_outcome") or {}).get(
                "collateral_damage", False)) for r in recT],
            "rt_Ttot": rt_Ttot, "rt_Btot": rt_Btot,
            "rt_Ttgt": rt_Ttgt, "rt_Btgt": rt_Btgt,
            "rt_Tdwn": rt_Tdwn, "rt_Bdwn": rt_Bdwn,
            **dm,
        })

    # ---- stratification tables (noise-corrected version) ----
    keys = [("node", "node_pattern"), ("position", "position"),
            ("cross_app", "cross_app"), ("control_phase", "control_phase"),
            ("inside_loop", "inside_loop"), ("state_effect", "state_effect"),
            ("output_contract", "output_contract"), ("verifier_type", "verifier_type"),
            ("context_source", "context_source")]
    strata = []
    for key, label in keys:
        groups = defaultdict(list)
        for s in rows:
            groups[str(s[key])].append(s)
        for name, g in sorted(groups.items()):
            lcb_d, ucb_d = _boot2_stat(g, ("T", "B"))
            lcb_n, ucb_n = _boot2_stat(g, ("Ttot", "Btot"))
            lcb_did, ucb_did = _boot2_did(g)
            rho = [s["rho_comp"] for s in g if abs(s["rho_comp"]) < 1e4]
            coll_n = sum(len(s["rt_T"]) for s in g)
            coll_x = sum(sum(s["rt_colT"]) for s in g)
            u1_st, p_st = mw_test([x for s in g for x in s["rt_T"]],
                                  [x for s in g for x in s["rt_B"]])
            u1_tk, p_tk = mw_test([x for s in g for x in s["rt_Ttot"]],
                                  [x for s in g for x in s["rt_Btot"]])
            strata.append({
                "key": label, "stratum": name, "n": len(g),
                "strict_T": statistics.mean(s["strict_T"] for s in g),
                "strict_B": statistics.mean(s["strict_B"] for s in g),
                "strict_I": statistics.mean(s["strict_I"] for s in g),
                "d_success": statistics.mean(s["d_success"] for s in g),
                "d_success_lcb": lcb_d, "d_success_ucb": ucb_d,
                "d_IB": statistics.mean(s["d_success_IB"] for s in g),
                "did": statistics.mean(s["d_success"] - s["d_success_IB"]
                                       for s in g),
                "did_lcb": lcb_did, "did_ucb": ucb_did,
                "net_saving": statistics.mean(s["net_saving"] for s in g),
                "net_saving_lcb": lcb_n, "net_saving_ucb": ucb_n,
                "rho_comp_median": statistics.median(rho) if rho else None,
                "zero_detour_ratio": sum(s["zero_detour"] for s in g) / len(g),
                "zero_detour_delta_ratio": sum(s["zero_detour_delta"]
                                               for s in g) / len(g),
                "positive_ns": sum(1 for s in g if s["net_saving"] > 0),
                "collateral_wilson_ucb": wilson_ucb(coll_x, coll_n),
                "mw_strict_p": p_st, "mw_tokens_p": p_tk,
            })
    # print three key tables
    for key, label in [("node", "node_pattern"), ("position", "position"),
                       ("cross_app", "cross_app")]:
        print("\n== %s (noise-corrected) == " % label)
        print("%-24s %2s %5s %5s %5s %8s %16s %8s %16s %9s %16s %6s %5s %6s %8s %8s" % (
            "stratum", "n", "stT", "stB", "stI", "ΔSucc", "[LCB,UCB](2-layer)",
            "DiD", "[LCB,UCB]", "NetSav", "[LCB,UCB](2-layer)",
            "ρ_med", "zeroDet%", "collUCB", "MWp(strict)", "MWp(tok)"))
        for s_ in strata:
            if s_["key"] != label:
                continue
            print("%-24s %2d %5.2f %5.2f %5.2f %+8.3f [%+.3f,%+.3f] %+8.3f [%+.3f,%+.3f] %+9.0f [%+.0f,%+.0f] %6.1f %5.0f%% %6.3f %8.3f %8.3f" % (
                s_["stratum"], s_["n"], s_["strict_T"], s_["strict_B"],
                s_["strict_I"], s_["d_success"], s_["d_success_lcb"],
                s_["d_success_ucb"], s_["did"], s_["did_lcb"], s_["did_ucb"],
                s_["net_saving"], s_["net_saving_lcb"], s_["net_saving_ucb"],
                s_["rho_comp_median"] or 0, 100 * s_["zero_detour_delta_ratio"],
                s_["collateral_wilson_ucb"], s_["mw_strict_p"], s_["mw_tokens_p"]))

    # ---- continuous features × outcomes (Spearman, two-layer bootstrap CI) ----
    rng = random.Random(SEED)
    # tuple = (ak, bk), ys = mean(bk) − mean(ak)
    targets = {"net_saving": ("rt_Ttot", "rt_Btot"),        # B − T
               "direct_saving": ("rt_Ttgt", "rt_Btgt"),     # B − T
               "downstream_comp": ("rt_Bdwn", "rt_Tdwn"),   # T − B
               "d_success": ("rt_B", "rt_T")}               # T − B
    corr = []
    for f in ("baseline_tokens", "max_remaining_steps", "n_allowed_tools"):
        for tg, (ak, bk) in targets.items():
            pairs = [(s[f], s) for s in rows if s[f] is not None]
            if len(pairs) < 8 or len({p[0] for p in pairs}) < 3:
                continue
            xs = [p[0] for p in pairs]
            ys = [statistics.mean(p[1][bk]) - statistics.mean(p[1][ak])
                  for p in pairs]

            def _rho(xs_, ys_):
                order_x = sorted(range(len(xs_)), key=lambda i: xs_[i])
                order_y = sorted(range(len(ys_)), key=lambda i: ys_[i])
                rx, ry = [0.0] * len(xs_), [0.0] * len(ys_)
                for p_, i_ in enumerate(order_x):
                    rx[i_] = p_
                for p_, i_ in enumerate(order_y):
                    ry[i_] = p_
                mx_, my_ = statistics.mean(rx), statistics.mean(ry)
                num = sum((a - mx_) * (b - my_) for a, b in zip(rx, ry))
                den = ((sum((a - mx_) ** 2 for a in rx)
                        * sum((b - my_) ** 2 for b in ry)) ** 0.5)
                return num / den if den else 0.0

            rho = _rho(xs, ys)
            draws = []
            for _ in range(BOOT):
                sample = rng.choices(pairs, k=len(pairs))
                xs_b = [p[0] for p in sample]
                ys_b = [statistics.mean(rng.choices(p[1][bk], k=MIN_ROUNDS))
                        - statistics.mean(rng.choices(p[1][ak], k=MIN_ROUNDS))
                        for p in sample]
                draws.append(_rho(xs_b, ys_b))
            draws.sort()
            lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]
            corr.append({"feature": f, "target": tg, "n": len(pairs),
                         "rho": rho, "ci_lo": lo, "ci_hi": hi,
                         "significant": bool(lo > 0 or hi < 0)})
    # d_norm (no round-level array, span-layer bootstrap)
    for f in ("baseline_tokens", "max_remaining_steps", "n_allowed_tools"):
        pairs = [(s[f], s["d_delta"]) for s in rows if s[f] is not None]
        if len(pairs) < 8 or len({p[0] for p in pairs}) < 3:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        rho = _rho(xs, ys)
        draws = []
        for _ in range(BOOT):
            sample = rng.choices(pairs, k=len(pairs))
            draws.append(_rho([p[0] for p in sample], [p[1] for p in sample]))
        draws.sort()
        lo, hi = draws[int(0.025 * BOOT)], draws[int(0.975 * BOOT)]
        corr.append({"feature": f, "target": "d_delta", "n": len(pairs),
                     "rho": rho, "ci_lo": lo, "ci_hi": hi,
                     "significant": bool(lo > 0 or hi < 0),
                     "note": "span 层 bootstrap(无轮级数组)"})
    print("\n== continuous features × outcomes (Spearman, noise-corrected CI) ==")
    print("%-20s %-18s %3s %+7.3f [%+.3f,%+.3f] %s" % ("feature", "outcome", "n", 0, 0, 0, ""))
    for c in corr:
        print("%-20s %-18s %3d %+7.3f [%+.3f,%+.3f] %s" % (
            c["feature"], c["target"], c["n"], c["rho"], c["ci_lo"], c["ci_hi"],
            "★" if c["significant"] else ""))

    out = {"n_spans": len(rows), "per_span": rows, "strata": strata, "corr": corr}
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
