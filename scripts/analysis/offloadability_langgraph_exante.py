#!/usr/bin/env python
"""langgraph (LoCoMo) arm ex-ante offloadability analysis (noise-corrected version, Δ_TB caliber).

Based on the methodology of stratified_offloadability_48.py, adapted to the langgraph batch:
  - two-layer bootstrap (span layer × round layer) for CIs of ΔSucc / NetSaving / DiD
  - denoised zero-detour: d_delta = d_cross − min(d_self_T, d_self_B)
  - collateral reported as Wilson one-sided 95% UCB
  - stratified DiD: (T−B) − (I−B) = net effect beyond natural fluctuation (I injection)
  - key strata T vs B Mann-Whitney (strict / total tokens)
  - input-side continuous features (baseline_tokens / max_remaining_steps /
    n_allowed_tools) × outcome Spearman correlation (two-layer bootstrap CI)

Output:
  A. combined-strata metrics (cross classes, strongest evidence): node×state_effect, node×position
  B. single-axis strata metrics: node / position / state_effect / control_phase /
     inside_loop / output_contract / verifier_type / context_source /
     cross_app
  C. input-side metrics (computable before offload): continuous feature correlation + feature-grouped outcome table

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/analysis/offloadability_langgraph_exante.py \
      --records /tmp/langgraph_merged.jsonl \
      --spans out_span_offload/langgraph/batch1/spans_langgraph_conv-26.jsonl \
      --traces-dir check_point_traces/locomo_langgraph_qwen3_8-27b \
      --out results/locomo/offloadability_langgraph_exante.json
"""
from __future__ import annotations

import argparse
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
    build_base_map, cf_metrics,
)

MIN_ROUNDS = 10
ZERO_DETOUR_DNORM = 0.1
BOOT = 2000
SEED = 42
Z = 1.644854


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


def _boot2_stat(g, key, nboot=BOOT, seed=SEED):
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


def _spearman(xs, ys):
    order_x = sorted(range(len(xs)), key=lambda i: xs[i])
    order_y = sorted(range(len(ys)), key=lambda i: ys[i])
    rx, ry = [0.0] * len(xs), [0.0] * len(ys)
    for p, i in enumerate(order_x):
        rx[i] = p
    for p, i in enumerate(order_y):
        ry[i] = p
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = ((sum((a - mx) ** 2 for a in rx)
            * sum((b - my) ** 2 for b in ry)) ** 0.5)
    return num / den if den else 0.0


def _expand(paths: str) -> list:
    """Space-separated paths/globs, expanded into a file list (backward compatible with a single file)."""
    import glob as _glob
    out = []
    for p in paths.split():
        out.extend(sorted(_glob.glob(p)) or [p])
    return out


def build_rows(records_path, spans_path, traces_dir, manifest_path=None):
    records = []
    for p in _expand(records_path):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            records.append(json.loads(l))
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        arm = (r.get("replay_meta") or {}).get("arm", "")
        if arm in ("T_span", "B", "I"):
            by[(r["task_id"], r["invocation_index"])][arm].append(r)
    complete = {k: v for k, v in by.items()
                if all(len(v.get(a, [])) >= MIN_ROUNDS
                       for a in ("T_span", "B", "I"))}
    print("complete paired spans: %d (each arm ≥%d rounds)" % (len(complete), MIN_ROUNDS))
    spans = {}
    for p in _expand(spans_path):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            s = json.loads(l)
            spans[(s["task_id"], s["invocation_index"])] = s
    # task-level category (gold qa category 1-5): locomo:<sample>:q<qi> → qa[qi].category
    category_of = {}
    if manifest_path:
        from span_offload.trace_io import load_manifest as _load_manifest
        _m = _load_manifest(manifest_path)
        for _tid in set(t for (t, _), _ in spans.items()):
            _parts = _tid.split(":")
            if len(_parts) < 3:
                continue
            _sample, _qi = _parts[1], int(_parts[2][1:])
            _mt = _m.get(_sample)
            if _mt is None:
                continue
            _qa = list(_mt.evaluator.get("qa") or [])
            if _qi < len(_qa):
                category_of[_tid] = _qa[_qi].get("category")
    max_inv = defaultdict(int)
    for (t, _), s in spans.items():
        max_inv[t] = max(max_inv[t], s["invocation_index"])
    base_map = build_base_map(traces_dir)
    rows = []
    for (t, i), arms in sorted(complete.items()):
        recT, recB, recI = (arms[a][:MIN_ROUNDS]
                            for a in ("T_span", "B", "I"))
        sd = recT[0].get("span_descriptor") or {}
        np_ = sd.get("node_pattern", "other")
        se = (tuple(sorted(sd.get("mechanism_tags", {}).get(
            "state_effect", [])))[0] if sd.get("mechanism_tags", {}).get(
                "state_effect") else "read_only")
        pos = ("early" if i / max(max_inv[t], 1) <= 0.34 else
               ("late" if i / max(max_inv[t], 1) >= 0.67 else "middle"))
        sp = spans.get((t, i), {})
        gc = ((sp.get("span_descriptor") or {}).get("graph_context") or {})
        oc = "+".join(sorted(sd.get("mechanism_tags", {}).get(
            "output_contract", []))) or "none"
        cs = "+".join(sorted(sd.get("mechanism_tags", {}).get(
            "context_source", []))) or "none"
        vt = "+".join(sorted(sd.get("verifier_type", []) or [])) or "none"
        tc = (sp.get("task_condition") or {}).get("cross_app", False)
        base_tok = (recT[0].get("replay_meta") or {}).get("baseline_tokens")
        if base_tok is None:
            base_tok = sp.get("baseline_tokens")
        n_allowed = len(((sp.get("replay_input") or {})
                         .get("output_contract_runtime") or {})
                        .get("allowed_tools") or [])
        max_rem = ((sp.get("replay_input") or {}).get("continuation") or {}) \
            .get("max_remaining_steps")
        rt_T = [int(bool(r["strict_success"])) for r in recT]
        rt_B = [int(bool(r["strict_success"])) for r in recB]
        rt_I = [int(bool(r["strict_success"])) for r in recI]
        rt_Ttot = [_total(r) for r in recT]
        rt_Btot = [_total(r) for r in recB]
        rt_Ttgt = [_target(r) for r in recT]
        rt_Btgt = [_target(r) for r in recB]
        rt_Tdwn = [_down(r) for r in recT]
        rt_Bdwn = [_down(r) for r in recB]
        dcross = _d_cross(recT, recB)
        ddelta = max(0.0, dcross - min(_d_self(recT), _d_self(recB)))
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
            "category": category_of.get(t),
            "node": np_, "state_effect": se, "position": pos,
            "control_phase": gc.get("control_phase", ""),
            "inside_loop": bool(gc.get("inside_loop", False)),
            "output_contract": oc, "context_source": cs,
            "verifier_type": vt, "cross_app": bool(tc),
            "baseline_tokens": base_tok,
            "n_allowed_tools": n_allowed,
            "max_remaining_steps": max_rem,
            "cls": "%s_%s" % (np_, se),
            "cls2": "%s_%s" % (np_, pos),
            "strict_T": stT, "strict_B": stB, "strict_I": stI,
            # offload success (background-removed caliber): DiD_i = (T−B) − (I−B) = T − I
            # ≥ 0 ⟺ offload is not worse than the fallback (injection-collected answer); must not judge
            # by T's absolute success — spans on intrinsically hard tasks would be mislabeled as "offload failure"
            "offload_ok": stT >= stI,
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
    return rows


def strata_table(rows, keys):
    strata = []
    for key in keys:
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
            _, p_st = mw_test([x for s in g for x in s["rt_T"]],
                              [x for s in g for x in s["rt_B"]])
            _, p_tk = mw_test([x for s in g for x in s["rt_Ttot"]],
                              [x for s in g for x in s["rt_Btot"]])
            strata.append({
                "key": key, "stratum": name, "n": len(g),
                "strict_T": statistics.mean(s["strict_T"] for s in g),
                "strict_B": statistics.mean(s["strict_B"] for s in g),
                "strict_I": statistics.mean(s["strict_I"] for s in g),
                "d_success": statistics.mean(s["d_success"] for s in g),
                "d_success_lcb": lcb_d, "d_success_ucb": ucb_d,
                "d_IB": statistics.mean(s["d_success_IB"] for s in g),
                "did": statistics.mean(
                    s["d_success"] - s["d_success_IB"] for s in g),
                "did_lcb": lcb_did, "did_ucb": ucb_did,
                "net_saving": statistics.mean(s["net_saving"] for s in g),
                "net_saving_lcb": lcb_n, "net_saving_ucb": ucb_n,
                "rho_comp_median": statistics.median(rho) if rho else None,
                "zero_detour_ratio": sum(s["zero_detour"]
                                         for s in g) / len(g),
                "zero_detour_delta_ratio": sum(s["zero_detour_delta"]
                                               for s in g) / len(g),
                "positive_ns": sum(1 for s in g if s["net_saving"] > 0),
                "collateral_wilson_ucb": wilson_ucb(coll_x, coll_n),
                "mw_strict_p": p_st, "mw_tokens_p": p_tk,
            })
    return strata


def corr_table(rows):
    targets = {"net_saving": ("rt_Ttot", "rt_Btot"),       # B − T
               "direct_saving": ("rt_Ttgt", "rt_Btgt"),    # B − T
               "downstream_comp": ("rt_Bdwn", "rt_Tdwn"),  # T − B
               "d_success": ("rt_B", "rt_T")}              # T − B
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
                        - statistics.mean(rng.choices(p[1][ak],
                                                      k=MIN_ROUNDS))
                        for p in sample]
                draws.append(_spearman(xs_b, ys_b))
            draws.sort()
            corr.append({
                "feature": f, "target": tg, "rho": rho,
                "ci_lo": draws[int(0.025 * BOOT)],
                "ci_hi": draws[int(0.975 * BOOT)],
                "n": len(pairs),
                "significant": draws[int(0.025 * BOOT)] > 0
                or draws[int(0.975 * BOOT)] < 0,
            })
    return corr


def success_table(rows, keys):
    """Offload success rate table with corrected caliber: per-span offload success = DiD_i ≥ 0
    (T not worse than identity injection); reports P(success) by stratum + bootstrap CI + lift +
    two-sample rate test against the whole. """
    n_all = len(rows)
    k_all = sum(1 for s in rows if s["offload_ok"])
    out = []
    for key in keys:
        groups = defaultdict(list)
        for s in rows:
            groups[str(s[key])].append(s)
        for name, g in sorted(groups.items()):
            if len(g) == n_all:  # single-value stratum carries no discriminative information
                continue
            k = sum(1 for s in g if s["offload_ok"])
            rate = k / len(g)
            rng = random.Random(SEED)
            ds = []
            for _ in range(BOOT):
                sample = rng.choices(g, k=len(g))
                ds.append(sum(1 for s in sample if s["offload_ok"]) / len(sample))
            ds.sort()
            p_all = k_all / n_all
            se = math.sqrt(p_all * (1 - p_all) * (
                1 / len(g) + 1 / max(n_all - len(g), 1)))
            z = abs(rate - p_all) / se if se > 0 else 0.0
            p = 2 * (1 - _norm_cdf(z))
            out.append({
                "key": key, "stratum": name, "n": len(g),
                "offload_ok_rate": rate,
                "ci_lo": ds[int(0.025 * BOOT)],
                "ci_hi": ds[int(0.975 * BOOT)],
                "lift": rate - p_all,
                "p_vs_all": p,
            })
    return out


def feature_discrimination(rows):
    """Discriminability of continuous features: means of offload-success group vs failure group + Mann-Whitney."""
    out = []
    for f in ("baseline_tokens", "max_remaining_steps", "n_allowed_tools"):
        ok = [s[f] for s in rows if s["offload_ok"] and s[f] is not None]
        bad = [s[f] for s in rows
               if not s["offload_ok"] and s[f] is not None]
        if len(ok) < 5 or len(bad) < 5:
            continue
        _, p = mw_test(ok, bad)
        out.append({
            "feature": f,
            "mean_ok": statistics.mean(ok),
            "mean_fail": statistics.mean(bad),
            "mw_p": p,
            "significant": bool(p is not None and p < 0.05),
        })
    return out


def print_tables(strata, corr, rows, succ, fdisc):
    def block(keys, title):
        print("\n== %s (noise-corrected, Δ_TB caliber) == " % title)
        print("%-34s %3s %5s %5s %5s %8s %16s %8s %16s %9s %16s %6s %5s %6s %7s %7s" % (
            "stratum", "n", "stT", "stB", "stI", "ΔSucc", "[LCB,UCB]2layer",
            "DiD", "[LCB,UCB]", "NetSav", "[LCB,UCB]2layer",
            "ρ_med", "zeroDet%", "collUCB", "MWp(st)", "MWp(tok)"))
        for s_ in strata:
            if s_["key"] not in keys:
                continue
            print("%-34s %3d %5.2f %5.2f %5.2f %+8.3f [%+.3f,%+.3f] %+8.3f [%+.3f,%+.3f] %+9.0f [%+.0f,%+.0f] %6.1f %5.0f%% %6.3f %7.3f %7.3f" % (
                s_["stratum"], s_["n"], s_["strict_T"], s_["strict_B"],
                s_["strict_I"], s_["d_success"], s_["d_success_lcb"],
                s_["d_success_ucb"], s_["did"], s_["did_lcb"], s_["did_ucb"],
                s_["net_saving"], s_["net_saving_lcb"],
                s_["net_saving_ucb"], s_["rho_comp_median"] or 0,
                100 * s_["zero_detour_delta_ratio"],
                s_["collateral_wilson_ucb"],
                s_["mw_strict_p"] or 1, s_["mw_tokens_p"] or 1))
    block(["cls", "cls2"], "A. combined-strata metrics (cross classes)")
    block(["node", "position", "state_effect", "control_phase",
           "inside_loop", "output_contract", "verifier_type",
           "context_source", "cross_app"], "B. single-axis strata metrics")
    print("\n== C. input-side metrics (computable before offload): continuous features × outcomes (Spearman, two-layer bootstrap CI) == ")
    print("%-22s %-18s %5s %6s %16s %5s %s" % (
        "feature", "outcome", "n", "ρ", "[LCB,UCB]", "sig", ""))
    for c in corr:
        print("%-22s %-18s %5d %+6.3f [%+.3f,%+.3f] %5s" % (
            c["feature"], c["target"], c["n"], c["rho"],
            c["ci_lo"], c["ci_hi"], "★" if c["significant"] else ""))

    print("\n== D. offload success rate table (corrected caliber: per-span offload success = DiD≥0,"
          "i.e. T not worse than identity injection) == ")
    print("%-32s %3s %8s %16s %8s %8s" % (
        "stratum", "n", "success%", "95%CI(boot)", "lift", "p(vs all)"))
    for r_ in succ:
        print("%-32s %3d %7.0f%% [%5.0f%%,%5.0f%%] %+7.0f%% %8.3f %s" % (
            r_["stratum"], r_["n"], 100 * r_["offload_ok_rate"],
            100 * r_["ci_lo"], 100 * r_["ci_hi"], 100 * r_["lift"],
            r_["p_vs_all"], "★" if r_["p_vs_all"] < 0.05 else ""))
    print("\n== E. continuous feature discriminability (success group vs failure group, Mann-Whitney) == ")
    for f_ in fdisc:
        print("  %-22s success group=%9.1f failure group=%9.1f MW p=%.3f %s" % (
            f_["feature"], f_["mean_ok"], f_["mean_fail"],
            f_["mw_p"] or 1, "★" if f_["significant"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--manifest", default="data/locomo/task_manifest.json")
    ap.add_argument("--out", default="results/locomo/offloadability_langgraph_exante.json")
    args = ap.parse_args()
    rows = build_rows(args.records, args.spans, args.traces_dir, args.manifest)
    strata = strata_table(
        rows, ["category", "cls", "cls2", "node", "position", "state_effect",
               "control_phase", "inside_loop", "output_contract",
               "verifier_type", "context_source", "cross_app"])
    corr = corr_table(rows)
    succ = success_table(rows, ["category", "cls", "cls2", "node", "position",
                                "control_phase", "inside_loop",
                                "output_contract"])
    fdisc = feature_discrimination(rows)
    print_tables(strata, corr, rows, succ, fdisc)
    json.dump({"spans": rows, "strata": strata, "corr": corr,
               "success": succ, "feature_disc": fdisc},
              open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("\nDetails:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
