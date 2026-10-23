#!/usr/bin/env python
"""tau2 batch span offload analysis v2 (three-way outcome caliber).

Fixes two problems in v1:
  1) no longer filter by "downstream token > 0" — v1 mistakenly dropped "inject-end" rounds
     (the injected output itself calls complete_task, downstream zero calls) as truncated
     rounds, biasing the downstream mean of later spans;
  2) outcomes are split into three classes, reported separately:
       inject_end   inject-end (downstream 0 calls, empty terminal, first span contains
                    complete_task) — the intervention's downstream cost is zero
       normal_done  normal completion (terminal.status == completed)
       normal_fail  normal failure (continuation ran but did not complete)

Four parts:
  1. metrics summary table (all rounds, 41 spans, T:B / T:I / B:I paired bootstrap)
  2. success group vs failure group (6 round-level groups)
  3. offloadability metrics (structure/fine-class stratified gate + continuous feature correlation, signs fixed)
  4. inject-end channel analysis (inject-end rate × strict per arm, by span position)
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
from span_offload.appworld_adapter import (  # noqa: E402
    parse_llm_output, APWORLD_TOOL_EFFECTS,
)

MIN_ROUNDS = 5
ZERO_DETOUR_DNORM = 0.1
BOOT = 2000
SEED = 42
Z = 1.644854

RECORD_DIR = "out_span_offload/tau2/screening"
TRACES_DIR = "check_point_traces/appworld_tau2_qwen3_8-27b_batch1"
SPANS_PATH = "out_span_offload/tau2/spans_tau2_qwen27b.jsonl"
OUT = "results/tau2/tau2_analysis_v2.json"


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


def outcome_type(r):
    """Three-way classification: inject_end / normal_done / normal_fail."""
    dt = r.get("downstream_trace") or {}
    spans = dt.get("spans") or []
    terminal = dt.get("terminal") or {}
    if _down(r) == 0 and spans:
        s0 = spans[0]
        calls = (s0.get("output") or {}).get("tool_calls", [])
        if any("complete_task" in str(c.get("name", "")) for c in calls):
            return "inject_end"
    if terminal.get("status") == "completed":
        return "normal_done"
    return "normal_fail"


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


def _boot2(g, ka, kb, nboot=BOOT, seed=SEED):
    rng = random.Random(seed)
    draws = []
    for _ in range(nboot):
        sample = rng.choices(g, k=len(g))
        s = 0.0
        for sp in sample:
            a = statistics.mean(rng.choices(sp[ka], k=MIN_ROUNDS))
            b = statistics.mean(rng.choices(sp[kb], k=MIN_ROUNDS))
            s += a - b
        draws.append(s / len(sample))
    draws.sort()
    return draws[int(0.025 * nboot)], draws[int(0.975 * nboot)]


def effect_of(name):
    key = str(name or "").replace("__", ".")
    return APWORLD_TOOL_EFFECTS.get(key, "external_state_mutation")


def main() -> int:
    records = []
    for p in sorted(glob.glob(os.path.join(RECORD_DIR, "*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                records.append(json.loads(l))
            except ValueError:
                continue
    arm = lambda r: (r.get("replay_meta") or {}).get("arm", "")  # noqa: E731
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        if arm(r) in ("T_span", "B", "I"):
            by[(r["task_id"], r["invocation_index"])][arm(r)].append(r)
    complete = {k: v for k, v in by.items()
                if all(len(v.get(a, [])) >= MIN_ROUNDS for a in ("T_span", "B", "I"))}
    print("all-round spans %d (≥%d rounds/arm: %d)" % (len(by), MIN_ROUNDS, len(complete)))

    spans_map = {}
    for l in open(SPANS_PATH, encoding="utf-8"):
        l = l.strip()
        if l:
            s = json.loads(l)
            spans_map[(s["task_id"], s["invocation_index"])] = s
    max_inv = defaultdict(int)
    for (t, _), s in spans_map.items():
        max_inv[t] = max(max_inv[t], s["invocation_index"])
    base_map = build_base_map(TRACES_DIR)

    out_feat = {}
    for t in set(k[0] for k in complete):
        try:
            d = json.load(open(os.path.join(TRACES_DIR, "%s.json" % t), encoding="utf-8"))
        except Exception:
            continue
        llms = [s for s in d["spans"] if s.get("type") == "llm.call"]
        seen = set()
        for i, s in enumerate(llms):
            a = s.get("attributes") or {}
            parsed, kind = parse_llm_output(a.get("gen_ai.output.messages", ""))
            calls = (parsed or {}).get("tool_calls", []) if isinstance(parsed, dict) else []
            names = [str(c.get("name", "")) for c in calls]
            novel = any(n not in seen for n in names) if names else False
            for n in names:
                seen.add(n)
            nc = len(names)
            out_feat[(t, i)] = {
                "call": "no_call" if nc == 0 else ("single" if nc == 1 else "multi"),
                "write": ("write" if any(effect_of(n) == "external_state_mutation"
                                         for n in names) else "read_only") if names else "no_call",
                "novel": "new" if novel else "repeat",
                "n_calls": nc,
            }

    rows = []
    for (t, i), arms in sorted(complete.items()):
        recT, recB, recI = (arms[a][:10] for a in ("T_span", "B", "I"))
        sd = recT[0].get("span_descriptor") or {}
        np_ = sd.get("node_pattern", "other")
        pos = ("early" if i / max(max_inv[t], 1) <= 0.34 else
               ("late" if i / max(max_inv[t], 1) >= 0.67 else "middle"))
        f = out_feat.get((t, i), {})
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
            "task": t, "inv": i, "node": np_, "position": pos,
            "call": f.get("call", "?"), "write": f.get("write", "?"),
            "novel": f.get("novel", "?"), "n_calls": f.get("n_calls"),
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
            "type_T": dict(Counter(outcome_type(r) for r in recT)),
            "type_B": dict(Counter(outcome_type(r) for r in recB)),
            "type_I": dict(Counter(outcome_type(r) for r in recI)),
            "rt_T": rt_T, "rt_B": rt_B, "rt_I": rt_I,
            "rt_Ttot": rt_Ttot, "rt_Btot": rt_Btot,
            "rt_Ttgt": rt_Ttgt, "rt_Btgt": rt_Btgt,
            "rt_Tdwn": rt_Tdwn, "rt_Bdwn": rt_Bdwn,
            "rt_colT": rt_colT,
        })
    print("node distribution:", dict(Counter(s["node"] for s in rows)))
    print("fine-class distribution: call=%s write=%s novel=%s" % (
        dict(Counter(s["call"] for s in rows)),
        dict(Counter(s["write"] for s in rows)),
        dict(Counter(s["novel"] for s in rows))))

    # ============ 4. inject-end channel (reported first, it is the focus of this version) ============
    print("\n" + "=" * 88)
    print("4. inject-end channel analysis (all rounds)")
    print("=" * 88)
    all_by_arm = {"T_span": [], "B": [], "I": []}
    for (t, i), arms in complete.items():
        for a, recs in (("T_span", "T_span"), ("B", "B"), ("I", "I")):
            all_by_arm[a].extend(arms[recs][:10])
    for a in ("T_span", "B", "I"):
        rs = all_by_arm[a]
        types = Counter(outcome_type(r) for r in rs)
        n = len(rs)
        ie = [r for r in rs if outcome_type(r) == "inject_end"]
        nd = [r for r in rs if outcome_type(r) == "normal_done"]
        nf = [r for r in rs if outcome_type(r) == "normal_fail"]
        print("%-7s n=%d: inject_end %3d (%.0f%%, strict=%.2f) | normal_done %3d (%.0f%%, strict=%.2f) | normal_fail %3d (%.0f%%, strict=%.2f)" % (
            a, n, len(ie), 100 * len(ie) / n,
            sum(1 for r in ie if r["strict_success"]) / max(len(ie), 1),
            len(nd), 100 * len(nd) / n,
            sum(1 for r in nd if r["strict_success"]) / max(len(nd), 1),
            len(nf), 100 * len(nf) / n,
            sum(1 for r in nf if r["strict_success"]) / max(len(nf), 1)))
    # position distribution of inject-end rounds
    print("\nspan position of inject-end rounds (T arm):")
    ie_spans = [s for s in rows if s["type_T"].get("inject_end", 0) > 0]
    print("  spans containing inject-end rounds: %d/%d" % (len(ie_spans), len(rows)))
    for s in ie_spans:
        print("  %-10s inv%-2d %-24s pos=%-7s T:%d/10 B:%d/10 I:%d/10" % (
            s["task"], s["inv"], s["node"], s["position"],
            s["type_T"].get("inject_end", 0), s["type_B"].get("inject_end", 0),
            s["type_I"].get("inject_end", 0)))

    # ============ 1. metrics summary table ============
    print("\n" + "=" * 88)
    print("1. metrics summary table (all rounds, %d spans)" % len(rows))
    print("=" * 88)
    span_means = {}
    for s in rows:
        k = (s["task"], s["inv"])
        span_means[k] = {}
        for a, recs in (("T_span", complete[k]["T_span"][:10]),
                        ("B", complete[k]["B"][:10]),
                        ("I", complete[k]["I"][:10])):
            span_means[k][a] = {}
            for v in CORE_TABLE:
                xs = [x for x in (cf_metrics(r, base_map.get(k))[v] for r in recs)
                      if x == x]
                span_means[k][a][v] = statistics.mean(xs) if xs else float("nan")
    for pair in ("T_span:B", "T_span:I", "B:I"):
        a, b = pair.split(":")
        print("\n--- %s vs %s ---" % (a, b))
        print("%-24s %9s %9s %+9s %s" % ("metric", "mu_" + a, "mu_" + b, "Δ", "95%CI"))
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
            print("%-24s %9.3f %9.3f %+9.3f [%+.3f,%+.3f] %s" % (
                v, statistics.mean(span_means[(s["task"], s["inv"])][a][v] for s in rows),
                statistics.mean(span_means[(s["task"], s["inv"])][b][v] for s in rows),
                mu, lo, hi, "★" if (lo > 0 or hi < 0) else ""))
        tk = {"T_span": "total_T", "B": "total_B", "I": "total_I"}
        for name in ("total",):
            diffs = [s[tk[a]] - s[tk[b]] for s in rows]
            mu = statistics.mean(diffs)
            rng = random.Random(SEED)
            boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                           for _ in range(BOOT))
            print("%-24s %9.0f %9.0f %+9.0f [%+.0f,%+.0f] %s" % (
                "TotalTokens", statistics.mean(s[tk[a]] for s in rows),
                statistics.mean(s[tk[b]] for s in rows), mu, boots[50], boots[1949],
                "★" if (boots[50] > 0 or boots[1949] < 0) else ""))
        if pair == "T_span:B":
            for name in ("direct_saving", "downstream_comp", "net_saving"):
                diffs = [s[name] for s in rows]
                mu = statistics.mean(diffs)
                rng = random.Random(SEED)
                boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                               for _ in range(BOOT))
                print("%-24s %9s %9s %+9.0f [%+.0f,%+.0f] %s" % (
                    name, "-", "-", mu, boots[50], boots[1949],
                    "★" if (boots[50] > 0 or boots[1949] < 0) else ""))

    # ============ 2. success group vs failure group ============
    print("\n" + "=" * 88)
    print("2. success group vs failure group (pooled across rounds, 6 groups)")
    print("=" * 88)
    groups = defaultdict(list)
    for (t, i), arms in complete.items():
        for a, recs in (("T_span", arms["T_span"][:10]), ("B", arms["B"][:10]),
                        ("I", arms["I"][:10])):
            for r in recs:
                groups["%s%s" % (a, "成" if r["strict_success"] else "败")].append((t, i, r))
    print("%-24s %10s %10s %10s %10s %10s %10s" % (
        "variable", "Tsucc", "Tfail", "Bsucc", "Bfail", "Isucc", "Ifail"))
    sf = []
    for v in CORE_TABLE:
        row = {"var": v}
        for gname in ("T_span成", "T_span败", "B成", "B败", "I成", "I败"):
            items = groups.get(gname, [])
            if not items:
                row[gname] = None
                continue
            t0, i0, r0 = items[0]
            row[gname] = statistics.mean(
                cf_metrics(r, base_map.get((t0, i0)))[v] for _, _, r in items)
        sf.append(row)
        print("%-24s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f" % (
            v, row["T_span成"], row["T_span败"], row["B成"], row["B败"],
            row["I成"], row["I败"]))
    print("(rounds per group:", {k: len(v) for k, v in groups.items()}, ")")

    # ============ 3. offloadability metrics ============
    print("\n" + "=" * 88)
    print("3. offloadability metrics (stratified gate + continuous features)")
    print("=" * 88)
    strata = []
    for key, label in (("node", "node"), ("position", "position"),
                       ("call", "call"), ("write", "write"), ("novel", "novel")):
        grp = defaultdict(list)
        for s in rows:
            grp[str(s[key])].append(s)
        for name, g in sorted(grp.items()):
            if len(g) < 3:
                continue
            lcb_d, ucb_d = _boot2(g, "rt_T", "rt_B")
            lcb_n, ucb_n = _boot2(g, "rt_Ttot", "rt_Btot")
            rho = [s["rho_comp"] for s in g if abs(s["rho_comp"]) < 1e4]
            coll_n = sum(len(s["rt_colT"]) for s in g)
            coll_x = sum(sum(s["rt_colT"]) for s in g)
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
            })
        print("\n== %s ==" % label)
        print("%-14s %2s %5s %5s %5s %8s %16s %9s %16s %6s %5s %6s" % (
            "stratum", "n", "stT", "stB", "stI", "ΔSucc", "[LCB,UCB](2-layer)",
            "NetSav", "[LCB,UCB](2-layer)", "ρ_med", "zeroDet%", "collUCB"))
        for s_ in strata[-len(grp):]:
            print("%-14s %2d %5.2f %5.2f %5.2f %+8.3f [%+.3f,%+.3f] %+9.0f [%+.0f,%+.0f] %6.1f %5.0f%% %6.3f" % (
                s_["stratum"], s_["n"], s_["strict_T"], s_["strict_B"],
                s_["strict_I"], s_["d_success"], s_["d_success_lcb"],
                s_["d_success_ucb"], s_["net_saving"], s_["net_saving_lcb"],
                s_["net_saving_ucb"], s_["rho_comp_median"] or 0,
                100 * s_["zero_detour_delta_ratio"], s_["collateral_wilson_ucb"]))

    # continuous features (sign-fixed version)
    print("\n== continuous features × outcomes (Spearman, two-layer bootstrap CI) ==")
    # tuple = (ak, bk), ys = mean(bk) − mean(ak)
    targets = {"net_saving": ("rt_Ttot", "rt_Btot"),
               "direct_saving": ("rt_Ttgt", "rt_Btgt"),
               "downstream_comp": ("rt_Bdwn", "rt_Tdwn"),
               "d_success": ("rt_B", "rt_T")}
    corr = []
    rng = random.Random(SEED)
    for f in ("baseline_tokens", "max_remaining_steps", "n_allowed_tools"):
        if not all(s.get(f) is not None for s in rows):
            pass
    # baseline_tokens backfilled from the spans file
    for s in rows:
        sp = spans_map.get((s["task"], s["inv"]), {})
        if s.get("baseline_tokens") is None:
            s["baseline_tokens"] = sp.get("baseline_tokens")
        if s.get("max_remaining_steps") is None:
            s["max_remaining_steps"] = ((sp.get("replay_input") or {}).get("continuation")
                                        or {}).get("max_remaining_steps")
        if s.get("n_allowed_tools") is None:
            s["n_allowed_tools"] = len(((sp.get("replay_input") or {})
                                        .get("output_contract_runtime") or {})
                                       .get("allowed_tools") or [])
    for f in ("baseline_tokens", "max_remaining_steps", "n_allowed_tools"):
        for tg, (ak, bk) in targets.items():
            pairs = [(s[f], s) for s in rows if s[f] is not None]
            if len(pairs) < 8 or len({p[0] for p in pairs}) < 3:
                continue
            xs = [p[0] for p in pairs]
            ys = [statistics.mean(p[1][bk]) - statistics.mean(p[1][ak]) for p in pairs]
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

    out = {"n_spans": len(rows), "per_span": rows, "strata": strata, "corr": corr,
           "success_failure": sf}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
