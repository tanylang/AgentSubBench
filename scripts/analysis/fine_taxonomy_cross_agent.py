#!/usr/bin/env python
"""Fine-grained span type taxonomy (cross-agent offloadability metrics).

All four axes are parsed from trace record outputs, available before offload, same caliber for both agents:
  pos    position (early/middle/late, tertiles of position within the trace)
  call   output call shape (no_call=free text/answer / single / multi)
  write  write-ness (read_only=recorded output is all read-only tools / write=contains a write tool)
  novel  tool novelty (new=at least one tool was never called before / repeat=all old tools)
Per-stratum gate (noise-corrected): ΔSucc / NetSaving (two-layer bootstrap CI), ρ_med,
zero-detour (d_delta), collateral UCB, MW p; classes present in both agents are compared.
"""

from __future__ import annotations
import glob
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from trace_edit_distance import edit_distance, signature  # noqa: E402
from span_offload.appworld_adapter import (  # noqa: E402
    parse_llm_output, APWORLD_TOOL_EFFECTS,
)

MIN_ROUNDS = 5
ZERO_DETOUR_DNORM = 0.1
BOOT = 2000
SEED = 42


def effect_of(name):
    key = str(name or "").replace("__", ".")
    return APWORLD_TOOL_EFFECTS.get(key, "external_state_mutation")


def _total(r):
    res = r.get("resource") or {}
    return ((res.get("target_prompt_tokens", 0) or 0)
            + (res.get("target_completion_tokens", 0) or 0)
            + (res.get("downstream_prompt_tokens", 0) or 0)
            + (res.get("downstream_completion_tokens", 0) or 0))


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


def output_features(trace_dir, spans_keys):
    """Parse per-span call shape / write-ness / novelty from trace record outputs."""
    feats = {}
    for t in sorted({k[0] for k in spans_keys}):
        p = os.path.join(trace_dir, "%s.json" % t)
        try:
            d = json.load(open(p, encoding="utf-8"))
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
            n_calls = len(names)
            call = "no_call" if n_calls == 0 else ("single" if n_calls == 1 else "multi")
            write = ("write" if any(effect_of(n) == "external_state_mutation" for n in names)
                     else "read_only") if names else "no_call"
            feats[(t, i)] = {"call": call, "write": write, "n_calls": n_calls,
                             "novel": "new" if novel else "repeat"}
    return feats


def build_rows(record_glob, spans_path, trace_dir, base_map, valid_filter=None,
               max_rounds=10, pick_dedup=False):
    files = sorted(glob.glob(record_glob))
    if pick_dedup:
        chosen = {}
        for p in files:
            m = re.search(r"([0-9a-f]{6,8}_\d)\.jsonl$", os.path.basename(p))
            if not m:
                continue
            task = m.group(1)
            is_new = "qwen3_8-27b_Qwen" in os.path.basename(p)
            if task not in chosen or is_new:
                chosen[task] = p
        files = sorted(chosen.values())
    records = []
    for p in files:
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except ValueError:
                continue
            if valid_filter and not valid_filter(r):
                continue
            records.append(r)
    arm = lambda r: (r.get("replay_meta") or {}).get("arm", "")  # noqa: E731
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        if arm(r) in ("T_span", "B", "I"):
            by[(r["task_id"], r["invocation_index"])][arm(r)].append(r)
    complete = {k: v for k, v in by.items()
                if all(len(v.get(a, [])) >= MIN_ROUNDS for a in ("T_span", "B", "I"))}
    spans_map = {}
    for l in open(spans_path, encoding="utf-8"):
        l = l.strip()
        if l:
            s = json.loads(l)
            spans_map[(s["task_id"], s["invocation_index"])] = s
    max_inv = defaultdict(int)
    for (t, _), s in spans_map.items():
        max_inv[t] = max(max_inv[t], s["invocation_index"])
    feats = output_features(trace_dir, set(complete))
    rows = []
    for (t, i), arms in sorted(complete.items()):
        recT, recB = arms["T_span"][:max_rounds], arms["B"][:max_rounds]
        f = feats.get((t, i), {})
        pos = ("early" if i / max(max_inv[t], 1) <= 0.34 else
               ("late" if i / max(max_inv[t], 1) >= 0.67 else "middle"))
        rt_T = [int(bool(r["strict_success"])) for r in recT]
        rt_B = [int(bool(r["strict_success"])) for r in recB]
        rt_Ttot = [_total(r) for r in recT]
        rt_Btot = [_total(r) for r in recB]
        rt_Ttgt = [_total(r) - _down(r) for r in recT]
        rt_Btgt = [_total(r) - _down(r) for r in recB]
        rt_Tdwn = [_down(r) for r in recT]
        rt_Bdwn = [_down(r) for r in recB]
        rt_colT = [int((r.get("final_outcome") or {}).get("collateral_damage", False))
                   for r in recT]
        dcross = _d_cross(recT, recB)
        ddelta = max(0.0, dcross - min(_d_self(recT), _d_self(recB)))
        sd = recT[0].get("span_descriptor") or {}
        rows.append({
            "task": t, "inv": i, "pos": pos,
            "call": f.get("call", "?"), "write": f.get("write", "?"),
            "novel": f.get("novel", "?"), "n_calls": f.get("n_calls"),
            "node": sd.get("node_pattern", "other"),
            "strict_T": statistics.mean(rt_T), "strict_B": statistics.mean(rt_B),
            "d_success": statistics.mean(rt_T) - statistics.mean(rt_B),
            "net_saving": statistics.mean(rt_Btot) - statistics.mean(rt_Ttot),
            "direct_saving": statistics.mean(rt_Btgt) - statistics.mean(rt_Ttgt),
            "downstream_comp": statistics.mean(rt_Tdwn) - statistics.mean(rt_Bdwn),
            "rho_comp": (statistics.mean(rt_Tdwn) - statistics.mean(rt_Bdwn))
                        / max(statistics.mean(rt_Btgt) - statistics.mean(rt_Ttgt), 1e-9),
            "d_delta": ddelta, "zero_detour_delta": ddelta < ZERO_DETOUR_DNORM,
            "rt_T": rt_T, "rt_B": rt_B, "rt_Ttot": rt_Ttot, "rt_Btot": rt_Btot,
            "rt_colT": rt_colT,
        })
    return rows


def stratum_table(rows, key, label, min_n=3):
    grp = defaultdict(list)
    for s in rows:
        grp[str(s[key])].append(s)
    out = []
    for name, g in sorted(grp.items()):
        if len(g) < min_n:
            continue
        lcb_d, ucb_d = _boot2(g, "rt_T", "rt_B")
        lcb_n, ucb_n = _boot2(g, "rt_Ttot", "rt_Btot")
        rho = [s["rho_comp"] for s in g if abs(s["rho_comp"]) < 1e4]
        coll_n = sum(len(s["rt_colT"]) for s in g)
        coll_x = sum(sum(s["rt_colT"]) for s in g)
        out.append({
            "key": label, "stratum": name, "n": len(g),
            "strict_T": statistics.mean(s["strict_T"] for s in g),
            "strict_B": statistics.mean(s["strict_B"] for s in g),
            "d_success": statistics.mean(s["d_success"] for s in g),
            "d_success_lcb": lcb_d, "d_success_ucb": ucb_d,
            "net_saving": statistics.mean(s["net_saving"] for s in g),
            "net_saving_lcb": lcb_n, "net_saving_ucb": ucb_n,
            "rho_comp_median": statistics.median(rho) if rho else None,
            "zero_detour_delta_ratio": sum(s["zero_detour_delta"] for s in g) / len(g),
            "collateral_wilson_ucb": wilson_ucb(coll_x, coll_n),
        })
    return out


def wilson_ucb(x, n, z=1.644854):
    if n == 0:
        return 1.0
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    import math
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return min(1.0, center + half)


def main() -> int:
    from span_offload.controlflow_metrics import build_base_map
    sim_rows = build_rows(
        "out_span_offload/simplified/screening/offload_profile_simplified_*.jsonl",
        "out_span_offload/spans/spans_simplified_qwen27b_50.jsonl",
        "check_point_traces/appworld_simplified_qwen3_8-27b_batch1",
        build_base_map("check_point_traces/appworld_simplified_qwen3_8-27b_batch1"),
        max_rounds=5, pick_dedup=True)
    t2_rows = build_rows(
        "out_span_offload/tau2/screening/*.jsonl",
        "out_span_offload/tau2/spans_tau2_qwen27b.jsonl",
        "check_point_traces/appworld_tau2_qwen3_8-27b_batch1",
        build_base_map("check_point_traces/appworld_tau2_qwen3_8-27b_batch1"),
        valid_filter=lambda r: _down(r) > 0)
    print("simplified rows: %d | tau2 rows: %d" % (len(sim_rows), len(t2_rows)))
    for name, rows in (("simplified", sim_rows), ("tau2", t2_rows)):
        from collections import Counter
        print("%s fine-class distribution:" % name)
        print("  pos:  ", dict(Counter(s["pos"] for s in rows)))
        print("  call: ", dict(Counter(s["call"] for s in rows)))
        print("  write:", dict(Counter(s["write"] for s in rows)))
        print("  novel:", dict(Counter(s["novel"] for s in rows)))

    all_strata = {}
    for name, rows in (("simplified", sim_rows), ("tau2", t2_rows)):
        print("\n" + "=" * 100)
        print("%s fine-taxonomy gate table" % name)
        print("=" * 100)
        all_strata[name] = {}
        for key, label in (("pos", "pos"), ("call", "call"), ("write", "write"),
                           ("novel", "novel")):
            tab = stratum_table(rows, key, label)
            all_strata[name][label] = tab
            print("\n== %s ==" % label)
            print("%-10s %2s %5s %5s %8s %16s %9s %16s %6s %5s %6s" % (
                "stratum", "n", "stT", "stB", "ΔSucc", "[LCB,UCB]", "NetSav",
                "[LCB,UCB]", "ρ_med", "zeroDet%", "collUCB"))
            for s_ in tab:
                print("%-10s %2d %5.2f %5.2f %+8.3f [%+.3f,%+.3f] %+9.0f [%+.0f,%+.0f] %6.1f %5.0f%% %6.3f" % (
                    s_["stratum"], s_["n"], s_["strict_T"], s_["strict_B"],
                    s_["d_success"], s_["d_success_lcb"], s_["d_success_ucb"],
                    s_["net_saving"], s_["net_saving_lcb"], s_["net_saving_ucb"],
                    s_["rho_comp_median"] or 0, 100 * s_["zero_detour_delta_ratio"],
                    s_["collateral_wilson_ucb"]))

    # cross dimensions (2-way, only when n>=4)
    print("\n" + "=" * 100)
    print("two-way cross classes (comparison of classes shared by both agents)")
    print("=" * 100)
    cross = []
    for k1, k2 in (("pos", "call"), ("pos", "write"), ("write", "novel")):
        print("\n== %s × %s ==" % (k1, k2))
        for name, rows in (("simplified", sim_rows), ("tau2", t2_rows)):
            grp = defaultdict(list)
            for s in rows:
                grp["%s×%s" % (s[k1], s[k2])].append(s)
            print("  %-10s:" % name, {k: len(v) for k, v in sorted(grp.items())
                                      if len(v) >= 4})
        # gate comparison for shared classes
        common = {}
        for cls in ("early×multi", "early×single", "middle×multi", "early×write",
                    "early×read_only", "middle×write", "write×new", "write×repeat",
                    "read_only×new", "read_only×repeat"):
            for name, rows in (("simplified", sim_rows), ("tau2", t2_rows)):
                g = [s for s in rows if "%s×%s" % (s[k1], s[k2]) == cls]
                if len(g) >= 4:
                    lcb_d, _ = _boot2(g, "rt_T", "rt_B")
                    rho = [s["rho_comp"] for s in g if abs(s["rho_comp"]) < 1e4]
                    common.setdefault(cls, {})[name] = {
                        "n": len(g),
                        "d_success": statistics.mean(s["d_success"] for s in g),
                        "d_success_lcb": lcb_d,
                        "rho_median": statistics.median(rho) if rho else None,
                        "zero": statistics.mean(1 if s["zero_detour_delta"] else 0 for s in g),
                    }
            if cls in common and len(common[cls]) == 2:
                cross.append((cls, common[cls]))
                a, b = common[cls]["simplified"], common[cls]["tau2"]
                same_d = (a["d_success"] > 0) == (b["d_success"] > 0)
                same_rho = (a["rho_median"] > 1) == (b["rho_median"] > 1)
                print("%-16s sim(n=%d): ΔSucc %+.3f ρ=%s zeroDet %.0f%% | tau2(n=%d): ΔSucc %+.3f ρ=%s zeroDet %.0f%% | Δ same sign:%s ρ>1 consistent:%s" % (
                    cls, a["n"], a["d_success"],
                    ("%.0f" % a["rho_median"]) if a["rho_median"] is not None else "—",
                    100 * a["zero"], b["n"], b["d_success"],
                    ("%.0f" % b["rho_median"]) if b["rho_median"] is not None else "—",
                    100 * b["zero"], same_d, same_rho))

    out = {"strata": all_strata, "cross": cross,
           "sim_rows": sim_rows, "tau2_rows": t2_rows}
    with open("results/cross_agent/fine_taxonomy_cross_agent.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/cross_agent/fine_taxonomy_cross_agent.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
