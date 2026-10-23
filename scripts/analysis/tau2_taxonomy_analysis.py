#!/usr/bin/env python
"""tau2 span taxonomy redefinition (output type × task stage) stratified gate analysis.

The tau2 agent has no simplified-style three-stage separation (node_pattern is all in-loop argument
generation); use two axes available before offload instead:
  task stage: position (early/middle/late, by position within the trace)
  output type: batch (single-shot 1 / batched 2-4, tool call count of the recorded output)
               write (output contains a write tool / all read-only)
Per-stratum noise-corrected gate: ΔSucc (two-layer CI) / NetSaving (two-layer CI) / ρ_med /
zero-detour (d_delta) / collateral UCB / MW p.
"""
from __future__ import annotations
import glob
import json
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from tau2_analysis import (_total, _down, _valid, _d_cross, _d_self,
                           wilson_ucb, mw_test, _boot2)  # noqa: E402
from span_offload.appworld_adapter import (  # noqa: E402
    parse_llm_output, APWORLD_TOOL_EFFECTS,
)

MIN_ROUNDS = 5
ZERO_DETOUR_DNORM = 0.1


def effect_of(name):
    key = str(name or "").replace("__", ".")
    return APWORLD_TOOL_EFFECTS.get(key, "external_state_mutation")


def main() -> int:
    records = []
    for p in sorted(glob.glob("out_span_offload/tau2/screening/*.jsonl")):
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

    out_feat = {}
    for t in set(k[0] for k in complete):
        d = json.load(open(
            "check_point_traces/appworld_tau2_qwen3_8-27b_batch1/%s.json" % t,
            encoding="utf-8"))
        llms = [s for s in d["spans"] if s.get("type") == "llm.call"]
        for i, s in enumerate(llms):
            a = s.get("attributes") or {}
            parsed, kind = parse_llm_output(a.get("gen_ai.output.messages", ""))
            calls = (parsed or {}).get("tool_calls", []) if isinstance(parsed, dict) else []
            names = [str(c.get("name", "")) for c in calls]
            out_feat[(t, i)] = {
                "n_calls": len(names),
                "has_write": any(effect_of(n) == "external_state_mutation"
                                 for n in names),
            }

    spans_map = {}
    for l in open("out_span_offload/tau2/spans_tau2_qwen27b.jsonl", encoding="utf-8"):
        l = l.strip()
        if l:
            s = json.loads(l)
            spans_map[(s["task_id"], s["invocation_index"])] = s
    max_inv = defaultdict(int)
    for (t, _), s in spans_map.items():
        max_inv[t] = max(max_inv[t], s["invocation_index"])

    rows = []
    for (t, i), arms in sorted(complete.items()):
        recT, recB, recI = arms["T_span"], arms["B"], arms["I"]
        f = out_feat.get((t, i), {})
        batch = "single" if f.get("n_calls", 0) <= 1 else "multi"
        wflag = "write" if f.get("has_write") else "read_only"
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
        rows.append({
            "task": t, "inv": i, "position": pos, "batch": batch, "write": wflag,
            "n_calls": f.get("n_calls"),
            "strict_T": statistics.mean(rt_T), "strict_B": statistics.mean(rt_B),
            "strict_I": statistics.mean(int(bool(r["strict_success"])) for r in recI),
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

    from collections import Counter
    print("taxonomy distribution:")
    print("  position:", dict(Counter(s["position"] for s in rows)))
    print("  batch:   ", dict(Counter(s["batch"] for s in rows)))
    print("  write:   ", dict(Counter(s["write"] for s in rows)))

    strata = []
    for key, label in (("position", "position"), ("batch", "batch"),
                       ("write", "write")):
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
        print("%-12s %2s %5s %5s %5s %8s %16s %9s %16s %6s %5s %6s %8s %8s" % (
            "stratum", "n", "stT", "stB", "stI", "ΔSucc", "[LCB,UCB](2-layer)",
            "NetSav", "[LCB,UCB](2-layer)", "ρ_med", "zeroDet%", "collUCB",
            "MWp(strict)", "MWp(tok)"))
        for s_ in strata[-len(grp):]:
            print("%-12s %2d %5.2f %5.2f %5.2f %+8.3f [%+.3f,%+.3f] %+9.0f [%+.0f,%+.0f] %6.1f %5.0f%% %6.3f %8.3f %8.3f" % (
                s_["stratum"], s_["n"], s_["strict_T"], s_["strict_B"],
                s_["strict_I"], s_["d_success"], s_["d_success_lcb"],
                s_["d_success_ucb"], s_["net_saving"], s_["net_saving_lcb"],
                s_["net_saving_ucb"], s_["rho_comp_median"] or 0,
                100 * s_["zero_detour_delta_ratio"], s_["collateral_wilson_ucb"],
                s_["mw_strict_p"], s_["mw_tokens_p"]))

    with open("results/tau2/tau2_taxonomy_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"n_spans": len(rows), "per_span": rows, "strata": strata},
                  f, ensure_ascii=False, indent=1)
    print("\nDetails: results/tau2/tau2_taxonomy_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
