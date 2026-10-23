#!/usr/bin/env python
"""Intervention dynamics analysis (gap 4): d_t curve + propagation/recovery distance.

Computed from downstream trajectories of 53 spans × (SLM 5 rounds + identity 5 rounds):
  - per-position divergence d_t(j) = match_i(t) − match_s(t)
    match_s(t) = fraction of SLM rounds whose tool name at position t matches the identity group mode
    match_i(t) = fraction of identity rounds matching their own mode (per-position noise floor)
  - derived quantities: τ_on (position where divergence appears), L_prop (propagation depth), L_rec (recovery distance),
    persistence (sustained fraction), amplification (magnification factor)
  - mean curves and sensitivity matrices by response class / node_pattern

Position alignment: position t = the t-th tool execution event after the injection point (repair→REPAIR,
failure→FAIL, ok→tool name; overlong→END). Common support t ≤ P75 of identity group lengths.

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/dynamics_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SLM_FILES = [
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_batch1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_1.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_383cbac_3.jsonl",
    "results/offload_profile_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl",
]
ID_FILES = [
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_21abae1_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_1.jsonl",
    "results/offload_profile_id_simplified_deepseek-v4-flash_Qwen_Qwen3_5-9B_23cf851_2.jsonl",
]
CLASS_FILE = "results/appworld_53span/response_analysis_clean53.json"
THETA = 0.4  # d_t divergence threshold (d_t=0.6 when ≥3 of 5 rounds diverge, 0.2 when ≥2)


def load(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs.extend(json.loads(l) for l in f if l.strip())
    return recs


def tool_sequence(record):
    """Tool execution sequence: repair→REPAIR, failure→FAIL, ok→tool name."""
    seq = []
    for e in (record.get("downstream_trace") or {}).get("spans", []):
        if e.get("span_type") != "tool.execution":
            continue
        if e.get("source") == "repair":
            seq.append("REPAIR")
        elif not e.get("ok"):
            seq.append("FAIL")
        else:
            seq.append(str(e.get("tool", "")))
    return seq


def dynamics(span_key, slm_seqs, id_seqs):
    """Per-position divergence curve + derived quantities. id_seqs provides the mode and noise floor."""
    lens = [len(s) for s in id_seqs]
    t_max = sorted(lens)[int(0.75 * (len(lens) - 1))] if lens else 0
    if t_max < 1:
        return None
    curve = []
    for t in range(1, t_max + 1):
        id_at = [s[t - 1] if len(s) >= t else "END" for s in id_seqs]
        mode = Counter(id_at).most_common(1)[0][0]
        match_i = sum(1 for x in id_at if x == mode) / len(id_at)
        sl_at = [s[t - 1] if len(s) >= t else "END" for s in slm_seqs]
        match_s = sum(1 for x in sl_at if x == mode) / len(sl_at)
        curve.append({"t": t, "d": match_i - match_s, "mode": mode})
    d = [c["d"] for c in curve]
    over = [i for i, x in enumerate(d) if x > THETA]  # 0-based
    if not over:
        onset, l_prop, l_rec = None, 0, 0
        persistence = 0.0
        amplification = 0.0
    else:
        onset = over[0] + 1
        l_prop = over[-1] + 1
        # recovery distance: first position after l_prop where "everything after is ≤θ" (1-based)
        rec = None
        for i in range(l_prop, len(d)):
            if all(x <= THETA for x in d[i:]):
                rec = i + 1
                break
        l_rec = rec  # None = never recovered (∞)
        persistence = len(over) / len(d)
        amplification = max(d[over[0]:]) / max(d[over[0]], 0.01)
    return {"curve": curve, "onset": onset, "l_prop": l_prop,
            "l_rec": l_rec, "persistence": persistence,
            "amplification": amplification, "t_max": t_max}


def main() -> int:
    slm = defaultdict(list)
    idd = defaultdict(list)
    # Note: new-task files also contain identity records inside (distinguished by the injection field)
    for r in load(SLM_FILES + ID_FILES):
        k = (r["task_id"], r["invocation_index"])
        (idd if r.get("replay_meta", {}).get("injection") == "identity"
         else slm)[k].append(r)
    classes = {r["task"] + "__" + str(r["inv"]): r["response_class"]
               for r in json.load(open(CLASS_FILE, encoding="utf-8"))["per_span"]}

    keys = sorted(set(slm) & set(idd))
    print("span count: %d" % len(keys))

    rows = []
    by_class = defaultdict(list)
    by_pattern = defaultdict(list)
    for k in keys:
        dyn = dynamics(k, [tool_sequence(r) for r in slm[k]],
                       [tool_sequence(r) for r in idd[k]])
        if dyn is None:
            continue
        cls = classes.get(k[0] + "__" + str(k[1]), "?")
        np_ = slm[k][0]["span_descriptor"]["node_pattern"]
        row = {"task": k[0], "inv": k[1], "node_pattern": np_,
               "class": cls, **{x: dyn[x] for x in
                                ("onset", "l_prop", "l_rec", "persistence",
                                 "amplification", "t_max")}}
        rows.append(row)
        by_class[cls].append(dyn)
        by_pattern[np_].append(dyn)

    # ---------- summary table ----------
    print("\n=== per-span dynamics (θ=%.1f) ===" % THETA)
    print("%-14s %-3s %-22s %6s %6s %6s %8s %8s" % (
        "task", "inv", "class", "onset", "L_prop", "L_rec", "persist", "amplif"))
    for r in rows:
        print("%-14s %-3d %-22s %6s %6s %6s %8.2f %8.2f" % (
            r["task"], r["inv"], r["class"],
            "∞" if r["onset"] is None else r["onset"],
            r["l_prop"], "∞" if r["l_rec"] is None else r["l_rec"],
            r["persistence"], r["amplification"]))

    # ---------- class aggregation ----------
    print("\n=== response class × dynamics shape ===")
    for cls in sorted(by_class):
        g = by_class[cls]
        n_never = sum(1 for d in g if d["onset"] is None)
        n_rec = sum(1 for d in g if d["onset"] is not None and d["l_rec"] is not None)
        n_pers = sum(1 for d in g if d["onset"] is not None and d["l_rec"] is None)
        print("  %-24s n=%2d | never diverged %d | recovered after diverging %d | sustained divergence %d" % (
            cls, len(g), n_never, n_rec, n_pers))

    # ---------- class mean curves (common support aligned per span) ----------
    print("\n=== class mean d_t curves (positions 1..8, '-' for no data) ===")
    for cls in sorted(by_class):
        g = by_class[cls]
        vals = []
        for t in range(1, 9):
            xs = [d["curve"][t - 1]["d"] for d in g if len(d["curve"]) >= t]
            vals.append("%+.2f" % (sum(xs) / len(xs)) if xs else "   -")
        print("  %-24s %s" % (cls, " ".join(vals)))

    # ---------- node_pattern × position sensitivity matrix ----------
    print("\n=== node_pattern × position mean d_t ===")
    for np_ in sorted(by_pattern):
        g = by_pattern[np_]
        vals = []
        for t in range(1, 9):
            xs = [d["curve"][t - 1]["d"] for d in g if len(d["curve"]) >= t]
            vals.append("%+.2f" % (sum(xs) / len(xs)) if xs else "   -")
        print("  %-26s n=%2d %s" % (np_, len(g), " ".join(vals)))

    # ---------- class C case studies: repair event positions ----------
    print("\n=== class C case studies: d_t curves ===")
    for k in keys:
        cls = classes.get(k[0] + "__" + str(k[1]), "")
        if cls.startswith("C"):
            dyn = dynamics(k, [tool_sequence(r) for r in slm[k]],
                           [tool_sequence(r) for r in idd[k]])
            print("  %s inv=%d:" % k, " ".join(
                "%+.1f" % c["d"] for c in dyn["curve"][:12]))
            for r in slm[k]:
                rp = [i + 1 for i, s in enumerate(tool_sequence(r))
                      if s == "REPAIR"]
                if rp:
                    print("    SLM round repair positions:", rp)
                    break

    # ---------- threshold sensitivity ----------
    print("\n=== threshold sensitivity (span count that never diverged) ===")
    for th in (0.3, 0.4, 0.5):
        n_never = sum(1 for r in rows
                      if dynamics((r["task"], r["inv"]),
                                  [tool_sequence(x) for x in slm[(r["task"], r["inv"])]],
                                  [tool_sequence(x) for x in idd[(r["task"], r["inv"])]]
                                  )["onset"] is None)
        print("  θ=%.1f → %d/%d never diverged" % (th, n_never, len(rows)))

    out = {"per_span": rows,
           "by_class": {c: len(g) for c, g in by_class.items()}}
    with open("results/appworld_53span/dynamics_53.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nDetails: results/appworld_53span/dynamics_53.json")

    # ---------- figure (optional) ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for cls in sorted(by_class):
            g = by_class[cls]
            ts, ms = [], []
            for t in range(1, 13):
                xs = [d["curve"][t - 1]["d"] for d in g if len(d["curve"]) >= t]
                if xs:
                    ts.append(t)
                    ms.append(sum(xs) / len(xs))
            if ts:
                ax.plot(ts, ms, marker="o", ms=3, label="%s (n=%d)" % (cls[:8], len(g)))
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(THETA, color="red", lw=0.8, ls="--",
                   label="θ=%.1f" % THETA)
        ax.set_xlabel("position t (tool executions after injection)")
        ax.set_ylabel("mean d_t (divergence beyond identity floor)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("results/misc/dynamics_by_class.png", dpi=150)
        print("figure: results/misc/dynamics_by_class.png")
    except ImportError:
        print("(matplotlib unavailable, skipping plot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
