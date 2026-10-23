#!/usr/bin/env python
"""Offline downstream token reconstruction (backfill for the engine 2026-09-12.3 batch).

Principle: the downstream trajectory fully records the prefix session + each round's model output and
tool observations, so the approximate prompt length of each round can be reconstructed from the message
sequence. Because the reconstructed message format is approximate (the simplified agent's real message
construction happens inside the harness, and the server-side tools protocol overhead is also not in the
records), records **with real usage** (the last 3 traces of the same batch) are used for linear calibration:

    real_prompt       = a_p * est_prompt       + b_p
    real_completion   = a_c * est_completion   + b_c

Calibration coefficients are written into replay_meta.token_reconstruction of each record, keeping the data
caliber traceable. Only records with missing accounting (downstream_prompt_tokens == 0) are backfilled;
records that already have real accounting are left untouched.

Usage (local, repository root):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/reconstruct_downstream_tokens.py \
      --calib out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_383cbac_1.jsonl \
      --calib out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_383cbac_3.jsonl \
      --calib out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_4ec8de5_1.jsonl \
      --files out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_21abae1_1.jsonl \
              out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_23cf851_1.jsonl \
              out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_23cf851_2.jsonl \
              out_span_offload/offload_profile_simplified_Qwen3_8-27B_Qwen_Qwen3_5-9B_23cf851_3.jsonl \
      --in-place
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ENCODING_NAME = "cl100k_base"  # tiktoken general encoding (approximate for Qwen family, calibration absorbs the bias)


def _load_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding(ENCODING_NAME)
    except Exception:
        return None


_ENC = _load_tokenizer()


def ntok(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)  # rough fallback when tiktoken is unavailable


def estimate(rec: dict) -> tuple:
    """Reconstruct (est_prompt, est_completion, n_llm_calls) from the record's downstream trajectory."""
    dt = rec.get("downstream_trace") or {}
    msgs = [dict(m) for m in (dt.get("prefix") or {}).get("messages", [])
            if isinstance(m, dict)]
    est_p = 0
    est_c = 0
    n_llm = 0
    for e in (dt.get("spans") or []):
        if e.get("span_type") == "llm.call":
            n_llm += 1
            est_p += ntok(json.dumps(msgs, ensure_ascii=False))
            out = e.get("output") or {}
            est_c += ntok(json.dumps(out, ensure_ascii=False))
            msgs.append({"role": "assistant",
                         "content": json.dumps(out, ensure_ascii=False)})
        elif e.get("span_type") == "tool.execution":
            txt = str(e.get("result") or e.get("error") or "")
            msgs.append({"role": "tool", "content": txt})
    return est_p, est_c, n_llm


def ols(xs, ys):
    """Univariate linear regression y = a*x + b (hand-written least squares, avoiding numpy dependency)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 0 else 1.0
    b = my - a * mx
    # R^2
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", action="append", default=[],
                    help="record files with real downstream usage (calibration set)")
    ap.add_argument("--files", nargs="+", required=True,
                    help="jsonl files that need backfilling")
    ap.add_argument("--in-place", action="store_true",
                    help="write back in place (default only prints, does not write)")
    args = ap.parse_args()

    # ---- calibration set ----
    calib = []
    for p in args.calib:
        with open(p, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if l:
                    calib.append(json.loads(l))
    print("calibration record count: %d" % len(calib))
    xs_p, ys_p, xs_c, ys_c, per_round_c = [], [], [], [], []
    for r in calib:
        res = r.get("resource") or {}
        rp = res.get("downstream_prompt_tokens", 0) or 0
        rc = res.get("downstream_completion_tokens", 0) or 0
        if rp <= 0 and rc <= 0:
            continue  # this record itself has no real accounting (not captured), skip
        ep, ec, n_llm = estimate(r)
        if ep <= 0:
            continue
        xs_p.append(ep); ys_p.append(rp)
        xs_c.append(ec); ys_c.append(rc)
        if n_llm > 0:
            per_round_c.append(rc / n_llm)
    if not xs_p:
        print("error: calibration set has no usable records"); return 1
    a_p, b_p, r2_p = ols(xs_p, ys_p)
    # completion barely correlates with the recorded output (R^2~0.09): Qwen 27B is a reasoning
    # model, real completion is dominated by thinking, and only the final tool_calls are recorded.
    # Best estimate = calibration-set per-round completion median × number of rounds
    mu_c_round = sorted(per_round_c)[len(per_round_c) // 2] if per_round_c else 0.0
    print("prompt    calibration: real = %.4f*est %+.0f  (R^2=%.3f, n=%d)" % (
        a_p, b_p, r2_p, len(xs_p)))
    print("completion calibration: per-round median %.0f tokens (estimate = rounds × median, n=%d round observations)" % (
        mu_c_round, len(per_round_c)))

    def cal(p, c, n_llm):
        return (max(0, int(round(a_p * p + b_p))),
                max(0, int(round(n_llm * mu_c_round))))

    # ---- backfill ----
    n_patched = 0
    for path in args.files:
        lines = open(path, encoding="utf-8").read().splitlines()
        out = []
        n_file = 0
        for l in lines:
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            res = r.get("resource") or {}
            rp = res.get("downstream_prompt_tokens", 0) or 0
            rc = res.get("downstream_completion_tokens", 0) or 0
            if rp > 0 or rc > 0:
                out.append(l)  # already has real accounting, leave untouched
                continue
            ep, ec, n_llm = estimate(r)
            np_, nc_ = cal(ep, ec, n_llm)
            res["downstream_prompt_tokens"] = np_
            res["downstream_completion_tokens"] = nc_
            res["context_load"] = (res.get("target_prompt_tokens", 0) or 0) + np_
            r.setdefault("replay_meta", {})["token_reconstruction"] = {
                "method": "offline-calibrated-v1",
                "encoding": ENCODING_NAME,
                "prompt_fit": [a_p, b_p, r2_p],
                "completion_per_round_median": mu_c_round,
                "n_llm_calls": n_llm,
                "raw_estimate": [ep, ec],
            }
            out.append(json.dumps(r, ensure_ascii=False))
            n_patched += 1
            n_file += 1
        print("%s: backfilled %d records" % (path, n_file))
        if args.in_place:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(out) + "\n")
    print("total backfilled: %d records" % n_patched)
    if not args.in_place:
        print("(dry-run, not written; add --in-place after confirming the numbers look right)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
