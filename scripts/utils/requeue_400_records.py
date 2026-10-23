#!/usr/bin/env python
"""Requeue-prep and backfill tool for 400-failed records (2026-09-20 batch1 only).

Records whose terminal.status starts with 'failed: Error code: 400' are
deterministic failures of "27B over-long thinking -> empty content -> illegal assistant message in the protocol" (same seed always
reproduces, already 15-18% of batch1). These records occupy
(span, arm, round) slots and pollute the failure-rate statistics.

Three-stage workflow (only the failed slots need rerunning, not all 10 rounds):
  1) --quarantine: move the 400 records out of the main jsonl, archive to
     <stem>_400failed_<timestamp>.jsonl, and record each record's **original line number** into <archive>.manifest.json (for backfill in place);
  2) rerun with the same parameters as the original batch + RESUME=1 (see the
     sbatch guide printed at the end of the script) -- run_offload_profile's
     resume mechanism skips (span, arm, round) still in the files, **only
     backfilling the removed slots** (one rerun round per 400 record);
  3) --merge-back <manifest>: insert the new records from the rerun back into
     the **original positions** of the original jsonl by original line number,
     file order identical to the original run; slots not backfilled and
     records still 400 after the rerun are reported explicitly.

⚠ Before rerunning, the trajectory must change, otherwise the same seed deterministically reproduces the same 400:
  - preferred: server-side/request-level reasoning_effort=medium or thinking_token_budget has taken
    effect (A/B verify first, templates may silently ignore it);
  - alternative: change the baseline seed (multi_replay_slurm.sh supports the BASELINE_SEED env var, default
    100) -- note this changes that round's downstream sampling, mark it in the analysis.
⚠ The rerun must keep ROUNDS / IDENTITY_ROUNDS / BASELINE_ROUNDS / BLOCK_SEED identical to the original run,
  otherwise the (arm, round) plan drifts and resume breaks, causing a full rerun.

Usage (server, inside container):
  python scripts/utils/requeue_400_records.py \
      --results out_span_offload/simplified/batch1/simplified/screening \
      --dry-run
  python scripts/utils/requeue_400_records.py \
      --results out_span_offload/simplified/batch1/simplified/screening \
      --quarantine
  # ... after the RESUME=1 rerun finishes ...
  python scripts/utils/requeue_400_records.py \
      --results out_span_offload/simplified/batch1/simplified/screening \
      --merge-back <manifest path printed by quarantine>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

FAIL_PREFIX = "failed: Error code: 400"


def _scan(results: str, pattern: str):
    """Scans the results dir (recursively) for jsonl files matching pattern, returns
    {file: {"keep": [lines], "bad": [dict(with line/index/keys)]}},
    file is the path relative to results (used to locate backfill positions via manifest)."""
    files = sorted(glob.glob(os.path.join(results, "**", pattern),
                             recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(results, pattern)))
    if not files:
        print("no files matching %s found (results=%s) -- no historical results to process"
              % (pattern, results))
        return {}
    out = {}
    for fp in files:
        keep, bad = [], []
        with open(fp, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    keep.append(line)  # leftover lines kept as-is, excluded from the verdict
                    continue
                term = (((d.get("downstream_trace") or {}).get("terminal")
                         or {}).get("status") or "")
                if term.startswith(FAIL_PREFIX):
                    bad.append({
                        "line": line,
                        "index": idx,  # original line number (for backfill in place)
                        "task": d.get("task_id"),
                        "inv": d.get("invocation_index"),
                        "span_id": d.get("span_id"),
                        "arm": (d.get("slm_output") or {}).get("target_arm"),
                        "round": (d.get("replay_meta") or {}).get("profile_round"),
                    })
                else:
                    keep.append(line)
        out[os.path.relpath(fp, results)] = {"keep": keep, "bad": bad}
    return out


def _report(scan):
    bad_all = [b for v in scan.values() for b in v["bad"]]
    print("=" * 72)
    print("400 failure scan: %d records total (across %d files)" % (len(bad_all), len(scan)))
    if not bad_all:
        print("no 400 records, nothing to do.")
        return []
    per_file = {fp: len(v["bad"]) for fp, v in scan.items()}
    for fp, n in sorted(per_file.items()):
        print("  %-4d  %s" % (n, os.path.basename(fp)))
    per_span = defaultdict(list)
    for b in bad_all:
        per_span[(b["task"], b["inv"])].append(b)
    print()
    print("by span (printed when this span has >=4 400 records):")
    for (task, inv), bs in sorted(per_span.items()):
        arm_c = Counter(b["arm"] for b in bs)
        flag = ""
        if len(bs) >= 4:
            flag = " <- multi-round failure"
        print("  %-10s inv=%-2d n=%d arms=%s%s" % (
            task, inv, len(bs), dict(arm_c), flag))
    # whole-arm write-off detection: vs the keep side, an arm is truly dead only if its good rounds = 0
    total_per_arm = defaultdict(lambda: defaultdict(int))
    for v in scan.values():
        for line in v["keep"]:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            arm = (d.get("slm_output") or {}).get("target_arm")
            if arm:
                total_per_arm[(d.get("task_id"),
                               d.get("invocation_index"))][arm] += 1
    for b in bad_all:
        total_per_arm[(b["task"], b["inv"])][b["arm"]] += 1
    print()
    print("arm write-off detection (arm total rounds > 0 and good rounds = 0):")
    arm_dead = []
    for (task, inv), bs in sorted(per_span.items()):
        for arm in ("I", "T_span", "B"):
            n_bad = len([b for b in bs if b["arm"] == arm])
            n_total = total_per_arm[(task, inv)].get(arm, 0)
            if n_total and n_bad == n_total:
                arm_dead.append((task, inv, arm, n_total))
    for t, inv, a, n in arm_dead:
        print("  %-10s inv=%-2d %-7s %d rounds all 400 <- truly dead" % (t, inv, a, n))
    if not arm_dead:
        print("  (none)")
    return bad_all


def _quarantine(scan, results: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive = os.path.join(results, "400failed_%s.jsonl" % ts)
    manifest_path = archive + ".manifest.json"
    manifest = []
    total_bad = 0
    with open(archive, "w", encoding="utf-8") as af:
        for rel, v in scan.items():
            if not v["bad"]:
                continue
            fp = os.path.join(results, rel)
            total_bad += len(v["bad"])
            with open(fp, "w", encoding="utf-8") as f:
                for line in v["keep"]:
                    f.write(line + "\n")
            for b in v["bad"]:
                af.write(b["line"] + "\n")
                manifest.append({
                    "file": rel,              # path relative to results
                    "index": b["index"],      # original line number, for backfill in place
                    "span_id": b["span_id"],
                    "arm": b["arm"],
                    "round": b["round"],
                })
            print("  %-4d moved out  %s" % (len(v["bad"]), rel))
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=1)
    # pointer file: lets the slurm wrapper locate the manifest after the backfill run
    with open(os.path.join(results, "requeue_manifest.latest"),
              "w", encoding="utf-8") as pf:
        pf.write(manifest_path)
    print()
    print("moved out %d 400 records total, archive: %s" % (total_bad, archive))
    print("line-number manifest (for backfill): %s" % manifest_path)
    print("main files now keep only normal records (usable record counts no longer include 400s).")
    return archive, manifest_path


def _merge_back(results: str, manifest_path: str):
    """Inserts the new records produced by the rerun back into the original jsonl at their original line numbers."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    by_file = defaultdict(list)
    for m in manifest:
        by_file[m["file"]].append(m)
    n_inserted = n_missing = n_still400 = 0
    missing = []
    for fname, entries in by_file.items():
        fp = os.path.join(results, fname)  # fname = path relative to results
        if not os.path.exists(fp):
            missing.extend((e["span_id"], e["arm"], e["round"]) for e in entries)
            n_missing += len(entries)
            print("  [missing file] %s (%d slots not backfilled)" % (fname, len(entries)))
            continue
        keys = {(e["span_id"], e["arm"], e["round"]) for e in entries}
        new_by_key = {}
        kept = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    kept.append(line)
                    continue
                k = (d.get("span_id"),
                     (d.get("slm_output") or {}).get("target_arm"),
                     (d.get("replay_meta") or {}).get("profile_round"))
                if k in keys:
                    new_by_key[k] = line
                else:
                    kept.append(line)
        # insert back in ascending original line order, rebuilding the original order
        rebuilt = list(kept)
        for e in sorted(entries, key=lambda x: x["index"]):
            k = (e["span_id"], e["arm"], e["round"])
            if k not in new_by_key:
                missing.append(k)
                n_missing += 1
                continue
            line = new_by_key[k]
            try:
                d = json.loads(line)
                term = (((d.get("downstream_trace") or {}).get("terminal")
                         or {}).get("status") or "")
                if term.startswith(FAIL_PREFIX):
                    n_still400 += 1
            except ValueError:
                pass
            rebuilt.insert(min(e["index"], len(rebuilt)), line)
            n_inserted += 1
        with open(fp, "w", encoding="utf-8") as f:
            for line in rebuilt:
                f.write(line + "\n")
        print("  %-4d backfilled  %s" % (
            sum(1 for e in entries
                if (e["span_id"], e["arm"], e["round"]) in new_by_key),
            fname))
    print()
    print("backfill done: %d inserted, %d not filled, %d still 400 after the rerun"
          % (n_inserted, n_missing, n_still400))
    if missing:
        print("unfilled slots (first 10): %s" % missing[:10])
    if n_still400:
        print("note: %d backfill records are still 400 -- thinking limit not"
              " effective or seed unchanged; fix it before rerunning." % n_still400)
    return 0 if not n_missing else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="batch output dir (contains *batch1_*.jsonl etc.)")
    ap.add_argument("--pattern", default="*batch1_*.jsonl",
                    help="filename glob (default *batch1_*.jsonl)")
    ap.add_argument("--dry-run", action="store_true",
                    help="scan and report only, do not touch files")
    ap.add_argument("--quarantine", action="store_true",
                    help="move 400 records out of the main files, archive + record line numbers")
    ap.add_argument("--merge-back", default="",
                    help="after the rerun finishes, insert backfill records in place by <manifest.json>")
    args = ap.parse_args()
    modes = [args.dry_run, args.quarantine, bool(args.merge_back)]
    if sum(modes) != 1:
        ap.error("must specify exactly one of --dry-run / --quarantine / --merge-back")

    if args.merge_back:
        return _merge_back(args.results, args.merge_back)

    scan = _scan(args.results, args.pattern)
    _report(scan)

    if args.quarantine:
        if not scan or not any(v["bad"] for v in scan.values()):
            print()
            print("no 400 records to quarantine, no manifest generated (in"
                  " REQUEUE_400 mode the backfill step will be skipped).")
            return 0
        print()
        print("=" * 72)
        _, manifest_path = _quarantine(scan, args.results)

    print()
    print("=" * 72)
    if args.quarantine:
        print("rerun guide (only backfill the removed slots, no need for all 10 rounds):")
    else:
        print("rerun guide (follow this after --quarantine):")
    print("  1. confirm the thinking limit has taken effect (otherwise the same seed will reproduce the same 400):")
    print("     A/B verify reasoning_effort=medium / thinking_token_budget,")
    print("     or switch to BASELINE_SEED to change downstream sampling (mark the convention!)")
    print("  2. rerun with the same parameters as the original batch (ROUNDS=10/10/10,BLOCK_SEED=0,")
    print("     RESUME=1, same OUT_SUFFIX; SPANS_FILE/addresses as the original batch):")
    print()
    print("     TRACE_DIR=check_point_traces/appworld_simplified_qwen3_8-27b_batch1 \\")
    print("     SPANS_FILE=<spans file of the original batch> \\")
    print("     ROUNDS=10 IDENTITY_ROUNDS=10 BASELINE_ROUNDS=10 MODE=both \\")
    print("     BLOCK_SEED=0 FALLBACK=none \\")
    print("     SLM_BASE_URL=<SLM address> SLM_MODEL=Qwen/Qwen3.5-9B \\")
    print("     BASELINE_MODEL=Qwen3.8-27B BASELINE_BASE_URL=<27B address> \\")
    print("     SPAN_OFFLOAD_CONT_BACKEND=simplified RESUME=1 \\")
    print("     OUT_SUBDIR=simplified/batch1 OUT_SUFFIX=_batch1 \\")
    print("     sbatch slurm/multi_replay_slurm.sh")
    print()
    print("     # if the thinking limit did not take effect, alternatively change downstream sampling (must mark the convention in analysis):")
    print("     # BASELINE_SEED=101 sbatch ...")
    print("  3. after the rerun finishes, insert the backfill records back in place by original line number:")
    if args.quarantine:
        print("     python scripts/utils/requeue_400_records.py \\")
        print("         --results %s \\" % args.results)
        print("         --merge-back %s" % manifest_path)
    else:
        print("     python scripts/utils/requeue_400_records.py --results <dir> \\")
        print("         --merge-back <manifest path printed by quarantine>")
    print("  4. check: total record count returns to the pre-move level, 400")
    print("     share drops significantly, file order matches the original run; keep 400failed_<ts>.jsonl archive for audit.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
