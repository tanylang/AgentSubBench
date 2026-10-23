#!/usr/bin/env python
"""Builds the SWE-bench TaskManifest (data/swebench/task_manifest.json).

Data sources: the trace_record project
  - tasks/swebench_tasks/swebench_verified_tasks.json (500 records, all fields)
  - tasks/swebench_tasks/swebench_verified_50.json (list of 50 instance_ids)
  - traces/swebench_mini-swe-agent/*.json (collected traces, one task each)

split assignment:
  - collected ∩ verified_50 -> pilot (pilot set)
  - collected − verified_50  -> heldout (natural held-out set)
Tasks not collected do not enter the manifest (not needed for replay).

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/manifests/build_swebench_manifest.py \
      --trace-record-root C:/Users/Lenovo/Desktop/trace_record \
      --traces-dir-name swebench_mini-swe-agent \
      --out data/swebench/task_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from span_offload.schemas import MANIFEST_SCHEMA_VERSION, ManifestTask  # noqa: E402
from span_offload.trace_io import load_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-record-root", required=True)
    ap.add_argument("--traces-dir-name", default="swebench_mini-swe-agent")
    ap.add_argument("--out", default="data/swebench/task_manifest.json")
    args = ap.parse_args()

    root = args.trace_record_root
    tasks_file = os.path.join(root, "tasks", "swebench_tasks",
                              "swebench_verified_tasks.json")
    sel_file = os.path.join(root, "tasks", "swebench_tasks",
                            "swebench_verified_50.json")
    traces_dir = os.path.join(root, "traces", args.traces_dir_name)
    tasks = {t["instance_id"]: t for t in
             json.load(open(tasks_file, encoding="utf-8"))}
    selected = set(json.load(open(sel_file, encoding="utf-8")))
    collected = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(traces_dir)
                       if f.endswith(".json"))

    manifest_tasks = []
    missing = []
    for iid in collected:
        t = tasks.get(iid)
        if t is None:
            missing.append(iid)
            continue
        split = "pilot" if iid in selected else "heldout"
        manifest_tasks.append(ManifestTask(
            task_id=iid,
            statement=t.get("problem_statement", ""),
            task_type="swebench",
            involved_apps=[t.get("repo", "")],
            cross_app=False,
            horizon="long",
            task_state_effect="state_mutating",
            branching="loop",
            evaluator_type="unit_tests",
            collateral_risk="low",
            split=split,
            world_variant="swebench_verified",
            evaluator={
                "instance_id": iid,
                "repo": t.get("repo"),
                "base_commit": t.get("base_commit"),
                "FAIL_TO_PASS": t.get("FAIL_TO_PASS", []),
                "PASS_TO_PASS": t.get("PASS_TO_PASS", []),
                "test_patch": t.get("test_patch", "") or "",
            },
        ))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "manifest_version": MANIFEST_SCHEMA_VERSION,
            "tasks": [t.__dict__ for t in manifest_tasks],
        }, f, ensure_ascii=False, indent=1)
    # verify it can be read back by trace_io.load_manifest
    m = load_manifest(args.out)
    print("manifest: %d tasks (pilot=%d heldout=%d)" % (
        len(m.tasks),
        sum(1 for t in m.tasks if t.split == "pilot"),
        sum(1 for t in m.tasks if t.split == "heldout")))
    if missing:
        print("warning: %d collected tasks not in the 500 list: %s" % (len(missing), missing))
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
