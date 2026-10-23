#!/usr/bin/env python
"""Builds the frozen task manifest from the six agents' checkpoint traces (pure trace derivation).

TaskCondition may only be read from the manifest (14.2.3). The original
pilot manifest has only the 0d8a4ee_1 entry; this script scans all task_ids
of the six batches under check_point_traces/ **without reading the original
task data** -- all conditioning fields are derived from the traces themselves:

  - task_id / statement: trace top-level fields (check_point.task_id / task)
  - involved_apps:   the set of apps from tool.execution records across all
                     traces of the task (the app part of tool name "app.api"; union over the task)
  - cross_app:       len(involved_apps) > 1
  - task_type:       "+".join(sorted(involved_apps)) (no official category, honestly labeled)
  - horizon / task_state_effect / branching / evaluator_type /
    collateral_risk: fixed defaults (AppWorld tasks are all multi-step
                     state-mutating; original task data not involved)

Usage:
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 python scripts/manifests/build_task_manifest.py
    [--out data/appworld/task_manifest.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRACE_DIRS = [
    "check_point_traces/appworld_simplified_batch1",
    "check_point_traces/appworld_autogen_batch1",
    "check_point_traces/appworld_legacy_batch1",
    "check_point_traces/appworld_openai_agents_batch1",
    "check_point_traces/appworld_smolagents_batch1",
    "check_point_traces/appworld_tau2_batch1",
]


def collect_from_traces(root: str) -> dict[str, dict]:
    """{task_id: {statement, apps}} -- all from the trace files themselves."""
    found: dict[str, dict] = {}
    for d in TRACE_DIRS:
        tdir = os.path.join(root, d)
        if not os.path.isdir(tdir):
            continue
        for f in sorted(os.listdir(tdir)):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(tdir, f), encoding="utf-8") as fp:
                    data = json.load(fp)
            except (ValueError, OSError):
                continue
            task_id = None
            for s in data.get("spans", []):
                cp = s.get("check_point")
                if cp and cp.get("task_id"):
                    task_id = cp["task_id"]
                    break
            if not task_id:
                continue
            entry = found.setdefault(task_id, {"statement": "", "apps": set()})
            if not entry["statement"]:
                entry["statement"] = data.get("task", "") or ""
            for s in data.get("spans", []):
                if s.get("type") != "tool.execution":
                    continue
                name = str((s.get("attributes") or {}).get(
                    "gen_ai.tool.name", ""))
                # tool names come in two separator forms (app__api / app.api); ignore invalid names
                sep = "__" if "__" in name else ("." if "." in name else None)
                if sep:
                    app = name.split(sep, 1)[0]
                    if app:
                        entry["apps"].add(app)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(root, "data/appworld",
                                                  "task_manifest.json"))
    args = ap.parse_args()

    found = collect_from_traces(root)
    print("trace tasks found: %d" % len(found))

    out_path = args.out
    existing: dict = {}
    if os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as fp:
            existing = json.load(fp)

    # keep the pilot hand-crafted entry as-is (its task_type is the manually labeled phone_messaging)
    entries: list[dict] = list(existing.get("tasks") or [])
    known = {t["task_id"] for t in entries}
    for tid in sorted(found):
        if tid in known:
            continue
        info = found[tid]
        apps = sorted(info["apps"])
        entries.append({
            "task_id": tid,
            "statement": info["statement"],
            # no official category: honestly label by the app combination actually executed in the traces
            "task_type": "+".join(apps) if apps else "appworld",
            "involved_apps": apps,
            "cross_app": len(apps) > 1,
            "horizon": "long",  # AppWorld tasks are all multi-step long-horizon
            "task_state_effect": "state_mutating",
            "branching": "none",
            "evaluator_type": "appworld_final_evaluator",
            "collateral_risk": "unknown",
            "split": "batch1",
            "world_variant": "default",
            "evaluator": {},
        })
    entries.sort(key=lambda t: t["task_id"])

    manifest = {
        "manifest_version": existing.get("manifest_version")
        or "stage10-task-manifest/v1",
        "description": ("Manifest for the six check_point_traces batches. "
                        "TaskCondition is read ONLY from this manifest (14.2.3); "
                        "entries derived from the traces themselves (executed "
                        "tool apps), the pilot entry is hand-crafted."),
        "tasks": entries,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    print("manifest written: %s (%d tasks)" % (out_path, len(entries)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
