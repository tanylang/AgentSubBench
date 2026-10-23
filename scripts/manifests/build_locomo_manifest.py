#!/usr/bin/env python
"""Builds the LocoMo TaskManifest (data/locomo/task_manifest.json).

Data source: data/locomo/locomo10.json (the official ACL 2024 full set, 10
samples). Each sample = one task (task_id = sample_id); the gold qa list goes
into the evaluator payload (the full corpus is loaded directly from
locomo10.json at runtime, not stored in the manifest).

Usage:
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe scripts/manifests/build_locomo_manifest.py \
      --out data/locomo/task_manifest.json
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
    ap.add_argument("--data-file",
                    default="data/locomo/locomo10.json")
    ap.add_argument("--out", default="data/locomo/task_manifest.json")
    args = ap.parse_args()

    data_file = args.data_file
    if not os.path.isabs(data_file):
        data_file = os.path.join(ROOT, data_file)
    samples = json.load(open(data_file, encoding="utf-8"))

    manifest_tasks = []
    for s in samples:
        sample_id = s["sample_id"]
        conv = s["conversation"]
        speaker_a = conv.get("speaker_a", "")
        speaker_b = conv.get("speaker_b", "")
        n_sessions = sum(1 for k in conv if k.startswith("session_")
                         and isinstance(conv[k], list))
        statement = (
            "LocoMo %s: answer questions about a %d-session conversation "
            "between %s and %s" % (sample_id, n_sessions, speaker_a, speaker_b))
        manifest_tasks.append(ManifestTask(
            task_id=sample_id,
            statement=statement,
            task_type="locomo",
            involved_apps=[],
            cross_app=False,
            horizon="long",
            task_state_effect="read_only",
            branching="none",
            evaluator_type="answer_match",
            collateral_risk="low",
            split="pilot",
            world_variant="locomo10",
            evaluator={
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "qa": s.get("qa", []),
            },
        ))
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "manifest_version": MANIFEST_SCHEMA_VERSION,
            "tasks": [t.__dict__ for t in manifest_tasks],
        }, f, ensure_ascii=False, indent=1)
    m = load_manifest(args.out)
    print("manifest: %d tasks (pilot=%d heldout=%d)" % (
        len(m.tasks),
        sum(1 for t in m.tasks if t.split == "pilot"),
        sum(1 for t in m.tasks if t.split == "heldout")))
    print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
