"""Native trace IO for Stage 10.

A native trace is one successful baseline run r of a frozen representative
agent H on an AppWorld task t, stored as a single JSON file per run:

    traces/<agent_id>__<task_id>__<run_id>.json

The trace keeps bulky payloads (state snapshots, message lists, tool schemas)
in a top-level `artifacts` dict referenced by *_ref fields, so spans stay
compact and the same snapshot object is shared by every replay started from it.
"""
from __future__ import annotations

import glob
import json
import os
from typing import List

from .schemas import TraceRun, TaskManifest, from_dict, to_dict, TRACE_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION


def save_trace(trace: TraceRun, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(trace), f, ensure_ascii=False, indent=1)
    return path


def load_trace(path: str) -> TraceRun:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trace = from_dict(TraceRun, data)
    if trace.schema_version != TRACE_SCHEMA_VERSION:
        raise ValueError("trace %s has schema %r, expected %r"
                         % (path, trace.schema_version, TRACE_SCHEMA_VERSION))
    return trace


def iter_traces(traces_dir: str) -> List[TraceRun]:
    traces: List[TraceRun] = []
    pattern = os.path.join(traces_dir, "*.json")
    for path in sorted(glob.glob(pattern)):
        # skip manifest files stored alongside traces
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("schema_version") != TRACE_SCHEMA_VERSION:
            continue
        traces.append(from_dict(TraceRun, data))
    return traces


def save_manifest(manifest: TaskManifest, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(manifest), f, ensure_ascii=False, indent=1)
    return path


def load_manifest(path: str) -> TaskManifest:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = from_dict(TaskManifest, data)
    if manifest.manifest_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest %s has version %r, expected %r"
                         % (path, manifest.manifest_version, MANIFEST_SCHEMA_VERSION))
    return manifest
