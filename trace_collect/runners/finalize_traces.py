"""Required post-collection finalization for traces: validation + annotation.

Must be run after collecting traces for every benchmark:
  1. Validator (evaluate_one in runners/evaluate_traces.py):
     replay env_changes to restore the final state, run the official AppWorld evaluator,
     and write pass / pass_details back into each trace file
  2. event_type annotation (classify_spans in analysis/add_event_types.py):
     annotate every span with its event type (planning/action/observation/...)

Usage (standalone):
    python runners/finalize_traces.py <trace_dir> [--experiment-name NAME]

As a library (called automatically at the end of benchmark scripts' main()):
    from runners.finalize_traces import finalize_trace_dir
    finalize_trace_dir(OUTPUT_DIR, experiment_name="offline_eval_xxx")
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "runners"), os.path.join(ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _trace_files(trace_dir: str) -> list[str]:
    if not os.path.isdir(trace_dir):
        return []
    return sorted(
        os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
        if f.endswith(".json") and not f.endswith(".tmp")
    )


def evaluate_dir(trace_dir: str, experiment_name: str) -> dict:
    """Step 1: run the official validator over all traces in the directory, writing pass/pass_details back.

    Returns {evaluated, success, fail, errors} statistics.
    """
    from evaluate_traces import evaluate_one  # runners/evaluate_traces.py

    stats = {"evaluated": 0, "success": 0, "fail": 0, "errors": 0}
    files = _trace_files(trace_dir)
    for i, path in enumerate(files, 1):
        try:
            r = evaluate_one(path, experiment_name)
            if "error" in r:
                stats["errors"] += 1
                print(f"    [{i}/{len(files)}] {r['trace']}: ERROR {r['error']}")
            else:
                stats["evaluated"] += 1
                if r["success"]:
                    stats["success"] += 1
                else:
                    stats["fail"] += 1
                print(f"    [{i}/{len(files)}] {r['task_id']}: "
                      f"success={r['success']} pass={r['pass_count']}/{r['total_count']} "
                      f"({r['pass_percentage']:.0f}%)")
        except Exception as e:
            stats["errors"] += 1
            print(f"    [{i}/{len(files)}] {os.path.basename(path)}: EXCEPTION {e}")
    return stats


def annotate_dir(trace_dir: str) -> int:
    """Step 2: annotate event_type on the spans of all traces in the directory; returns the number of files processed.

    Delegates to annotate_trace_dir in analysis/add_event_types.py —
    the same authoritative implementation as the SWE-bench side (finalize_swe_traces), with consistent criteria.
    """
    from add_event_types import annotate_trace_dir  # analysis/add_event_types.py

    return annotate_trace_dir(trace_dir)


def _is_remote_agent_dir(trace_dir: str) -> bool:
    """Detect whether this directory needs the server-side final DB evaluation path.

    Remote-execution agents (OpenAI) under the old configuration (server-side in-memory) cannot collect
    env_changes (all empty) → fall back to server-side final DB evaluation;
    under the new configuration (on_disk=True) env_changes are truly collected → use the standard replay path.
    Therefore: only OpenAI traces with all-empty env_changes use the remote DB mode.
    """
    files = _trace_files(trace_dir)
    if not files:
        return False
    has_openai = False
    total_ec = 0
    has_snapshot = False
    for path in files[:3]:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        if "OpenAI" in str(data.get("agent_name", "")):
            has_openai = True
        for s in data.get("spans", []):
            cp = s.get("check_point")
            if not cp:
                continue
            if cp.get("snapshot_id"):
                has_snapshot = True
            for ch in cp.get("env_changes", {}).values():
                if isinstance(ch, list):
                    total_ec += len(ch)
    # Traces with a snapshot or env_changes use the standard replay path; old OpenAI traces with neither fall back to the remote DB
    return has_openai and total_ec == 0 and not has_snapshot


def evaluate_dir_remote_db(trace_dir: str, experiment_name: str) -> dict:
    """Evaluation for remote-execution agents: skip env_changes replay and use the server-side final DB directly.

    The AppWorld initializer persists the real final DB of each run at
    <appworld>/experiments/outputs/trace_builtin_oai_<task_id>/tasks/<task_id>/dbs;
    copy it to the location expected by evaluate_task and run the official evaluator directly.
    No changes to any appworld/SDK/agent source logic; pure collection-layer adaptation.
    """
    import evaluate_traces  # noqa: F401  (performs chdir + sys.path setup on import)
    from appworld.evaluator import evaluate_task as _evaluate_task
    import shutil

    OUTPUTS = os.path.join(ROOT, "benchmarks", "appworld", "experiments", "outputs")
    stats = {"evaluated": 0, "success": 0, "fail": 0, "errors": 0}
    files = _trace_files(trace_dir)
    for i, path in enumerate(files, 1):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        # Get the real task_id from the spans
        task_id = None
        for s in data.get("spans", []):
            cp = s.get("check_point")
            if cp and cp.get("task_id"):
                task_id = cp["task_id"]; break
            a = s.get("attributes", {})
            if a.get("benchmark_task_id"):
                task_id = a["benchmark_task_id"]; break
        if not task_id:
            stats["errors"] += 1
            print(f"    [{i}/{len(files)}] {os.path.basename(path)}: cannot determine task_id")
            continue
        # Server-side final DB location: data/.tmp.base_dbs when ON_DISK=true (overwritten on
        # every run), otherwise the client experiment directory (historical in-memory runs only have the client skeleton).
        if os.getenv("ON_DISK", "false").lower() == "true":
            from appworld.apps.lib.models.db import get_db_home_path
            src = get_db_home_path(storage_type="disk", type="base", temporary=True)
        else:
            src = os.path.join(OUTPUTS, f"trace_builtin_oai_{task_id}", "tasks", task_id, "dbs")
        if not os.path.isdir(src):
            stats["errors"] += 1
            print(f"    [{i}/{len(files)}] {task_id}: server-side DB does not exist (no matching run record): {src}")
            continue
        dst = os.path.join(OUTPUTS, experiment_name, "tasks", task_id, "dbs")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        try:
            tracker = _evaluate_task(task_id=task_id, experiment_name=experiment_name,
                                     suppress_errors=True)
            success = bool(tracker.success)
            data["pass"] = "success" if success else "fail"
            data["pass_details"] = {
                "pass_count": tracker.pass_count, "fail_count": tracker.fail_count,
                "total_count": tracker.total_count,
                "pass_percentage": tracker.pass_percentage,
            }
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2, default=str)
            stats["evaluated"] += 1
            if success:
                stats["success"] += 1
            else:
                stats["fail"] += 1
            print(f"    [{i}/{len(files)}] {task_id}: success={success} "
                  f"pass={tracker.pass_count}/{tracker.total_count} "
                  f"({tracker.pass_percentage:.0f}%)")
        except Exception as e:
            stats["errors"] += 1
            print(f"    [{i}/{len(files)}] {task_id}: EXCEPTION {e}")
    return stats


def finalize_trace_dir(trace_dir: str, experiment_name: str = "offline_eval") -> None:
    """Required post-collection operations: validation (write pass) → annotation (write event_type).

    Remote-execution agents (OpenAI) automatically use the server-side final DB evaluation path;
    all other agents use the env_changes replay path.
    """
    trace_dir = os.path.abspath(trace_dir)
    if not os.path.isdir(trace_dir):
        print(f"[finalize] directory does not exist, skipping: {trace_dir}")
        return
    remote = _is_remote_agent_dir(trace_dir)
    print(f"[finalize] step 1/2: validator (write pass) -> {trace_dir}"
          f"{' (remote DB mode)' if remote else ''}")
    if remote:
        stats = evaluate_dir_remote_db(trace_dir, experiment_name)
    else:
        stats = evaluate_dir(trace_dir, experiment_name)
    print(f"[finalize] validation: {stats['evaluated']} valid evaluations, "
          f"success={stats['success']}, fail={stats['fail']}, errors={stats['errors']}")
    print(f"[finalize] step 2/2: event_type annotation")
    n = annotate_dir(trace_dir)
    print(f"[finalize] done: {n} files annotated with event_type")


def main() -> None:
    parser = argparse.ArgumentParser(description="Required finalization after trace collection (validation + annotation)")
    parser.add_argument("trace_dir", help="trace directory")
    parser.add_argument("--experiment-name", default="offline_eval",
                        help="subdirectory name under experiment_outputs")
    args = parser.parse_args()
    finalize_trace_dir(args.trace_dir, experiment_name=args.experiment_name)


if __name__ == "__main__":
    main()
