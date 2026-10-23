"""Offline evaluation script — restore the final state from collected traces and feed it to the AppWorld evaluator to get the real success/fail.

Usage:
    python runners/evaluate_traces.py <trace_dir> [--experiment-name NAME] [--limit N]

For each <task_id>.json under trace_dir:
  1. replay restores the "final state" (apply every span's env_changes)
  2. save the DB snapshot to experiment_outputs/<name>/tasks/<task_id>/dbs
  3. call appworld.evaluator.evaluate_task to compare against ground truth
  4. write back pass/pass_details, and also annotate event_type (built-in finalization of the collection pipeline)
  5. summarize success/fail + pass/fail statistics

Relationship with the collection side:
  - The collection side stores env_changes (SQL deltas) + application_state_hash in each span's check_point.
  - This script rebuilds the DB from these deltas and then runs the official evaluator, so there is no need to collect the trace again.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld", "src"))
sys.path.insert(0, os.path.join(ROOT, "runners"))
_ORIGINAL_CWD = os.getcwd()
os.chdir(os.path.join(ROOT, "benchmarks", "appworld"))

from replay_env import (restore_world_at, load_trace, spans_with_checkpoint,  # noqa: E402
                        apply_changes)
from appworld.apps.lib.apis.local_remote import save_local_dbs  # noqa: E402
from appworld.common.path_store import path_store  # noqa: E402


def _annotate_event_types(trace_path: str) -> None:
    """Annotate event_type on the trace's spans (authoritative implementation: analysis/add_event_types.py).

    Used as the built-in finalization of the collection pipeline: called after evaluation writes pass back; idempotent, re-annotation has no side effects.
    """
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from add_event_types import annotate_trace_file
    annotate_trace_file(trace_path)


def _final_step(data: dict) -> int:
    """Return the step needed to "apply all span deltas" (= max agent.step + 1)."""
    spans = spans_with_checkpoint(data)
    if not spans:
        return 0
    return max(s.get("attributes", {}).get("agent.step", 0) for s in spans) + 1


def _apply_task_output_changes(world, task_id: str) -> dict | None:
    """OAI remote-mode fallback: when the trace has no per-span env_changes, use the full
    jsonl change log from the task output to restore the final state (experiments/outputs/<exp>/tasks/<tid>/dbs/*.jsonl).

    The changes saved by the server's world.save() = initialization writes + task writes. The initialization writes are
    deterministic (same task, same seed), exactly matching the tracker changes when the current evaluation world is constructed,
    so skip the prefix according to each app's initialization change count and apply only the real deltas from the task period.
    Returns {app: [Change tuples]} (also written into the trace's env_changes_final, so the data is not lost).
    """
    from appworld.common.imports import import_models_module

    dbs_dir = os.path.join(path_store.experiment_outputs,
                           f"trace_builtin_oai_{task_id}", "tasks", task_id, "dbs")
    if not os.path.isdir(dbs_dir):
        return None
    final_changes: dict = {}
    for jf in sorted(os.listdir(dbs_dir)):
        if not jf.endswith(".jsonl"):
            continue
        app = jf[:-len(".jsonl")]
        raw = [json.loads(line) for line in
               open(os.path.join(dbs_dir, jf), encoding="utf-8") if line.strip()]
        try:
            n_init = len(import_models_module(app).SQLModel.db.tracker.changes)
        except Exception:
            n_init = 0
        task_changes = raw[n_init:]
        if not task_changes:
            continue
        apply_changes(world, {app: task_changes})
        final_changes[app] = task_changes
    return final_changes or None


def _write_content_model_hashes(world, output_dir: str) -> None:
    """Compute per-table hashes from the actual DB content and write model_hashes.json into the dump directory as a supplement.

    Replay (apply_changes) writes to the DB via raw SQL, bypassing the ORM's increment_model_hash;
    save_local_dbs(save_model_hashes=False) then discards the hash file → on the evaluation side
    ModelCollection.load cannot read the hashes, both start/end sides are all 0, and _changed_model_names'
    coarse candidate filter is always empty → "assert model changes match {non-empty}" always fails.
    Here the content hash is computed by aggregating the record_hash of all records per table; the coarse filter only produces
    candidates, with the size/record-hash comparison as the fallback, so candidates that slip in by mistake are not misjudged as changed.
    """
    import hashlib

    from appworld.common.imports import import_models_module

    data: dict[str, dict[str, int]] = {}
    for app_name in [*world.task.allowed_apps, "admin", "supervisor"]:
        try:
            mod = import_models_module(app_name)
        except Exception:
            continue
        app_hashes: dict[str, int] = {}
        for model_name in mod.SQLModel.model_names():
            try:
                records = mod.SQLModel.model(model_name).all()
            except Exception:
                continue
            digest = hashlib.sha256()
            for record_hash in sorted(str(r.record_hash) for r in records):
                digest.update(record_hash.encode("utf-8"))
            app_hashes[model_name] = int.from_bytes(digest.digest()[:8], "big")
        if app_hashes:
            data[app_name] = app_hashes
    with open(os.path.join(output_dir, "model_hashes.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _save_end_db(world, experiment_name: str, task_id: str) -> str:
    """Save the restored final DB to the location expected by evaluate_task, and return that directory path."""
    output_dir = os.path.join(path_store.experiment_outputs, experiment_name, "tasks", task_id, "dbs")
    save_local_dbs(
        from_db_home_path=world.models_to_db_home_path,
        to_db_home_path=output_dir,
        format="full",
        app_names=[*world.task.allowed_apps, "admin", "supervisor"],
        delete_if_exists=True,
        skip_mandatory_apps=False,
        save_model_hashes=False,
    )
    _write_content_model_hashes(world, output_dir)
    return output_dir


def evaluate_one(trace_path: str, experiment_name: str) -> dict:
    """Restore + evaluate a single trace, returning a result dict."""
    from appworld.evaluator import evaluate_task

    data = load_trace(trace_path)
    spans = spans_with_checkpoint(data)
    if not spans:
        return {"trace": os.path.basename(trace_path), "error": "no check_point data"}

    task_id = (spans[0]["check_point"].get("task_id")
               or data["spans"][0]["attributes"].get("benchmark_task_id"))
    if not task_id:
        return {"trace": os.path.basename(trace_path), "error": "cannot determine task_id"}

    # 1. Restore the final state (apply all env_changes)
    world, info = restore_world_at(trace_path, _final_step(data))

    # 1b. OAI remote-mode fallback: when no span has env_changes, restore the final state from the task output jsonl
    n_span_changes = sum(
        len((s.get("check_point") or {}).get("env_changes") or {})
        for s in spans)
    if n_span_changes == 0:
        final_changes = _apply_task_output_changes(world, task_id)
        if final_changes:
            data["env_changes_final"] = final_changes
            print(f"    [oai-fallback] applied "
                  f"{sum(len(v) for v in final_changes.values())} final changes from the task output jsonl")

    try:
        # 2. Save the final DB snapshot
        _save_end_db(world, experiment_name, task_id)

        # 3. Run the official evaluator
        tracker = evaluate_task(task_id=task_id, experiment_name=experiment_name, suppress_errors=True)

        success = bool(tracker.success)

        # 4. Write the real evaluation result back into the trace data (top-level pass field + details)
        data["pass"] = "success" if success else "fail"
        data["pass_details"] = {
            "pass_count": tracker.pass_count,
            "fail_count": tracker.fail_count,
            "total_count": tracker.total_count,
            "pass_percentage": tracker.pass_percentage,
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        # Built-in finalization: annotate event_type (same source as finalize_trace_dir step 2, idempotent)
        try:
            _annotate_event_types(trace_path)
        except Exception as _e:
            print(f"    [annotate] {os.path.basename(trace_path)}: "
                  f"{type(_e).__name__}: {_e}")

        return {
            "task_id": task_id,
            "success": success,
            "pass_count": tracker.pass_count,
            "fail_count": tracker.fail_count,
            "total_count": tracker.total_count,
            "pass_percentage": tracker.pass_percentage,
        }
    finally:
        try:
            world.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation of collected traces (compared against ground truth)")
    parser.add_argument("trace_dir", nargs="?", help="trace directory (containing <task_id>.json)")
    parser.add_argument("--trace", help="evaluate only a single trace file")
    parser.add_argument("--experiment-name", default="offline_eval",
                        help="subdirectory name under experiment_outputs")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N (0=all)")
    args = parser.parse_args()

    # The script chdir's into appworld/ at the top; convert relative paths back to absolute paths here
    def _abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(_ORIGINAL_CWD, p)

    if args.trace:
        trace_paths = [_abs(args.trace)]
    else:
        if not args.trace_dir:
            parser.error("provide either trace_dir or --trace")
        trace_dir = _abs(args.trace_dir)
        trace_paths = sorted(
            p for p in
            (os.path.join(trace_dir, f) for f in os.listdir(trace_dir))
            if os.path.isfile(p) and p.endswith(".json")
        )
        if args.limit:
            trace_paths = trace_paths[: args.limit]

    results = []
    for i, tp in enumerate(trace_paths, 1):
        try:
            r = evaluate_one(tp, args.experiment_name)
            results.append(r)
            if "error" in r:
                print(f"[{i}/{len(trace_paths)}] {r['trace']}: ERROR {r['error']}")
            else:
                print(f"[{i}/{len(trace_paths)}] {r['task_id']}: "
                      f"success={r['success']} pass={r['pass_count']}/{r['total_count']} "
                      f"({r['pass_percentage']:.0f}%)")
        except Exception as e:
            results.append({"trace": os.path.basename(tp), "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(trace_paths)}] {os.path.basename(tp)}: EXCEPTION {e}")

    ok = [r for r in results if "error" not in r]
    success = [r for r in ok if r["success"]]
    print("\n" + "=" * 60)
    print(f"  evaluation done: {len(ok)}/{len(results)} evaluated successfully, "
          f"of which success={len(success)}, fail={len(ok) - len(success)}")
    print(f"  true success rate: {len(success)}/{len(ok)} = "
          f"{len(success) / len(ok) * 100:.1f}%" if ok else "  no valid results")


if __name__ == "__main__":
    main()
