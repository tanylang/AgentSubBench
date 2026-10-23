"""Environment replay tool — restore the AppWorld environment state from a trace's check_point chain.

Usage:
    python replay_env.py verify <trace_path>
        Verify the entire hash chain (compare application_state_hash span by span), reporting the consistency of each span

    python replay_env.py restore <trace_path> <step> [--save-dir DIR]
        Restore the environment state before the step-th span executes; with --save-dir, save a full DB snapshot to DIR
        (which can later be loaded in any experiment via world.load_state(DIR))

Principle:
    Task initial state (AppWorld reproducible) → open a brand-new world (AppWorld completes initialization automatically)
    → apply each span's env_changes (SQL delta diff) in order → obtain the DB state before any span.
    Each step verifies replay correctness with application_state_hash.
    The first span's env_changes_baseline (world initialization writes) is only archived, not replayed.

Use cases:
    1. Verify the integrity of a trace's snapshot chain
    2. Counterfactual experiments: restore to the same state anchor and continue with a different agent/model/prompt
    3. Offline reproduction of failure cases (without rerunning the whole task)
    4. Offline evaluation: restore the "declared-done" state and feed it to the AppWorld evaluator to compare against ground truth
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(os.path.join(ROOT, "benchmarks", "appworld"))


def _chain_update(running, env_changes: dict):
    """Feed each app's deltas from env_changes into the hash in record order (exactly matching the collection side)."""
    for app_changes in env_changes.values():
        if app_changes:
            running.update(
                json.dumps(app_changes, default=str, ensure_ascii=False).encode("utf-8"))
    return running


def load_trace(trace_path: str) -> dict:
    with open(trace_path, encoding="utf-8") as f:
        return json.load(f)


def spans_with_checkpoint(data: dict) -> list[dict]:
    return [s for s in data["spans"] if "check_point" in s]


def verify_chain(trace_path: str) -> list[dict]:
    """Verify the hash chain span by span. Returns [{step, type, ok, expected, got}]."""
    data = load_trace(trace_path)
    spans = spans_with_checkpoint(data)
    running = hashlib.sha256()
    # The baseline participates in the hash chain (the collection side feeds the baseline into the running hash at init):
    # take the baseline from the first span's env_changes_baseline (order = recorded app order)
    if spans and spans[0]["check_point"].get("env_changes_baseline"):
        _chain_update(running, spans[0]["check_point"]["env_changes_baseline"])
    results = []
    for s in spans:
        cp = s["check_point"]
        expected = cp.get("application_state_hash")
        step = s["attributes"].get("agent.step")
        if not expected:
            results.append({"step": step, "type": s["type"], "ok": None,
                            "reason": "no hash recorded"})
            continue
        got = running.hexdigest()
        results.append({"step": step, "type": s["type"], "ok": got == expected,
                        "expected": expected[:12], "got": got[:12]})
        _chain_update(running, cp.get("env_changes", {}))
    return results


def apply_changes(world, env_changes: dict) -> None:
    """Apply Change tuples (statement, parameters, executemany) to each app DB of the world."""
    from appworld.common.imports import import_models_module

    for app, changes in env_changes.items():
        if not changes:
            continue
        mod = import_models_module(app)
        conn = mod.SQLModel.db.connection
        for statement, parameters, executemany in changes:
            if executemany:
                conn.executemany(statement, parameters)
            else:
                conn.execute(statement, parameters)
        conn.commit()


def restore_world_at(trace_path: str, step: int):
    """Restore the environment before the step-th span executes, returning (world, data dict).

    The replay starts from an "already-initialized world" (initialization completes automatically when AppWorld is constructed),
    so env_changes_baseline is skipped and only pure span deltas are applied.
    """
    from appworld import AppWorld

    data = load_trace(trace_path)
    spans = spans_with_checkpoint(data)
    if not spans:
        raise ValueError("trace has no check_point data")

    task_id = spans[0]["check_point"].get("task_id") \
        or data["spans"][0]["attributes"].get("benchmark_task_id")
    if not task_id:
        raise ValueError("cannot determine task_id")

    world = AppWorld(task_id=task_id, experiment_name="replay_env")
    running = hashlib.sha256()
    if spans[0]["check_point"].get("env_changes_baseline"):
        _chain_update(running, spans[0]["check_point"]["env_changes_baseline"])
    applied = 0
    for s in spans:
        if s["attributes"].get("agent.step", 0) >= step:
            break
        cp = s["check_point"]
        expected = cp.get("application_state_hash")
        got = running.hexdigest()
        if expected and got != expected:
            raise ValueError(
                f"hash chain inconsistent at span step={s['attributes'].get('agent.step')}: "
                f"expected {expected[:12]} got {got[:12]}")
        apply_changes(world, cp.get("env_changes", {}))
        _chain_update(running, cp.get("env_changes", {}))
        applied += 1
    return world, {"task_id": task_id, "applied_spans": applied, "target_step": step}


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        return

    cmd = sys.argv[1]
    trace_path = sys.argv[2]

    if cmd == "verify":
        results = verify_chain(trace_path)
        bad = [r for r in results if r["ok"] is False]
        print(f"verify: {len(results)} spans, "
              f"consistent {sum(1 for r in results if r['ok'])} / "
              f"inconsistent {len(bad)} / no hash {sum(1 for r in results if r['ok'] is None)}")
        for r in bad[:10]:
            print(f"  MISMATCH step={r['step']} {r['type']}: "
                  f"expected {r['expected']} got {r['got']}")
        if not bad:
            print("  hash chain fully OK")
        return

    if cmd == "restore":
        step = int(sys.argv[3])
        save_dir = None
        if "--save-dir" in sys.argv:
            save_dir = sys.argv[sys.argv.index("--save-dir") + 1]
        world, info = restore_world_at(trace_path, step)
        print(f"restore OK: task={info['task_id']}, "
              f"replayed the deltas of {info['applied_spans']} spans, target step={step}")
        if save_dir:
            # Replayed writes go through the raw sqlite connection and do not enter AppWorld's changes tracker,
            # so world.save_state cannot be used (the changes format would be incomplete); export fully with the full format instead
            from appworld.apps.lib.apis.local_remote import save_local_dbs
            save_local_dbs(
                from_db_home_path=world.models_to_db_home_path,
                to_db_home_path=save_dir,
                format="full",
                app_names=[*world.task.allowed_apps, "admin", "supervisor"],
                delete_if_exists=True,
                skip_mandatory_apps=False,
                save_model_hashes=False,
            )
            print(f"  state saved to: {save_dir} (full format, loadable via world.load_state('{save_dir}'))")
        return

    print(__doc__)


if __name__ == "__main__":
    main()
