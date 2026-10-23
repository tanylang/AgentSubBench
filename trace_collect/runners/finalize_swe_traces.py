"""
SWE-bench trace finalization: evaluation (pass) + event_type annotation.

Evaluation criteria match the official SWE-bench:
- Replay: clean checkout of base_commit + apply each span's env_changes in order
- After applying test_patch, all FAIL_TO_PASS pass and all PASS_TO_PASS pass → success
- Result written back to the trace's top-level pass / pass_details (same fields as AppWorld traces)

Test execution mode (choose one):
- docker: reuse the SWE-bench harness container evaluation (recommended, safe)
- local: run pytest directly in the host repo directory (arbitrary code execution risk, personal use only)

Usage:
    python runners/finalize_swe_traces.py <trace_dir> [--agent NAME] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"),
          str(ROOT / "runners"), str(ROOT / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter.swebench import (  # noqa: E402
    SWETask, apply_env_changes, load_swebench_tasks, prepare_repo, repo_diff,
)


def _read_trace(path: Path) -> dict:
    return json.load(open(path, encoding="utf-8"))


def _spans_with_checkpoint(data: dict) -> list[dict]:
    return [s for s in data.get("spans", []) if "check_point" in s]


def _reconstruct_repo(task: SWETask, trace_path: Path, work_root: Path,
                      mirror_prefix: str = "") -> Path:
    """Replay all env_changes from the trace onto the base checkout, returning the final repo directory.

    Prefer reusing the workspace from the collection phase (swe_workspaces/<agent>/<instance_id>)
    to avoid re-downloading; create a new one only if none exists.
    """
    # Reuse the collection workspace (an already-prepared repo for the same task under any agent subdirectory)
    repo_dir = work_root / task.instance_id
    if not (repo_dir / ".git").exists():
        reused = None
        for sub in work_root.iterdir() if work_root.exists() else []:
            cand = sub / task.instance_id
            if cand.is_dir() and (cand / ".git").exists():
                reused = cand
                break
        if reused:
            shutil.rmtree(repo_dir, ignore_errors=True)
            shutil.copytree(reused, repo_dir)
    if not (repo_dir / ".git").exists():
        repo_dir = prepare_repo(task, work_root, mirror_prefix)
    # The workspace may retain the agent's modifications (from the collection phase); reset to base first, then replay
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet", "--", "."],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    subprocess.run(["git", "-C", str(repo_dir), "clean", "-fdq"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    data = _read_trace(trace_path)
    for s in _spans_with_checkpoint(data):
        apply_env_changes(repo_dir, s["check_point"].get("env_changes", {}))
    return repo_dir


def _run_tests_local(repo_dir: Path, task: SWETask) -> dict:
    """Local pytest evaluation (simplified criteria: FAIL_TO_PASS ∪ PASS_TO_PASS)."""
    _apply_test_patch(repo_dir, task)
    results = {}
    for test in task.FAIL_TO_PASS + task.PASS_TO_PASS:
        p = subprocess.run(
            [sys.executable, "-m", "pytest", test, "-x", "-q"],
            cwd=repo_dir, capture_output=True, text=True, timeout=1200)
        results[test] = (p.returncode == 0, (p.stdout or "")[-300:])
    return results


def _run_tests_docker(repo_dir: Path, task: SWETask, agent_name: str = "") -> dict:
    """Docker evaluation — directly call the official SWE-bench harness (benchmarks/swebench).

    Flow: generate model_patch from the replayed repo → write the official predictions.json →
    swebench.harness.run_evaluation.main builds the container / runs the tests / scores → read the official report.
    test_patch is applied by the official harness inside the container itself (manual application is only needed in local mode).
    """
    sys.path.insert(0, str(ROOT / "benchmarks" / "swebench"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # dataset loading goes through the mirror
    from swebench.harness.run_evaluation import main as harness_main

    patch = repo_diff(repo_dir)
    predictions = {task.instance_id: {
        "model_name_or_path": agent_name or "agent",
        "model_patch": patch,
    }}
    pred_path = repo_dir.parent / f"preds_{task.instance_id}.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")
    run_id = f"eval_{task.instance_id}"
    report_dir = ROOT / "swe_workspaces" / "eval_reports"
    harness_main(
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        instance_ids=[task.instance_id],
        predictions_path=str(pred_path),
        max_workers=1,
        open_file_limit=4096,
        run_id=run_id,
        timeout=900,
        rewrite_reports=False,
        modal=False,
        report_dir=str(report_dir),
    )
    # Official report: <report_dir>/<run_id>/<instance_id>.json
    report_path = report_dir / run_id / f"{task.instance_id}.json"
    try:
        report = json.load(open(report_path, encoding="utf-8"))
    except FileNotFoundError:
        return {"error": f"official report missing: {report_path}"}
    entry = report.get(task.instance_id, {})
    return {
        "resolved": bool(entry.get("resolved", False)),
        "report": {k: v for k, v in entry.items()
                   if k in ("resolved", "FAIL_TO_PASS", "PASS_TO_PASS",
                            "test_status_map")},
    }


def _apply_test_patch(repo_dir: Path, task: SWETask) -> None:
    if task.test_patch:
        p = subprocess.run(["git", "-C", str(repo_dir), "apply"],
                           input=task.test_patch, capture_output=True, text=True)
        if p.returncode != 0:
            subprocess.run(["git", "-C", str(repo_dir), "apply", "--3way"],
                           input=task.test_patch, capture_output=True, text=True)


def evaluate_swe_trace(trace_path: Path, agent_name: str = "",
                       mode: str = "local", mirror_prefix: str = "") -> dict:
    data = _read_trace(trace_path)
    meta = data.get("swebench") or {}
    instance_id = meta.get("instance_id") or Path(trace_path).stem
    tasks = load_swebench_tasks(task_ids=[instance_id])
    if not tasks:
        return {"trace": trace_path.name, "error": f"task {instance_id} is not in the task library"}
    task = tasks[0]

    work_root = ROOT / "swe_workspaces" / "eval"
    try:
        repo_dir = _reconstruct_repo(task, trace_path, work_root, mirror_prefix)
    except Exception as e:
        return {"trace": trace_path.name, "error": f"replay failed: {e}"}

    try:
        if mode == "docker":
            res = _run_tests_docker(repo_dir, task, agent_name)
            success = bool(res.get("resolved", False))
            ftp_ok = ptp_ok = None
            details = res.get("report", res)
        else:
            res = _run_tests_local(repo_dir, task)
            ftp_ok = all(res[t][0] for t in task.FAIL_TO_PASS) if task.FAIL_TO_PASS else True
            ptp_ok = all(res[t][0] for t in task.PASS_TO_PASS) if task.PASS_TO_PASS else True
            success = bool(ftp_ok and ptp_ok)
            details = {t: r[0] for t, r in res.items()}
    except Exception as e:
        return {"trace": trace_path.name, "error": f"evaluation run failed: {e}"}

    data["pass"] = "success" if success else "fail"
    data["pass_details"] = {
        "success": success,
        "FAIL_TO_PASS_ok": ftp_ok,
        "PASS_TO_PASS_ok": ptp_ok,
        "total_tests": len(task.FAIL_TO_PASS) + len(task.PASS_TO_PASS),
        "details": details,
    }
    json.dump(data, open(trace_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return {"task_id": instance_id, "success": success}


def finalize_swe_trace_dir(trace_dir: str, agent_name: str = "",
                           mode: str = "local", mirror_prefix: str = "",
                           limit: int = 0) -> None:
    """Required post-collection operations: evaluation (write pass) → event_type annotation."""
    from add_event_types import annotate_trace_file

    files = sorted(p for p in Path(trace_dir).glob("*.json"))
    if limit:
        files = files[:limit]
    stats = {"evaluated": 0, "success": 0, "fail": 0, "errors": 0}
    for i, fp in enumerate(files, 1):
        r = evaluate_swe_trace(fp, agent_name, mode, mirror_prefix)
        if "error" in r:
            stats["errors"] += 1
            print(f"    [{i}/{len(files)}] {r['trace']}: ERROR {r['error']}")
        else:
            stats["evaluated"] += 1
            if r["success"]:
                stats["success"] += 1
            else:
                stats["fail"] += 1
            print(f"    [{i}/{len(files)}] {r['task_id']}: success={r['success']}")

    # event_type annotation: the same authoritative implementation as the AppWorld side (add_event_types.annotate_*)
    n = 0
    for fp in files:
        annotate_trace_file(str(fp))
        n += 1
    print(f"[finalize] evaluation: {stats['evaluated']} valid, success={stats['success']}, "
          f"fail={stats['fail']}, errors={stats['errors']}; annotated {n} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench trace finalization (evaluation + annotation)")
    parser.add_argument("trace_dir")
    parser.add_argument("--agent", default="")
    parser.add_argument("--mode", default="local", choices=["local", "docker"])
    parser.add_argument("--mirror", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    finalize_swe_trace_dir(args.trace_dir, args.agent, args.mode, args.mirror,
                           args.limit)


if __name__ == "__main__":
    main()
