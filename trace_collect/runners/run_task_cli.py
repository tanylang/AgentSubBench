"""Run one task or the full task list of a benchmark module — shared CLI for local / sbatch.

Usage:
    python run_task_cli.py <module>              # full collection (sequential)
    python run_task_cli.py <module> all [N]      # full collection (N concurrent processes)
    python run_task_cli.py <module> <task_index> # single task
    TASK_INDEX=5 python run_task_cli.py <module> # run a single task via env var for sbatch arrays

- module: e.g. scripts.run_benchmark_builtin_oai (see the run_benchmark_*.py files under scripts/)
- completed tasks are skipped by the skip-if-exists logic inside run_one; safe to rerun (resumable)
"""
import importlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

# Windows: stdout/stderr default to GBK after redirection; AppWorld printing emoji would crash the child process
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
# Clear other projects' venv variables left in the shell profile (they previously polluted sys.path and broke jiter imports)
os.environ.pop("VIRTUAL_ENV", None)
os.environ.pop("CONDA_PREFIX", None)
# simplified and other agents' model config needs this variable (missing it crashes directly with KeyError)
os.environ.setdefault("MODEL_SERVER_URL", "")


def run_one_task(mod_name: str, idx: int) -> None:
    mod = importlib.import_module(mod_name)
    real_tid, prefix = mod.TASKS[idx]
    tag = os.environ.get("SLURM_ARRAY_TASK_ID", str(idx))
    print(f"[task {tag}] {real_tid}", flush=True)
    out_path = mod.run_one(real_tid, prefix)

    # After the task finishes, evaluate offline and write the real pass result back into the trace (disable with EVAL_AFTER_COLLECT=0).
    # Run the evaluation in a subprocess to isolate DB state (in the same process it errors because Database.storage_type is not reset).
    if os.environ.get("EVAL_AFTER_COLLECT", "1") == "0":
        return
    if not out_path or not os.path.exists(out_path):
        return
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "runners" / "evaluate_traces.py"),
             "--trace", str(out_path), "--experiment-name", "offline_eval"],
            env=os.environ, check=False, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            # evaluation failure must be visible (previously the output was discarded, making a missing pass impossible to diagnose)
            print(f"    [eval] {prefix}: evaluation failed rc={proc.returncode}", flush=True)
            print(f"      stdout: {(proc.stdout or '')[-1200:]}", flush=True)
            print(f"      stderr: {(proc.stderr or '')[-1200:]}", flush=True)
    except Exception as e:
        print(f"    [eval] {prefix}: evaluation subprocess error {type(e).__name__}: {e}", flush=True)


def run_all(mod_name: str, workers: int = 1) -> None:
    mod = importlib.import_module(mod_name)
    n = len(mod.TASKS)
    print(f"=== {mod_name} : {n} tasks, {workers} workers ===", flush=True)
    if workers <= 1:
        for i in range(n):
            run_one_task(mod_name, i)
        return
    # Multi-concurrency: one independent subprocess per task (same behavior as run_parallel.sh, inter-process isolation)
    def _run(i: int) -> None:
        subprocess.run(
            [sys.executable, str(ROOT / "runners" / "run_task_cli.py"), mod_name, str(i)],
            env=os.environ, check=False,
        )

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_run, range(n)))


def main() -> None:
    mod_name = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] != "all":
        run_one_task(mod_name, int(sys.argv[2]))
        return
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run_all(mod_name, workers)


if __name__ == "__main__":
    main()
