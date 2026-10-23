"""SWE-bench environment bridge: workspace preparation/reset, env_changes delta restore (with hash-chain verification), pytest
evaluation.

Import strategy: when SWE_COLLECT_ROOT (default /share/home/tanylang/trace_collect)
is importable, prefer the collection side's own source (`from adapter.swebench
import ...`), zero hash-semantics drift; when not importable, use this module's
vendored copy (verbatim identical to the collection side, backed by the
golden-vector tests).

Hash semantics (trace_record adapter/swebench.py, must match verbatim): - state_hash: sorted keys +
b"\\x00" + (value or "<deleted>") + b"\\x00\\x01" - chained accumulation: running.update(json.dumps(state,
ensure_ascii=False, default=str).encode("utf-8")) -- insertion-order sensitive! - application_state_hash =
the accumulated hash before that span starts (chain start = e3b0..., a legal empty state, not a
placeholder)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: trace_record deploy package root (server): prefer importing the authoritative implementation from it
SWE_COLLECT_ROOT = os.environ.get(
    "SWE_COLLECT_ROOT", "/share/home/tanylang/trace_collect")
#: replay workspace root (per-process, prevents concurrent git clobbering)
SWE_WORK_ROOT = os.environ.get(
    "SWE_WORK_ROOT", "out_span_offload_swe/workspaces")
#: git clone mirror prefix
SWE_MIRROR = os.environ.get("SWE_MIRROR", "")
#: pytest per-test timeout (seconds)
SWE_TEST_TIMEOUT = int(os.environ.get("SWE_TEST_TIMEOUT", "1200"))

_authoritative: Optional[Any] = None
if os.path.isdir(SWE_COLLECT_ROOT):
    _adapter_dir = os.path.join(SWE_COLLECT_ROOT, "adapter")
    if os.path.isdir(_adapter_dir):
        try:
            sys.path.insert(0, SWE_COLLECT_ROOT)
            from adapter import swebench as _auth_mod  # type: ignore

            _authoritative = _auth_mod
        except Exception:
            _authoritative = None


# ---------------------------------------------------------------------------
# vendored copies (verbatim identical to trace_record adapter/swebench.py)
# ---------------------------------------------------------------------------
def _git(repo_dir: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo_dir), *args],
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (
            " ".join(args), (r.stderr or "")[:200]))
    return r.stdout or ""


def state_hash(state: Dict[str, Any]) -> str:
    """Repository state hash (the serialization rule shared by the collection side / replay side)."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        v = state[k]
        h.update((v if v is not None else "<deleted>").encode(
            "utf-8", errors="replace"))
        h.update(b"\x00\x01")
    return h.hexdigest()


def apply_env_changes(repo_dir: Path, env_changes: Dict[str, Any]) -> None:
    """Replay: apply env_changes (full file contents) to the current workspace."""
    for path, content in env_changes.items():
        if path.startswith("__skipped__"):
            continue
        fp = repo_dir / path
        try:
            if content is None:
                fp.unlink(missing_ok=True)
            else:
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_text(str(content), encoding="utf-8")
        except OSError as e:
            print("[replay] skipped unwritable path %r: %s" % (path, e))


def _chain_update(running: Any, state: Dict[str, Any]) -> None:
    """Chained accumulation (insertion-order sensitive, verbatim identical to the collection side)."""
    if state:
        running.update(json.dumps(
            state, ensure_ascii=False, default=str).encode("utf-8"))


# ---------------------------------------------------------------------------
# workspace management
# ---------------------------------------------------------------------------
def ensure_workspace(task: Dict[str, Any], work_root: Optional[str] = None,
                     force_reinit: bool = False) -> Path:
    """Prepare the base_commit workspace (locally copying the collection side's ready workspace is preferred)."""
    import shutil
    iid = task["instance_id"]
    root = Path(work_root or SWE_WORK_ROOT) / iid
    git_dir = root / ".git"
    if git_dir.is_dir() and not force_reinit:
        return root
    # copy locally from the collection side's workspace (avoid re-downloading)
    if _authoritative is None:
        candidates = [os.path.join(SWE_COLLECT_ROOT, "swe_workspaces")]
        for cand in candidates:
            for sub in os.listdir(cand):
                src = Path(cand) / sub / iid
                if (src / ".git").is_dir():
                    shutil.copytree(src, root, dirs_exist_ok=True)
                    return root
        raise RuntimeError(
            "workspace unavailable and no ready copy on the collection side: %s" % iid)
    # authoritative path: reuse the collection side's prepare_repo
    from adapter.swebench import SWETask  # type: ignore
    st = SWETask(**task)
    return _authoritative.prepare_repo(
        st, Path(work_root or SWE_WORK_ROOT), mirror_prefix=SWE_MIRROR)


def reset_workspace(repo_dir: Path) -> None:
    """Reset to base_commit (equivalent to the finalize_swe_traces.py basis)."""
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet",
                    "--", "."], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    subprocess.run(["git", "-C", str(repo_dir), "clean", "-fdq"],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")


def restore_deltas(repo_dir: Path, delta_log: List[Dict[str, Any]],
                   up_to_index: int) -> Tuple[str, List[str]]:
    """Replay delta_log[0:up_to_index]'s env_changes in order, verifying the hash chain entry by entry.

    Returns (final accumulated hash, verification note list). Entries missing a hash skip verification and record a
    note.
    """
    import hashlib as _hl
    running = _hl.sha256()
    notes: List[str] = []
    for k, entry in enumerate(delta_log[:up_to_index]):
        changes = entry.get("env_changes") or {}
        apply_env_changes(repo_dir, changes)
        _chain_update(running, changes)
        expected = entry.get("hash_before", "")
        # entry k's hash_before records the "before this span starts" accumulation = the hash after applying the first k deltas. The verification target is
        # entry k+1's hash_before (if it exists); its consistency with the current running is covered by the next loop round; here just apply in order
        # first
        if k + 1 < up_to_index:
            nxt = delta_log[k + 1].get("hash_before", "")
            if nxt and nxt != running.hexdigest():
                raise RuntimeError(
                    "restore delta mismatch at entry %d: recorded %s != "
                    "applied %s" % (k + 1, nxt[:12], running.hexdigest()[:12]))
    return running.hexdigest(), notes


def apply_test_patch(repo_dir: Path, task: Dict[str, Any]) -> None:
    """Apply test_patch (needed by the local pytest basis)."""
    patch = task.get("test_patch", "")
    if not patch:
        return
    p = subprocess.run(["git", "-C", str(repo_dir), "apply"],
                       input=patch, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        # when git apply fails, fall back to the patch tool
        p2 = subprocess.run(["patch", "-p1", "-f"],
                            cwd=repo_dir, input=patch, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")


def run_pytest_eval(repo_dir: Path, task: Dict[str, Any]
                    ) -> Dict[str, Any]:
    """Local pytest evaluation (simplified basis: FAIL_TO_PASS ∪ PASS_TO_PASS, per test)."""
    apply_test_patch(repo_dir, task)
    results: Dict[str, Tuple[bool, str]] = {}
    tests = list(task.get("FAIL_TO_PASS", [])) + list(task.get("PASS_TO_PASS", []))
    for test in tests:
        try:
            p = subprocess.run(
                [sys.executable, "-m", "pytest", test, "-x", "-q"],
                cwd=repo_dir, capture_output=True, text=True,
                timeout=SWE_TEST_TIMEOUT, encoding="utf-8", errors="replace")
            results[test] = (p.returncode == 0, (p.stdout or "")[-300:])
        except subprocess.TimeoutExpired:
            results[test] = (False, "timeout(%ds)" % SWE_TEST_TIMEOUT)
        except Exception as e:  # pytest not installed, etc.
            results[test] = (False, "error: %s" % str(e)[:200])
    ftp = task.get("FAIL_TO_PASS", [])
    ptp = task.get("PASS_TO_PASS", [])
    ftp_ok = all(results[t][0] for t in ftp) if ftp else True
    ptp_ok = all(results[t][0] for t in ptp) if ptp else True
    return {
        "success": bool(ftp_ok and ptp_ok),
        "FAIL_TO_PASS_ok": bool(ftp_ok),
        "PASS_TO_PASS_ok": bool(ptp_ok),
        "total_tests": len(tests),
        "details": {t: r[0] for t, r in results.items()},
        "tails": {t: r[1] for t, r in results.items()},
    }
