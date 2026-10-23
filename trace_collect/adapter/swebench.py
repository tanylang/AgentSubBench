"""
SWE-bench adapter layer — unified task/environment/tool integration for software-engineering agents.

Layering (fully isomorphic with the AppWorld side):
- Task layer: SWE-bench Verified (tasks/swebench_tasks/swebench_verified_tasks.json)
- Environment layer: git repository workspace (checked out at base_commit); all agent
  tools ultimately land on the "repository file state" — git serves as the unified
  state machine
- Tool layer: each agent brings its own tools (bash/edit/view/submit, etc.); the driver
  layer wraps them **externally** (without modifying agent source logic) and records
  them uniformly as canonical tool.execution spans
- LLM layer: these agents all go through the litellm / openai SDK — the driver layer
  registers a litellm global callback to intercept all LLM calls, likewise with zero
  agent changes

check_point conventions (consistent with the AppWorld trace structure):
- task_id          : SWE-bench instance_id
- application_state_hash : repository state hash (sha256 of the serialized git state, chained accumulation)
- env_changes      : repository changes during this span ({"<path>": "patch text", ...})
- env_changes_baseline : {} (the initial checkout is performed by the replay side itself)
Replay = clean checkout at base_commit + apply each span's env_changes in order.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
TASKS_FILE = ROOT / "tasks" / "swebench_tasks" / "swebench_verified_tasks.json"

# miscellaneous items in the git repository state that are "irrelevant to solving the task" (log files, etc.) are excluded from env_changes
IGNORE_PATTERNS = [".git", "__pycache__", "*.pyc"]


@dataclass
class SWETask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    patch: str = ""
    test_patch: str = ""
    environment_setup_commit: str = ""
    difficulty: str = ""


def load_swebench_tasks(limit: int = 0, task_ids: list[str] | None = None) -> list[SWETask]:
    """Load the SWE-bench Verified task list (limit=0 means all; filter by task_ids)."""
    with open(TASKS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    tasks = [SWETask(**{k: t.get(k) for k in SWETask.__dataclass_fields__})
             for t in raw]
    if task_ids:
        want = set(task_ids)
        tasks = [t for t in tasks if t.instance_id in want]
    return tasks[:limit] if limit else tasks


def _git(repo_dir: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo_dir), *args],
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(r.stderr or '')[:200]}")
    return r.stdout or ""


def prepare_repo(task: SWETask, work_root: Path, mirror_prefix: str = "") -> Path:
    """Prepare the repository workspace at base_commit under work_root/<instance_id>.

    git clone is preferred (mirror_prefix supported); on failure, fall back to the
    codeload tarball (a usable channel when the github.com git protocol is blocked)
    + git init to rebuild .git.
    """
    repo_dir = work_root / task.instance_id
    if not (repo_dir / ".git").exists():
        ok = False
        url = f"{mirror_prefix}https://github.com/{task.repo}.git" if mirror_prefix \
            else f"https://github.com/{task.repo}.git"
        try:
            subprocess.run(["git", "clone", "--quiet", url, str(repo_dir)],
                           capture_output=True, text=True, check=True, timeout=1200)
            _git(repo_dir, "checkout", "--quiet", task.base_commit)
            ok = True
        except Exception:
            pass
        if not ok:
            # codeload tarball channel + git init (no remote history; only a local .git for checkpoint use)
            repo_dir.mkdir(parents=True, exist_ok=True)
            tar = work_root / f".tmp_{task.instance_id}.tar.gz"
            subprocess.run(["curl", "-sSL", "--connect-timeout", "20", "-o", str(tar),
                            f"https://codeload.github.com/{task.repo}/tar.gz/{task.base_commit}"],
                           check=True, timeout=1800)
            import tarfile
            with tarfile.open(tar) as t:
                t.extractall(work_root)
            prefix = task.repo.split("/")[1]
            import re
            # match only codeload's extracted directory shape: <repo name>-<hex sha>
            # (the previous prefix matching could hit other instance directories in
            # the same workspace, e.g. "pytest-dev__pytest-10081" also starts with
            # "pytest-")
            extracted = [p for p in work_root.iterdir()
                         if p.is_dir()
                         and re.match(rf"^{re.escape(prefix)}-[0-9a-f]{{7,}}$", p.name)]
            if extracted:
                import shutil
                shutil.rmtree(repo_dir, ignore_errors=True)
                extracted[0].rename(repo_dir)
            tar.unlink(missing_ok=True)
            if not any(repo_dir.iterdir()):
                raise RuntimeError(f"{task.instance_id}: downloaded content is empty (codeload connection dropped), please retry")
            subprocess.run(["git", "-C", str(repo_dir), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "config", "user.email", "t@t"],
                           check=True)
            subprocess.run(["git", "-C", str(repo_dir), "config", "user.name", "t"],
                           check=True)
            subprocess.run(["git", "-C", str(repo_dir), "add", "-A"], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "commit", "-qm", "base",
                            "--allow-empty"], check=True)
    return repo_dir


def repo_state(repo_dir: Path) -> dict[str, str]:
    """Current repository state: full contents of changed files (tracked
    modifications/deletions + untracked additions).

    Returns {"<path>": "file content text" or None (deleted), ...} (empty = no
    changes). Files over 10MB are skipped and recorded with a "__skipped__"
    marker. The full-content form lets replay write files directly, naturally
    handling patches git apply cannot, such as "no trailing newline".
    """
    state: dict[str, Any] = {}
    changed = _git(repo_dir, "diff", "--name-only", "HEAD").splitlines()
    changed = [p for p in changed if p.strip()]
    untracked = _git(repo_dir, "ls-files", "--others", "--exclude-standard").splitlines()
    for p in dict.fromkeys(changed + [u for u in untracked if u.strip()]):
        fp = repo_dir / p
        if not fp.is_file():
            state[p] = None  # file deleted
            continue
        if fp.stat().st_size > 10_000_000:
            state[f"__skipped__: {p}"] = f"{fp.stat().st_size} bytes"
            continue
        try:
            state[p] = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    return state


def state_hash(state: dict[str, str]) -> str:
    """Repository state hash (serialization rule shared by the capture and replay sides)."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
        v = state[k]
        h.update((v if v is not None else "<deleted>").encode("utf-8", errors="replace"))
        h.update(b"\x00\x01")
    return h.hexdigest()


def make_swe_checkpoint_provider(repo_dir: Path, task_id: str):
    """SWE environment checkpoint collector — same interface as the appworld provider.

    provide(span_type) -> {task_id, application_state_hash, env_changes, ...}
    env_changes = the repository state delta during this span (the first call records
    the current state as the baseline).
    """
    _prev: dict[str, str] = {}
    _running = hashlib.sha256()
    _init_done = [False]

    def _chain_update(state: dict[str, str]) -> None:
        if state:
            _running.update(
                json.dumps(state, ensure_ascii=False, default=str).encode("utf-8"))

    def provide(span_type: str, span_id: str | None = None) -> dict:
        nonlocal _prev
        first = not _init_done[0]
        if first:
            _init_done[0] = True
        cur = repo_state(repo_dir)
        start_hash = _running.hexdigest()
        env_changes: dict[str, Any] = {}
        if first:
            # the baseline (base_commit state) is archived but not replayed: the replay side checks out base itself
            baseline = cur
        else:
            for k, v in cur.items():
                if _prev.get(k) != v:
                    env_changes[k] = v
            for k in list(_prev):
                if k not in cur:
                    env_changes[k] = None  # file deleted
        _chain_update(env_changes)
        _prev = cur
        cp: dict[str, Any] = {
            "task_id": task_id,
            "application_state_hash": start_hash,
            "env_changes": env_changes,
        }
        if first:
            cp["env_changes_baseline"] = baseline
        return cp

    return provide


def repo_diff(repo_dir: Path) -> str:
    """git diff of the current workspace relative to HEAD (base) — the agent's complete modification patch."""
    return _git(repo_dir, "diff", "HEAD")


def apply_env_changes(repo_dir: Path, env_changes: dict[str, Any]) -> None:
    """Replay: apply env_changes (full file contents) to the current workspace.

    content=None → delete the file; str → write the file content; keys with the
    "__skipped__" prefix denote over-limit files and are skipped during replay
    (they require manual handling before evaluation).
    """
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
            # garbage filenames written by agents are illegal on Windows (WinError 123,
            # etc.) — skip the individual file without aborting the whole replay
            # (such files cannot be valid patches)
            print(f"[replay] skipping unwritable path {path!r}: {e}")
