"""Fetch official LoCoMo data (pinned version) into tasks/locomo_tasks/.

The official repo is https://github.com/snap-research/LoCoMo (ACL 2024); the
data file data/locomo10.json ships with the repo (10 conversation samples,
~2.8MB). This script fetches via the GitHub API blob endpoint (pinning the git
blob sha guarantees data determinism), with git clone as a fallback. It does
not modify any official content; the file is only saved locally for read-only
collection/evaluation use.

Usage:
    python scripts/fetch_locomot.py                    # fetch the data file only
    python scripts/fetch_locomot.py --mirror https://ghfast.top/   # route the clone fallback through a mirror
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DEST = ROOT / "tasks" / "locomo_tasks" / "locomo10.json"
CLONE_DEST = ROOT / "source_code" / "locomo"

# Git blob sha of the official locomo10.json (pinned for data determinism; repo default branch is main)
BLOB_SHA = "d95b872480b413d935821fdc3c84f8a8f5f29e73"
GITHUB_REPO = "snap-research/LoCoMo"

# Observed counts from the official data (used to align validate(); see paper/README)
EXPECT_SAMPLES, EXPECT_SESSIONS, EXPECT_TURNS, EXPECT_QA = 10, 272, 5882, 1986


def fetch_via_github_api(sha: str, dest: Path) -> bool:
    """GitHub API blob channel (usable when raw.githubusercontent is blocked).

    GET https://api.github.com/repos/<repo>/git/blobs/<sha>
    → take the content field, base64-decode it → atomically write dest.
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/git/blobs/{sha}"
    p = subprocess.run(
        ["curl", "-sSL", "--connect-timeout", "30", "--max-time", "300",
         "-H", "Accept: application/vnd.github+json", "-o", "-", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0 or not p.stdout.strip():
        return False
    try:
        payload = json.loads(p.stdout)
        content = payload["content"]
        data = base64.b64decode(content)
    except Exception:
        return False
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    print(f"[blob] {GITHUB_REPO}@{sha[:12]} → {dest}")
    return True


def fetch_via_clone(dest: Path, mirror: str = "") -> bool:
    """git clone --depth 1 (direct or with a --mirror prefix), copy data/locomo10.json."""
    url = f"{mirror}https://github.com/{GITHUB_REPO}.git"
    p = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        return False
    src = dest / "data" / "locomo10.json"
    if src.exists():
        shutil.copy2(src, DATA_DEST)
        print(f"[clone] {dest}/data/locomo10.json → {DATA_DEST}")
        return True
    return False


def validate(data_path: Path) -> dict:
    """Check that the JSON parses and sample-check required fields; return counts."""
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data, "locomo10.json should be a non-empty array"
    sessions = turns = qa = 0
    for s in data:
        conv = s["conversation"]
        assert conv.get("speaker_a") and conv.get("speaker_b"), f"{s['sample_id']} missing speaker"
        for k, v in conv.items():
            if k.startswith("session_") and not k.endswith("date_time") and isinstance(v, list):
                sessions += 1
                turns += len(v)
        assert isinstance(s.get("qa"), list), f"{s['sample_id']} missing qa"
        qa += len(s["qa"])
    stats = {"samples": len(data), "sessions": sessions, "turns": turns, "qa": qa}
    print(f"[validate] {stats} (official observed counts 10/272/5882/1986)")
    if (stats["samples"], stats["sessions"], stats["turns"], stats["qa"]) != \
            (EXPECT_SAMPLES, EXPECT_SESSIONS, EXPECT_TURNS, EXPECT_QA):
        print("Warning: counts differ from the official observed counts -- check data version/integrity")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official LoCoMo data")
    parser.add_argument("--mirror", default="", help="git clone mirror prefix (e.g. https://ghfast.top/)")
    parser.add_argument("--method", default="auto", choices=["auto", "blob", "clone"])
    args = parser.parse_args()

    DATA_DEST.parent.mkdir(parents=True, exist_ok=True)
    ok = False
    if DATA_DEST.exists() and validate(DATA_DEST):
        ok = True
    if not ok and args.method in ("auto", "blob"):
        ok = fetch_via_github_api(BLOB_SHA, DATA_DEST) and bool(validate(DATA_DEST))
    if not ok and args.method in ("auto", "clone"):
        ok = fetch_via_clone(CLONE_DEST, args.mirror) and bool(validate(DATA_DEST))
    if not ok:
        print("Error: data fetch failed (network down or blocked) -- download manually and place at", DATA_DEST)
        sys.exit(1)
    print("Done [OK]")


if __name__ == "__main__":
    main()
