"""
Fetch the source code of 7 SWE agents into source_code/swe_agents/ (without
modifying any agent logic).

Strategy:
- Agents with a PyPI release: pip download the source package (no GitHub
  network needed)
- Others: git clone (direct or with a --mirror prefix, e.g. https://ghfast.top/)

Pins are built in (commits/versions come from each agent repo's stable
release; they can be updated in the table).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEST = ROOT / "source_code" / "swe_agents"

AGENTS = {
    # name: (fetch method, identifier, branch) -- pip: PyPI package name; tarball: codeload channel
    # (github.com's git protocol is often blocked; codeload.github.com is usually reachable -- verified)
    "mini-swe-agent": ("pip", "mini-swe-agent", ""),
    "swe-agent": ("tarball", "SWE-agent/SWE-agent", "main"),
    "openhands": ("tarball", "All-Hands-AI/OpenHands", "main"),
    "deepagents": ("tarball", "langchain-ai/deepagents", "main"),
    "autocoderover": ("tarball", "nus-apr/auto-code-rover", "main"),
    "devika": ("tarball", "stitionai/devika", "main"),
    "opengpts": ("tarball", "langchain-ai/opengpts", "main"),
}


def fetch_pip(name: str, pkg: str) -> Path:
    d = DEST / name
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps",
                    "-q", "-d", str(d), pkg], check=True)
    for f in d.iterdir():
        if f.suffix == ".whl":
            with zipfile.ZipFile(f) as z:
                z.extractall(d)
        elif f.name.endswith(".tar.gz"):
            with tarfile.open(f) as t:
                t.extractall(d)
    print(f"[pip] {name} done → {d}")
    return d


def fetch_tarball(name: str, repo: str, branch: str) -> Path:
    """codeload.github.com tarball channel (usable when github.com's git protocol is blocked)."""
    import shutil
    d = DEST / name
    if d.exists() and any(d.iterdir()):
        print(f"[tarball] {name} already exists, skipping")
        return d
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
    tmp = DEST / f".tmp_{name}.tar.gz"
    subprocess.run(["curl", "-sSL", "--connect-timeout", "20",
                    "-o", str(tmp), url], check=True, timeout=600)
    with tarfile.open(tmp) as t:
        t.extractall(DEST)
    prefix = repo.split("/")[1]
    for p in DEST.iterdir():
        if p.is_dir() and p.name.startswith(prefix + "-"):
            shutil.rmtree(d, ignore_errors=True)
            p.rename(d)
    tmp.unlink()
    print(f"[tarball] {name} done → {d}")
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch SWE agent source code")
    parser.add_argument("--only", help="fetch only the specified agents (comma-separated)")
    parser.add_argument("--mirror", default="", help="GitHub mirror prefix (ignored in tarball mode)")
    args = parser.parse_args()

    names = args.only.split(",") if args.only else list(AGENTS)
    for name in names:
        if name not in AGENTS:
            print(f"[skip] unknown agent: {name}")
            continue
        kind, ident, branch = AGENTS[name]
        try:
            if kind == "pip":
                fetch_pip(name, ident)
            elif kind == "tarball":
                fetch_tarball(name, ident, branch)
            else:
                fetch_git(name, ident, args.mirror)
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
