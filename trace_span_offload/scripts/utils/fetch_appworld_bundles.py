#!/usr/bin/env python
"""Extracts AppWorld's apps source from the official ghcr.io Docker image (fallback when the repo's LFS quota is exhausted).

Usage (no Docker needed, pure HTTP + tarfile):
    python fetch_appworld_bundles.py [output dir]

Steps:
  1. get an anonymous ghcr.io token + manifest
  2. download all image layers into .appworld_layers/
  3. unpack layers one by one, looking for src/appworld/.source/apps.bundle or an unpacked src/appworld/apps/*.py
  4. if a bundle is found: use the appworld library's unpack_bundle to decrypt it into the repo's src/appworld/
     if unpacked apps source is found: copy it directly into the repo's src/appworld/apps/
"""
import json
import os
import subprocess
import sys
import tarfile

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_APPWORLD = os.path.join(REPO_DIR, "benchmarks", "appworld")
LAYER_DIR = os.path.join(REPO_DIR, ".appworld_layers")
OUT_APPS = os.path.join(REPO_APPWORLD, "src", "appworld", "apps")
OUT_SOURCE = os.path.join(REPO_APPWORLD, "src", "appworld", ".source")

REGISTRY = "ghcr.io"
REPO = "stonybrooknlp/appworld"


def http(url, headers=None, raw=False):
    cmd = ["curl", "-sL", "--max-time", "900"]
    for k, v in (headers or {}).items():
        cmd += ["-H", "%s: %s" % (k, v)]
    cmd.append(url)
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError("curl failed: %s" % out.stderr.decode()[:300])
    return out.stdout if raw else out.stdout.decode("utf-8")


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print("[1/4] getting anonymous token ...")
    token = json.loads(http(
        "https://%s/token?service=ghcr.io&scope=repository:%s:pull" % (REGISTRY, REPO)
    ))["token"]
    auth = {"Authorization": "Bearer " + token}

    print("[2/4] getting manifest ...")
    manifest = json.loads(http(
        "https://%s/v2/%s/manifests/latest" % (REGISTRY, REPO),
        headers={**auth, "Accept": (
            "application/vnd.docker.distribution.manifest.v2+json, "
            "application/vnd.oci.image.manifest.v1+json")},
    ))
    layers = manifest["layers"]

    os.makedirs(LAYER_DIR, exist_ok=True)
    for i, layer in enumerate(layers):
        path = os.path.join(LAYER_DIR, "layer%d.tar.gz" % i)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print("  layer %d cached (%d MB)" % (i, os.path.getsize(path) // 1024 // 1024))
            continue
        print("  downloading layer %d/%d (%d MB) ..." % (
            i + 1, len(layers), layer["size"] // 1024 // 1024))
        blob = http("https://%s/v2/%s/blobs/%s" % (REGISTRY, REPO, layer["digest"]),
                    headers=auth, raw=True)
        with open(path, "wb") as f:
            f.write(blob)

    print("[3/4] searching layers for apps source ...")
    candidates = {
        "apps.bundle": "src/appworld/.source/apps.bundle",
        "tests.bundle": "src/appworld/.source/tests.bundle",
        "apps_init": "src/appworld/apps/__init__.py",
        "apps_phone": "src/appworld/apps/phone/__init__.py",
    }
    found = {}
    for i, layer in enumerate(layers):
        path = os.path.join(LAYER_DIR, "layer%d.tar.gz" % i)
        with tarfile.open(path, "r:gz") as tar:
            names = set(tar.getnames())
            for key, target in candidates.items():
                if key in found:
                    continue
                hits = [n for n in names
                        if n.endswith("/" + target) or n == target
                        or n.endswith("/appworld/" + target)]
                if hits:
                    found[key] = (path, hits[0])
                    print("  found %s -> %s (layer %d)" % (key, hits[0], i))

    if not found:
        print("ERROR: nothing found in image layers.")
        sys.exit(1)

    print("[4/4] extracting ...")
    for key, (layer_path, member) in found.items():
        with tarfile.open(layer_path, "r:gz") as tar:
            f = tar.extractfile(member)
            data = f.read()
        if key.endswith("bundle"):
            os.makedirs(OUT_SOURCE, exist_ok=True)
            dest = os.path.join(OUT_SOURCE, os.path.basename(member))
            with open(dest, "wb") as fh:
                fh.write(data)
            print("  saved bundle: %s (%d MB)" % (dest, len(data) // 1024 // 1024))
        else:
            # already-unpacked apps source: copy out the whole directory containing member
            base = member.rsplit("/", 1)[0]  # .../src/appworld/apps
            with tarfile.open(layer_path, "r:gz") as tar:
                for m in tar.getmembers():
                    if m.name.startswith(base + "/") and m.isfile():
                        rel = os.path.relpath(m.name, base)
                        dest = os.path.join(OUT_APPS, rel)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        with open(dest, "wb") as fh:
                            fh.write(tar.extractfile(m).read())
            print("  copied apps source from image to: %s" % OUT_APPS)

    print("DONE.")
    if "apps.bundle" in found:
        print("Next: run appworld install --repo inside .venv_appworld to decrypt the bundle")
    else:
        print("apps source is in place, you can use the appworld package in the repo directly")


if __name__ == "__main__":
    main()
