"""Vendored autogen source activation.

The continuation backend drives the real AutoGen agent.  Instead of the
installed autogen_agentchat / autogen_core / autogen_ext wheels, we run the
in-repo microsoft/autogen source checkout
(``<project_root>/source_code/autogen/python/packages/<pkg>/src``) so that patches to
the library (e.g. the run loop's "model text output ends the run" behavior
in ``_assistant_agent.py``) take effect without reinstalling anything.

``activate()`` prepends the three source trees to ``sys.path``; it is
idempotent and must run before any ``autogen_*`` import.  When the checkout
is missing (e.g. not synced to the server), a warning is printed and the
installed packages keep serving as a silent fallback.

The vendored repo root can be overridden with the
``SPAN_OFFLOAD_AUTOGEN_REPO`` environment variable.
"""
from __future__ import annotations

import os
import sys

_PACKAGES = ("autogen-core", "autogen-agentchat", "autogen-ext")

_activated = False


def _src_dirs() -> list:
    root = os.environ.get("SPAN_OFFLOAD_AUTOGEN_REPO", "").strip()
    if not root:
        # <project_root>/source_code/autogen/python/packages/<pkg>/src
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(here)
        root = os.path.join(project_root, "source_code", "autogen")
    return [os.path.join(root, "python", "packages", p, "src")
            for p in _PACKAGES]


def activate() -> bool:
    """Prepend the vendored autogen source trees to sys.path.

    Returns True when the vendored source is active (False = fallback to
    the installed packages).  Safe to call repeatedly."""
    global _activated
    if _activated:
        return True
    _activated = True
    dirs = _src_dirs()
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        print("[span_offload] WARNING: vendored autogen source not found "
              "(%s); falling back to installed autogen packages" % ", ".join(missing),
              flush=True)
        return False
    for d in reversed(dirs):
        if d not in sys.path:
            sys.path.insert(0, d)
    print("[span_offload] using vendored autogen source: %s" % os.path.dirname(dirs[0]),
          flush=True)
    return True
