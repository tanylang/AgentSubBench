"""Stratified deterministic span sampling (zero randomness: same input, same output).

Purpose: pick a fixed-size experiment subset from a batch's (agent × collection baseline)
eligible spans to control experiment cost (e.g. 50 spans × 15 rounds). Sampling rules:

  1. only sample eligible spans (same filter convention as run_offload_profile);
  2. stratify by control role, priority = tool_selection -> final_synthesis ->
     other -> tool_argument_generation;
  3. within each layer "take 1 per task first" (task_id lexicographic order,
     invocation_index ascending takes the first), maximizing task coverage;
  4. still short of n, fill with remaining tool_argument_generation spans (inv ascending);
  5. over n, trim from the tail of the append order (keeping role/task coverage,
     sacrificing the outermost tool_argument_generation);
  6. output sorted by (task_id, invocation_index).

Determinism guarantee: every sort key is (task_id, invocation_index), no random
source anywhere; rerunning the same spans file on any machine at any time yields
the exact same 50 spans.

Usage (library):
    from span_offload.span_sampling import stratified_sample
    picked = stratified_sample(spans, n=50)
Or CLI: python scripts/sample_spans.py --spans OUT_DIR/spans.jsonl \
        --out OUT_DIR/spans_50.jsonl --n 50
"""
from __future__ import annotations

from typing import Any, Dict, List

#: Control-role sampling priority (first come first served; 'other' captures everything outside the three main roles)
ROLE_PRIORITY = ["tool_selection", "final_synthesis", "other",
                 "tool_argument_generation"]


def _sort_key(s: Dict[str, Any]):
    return (str(s.get("task_id", "")), int(s.get("invocation_index", 0) or 0))


def _role_of(s: Dict[str, Any]) -> str:
    np_ = str((s.get("span_descriptor") or {}).get("node_pattern", ""))
    return np_ if np_ in ROLE_PRIORITY else "other"


def _state_effect(s: Dict[str, Any]) -> str:
    tags = ((s.get("span_descriptor") or {}).get("mechanism_tags") or {}) \
        .get("state_effect") or []
    return "write" if "external_state_mutation" in tags else "read"


def one_per_task(items: List[Dict[str, Any]],
                 prefer_write: bool = False) -> List[Dict[str, Any]]:
    """Take 1 per task (lexicographic order + inv ascending takes the first).

    prefer_write=True (for execution spans): prefer a write-state span per task — fixes a
    historical selection bias (taking the first by inv almost always picks a read span, so
    the largest class in the whole set, "write execution spans", was entirely absent from
    the three-layer experiment); when a task has no write span, fall back to its earliest read span."""
    if not prefer_write:
        seen = set()
        out = []
        for s in items:
            t = s.get("task_id")
            if t not in seen:
                seen.add(t)
                out.append(s)
        return out
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for s in items:
        by_task.setdefault(s.get("task_id"), []).append(s)
    out = []
    for t in sorted(by_task):
        g = by_task[t]
        writes = [s for s in g if _state_effect(s) == "write"]
        out.append((writes or g)[0])
    return out


def stratified_sample(spans: List[Dict[str, Any]],
                      n: int = 50) -> List[Dict[str, Any]]:
    """Deterministically sample n eligible spans from spans (a raw jsonl record list)."""
    elig = sorted((s for s in spans if s.get("eligible")), key=_sort_key)
    by_role: Dict[str, List[Dict[str, Any]]] = {}
    for s in elig:
        by_role.setdefault(_role_of(s), []).append(s)
    for role in by_role:
        by_role[role] = sorted(by_role[role], key=_sort_key)

    picked: List[Dict[str, Any]] = []
    picked_ids = set()
    # Round 1: by role priority, take 1 per task (execution spans prefer write state)
    for role in ROLE_PRIORITY:
        for s in one_per_task(by_role.get(role, []),
                              prefer_write=(role == "tool_argument_generation")):
            if s.get("span_id") not in picked_ids:
                picked_ids.add(s["span_id"])
                picked.append(s)
    # Round 2: short of n, fill with remaining tool_argument_generation (inv ascending)
    if len(picked) < n:
        rest = [s for s in by_role.get("tool_argument_generation", [])
                if s.get("span_id") not in picked_ids]
        picked += rest[: max(0, n - len(picked))]
    # Over n: trim the tail (the lowest-priority appended items)
    picked = picked[:n]
    return sorted(picked, key=_sort_key)
