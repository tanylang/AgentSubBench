"""Local-layer equivalence verifier: decide whether, after a span is offloaded to the
SLM, the SLM output matches the baseline (large-model) output.

The logic mirrors AppWorld's built-in task verifier (TestTracker + assert_plus semantics):
  - each check item is a "requirement", judged pass/fail one by one with details;
  - comparison semantics follow assert_plus normalization: numeric tolerance, normalize_text
    (ignore case / collapse whitespace), ignore_order, recursive comparison of structured values;
  - tool-call contract: compare tool-name multisets + per-call arguments + execution results
    (runtime environment observations, i.e. what the environment returned after each call);
  - freeform-text contract: semantic detection — API-list-like text uses line-set Jaccard, the rest uses
    word-set Jaccard and character-bigram overlap (a deterministic approximation without an embedding model).

This module does not depend on the appworld package and can be unit-tested standalone.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

#: Absolute tolerance for numeric comparison (AppWorld's tolerance semantics)
NUMERIC_TOLERANCE = 1e-6
#: Line-set Jaccard threshold for API-list-like text
API_LIST_JACCARD_THRESHOLD = 0.8
#: Freeform-text semantic overlap threshold (the larger of word Jaccard and character-bigram overlap)
TEXT_SIM_THRESHOLD = 0.5
#: Max length of result content in detail strings
DETAIL_MAX_LEN = 300

_API_NAME_RE = re.compile(r"^[A-Za-z_][\w.]*$")


def normalize_text(s: Any) -> str:
    """assert_plus's normalize_text equivalent: ignore case, strip, collapse whitespace."""
    return " ".join(str(s or "").lower().split())


def _normalize_tool_name(name: Any) -> str:
    return str(name or "").lower().replace(".", "__").strip()


def _num(a: Any) -> Any:
    if isinstance(a, bool):
        return a
    if isinstance(a, (int, float)):
        return a
    try:
        return float(str(a).strip())
    except (TypeError, ValueError):
        return a


def _is_number(a: Any) -> bool:
    return isinstance(a, (int, float)) and not isinstance(a, bool)


_JWT_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")


def _jwt_equal_except_exp(a: str, b: str) -> Tuple[bool, bool]:
    """Compare two JWTs: equivalent when their payloads match after stripping the exp claim.

    login's expires_in is a random 10-30 minutes (random state is not reproducible), so exp differs
    per login; under the frozen clock the tokens are semantically equivalent. Returns (is_pair, equal)."""
    if not (_JWT_RE.match(a) and _JWT_RE.match(b)):
        return False, False
    import base64 as _b64

    def payload(tok: str):
        p = tok.split(".")[1]
        p += "=" * (-len(p) % 4)
        try:
            return json.loads(_b64.urlsafe_b64decode(p))
        except Exception:
            return None

    pa, pb = payload(a), payload(b)
    if not isinstance(pa, dict) or not isinstance(pb, dict):
        return True, False
    pa = {k: v for k, v in pa.items() if k != "exp"}
    pb = {k: v for k, v in pb.items() if k != "exp"}
    return True, pa == pb


_EPHEMERAL_TS_RE = re.compile(r'"(created_at|updated_at)"\s*:\s*"[^"]*"')


def _masked_prefix_equal(left: str, right: str) -> Tuple[bool, str]:
    """Truncation-asymmetry rescue: mask timestamps then compare on the common prefix (verifiable
    window = the shorter side). Used for execution-result comparison where one side is full JSON
    and the other was truncated by the recorder (replay side TRACE_RESULT_MAX_LEN=2000)."""
    a = _EPHEMERAL_TS_RE.sub(r'"\1":"<ts>"', str(left))
    b = _EPHEMERAL_TS_RE.sub(r'"\1":"<ts>"', str(right))
    n = min(len(a), len(b))
    if n == 0:
        return False, "value: empty prefix"
    if a[:n] == b[:n]:
        return True, ""
    i = next((i for i in range(n) if a[i] != b[i]), n)
    return False, ("value: prefix mismatch at %d: %r... != %r..." % (
        i, a[max(0, i - 40): i + 40], b[max(0, i - 40): i + 40]))


def _parse_struct_value(s: str) -> Any:
    """Parse a tool-result string into a structured value: **every string is first tried as
    JSON** (pretty JSON starts with '{'/'[' and contains null, which literal_eval cannot parse),
    then a Python literal (repr single-quoted form) on failure; keep peeling the unwrapped inner
    strings (the collection side commonly double-wraps str(repr)). Returns None when unparseable."""
    v: Any = s
    for _ in range(4):  # peel at most 4 layers, guard against pathological nesting
        if not isinstance(v, str):
            return v
        t = v.strip()
        if not t:
            return None
        try:
            v = json.loads(t)
            continue
        except ValueError:
            pass
        try:
            return ast.literal_eval(t)
        except (ValueError, SyntaxError, MemoryError):
            return None
    return v


def compare_values(left: Any, right: Any, tolerance: float = NUMERIC_TOLERANCE,
                  ignore_order: bool = False, path: str = "",
                  skip_keys: Optional[Any] = None) -> Tuple[bool, str]:
    """Structured value comparison, returns (ok, detail). Numbers use tolerance; strings are
    first parsed into structured values (JSON or Python literal) before comparing — the
    serialization-format difference between collection-side records (repr single quotes) and
    replay results (JSON double quotes) is a common false-mismatch source; normalization
    dissolves the format difference and only real value differences count. Lists compare in
    sequence when ignore_order=True; dicts recurse per key, and keys in skip_keys are exempt
    (e.g. memory records' created_at/updated_at: replay time and collection time necessarily differ, matching the memory-hash semantics of excluding timestamps)."""
    if _is_number(left) and _is_number(right):
        if abs(left - right) <= tolerance:
            return True, ""
        return False, "%s: %r != %r" % (path or "value", left, right)
    # Numeric string vs number: try a normalized tolerance comparison (e.g. "1029.0" vs 1029)
    ln, rn = _num(left), _num(right)
    if _is_number(ln) and _is_number(rn):
        if abs(ln - rn) <= tolerance:
            return True, ""
        return False, "%s: %r != %r" % (path or "value", left, right)
    if isinstance(left, str) or isinstance(right, str):
        sl, sr = normalize_text(left), normalize_text(right)
        if sl == sr:
            return True, ""
        # JWT token: strip exp (login expiry offset is random, not reproducible) and compare semantically
        _jwt_pair, _jwt_eq = _jwt_equal_except_exp(str(left), str(right))
        if _jwt_pair:
            return (True, "") if _jwt_eq else \
                (False, "%s: jwt payload mismatch (exp excluded)" % (path or "value"))
        # When both sides parse into structured values, compare recursively by structure.
        # Note: structured parsing must use the raw text — normalize_text changes case
        # and destroys 'None'/'True' literals (measured: pretty JSON null fails to parse
        # because of this, and values with identical format are judged unequal)
        jl = _parse_struct_value(str(left))
        jr = _parse_struct_value(str(right))
        if jl is not None and jr is not None:
            return compare_values(jl, jr, tolerance, ignore_order, path,
                                  skip_keys)
        return False, "%s: %r != %r" % (path or "value", left, right)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False, "%s: list length %d != %d" % (path, len(left), len(right))
        if ignore_order:
            matched = [False] * len(right)
            for a in left:
                hit = False
                for i, b in enumerate(right):
                    if matched[i]:
                        continue
                    ok, _ = compare_values(a, b, tolerance, True, path,
                                           skip_keys)
                    if ok:
                        matched[i] = True
                        hit = True
                        break
                if not hit:
                    return False, "%s: %r not found in %r" % (path, a, right)
            return True, ""
        for i, (a, b) in enumerate(zip(left, right)):
            ok, det = compare_values(a, b, tolerance, False,
                                     "%s[%d]" % (path, i), skip_keys)
            if not ok:
                return False, det
        return True, ""
    if isinstance(left, dict) and isinstance(right, dict):
        skip = set(skip_keys or ())
        keys_l = set(left.keys()) - skip
        keys_r = set(right.keys()) - skip
        if keys_l != keys_r:
            return False, "%s: keys %r != %r" % (path, sorted(keys_l),
                                                 sorted(keys_r))
        for k in sorted(keys_l):
            ok, det = compare_values(left[k], right[k], tolerance, False,
                                     "%s.%s" % (path, k) if path else str(k),
                                     skip_keys)
            if not ok:
                return False, det
        return True, ""
    if left == right:
        return True, ""
    return False, "%s: %r != %r" % (path or "value", left, right)


def _calls_from_parsed(parsed: Any) -> List[Dict[str, Any]]:
    if not isinstance(parsed, dict):
        return []
    return list(parsed.get("tool_calls") or [])


#: Default values for tool arguments: the semantically equivalent value when some optional
#: argument is omitted (consistent with AppWorld supervisor__complete_task's status defaulting
#: to success). Fill default keys before comparing, to avoid judging "optional argument omitted" as a mismatch.
_ARG_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "complete_task": {"status": "success"},
}


def _fill_arg_defaults(name: Any, args: Any) -> Any:
    if not isinstance(args, dict):
        return args
    key = _normalize_tool_name(name)
    defaults = _ARG_DEFAULTS.get(key.rsplit("__", 1)[-1], {})
    if not defaults:
        return args
    out = dict(args)
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


def compare_tool_calls(slm_parsed: Any, baseline_parsed: Any) -> Dict[str, Any]:
    """Tool-call contract: SLM calls vs baseline calls (name multiset + per-call arguments)."""
    slm_calls = _calls_from_parsed(slm_parsed)
    base_calls = _calls_from_parsed(baseline_parsed)
    reqs: List[Dict[str, Any]] = []

    slm_names = sorted(_normalize_tool_name(c.get("name")) for c in slm_calls)
    base_names = sorted(_normalize_tool_name(c.get("name")) for c in base_calls)
    reqs.append({
        "requirement": "assert tool call names match (ignore order).",
        "passed": slm_names == base_names,
        "details": "slm=%s vs baseline=%s" % (slm_names, base_names)
        if slm_names != base_names else "",
    })

    if len(slm_calls) != len(base_calls):
        reqs.append({
            "requirement": "assert tool call arguments match.",
            "passed": False,
            "details": "call count mismatch: slm=%d baseline=%d" % (
                len(slm_calls), len(base_calls)),
        })
    else:
        ok_all = True
        details = []
        for i, (sc, bc) in enumerate(zip(slm_calls, base_calls)):
            # Fill default arguments by tool name first (e.g. complete_task's status→success)
            # before comparing, to avoid judging "optional argument omitted" as a mismatch
            s_args = _fill_arg_defaults(sc.get("name"), sc.get("arguments"))
            b_args = _fill_arg_defaults(bc.get("name"), bc.get("arguments"))
            ok, det = compare_values(s_args, b_args)
            if not ok:
                ok_all = False
                details.append("call %d (%s): %s" % (
                    i, _normalize_tool_name(sc.get("name")), det))
        reqs.append({
            "requirement": "assert tool call arguments match.",
            "passed": ok_all,
            "details": "; ".join(details),
        })
    return {"contract": "tool_call_list", "requirements": reqs,
            "passed": all(r["passed"] for r in reqs)}


def _error_status(txt: str):
    """Extract the HTTP status code from an error string (None if absent)."""
    m = re.search(r"status code is (\d+)", txt)
    return m.group(1) if m else None


def compare_execution_results(slm_results: List[Dict[str, Any]],
                              base_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tool execution results (runtime environment observations): compare per call the success flag and
    the result content.

    slm_results / base_results: [{"tool": ..., "ok": bool, "result": str}, ...], paired in order;
    normalized names must match.

    Semantics aligned with the C0 gate: names/count/ok flag = hard (blocks passed); the result **value** is
    also judged hard, but non-reproducible ephemeral keys are exempt — memory records' created_at/updated_at
    (replay time and collection time necessarily differ, matching the memory-hash semantics of excluding timestamps), skipped via compare_values's skip_keys.
    """
    reqs: List[Dict[str, Any]] = []
    slm_names = sorted(_normalize_tool_name(r.get("tool")) for r in slm_results)
    base_names = sorted(_normalize_tool_name(r.get("tool")) for r in base_results)
    reqs.append({
        "requirement": "assert executed tool names match (ignore order).",
        "passed": slm_names == base_names,
        "details": "slm=%s vs baseline=%s" % (slm_names, base_names)
        if slm_names != base_names else "",
    })
    ok_all = True
    details: List[str] = []
    if len(slm_results) != len(base_results):
        ok_all = False
        details.append("execution count mismatch: slm=%d baseline=%d" % (
            len(slm_results), len(base_results)))
    else:
        for i, (sr, br) in enumerate(zip(slm_results, base_results)):
            if bool(sr.get("ok")) != bool(br.get("ok")):
                ok_all = False
                details.append("call %d (%s): execution ok flag %r != %r" % (
                    i, _normalize_tool_name(sr.get("tool")),
                    sr.get("ok"), br.get("ok")))
                continue
            sr_txt, br_txt = str(sr.get("result", "")), str(br.get("result", ""))
            if not bool(sr.get("ok")) and not bool(br.get("ok")):
                # Both failed: compare the failure semantics (status code) rather than the
                # error string format (recorded side "Execution failed. Traceback:...", replay
                # side "Error: Exception: Response status code is 422:...", same cause,
                # different wording); the same status code counts as equivalent
                _sc_s = _error_status(sr_txt)
                _sc_b = _error_status(br_txt)
                if _sc_s is not None and _sc_s == _sc_b:
                    continue
            ok, det = compare_values(
                sr_txt, br_txt,
                skip_keys={"created_at", "updated_at"})
            if not ok:
                # Truncation-asymmetry rescue: one side is full JSON, the other was truncated
                # by the recorder (replay side TRACE_RESULT_MAX_LEN=2000 truncation, structural
                # parse fails) — mask timestamps then compare on the common prefix
                if (_parse_struct_value(str(sr_txt)) is None) != (
                        _parse_struct_value(str(br_txt)) is None):
                    ok2, det2 = _masked_prefix_equal(sr_txt, br_txt)
                    if ok2:
                        ok, det = True, det2
            if not ok:
                ok_all = False
                details.append("call %d (%s): %s" % (
                    i, _normalize_tool_name(sr.get("tool")),
                    (det or "result mismatch")[:DETAIL_MAX_LEN]))
    reqs.append({
        "requirement": "assert tool execution results match.",
        "passed": ok_all,
        "details": "; ".join(details)[:2000],
    })
    return {"contract": "tool_execution", "requirements": reqs,
            "passed": all(r["passed"] for r in reqs)}


def _is_api_list(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return (len(lines) >= 2
            and all(_API_NAME_RE.fullmatch(l.replace("__", ".")) for l in lines))


def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def _text_similarity(a: str, b: str) -> float:
    """Deterministic semantic approximation: the larger of word-set Jaccard and character-bigram overlap."""
    ta, tb = normalize_text(a), normalize_text(b)
    if ta == tb:
        return 1.0
    wa, wb = set(ta.split()), set(tb.split())
    word_jaccard = (len(wa & wb) / len(wa | wb)) if (wa and wb) else 0.0
    ba, bb = _bigrams(ta), _bigrams(tb)
    char_f1 = 2 * len(ba & bb) / max(1, len(ba) + len(bb))
    return max(word_jaccard, char_f1)


def compare_texts(slm_parsed: Any, baseline_parsed: Any) -> Dict[str, Any]:
    """Freeform-text contract: semantic detection. API-list-like text uses line-set
    Jaccard; the rest uses word/character overlap."""
    slm_text = ""
    base_text = ""
    if isinstance(slm_parsed, dict):
        slm_text = str(slm_parsed.get("text") or "")
    if isinstance(baseline_parsed, dict):
        base_text = str(baseline_parsed.get("text") or "")
    reqs: List[Dict[str, Any]] = []
    if _is_api_list(slm_text) and _is_api_list(base_text):
        slm_set = {_normalize_tool_name(l) for l in slm_text.splitlines() if l.strip()}
        base_set = {_normalize_tool_name(l) for l in base_text.splitlines() if l.strip()}
        score = len(slm_set & base_set) / len(slm_set | base_set) if (slm_set or base_set) else 0.0
        reqs.append({
            "requirement": "assert predicted api list matches (line-set Jaccard >= %.2f)."
                           % API_LIST_JACCARD_THRESHOLD,
            "passed": score >= API_LIST_JACCARD_THRESHOLD,
            "details": "jaccard=%.3f slm=%s baseline=%s" % (score, sorted(slm_set), sorted(base_set)),
        })
    else:
        score = _text_similarity(slm_text, base_text)
        reqs.append({
            "requirement": "assert text output semantically matches (overlap >= %.2f)."
                           % TEXT_SIM_THRESHOLD,
            "passed": score >= TEXT_SIM_THRESHOLD,
            "details": "similarity=%.3f slm=%r baseline=%r" % (
                score, slm_text[:DETAIL_MAX_LEN], base_text[:DETAIL_MAX_LEN]),
        })
    return {"contract": "freeform_text", "requirements": reqs,
            "passed": all(r["passed"] for r in reqs)}


def compare_answers(slm_parsed: Any, baseline_parsed: Any) -> Dict[str, Any]:
    """answer_json contract: structured comparison (numbers with tolerance)."""
    ok, det = compare_values(slm_parsed, baseline_parsed)
    reqs = [{
        "requirement": "assert answers match.",
        "passed": ok,
        "details": det[:DETAIL_MAX_LEN] if det else "",
    }]
    return {"contract": "answer_json", "requirements": reqs,
            "passed": all(r["passed"] for r in reqs)}


def verify_local_equivalence(contract: str, slm_parsed: Any, baseline_parsed: Any,
                             slm_results: List[Dict[str, Any]] | None = None,
                             base_results: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Local equivalence verifier entry: compare SLM output vs baseline output by contract type, return
    {contract, requirements[{requirement, passed, details}], passed}.

    slm_results / base_results only matter for the tool_call_list contract: the execution result
    (environment observation) comparison is added by the caller after injected execution completes."""
    if contract == "tool_call_list":
        out = compare_tool_calls(slm_parsed, baseline_parsed)
        if slm_results is not None and base_results is not None:
            exec_out = compare_execution_results(slm_results, base_results)
            out["requirements"].extend(exec_out["requirements"])
            out["passed"] = out["passed"] and exec_out["passed"]
        return out
    if contract == "answer_json":
        return compare_answers(slm_parsed, baseline_parsed)
    return compare_texts(slm_parsed, baseline_parsed)
