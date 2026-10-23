"""Local verifiers for Stage 10 (spec 14.2.2.E / 14.5).

A verifier_type is a pre-replay capability descriptor (schema_check,
tool_exec_success, policy_check, goal_state_match, db_state_match, none).
The runtime executes the corresponding checks on SLM output during replay and
reports a VerifierVerdict; the verdict feeds local_outcome and fallback, never
the span descriptor.

verifier_cost is kept at 0.0 in this pilot: all listed verifiers are local
(deterministic checks / world predicates), no verifier LLM calls are made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VerifierVerdict:
    triggered: bool = False
    detected_failure: bool = False
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    executed: List[str] = field(default_factory=list)

    def merge(self, other: "VerifierVerdict") -> "VerifierVerdict":
        return VerifierVerdict(
            triggered=self.triggered or other.triggered,
            detected_failure=self.detected_failure or other.detected_failure,
            passed=self.passed and other.passed,
            errors=list(self.errors) + list(other.errors),
            executed=list(self.executed) + list(other.executed),
        )


def no_verifier_verdict() -> VerifierVerdict:
    """Verdict when verifier_type is empty or ['none']: vacuously passed."""
    return VerifierVerdict(triggered=False, detected_failure=False, passed=True)


# ---------------------------------------------------------------------------
# Minimal JSON-schema validator (no external dependency).  Supports the subset
# needed by pilot output contracts: type, required, properties, items, enum,
# minimum/maximum, minLength/maxLength, additionalProperties.
# ---------------------------------------------------------------------------

def validate_schema(instance: Any, schema: Dict[str, Any], path: str = "$") -> List[str]:
    errors: List[str] = []

    def err(msg: str) -> None:
        errors.append("%s: %s" % (path, msg))

    if not isinstance(schema, dict):
        return errors

    if "enum" in schema and instance not in schema["enum"]:
        err("value %r not in enum %r" % (instance, schema["enum"]))

    expected = schema.get("type")
    if isinstance(expected, list):  # e.g. ["string", "null"]
        if not any(_type_matches(instance, t) for t in expected):
            err("expected one of %r, got %r" % (expected, type(instance).__name__))
    elif expected is not None and not _type_matches(instance, expected):
        err("expected %r, got %r" % (expected, type(instance).__name__))
        return errors

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            err("string shorter than minLength %s" % schema["minLength"])
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            err("string longer than maxLength %s" % schema["maxLength"])
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            err("%r below minimum %s" % (instance, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            err("%r above maximum %s" % (instance, schema["maximum"]))
    elif isinstance(instance, dict):
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        for key in required:
            if key not in instance:
                err("missing required property %r" % key)
        for key, value in instance.items():
            if props:
                if key not in props:
                    if schema.get("additionalProperties") is False:
                        err("unexpected property %r" % key)
                    continue
                errors.extend(validate_schema(value, props[key], path + "." + key))
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            err("fewer properties than minProperties %s" % schema["minProperties"])
    elif isinstance(instance, list):
        items = schema.get("items")
        if items is not None:
            for i, item in enumerate(instance):
                errors.extend(validate_schema(item, items, path + "[%d]" % i))
    return errors


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "boolean":
        return isinstance(instance, bool)
    return True
