"""
Classify each span into event types and compare distributions across agents.
Event types: planning, reasoning, action, observation, verification,
             repair, replanning, delegation, terminal, other
"""
import json, os, re
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

AGENTS = OrderedDict([
    ("appworld_autogen_batch1", "AutoGen-B1"),
    ("appworld_autogen_batch2", "AutoGen-B2"),
    ("appworld_legacy_batch1", "Legacy-B1"),
    ("appworld_legacy_batch2", "Legacy-B2"),
    ("appworld_openai_agents_batch1", "OpenAI-B1"),
    ("appworld_simplified_batch1", "Simplified-B1"),
    ("appworld_simplified_batch2", "Simplified-B2"),
    ("appworld_smolagents_batch1", "Smolagents-B1"),
    ("appworld_smolagents_batch2", "Smolagents-B2"),
    ("appworld_tau2_batch1", "Tau2-B1"),
    ("appworld_tau2_batch2", "Tau2-B2"),
])

OBSERVATION_KEYWORDS = [
    "show_", "search_", "get_", "list_", "read_", "find_", "lookup_",
    "display", "retrieve", "fetch", "query",
]
VERIFICATION_KEYWORDS = ["check", "verify", "confirm", "validate", "correct", "ensure"]
TERMINAL_KEYWORDS = ["complete_task"]

# Auth/session parameters — excluded from retry comparison
AUTH_PARAMS = {"access_token", "username", "password", "token"}


def _normalize_args(args_str: str) -> str:
    """Extract the key parts of tool call arguments for comparison (auth fields excluded)."""
    try:
        data = json.loads(args_str)
    except Exception:
        return str(args_str).strip()
    kwargs = data.get("kwargs", data) if isinstance(data, dict) else {}
    if not isinstance(kwargs, dict):
        return str(args_str).strip()
    semantic = {k: v for k, v in kwargs.items() if k not in AUTH_PARAMS}
    return json.dumps(semantic, sort_keys=True, ensure_ascii=False, default=str)


def classify_spans(spans):
    """Return list of (span, event_type) tuples."""
    labelled = []

    for i, s in enumerate(spans):
        stype = s["type"]
        attrs = s.get("attributes", {})

        # ── Root span: always planning (episode-level) ──
        if stype == "invoke_agent":
            continue  # root span — skip, no event type

        if stype == "tool.execution":
            tname = attrs.get("gen_ai.tool.name", "")
            result = str(attrs.get("gen_ai.tool.call.result", ""))

            # Terminal
            if any(kw in tname for kw in TERMINAL_KEYWORDS):
                labelled.append((s, "terminal"))
                continue

            # Repair: consecutive retries of the same tool with **identical args** (different args = supplementary exploration)
            if i >= 2:
                j = i - 1
                while j >= 0 and spans[j]["type"] != "tool.execution":
                    j -= 1
                if j >= 0 and spans[j]["type"] == "tool.execution":
                    prev_tname = spans[j]["attributes"].get("gen_ai.tool.name", "")
                    if tname == prev_tname:
                        cur_args = _normalize_args(
                            spans[i]["attributes"].get("gen_ai.tool.call.arguments", ""))
                        prev_args = _normalize_args(
                            spans[j]["attributes"].get("gen_ai.tool.call.arguments", ""))
                        if cur_args == prev_args:
                            labelled.append((s, "repair"))
                            continue

            # Observation: data-retrieval APIs
            if any(kw in tname.lower() for kw in OBSERVATION_KEYWORDS):
                labelled.append((s, "observation"))
                continue

            # Default: action
            labelled.append((s, "action"))
            continue

        if stype == "llm.call":
            in_msgs = str(attrs.get("gen_ai.input.messages", ""))
            out_msgs = str(attrs.get("gen_ai.output.messages", ""))
            out_tok = attrs.get("gen_ai.usage.output_tokens", 0)

            # API prediction: input contains the marker, or the output is a plain-text
            # API name list (contains complete_task but no JSON tool-call structure
            # -> initial planning, not task termination)
            has_name_key = ('"name"' in out_msgs) or ("'name'" in out_msgs)
            if ("API prediction" in in_msgs
                    or (any(kw in out_msgs for kw in TERMINAL_KEYWORDS)
                        and not has_name_key and "\n" in out_msgs)):
                labelled.append((s, "planning"))
                continue

            # Terminal: complete_task invoked in JSON tool-call format (the real task-termination decision)
            if any(kw in out_msgs for kw in TERMINAL_KEYWORDS) and has_name_key:
                labelled.append((s, "terminal"))
                continue

            # Terminal: task_completer's answer/status decision JSON (task-completion verdict)
            if '"answer"' in out_msgs and '"status"' in out_msgs:
                labelled.append((s, "terminal"))
                continue

            # Count tool calls in output
            tc_count = max(out_msgs.count('"name"'), out_msgs.count("'name'"))
            is_text_only = tc_count == 0

            # Find what happened before this LLM call
            prev_spans = spans[:i]
            prev_tool_err = False
            prev_is_llm = False
            if prev_spans:
                prev_s = prev_spans[-1]
                if prev_s["type"] == "llm.call":
                    prev_is_llm = True
                elif prev_s["type"] == "tool.execution":
                    pres = str(prev_s["attributes"].get("gen_ai.tool.call.result", ""))
                    if any(kw in pres.lower() for kw in ["401", "error", "failed", "exception"]):
                        prev_tool_err = True

            # Replanning: consecutive L->L thinking with text-only output = replanning (reflection)
            if prev_is_llm and not prev_tool_err and is_text_only:
                labelled.append((s, "replanning"))
                continue

            # Repair: consecutive L->L thinking (text-only) with a tool error in the previous round = repair
            if prev_is_llm and prev_tool_err and is_text_only:
                labelled.append((s, "repair"))
                continue

            # Reasoning: text-only response (no tool calls)
            if is_text_only:
                labelled.append((s, "reasoning"))
                continue

            # Planning: generates 2+ tool calls
            if tc_count >= 2:
                labelled.append((s, "planning"))
                continue

            # Verification: generates observation-type tool calls after tool execution
            if not prev_is_llm and tc_count == 1:
                out_lower = out_msgs.lower()
                if any(kw in out_lower for kw in VERIFICATION_KEYWORDS):
                    labelled.append((s, "verification"))
                    continue

            # Action: generates exactly 1 tool call
            if tc_count == 1:
                labelled.append((s, "action"))
                continue

            # Other: everything else
            labelled.append((s, "other"))
            continue

        # Unknown span type
        labelled.append((s, "other"))

    return labelled


# Process all traces
all_dist = {}
for adir, name in AGENTS.items():
    d = os.path.join(ROOT, "traces", adir)
    if not os.path.isdir(d):
        print(f"  [SKIP] {name}: directory does not exist (not collected)")
        continue
    totals = Counter()
    per_task = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(d, f), encoding="utf-8") as fp:
            data = json.load(fp)
        labelled = classify_spans(data["spans"])
        counts = Counter(e for _, e in labelled)
        totals += counts
        per_task.append(counts)
    all_dist[name] = {"total": totals, "per_task": per_task}

# ── Print comparison ──
EVENTS = ["planning", "reasoning", "action", "observation", "verification",
          "repair", "replanning", "terminal", "other"]
NAMES = list(AGENTS.values())

w = 155
print()
print("=" * w)
print("  EVENT TYPE DISTRIBUTION — classified from trace spans via heuristics")
print("=" * w)
print()

# Table 1: Absolute counts
print("  ABSOLUTE COUNTS (sum across all 13 tasks)")
print(f"  {'Event':<16}" + "".join(f"{n:>12}" for n in NAMES) + f"  {'Dominant Agent':<20}")
print("  " + "-" * (28 + 12 * 6 + 20))
for evt in EVENTS:
    counts = [all_dist[n]["total"].get(evt, 0) for n in NAMES]
    total = sum(counts)
    dominant = NAMES[counts.index(max(counts))] if max(counts) > 0 else "—"
    print(f"  {evt:<16}" + "".join(f"{c:>12}" for c in counts) + f"  {dominant:<20}")
print()

# Table 2: Percentage distribution per agent (normalize each column)
print("  PERCENTAGE DISTRIBUTION (% of agent's total spans)")
print(f"  {'Event':<16}" + "".join(f"{n:>12}" for n in NAMES))
print("  " + "-" * (28 + 12 * 6))
for evt in EVENTS:
    pcts = []
    for n in NAMES:
        total = sum(all_dist[n]["total"].values())
        c = all_dist[n]["total"].get(evt, 0)
        pcts.append(f"{100*c/total:.1f}%" if total > 0 else "—")
    print(f"  {evt:<16}" + "".join(f"{p:>12}" for p in pcts))
print()

# Table 3: Event type ratio: (planning+reasoning) / action — cognitive vs executive
print("  COGNITIVE-TO-EXECUTIVE RATIO: (planning+reasoning+replanning+repair) / (action+observation)")
print(f"  {'Ratio':<16}" + "".join(f"{n:>12}" for n in NAMES))
print("  " + "-" * (28 + 12 * 6))
for n in NAMES:
    d = all_dist[n]["total"]
    cognitive = d.get("planning", 0) + d.get("reasoning", 0) + d.get("replanning", 0) + d.get("repair", 0)
    executive = d.get("action", 0) + d.get("observation", 0)
    ratio = cognitive / max(executive, 1)
    print(f"  {'cog/exec':<16}" + "".join(f"{('%.2f'%ratio) if n2==n else '':>12}" for n2 in NAMES))

# Per agent
for n in NAMES:
    d = all_dist[n]["total"]
    cognitive = d.get("planning",0)+d.get("reasoning",0)+d.get("replanning",0)+d.get("repair",0)
    executive = d.get("action",0)+d.get("observation",0)
    ratio = cognitive/max(executive,1)
    print(f"  {n:<16}  cog={cognitive}  exec={executive}  ratio={ratio:.2f}")

print()
print("  CLASSIFICATION HEURISTICS:")
print("    planning   = API prediction call OR generates 2+ tool calls in one response")
print("    reasoning  = text-only LLM response (no tool calls)")
print("    action     = generates exactly 1 tool call OR non-observation tool execution")
print("    observation= tool name contains show_/search_/get_/list_/read_/find_/lookup_")
print("    verification= LLM output contains check/verify/confirm/validate keywords")
print("    repair     = consecutive retry of same tool OR L->L after tool error")
print("    replanning = L->L transition without preceding tool error")
print("    terminal   = tool/LLM output contains complete_task")
print("    other      = unclassified")
print()
print("=" * w)
