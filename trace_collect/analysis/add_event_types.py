"""
Add event_type field to every span in all trace JSON files.

Event types: planning, reasoning, action, observation, verification,
             repair, replanning, delegation, terminal, other
"""
import json, os, shutil
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

AGENTS = OrderedDict([
    # AppWorld
    ("appworld_autogen_batch1", "AutoGen-B1"),
    ("appworld_autogen_all_batch1", "AutoGen-All-B1"),
    ("appworld_autogen_batch2", "AutoGen-B2"),
    ("appworld_autogen_batch2_fix", "AutoGen-B2-Fix"),
    ("appworld_legacy_batch1", "Legacy-B1"),
    ("appworld_legacy_batch2", "Legacy-B2"),
    ("appworld_openai_agents_batch1", "OpenAI-B1"),
    ("appworld_openai_agents_batch2", "OpenAI-B2"),
    ("appworld_simplified_batch1", "Simplified-B1"),
    ("appworld_simplified_batch2", "Simplified-B2"),
    ("appworld_smolagents_batch1", "Smolagents-B1"),
    ("appworld_smolagents_batch2", "Smolagents-B2"),
    ("appworld_tau2_batch1", "Tau2-B1"),
    ("appworld_tau2_batch2", "Tau2-B2"),
    # SWE-bench Verified (each directory is one agent; no batch concept)
    ("swebench_autocoderover", "ACR"),
    ("swebench_deepagents", "DeepAgents"),
    ("swebench_devika", "Devika"),
    ("swebench_mini-swe-agent", "MiniSWE"),
    ("swebench_openhands", "OpenHands"),
    ("swebench_opengpts", "OpenGPTs"),
    ("swebench_sweagent", "SWEAgent"),
])

OBSERVATION_TOOLS = [
    "show_", "search_", "get_", "list_", "read_", "find_", "lookup_",
    "display", "retrieve", "fetch", "query",
]
VERIFICATION_KW = ["check", "verify", "confirm", "validate", "correct", "ensure"]

# Auth/session parameters — excluded from retry comparison (an access_token change does not indicate a semantic retry)
AUTH_PARAMS = {"access_token", "username", "password", "token"}


def _normalize_args(args_str: str) -> str:
    """Extract the key parts of tool call arguments for comparison (auth fields excluded; different args = supplementary exploration)."""
    try:
        data = json.loads(args_str)
    except Exception:
        return str(args_str).strip()
    kwargs = data.get("kwargs", data) if isinstance(data, dict) else {}
    if not isinstance(kwargs, dict):
        return str(args_str).strip()
    # Keep only semantic params (auth fields excluded), sorted for comparison
    semantic = {k: v for k, v in kwargs.items() if k not in AUTH_PARAMS}
    return json.dumps(semantic, sort_keys=True, ensure_ascii=False, default=str)


def classify_spans(spans):
    """Return list of event_type strings, one per span."""
    events = []
    for i, s in enumerate(spans):
        stype = s["type"]
        attrs = s.get("attributes", {})

        if stype == "invoke_agent":
            events.append(None)  # root span — no event type
            continue

        if stype == "tool.execution":
            tname = attrs.get("gen_ai.tool.name", "")

            if "complete_task" in tname:
                events.append("terminal")
                continue
            # SWE-side terminal tools: submit (swe-agent) / finish (openhands);
            # appworld tool names do not contain these words, so existing labels are unaffected
            if "submit" in tname or "finish" in tname:
                events.append("terminal")
                continue

            # Repair: consecutive retries of the same tool with **identical args**.
            # Consecutive same-name calls with different args are supplementary
            # exploration (e.g. search_contacts with a different relationship query), not repair.
            j = i - 1
            while j >= 0 and spans[j]["type"] != "tool.execution":
                j -= 1
            if j >= 0 and spans[j]["type"] == "tool.execution":
                prev_tname = spans[j]["attributes"].get("gen_ai.tool.name", "")
                if tname == prev_tname:
                    cur_args = _normalize_args(attrs.get("gen_ai.tool.call.arguments", ""))
                    prev_args = _normalize_args(
                        spans[j]["attributes"].get("gen_ai.tool.call.arguments", ""))
                    if cur_args == prev_args:
                        events.append("repair")
                        continue

            # Observation
            if any(kw in tname.lower() for kw in OBSERVATION_TOOLS):
                events.append("observation")
                continue

            events.append("action")
            continue

        if stype == "llm.call":
            in_msgs = str(attrs.get("gen_ai.input.messages", ""))
            out_msgs = str(attrs.get("gen_ai.output.messages", ""))

            # API prediction: input contains the marker, or the output is a plain-text
            # API name list (legacy's API prediction input is the task template, but the
            # output is a multi-line API name list containing complete_task —
            # it is not terminal, but the initial task planning)
            has_name_key = ('"name"' in out_msgs) or ("'name'" in out_msgs)
            if ("API prediction" in in_msgs
                    or ("complete_task" in out_msgs and not has_name_key and "\n" in out_msgs)):
                events.append("planning")
                continue

            # terminal: complete_task invoked in JSON tool-call format (the real task-termination decision)
            if "complete_task" in out_msgs and has_name_key:
                events.append("terminal")
                continue

            # terminal: task_completer's answer/status decision JSON (task-completion verdict)
            if '"answer"' in out_msgs and '"status"' in out_msgs:
                events.append("terminal")
                continue

            tc_count = max(out_msgs.count('"name"'), out_msgs.count("'name'"))
            is_text_only = (tc_count == 0)

            # What happened before?
            prev_is_llm = False
            prev_tool_err = False
            if i > 0:
                ps = spans[i - 1]
                if ps["type"] == "llm.call":
                    prev_is_llm = True
                elif ps["type"] == "tool.execution":
                    pres = str(ps["attributes"].get("gen_ai.tool.call.result", ""))
                    if any(kw in pres.lower() for kw in ["401", "error", "failed", "exception"]):
                        prev_tool_err = True

            if prev_is_llm and not prev_tool_err and is_text_only:
                # Consecutive thinking with text-only output = replanning (reflection);
                # if the output contains tool calls, fall through to the planning/action checks below
                events.append("replanning")
                continue
            if prev_is_llm and prev_tool_err and is_text_only:
                events.append("repair")
                continue
            if is_text_only:
                events.append("reasoning")
                continue
            if tc_count >= 2:
                events.append("planning")
                continue
            if not prev_is_llm and tc_count == 1:
                if any(kw in out_msgs.lower() for kw in VERIFICATION_KW):
                    events.append("verification")
                    continue
            if tc_count == 1:
                events.append("action")
                continue

            events.append("other")
            continue

        events.append("other")

    return events


def annotate_spans(spans):
    """Annotate event_type on a span list in place (the write-back step for classify_spans).

    This is the single authoritative implementation of event_type labelling:
    both the AppWorld and SWE-bench finalize flows call
    annotate_trace_file / annotate_trace_dir, avoiding duplicate loops that
    would cause the criteria to drift.
    """
    events = classify_spans(spans)
    for span, evt in zip(spans, events):
        if evt is not None:
            span["event_type"] = evt
        else:
            span.pop("event_type", None)  # remove if exists (e.g. invoke_agent)
    return events


def annotate_trace_file(path):
    """Read a single trace file -> annotate -> atomic write-back. Returns the number of spans."""
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    events = annotate_spans(data.get("spans", []))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)
    return len(events)


def annotate_trace_dir(trace_dir, limit=0):
    """Annotate event_type on all trace files under a directory. Returns the number of files processed."""
    if not os.path.isdir(trace_dir):
        return 0
    files = sorted(
        os.path.join(trace_dir, f) for f in os.listdir(trace_dir)
        if f.endswith(".json") and not f.endswith(".tmp")
    )
    if limit:
        files = files[:limit]
    for path in files:
        n = annotate_trace_file(path)
        print(f"  {os.path.basename(trace_dir)}/{os.path.basename(path)}  "
              f"({n} spans)")
    return len(files)


def main():
    total = 0
    for adir, name in AGENTS.items():
        d = os.path.join(ROOT, "traces", adir)
        if not os.path.isdir(d):
            print(f"  [SKIP] {name}: directory does not exist (not collected)")
            continue
        total += annotate_trace_dir(d)

    print(f"\n  Done: {total} trace files updated with 'event_type' field")


if __name__ == "__main__":
    main()
