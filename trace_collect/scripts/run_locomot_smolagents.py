"""LocoMo trace collection driver -- smolagents ToolCallingAgent.

Zero agent source changes -- LLM recording reuses the client patch pattern
from the AppWorld smolagents driver (inject seed/extra_body/max_completion_tokens
+ log_llm after completion); tool execution goes through the Tool subclasses
in adapter.locomot (forward → execute_tool).

Run:
    python scripts/run_locomot_smolagents.py [--samples id1,id2] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from adapter.locomot import (LocomoSample, LocomoMemory,  # noqa: E402
                             build_smolagents_tools, OAI_TOOL_SCHEMAS)
from adapter.model_config import (MAIN_MODEL_NAME, MAIN_BASE_URL,  # noqa: E402
                                  THINKING_MODE)
from locomot_common import (  # noqa: E402
    load_tasks_from_env, sample_questions, build_system_prompt,
    build_question_prompt, run_one_sample,
)

AGENT_NAME = "LocoMo-Smolagents"
TASKS = load_tasks_from_env()
OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or \
    os.path.join(ROOT, "traces", "locomo_smolagents_batch1")


def launch(sample: LocomoSample, env: LocomoMemory, collector) -> dict:
    from smolagents import ToolCallingAgent
    from smolagents import OpenAIServerModel

    model = OpenAIServerModel(model_id=MAIN_MODEL_NAME,
                              api_base=MAIN_BASE_URL,
                              api_key=os.environ.get("OPENAI_API_KEY", ""))

    # client patch: inject seed/extra_body/max_completion_tokens + record llm.call
    _orig_create = model.client.chat.completions.create

    def _patched_create(**kwargs):
        kwargs["extra_body"] = dict(kwargs.get("extra_body") or {}, **THINKING_MODE)
        kwargs.setdefault("seed", 100)
        kwargs.setdefault("max_completion_tokens", 8192)
        kwargs.setdefault("temperature", 0.0)
        # smolagents defaults to tool_choice="required", which conflicts with DeepSeek thinking mode
        # ("Thinking mode does not support this tool_choice" 400) → force auto uniformly
        if kwargs.get("tool_choice") == "required":
            kwargs["tool_choice"] = "auto"
        t0 = time.perf_counter()
        result = _orig_create(**kwargs)
        try:
            usage = result.usage
            msg = result.choices[0].message
            out = msg.content or ""
            tcs = getattr(msg, "tool_calls", None)
            struct = None
            if tcs:
                struct = [{"id": t.id, "name": t.function.name,
                           "arguments": t.function.arguments} for t in tcs]
            collector.log_llm(
                model=MAIN_MODEL_NAME,
                messages=[{"role": str(m.get("role", "unk")),
                           "content": str(m.get("content") or "")}
                          for m in kwargs.get("messages", [])],
                output_text=out, tools=kwargs.get("tools"),
                tool_calls=struct,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        except Exception:
            pass
        return result

    model.client.chat.completions.create = _patched_create

    tools = build_smolagents_tools(sample, env, collector)
    system_prompt = build_system_prompt(sample)
    qas = sample_questions(sample)
    answers = []

    # Default templates + override system_prompt (smolagents 1.19 has no system_prompt parameter)
    import importlib.resources
    import yaml
    _templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts")
        .joinpath("toolcalling_agent.yaml").read_text())
    _templates["system_prompt"] = system_prompt

    for i, qa in enumerate(qas):
        # New instance per question (the framework's internal memory resets automatically; persistent memory lives in LocomoMemory)
        agent = ToolCallingAgent(
            tools=tools, model=model,
            prompt_templates=_templates,
            max_steps=10, add_base_tools=False,
        )
        try:
            resp = agent.run(build_question_prompt(i, len(qas), qa.question))
            pred = (getattr(resp, "content", None) or "").strip()
        except Exception as e:
            pred = f"ERROR: {e}"
        answers.append({
            "idx": i, "question": qa.question, "pred": pred,
            "category": qa.category, "evidence": qa.evidence,
        })
    return {"answers": answers}


def run_one(real_tid: str, prefix: str) -> str:
    return run_one_sample(AGENT_NAME, real_tid, launch, OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="LocoMo smolagents driver")
    parser.add_argument("--samples", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tasks = [(s, s) for s in
             (args.samples.split(",") if args.samples else [t[0] for t in TASKS])
             if s.strip()]
    if args.limit:
        tasks = tasks[:args.limit]
    for real_tid, prefix in tasks:
        print(f"[sample] {real_tid}", flush=True)
        run_one(real_tid, prefix)


if __name__ == "__main__":
    main()
