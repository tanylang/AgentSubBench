"""LocoMo trace collection driver -- AutoGen AssistantAgent.

Zero agent source changes -- LLM recording reuses the `_trace_tracer` source
hook in vendored autogen-ext (AutoGenTracerShim, the same one used by the
AppWorld autogen driver); tool execution goes through the async tool functions
in adapter.locomot (execute_tool records everything uniformly).

Run:
    python scripts/run_locomot_autogen.py [--samples id1,id2] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from adapter.builtin_hooks import AutoGenTracerShim  # noqa: E402
from adapter.locomot import (LocomoSample, LocomoMemory,  # noqa: E402
                             build_autogen_tool_funcs)
from adapter.model_config import (MAIN_MODEL_NAME, MAIN_BASE_URL,  # noqa: E402
                                  THINKING_MODE)
from locomot_common import (  # noqa: E402
    load_tasks_from_env, sample_questions, build_system_prompt,
    build_question_prompt, run_one_sample,
)

AGENT_NAME = "LocoMo-AutoGen"
TASKS = load_tasks_from_env()
OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or \
    os.path.join(ROOT, "traces", "locomo_autogen_batch1")


def launch(sample: LocomoSample, env: LocomoMemory, collector) -> dict:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelInfo, ModelFamily
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat

    # Main-loop client: same parameters as the AppWorld autogen driver (incl. timeout/limits/seed/temperature)
    client = OpenAIChatCompletionClient(
        model=MAIN_MODEL_NAME,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=MAIN_BASE_URL,
        model_info=ModelInfo(vision=False, function_calling=True,
                             json_output=True, structured_output=False,
                             family=ModelFamily.UNKNOWN),
        extra_body=THINKING_MODE,
        timeout=600.0, max_retries=3, max_tokens=8192,
        seed=100, temperature=0.0,
    )
    client._trace_tracer = AutoGenTracerShim(collector, MAIN_MODEL_NAME)

    tools = build_autogen_tool_funcs(sample, env, collector)
    agent = AssistantAgent(
        name="assistant",
        model_client=client,
        tools=tools,
        system_message=build_system_prompt(sample),
        # Default is reflect_on_tool_use=False: tool results are returned
        # directly without reflection; when enabled the agent keeps reasoning
        # over tool results until it produces a text answer
        reflect_on_tool_use=True,
        max_tool_iterations=10,
    )

    qas = sample_questions(sample)
    answers = []

    async def _run() -> None:
        nonlocal answers
        for i, qa in enumerate(qas):
            task = build_question_prompt(i, len(qas), qa.question)
            result = await asyncio.wait_for(
                agent.run(task=task, cancellation_token=CancellationToken()),
                timeout=600)
            pred = ""
            for m in reversed(result.messages):
                if isinstance(m, TextMessage) and m.content and \
                        m.source == "assistant":
                    pred = m.content
                    break
            answers.append({
                "idx": i, "question": qa.question, "pred": pred,
                "category": qa.category, "evidence": qa.evidence,
            })

    asyncio.run(_run())
    return {"answers": answers}


def run_one(real_tid: str, prefix: str) -> str:
    return run_one_sample(AGENT_NAME, real_tid, launch, OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="LocoMo autogen driver")
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
