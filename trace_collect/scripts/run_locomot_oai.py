"""LocoMo trace collection driver -- OpenAI Agents SDK (no MCP).

Zero agent source changes -- LLM recording reuses the AsyncOpenAI client patch
(inject extra_body/seed + stash request tools/response tool_calls + log_llm,
the same _patched_create as the AppWorld oai driver, no MCP); tools are local
function_tool wrappers (wrapped by adapter.locomot, executed natively by the
SDK Runner).

Run:
    python scripts/run_locomot_oai.py [--samples id1,id2] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
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

from adapter.locomot import (LocomoSample, LocomoMemory,  # noqa: E402
                             build_oai_tools)
from adapter.model_config import (MAIN_MODEL_NAME, MAIN_BASE_URL,  # noqa: E402
                                  THINKING_MODE)
from locomot_common import (  # noqa: E402
    load_tasks_from_env, sample_questions, build_system_prompt,
    build_question_prompt, run_one_sample,
)

AGENT_NAME = "LocoMo-OpenAI-Agents"
TASKS = load_tasks_from_env()
OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or \
    os.path.join(ROOT, "traces", "locomo_oai_batch1")

_OAI_STASH: dict = {}


def launch(sample: LocomoSample, env: LocomoMemory, collector) -> dict:
    from agents import (Agent as SDKAgent, set_tracing_disabled,
                        set_default_openai_client)
    from agents.model_settings import ModelSettings
    from agents.run import RunConfig, Runner
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    client = AsyncOpenAI(base_url=MAIN_BASE_URL,
                         api_key=os.environ.get("OPENAI_API_KEY", ""),
                         timeout=600.0, max_retries=3)

    _orig_create = client.chat.completions.create

    async def _patched_create(**kwargs):
        if THINKING_MODE:
            kwargs["extra_body"] = dict(kwargs.get("extra_body") or {},
                                        **THINKING_MODE)
        kwargs.setdefault("seed", 100)
        kwargs.setdefault("temperature", 0.0)
        _OAI_STASH["tools"] = kwargs.get("tools")
        result = await _orig_create(**kwargs)
        try:
            msg = result.choices[0].message
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
                output_text=msg.content or "",
                input_tokens=getattr(result.usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(result.usage, "completion_tokens", 0) or 0,
                tools=kwargs.get("tools"), tool_calls=struct,
            )
        except Exception:
            pass
        return result

    client.chat.completions.create = _patched_create
    set_default_openai_client(client)

    tools = build_oai_tools(sample, env, collector)
    sdk_agent = SDKAgent(
        name="assistant",
        instructions=build_system_prompt(sample),
        tools=tools,
        model=MAIN_MODEL_NAME,
        model_settings=ModelSettings(temperature=0.0, max_output_tokens=8192),
    )

    qas = sample_questions(sample)
    answers = []

    async def _run() -> None:
        nonlocal answers
        for i, qa in enumerate(qas):
            try:
                result = await Runner.run(
                    sdk_agent,
                    input=build_question_prompt(i, len(qas), qa.question),
                    max_turns=10,
                    run_config=RunConfig(tracing_disabled=True),
                )
                pred = (result.final_output or "").strip()
            except Exception as e:
                pred = f"ERROR: {e}"
            answers.append({
                "idx": i, "question": qa.question, "pred": pred,
                "category": qa.category, "evidence": qa.evidence,
            })

    asyncio.run(_run())
    return {"answers": answers}


def run_one(real_tid: str, prefix: str) -> str:
    return run_one_sample(AGENT_NAME, real_tid, launch, OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="LocoMo oai driver")
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
