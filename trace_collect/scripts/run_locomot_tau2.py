"""LocoMo trace collection driver -- tau2-bench LLMAgent.

Zero agent source changes -- LLM recording reuses the tau2
`llm_utils._TRACE_TRACER` sentinel (_Tau2CollectorShim, the same one used by
the AppWorld tau2 driver); tool execution goes through the as_tool wrappers in
adapter.locomot. The driver maintains its own state loop (copied from the
AppWorld tau2 driver's loop pattern; no complete_task termination condition --
"no tool_calls means the answer" instead).

Run:
    python scripts/run_locomot_tau2.py [--samples id1,id2] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"),
          str(ROOT / "source_code" / "tau2-bench" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# litellm reads this env for the openai/ prefix; must be set before the tau2 import chain
os.environ["OPENAI_BASE_URL"] = os.environ.get(
    "MODEL_BASE_URL", "https://api.deepseek.com/v1")
os.environ.setdefault("LITELLM_REQUEST_TIMEOUT", "600")
os.environ.setdefault("LITELLM_NUM_RETRIES", "3")

from adapter.locomot import (LocomoSample, LocomoMemory,  # noqa: E402
                             build_tau2_tools, TOOL_DESCRIPTIONS)
from adapter.model_config import MAIN_MODEL_NAME, THINKING_MODE  # noqa: E402
from locomot_common import (  # noqa: E402
    load_tasks_from_env, sample_questions, build_system_prompt,
    build_question_prompt, run_one_sample,
)

AGENT_NAME = "LocoMo-Tau2"
TASKS = load_tasks_from_env()
OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or \
    os.path.join(ROOT, "traces", "locomo_tau2_batch1")


class _Tau2CollectorShim:
    """Forward the tau2 llm_utils._TRACE_TRACER interface to the collector."""

    def __init__(self, collector):
        self.collector = collector

    def log_llm_call(self, model, messages, tools=None):
        return {"model": model, "messages": messages, "tools": tools}

    def log_llm_response(self, completion, model="", input_tokens=0,
                         output_tokens=0, latency_ms=0.0, llm_call_span=None,
                         tool_calls=None, **kw):
        msgs = llm_call_span.get("messages", []) if isinstance(llm_call_span, dict) else []
        tools = llm_call_span.get("tools") if isinstance(llm_call_span, dict) else None
        self.collector.log_llm(
            model=model or MAIN_MODEL_NAME,
            messages=[{"role": str(m.get("role", "unk")),
                       "content": str(m.get("content") or "")} for m in msgs],
            output_text=str(completion),
            input_tokens=input_tokens, output_tokens=output_tokens,
            tools=tools, tool_calls=tool_calls,
            reasoning=str(kw.get("reasoning") or ""),
        )


def launch(sample: LocomoSample, env: LocomoMemory, collector) -> dict:
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import UserMessage, ToolMessage, MultiToolMessage
    from tau2.utils import llm_utils

    llm_utils._TRACE_TRACER = _Tau2CollectorShim(collector)

    tools, tool_map = build_tau2_tools(sample, env, collector)
    domain_policy = (
        f"{build_system_prompt(sample)}\n\n"
        "Answer the current question using the tools if needed. "
        "Stop when you have the answer."
    )
    agent = LLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=f"openai/{MAIN_MODEL_NAME}",
        llm_args={"temperature": 0, "max_tokens": 8192,
                  "seed": 100, "extra_body": THINKING_MODE},
    )

    qas = sample_questions(sample)
    answers = []
    state = agent.get_init_state()
    for i, qa in enumerate(qas):
        msg = UserMessage(role="user",
                          content=build_question_prompt(i, len(qas), qa.question))
        pred = ""
        for _ in range(50):
            assistant_msg, state = agent.generate_next_message(msg, state)
            tcs = assistant_msg.tool_calls
            if not tcs:
                pred = str(getattr(assistant_msg, "content", "") or "")
                break
            tool_msgs = []
            for tc in tcs:
                tool = tool_map.get(tc.name)
                try:
                    result = str(tool(**tc.arguments)) if tool is not None \
                        else f"Tool {tc.name} not found"
                except Exception as e:
                    result = f"Error: {e}"
                tool_msgs.append(ToolMessage(
                    role="tool", content=result, id=tc.id))
            msg = MultiToolMessage(role="tool", tool_messages=tool_msgs)
        answers.append({
            "idx": i, "question": qa.question, "pred": pred,
            "category": qa.category, "evidence": qa.evidence,
        })
    return {"answers": answers}


def run_one(real_tid: str, prefix: str) -> str:
    return run_one_sample(AGENT_NAME, real_tid, launch, OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="LocoMo tau2 driver")
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
