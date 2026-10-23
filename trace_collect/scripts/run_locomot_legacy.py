"""LocoMo trace collection driver -- legacy FunctionCallingAgent's OpenAILanguageModel.

Zero agent source changes -- LLM recording reuses the legacy
`openai_language_model` `_TRACE_TRACER` sentinel + `model_to_completion_function`
stash patch (the same one used by the AppWorld legacy driver, including
runtime model registry registration and openai global timeout hardening).
Tool execution goes through adapter.locomot's unified entry point
(execute_tool); the driver keeps its own message list loop (same structure as
the simplified driver).

Run:
    python scripts/run_locomot_legacy.py [--samples id1,id2] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"),
          str(ROOT / "benchmarks" / "appworld" / "src"),
          str(ROOT / "benchmarks" / "appworld" / "experiments" / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# openai SDK global base_url (legacy openai_lm reads this env)
os.environ["OPENAI_BASE_URL"] = os.environ.get(
    "MODEL_BASE_URL", "https://api.deepseek.com/v1")

from adapter.locomot import LocomoSample, LocomoMemory, OAI_TOOL_SCHEMAS  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, THINKING_MODE  # noqa: E402
from locomot_common import (  # noqa: E402
    load_tasks_from_env, sample_questions, build_system_prompt,
    build_question_prompt, run_question_loop, run_one_sample,
    MAX_TOOL_STEPS,
)

AGENT_NAME = "LocoMo-Legacy"
TASKS = load_tasks_from_env()
OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or \
    os.path.join(ROOT, "traces", "locomo_legacy_batch1")

_STASH: dict = {}


class _CollectorShim:
    """Forward the legacy openai_lm _TRACE_TRACER interface to the collector."""

    def __init__(self, collector):
        self.collector = collector

    def log_llm_call(self, model, messages):
        return {"model": model, "messages": messages}

    def log_llm_response(self, completion, model="", input_tokens=0,
                         output_tokens=0, latency_ms=0.0, llm_call_span=None,
                         **kw):
        msgs = llm_call_span.get("messages", []) if isinstance(llm_call_span, dict) else []
        tools = _STASH.pop("tools", None)
        resp = _STASH.pop("response", None)
        struct_calls = None
        if resp is not None:
            try:
                tcs = getattr(resp.choices[0].message, "tool_calls", None)
                if tcs:
                    struct_calls = [{"id": t.id, "name": t.function.name,
                                     "arguments": t.function.arguments}
                                    for t in tcs]
            except Exception:
                pass
        self.collector.log_llm(
            model=model or MAIN_MODEL_NAME,
            messages=[{"role": str(m.get("role", "unk")),
                       "content": str(m.get("content") or "")} for m in msgs],
            output_text=str(completion),
            input_tokens=input_tokens, output_tokens=output_tokens,
            tools=tools, tool_calls=struct_calls,
        )


def launch(sample: LocomoSample, env: LocomoMemory, collector) -> dict:
    import openai
    from appworld_agents.code.legacy.plain.language_models import \
        openai_language_model as lm_mod
    from appworld_agents.code.legacy.plain.language_models.openai_language_model import (
        ModelInformation, ModelsInformation)

    # Timeout hardening (openai SDK global client)
    openai.timeout = 600.0
    openai.max_retries = 3
    # Runtime model registry registration (unknown models crash with "No item found")
    _known = {m.name for m in ModelsInformation._list}
    if MAIN_MODEL_NAME not in _known:
        ModelsInformation._list.append(ModelInformation(
            name=MAIN_MODEL_NAME, provider="openai", completion_type="chat",
            token_limits={"input": 256000, "output": 8192}))

    lm_mod._TRACE_TRACER = _CollectorShim(collector)
    orig_mtf = lm_mod.model_to_completion_function

    def _patched_mtf(model):
        f = orig_mtf(model)

        def _wrapped(*a, **kw):
            _STASH["tools"] = kw.get("tools")
            result = f(*a, **kw)
            _STASH["response"] = result
            return result

        return _wrapped

    lm_mod.model_to_completion_function = _patched_mtf

    lm = lm_mod.OpenAILanguageModel(
        model=MAIN_MODEL_NAME, max_tokens=8192, temperature=0.0,
        seed=100, extra_body=THINKING_MODE, retry_after_n_seconds=15,
    )

    qas = sample_questions(sample)
    system_prompt = build_system_prompt(sample)
    base_messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def step_fn(messages):
        result = lm.generate(base_messages + messages,
                             tools=OAI_TOOL_SCHEMAS, tool_choice="auto")
        if isinstance(result, tuple):
            out = str(result[0] or "")
            second = result[1] or {}
            # The second element is a message dict (tool_calls live inside it, raw OpenAI format)
            tcs = second.get("tool_calls") if isinstance(second, dict) else second
        else:
            out, tcs = str(result or ""), None
        return out, tcs

    answers = []
    for i, qa in enumerate(qas):
        answer, _steps = run_question_loop(
            env, collector, build_question_prompt(i, len(qas), qa.question),
            step_fn, MAX_TOOL_STEPS)
        answers.append({
            "idx": i, "question": qa.question, "pred": answer,
            "category": qa.category, "evidence": qa.evidence,
        })
    lm_mod.model_to_completion_function = orig_mtf
    return {"answers": answers}


def run_one(real_tid: str, prefix: str) -> str:
    return run_one_sample(AGENT_NAME, real_tid, launch, OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="LocoMo legacy driver")
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
