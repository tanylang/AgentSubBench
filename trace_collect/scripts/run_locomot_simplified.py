"""LocoMo trace collection driver -- Simplified FunctionCallingAgent's LanguageModel.

Zero agent source changes -- the LLM layer reuses appworld_agents'
`code.simplified.language_model` (its `_TRACE_TRACER` sentinel is already
hooked in the source), and the protocol layer stash reuses the grlc patch
pattern from the AppWorld simplified driver (with timeout hardening). Tool
execution goes through adapter.locomot.execute_tool (LocoMo read-only
retrieval tools).

Run:
    python scripts/run_locomot_simplified.py [--samples id1,id2] [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"),
          str(ROOT / "benchmarks" / "appworld" / "src"),
          str(ROOT / "benchmarks" / "appworld" / "experiments" / "code")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Windows: default GBK after redirection (same convention as the other drivers)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from adapter.locomot import LocomoSample, LocomoMemory, OAI_TOOL_SCHEMAS  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL, THINKING_MODE  # noqa: E402
from locomot_common import (  # noqa: E402
    make_collector, load_tasks_from_env, sample_questions,
    build_system_prompt, build_question_prompt, run_question_loop,
    run_one_sample, MAX_TOOL_STEPS,
)

AGENT_NAME = "LocoMo-Simplified"
TASKS = load_tasks_from_env()
OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or \
    os.path.join(ROOT, "traces", "locomo_simplified_batch1")

_STASH: dict = {}  # Protocol layer data stash (request tools + raw response)


class _CollectorShim:
    """Forward the simplified LanguageModel _TRACE_TRACER interface to the collector."""

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
        reasoning = ""
        if resp is not None:
            try:
                msg = resp.choices[0].message
                tcs = getattr(msg, "tool_calls", None)
                if tcs:
                    struct_calls = [{"id": t.id, "name": t.function.name,
                                     "arguments": t.function.arguments}
                                    for t in tcs]
                # Reasoning content (same convention as the appworld simplified driver: reasoning_content/reasoning dual keys)
                _r = (getattr(msg, "reasoning_content", None)
                      or getattr(msg, "reasoning", None))
                if not _r:
                    for _ea in ("model_extra", "provider_specific_fields"):
                        _extra = getattr(msg, _ea, None)
                        if isinstance(_extra, dict):
                            _r = _extra.get("reasoning_content") or _extra.get("reasoning")
                            if _r:
                                break
                reasoning = str(_r) if _r else ""
            except Exception:
                pass
        self.collector.log_llm(
            model=model or MAIN_MODEL_NAME,
            messages=[{"role": str(m.get("role", "unk")),
                       "content": str(m.get("content") or "")} for m in msgs],
            output_text=str(completion),
            input_tokens=input_tokens, output_tokens=output_tokens,
            tools=tools, tool_calls=struct_calls,
            reasoning=reasoning,
            duration_ms=latency_ms,
        )


def _install_hooks(collector) -> None:
    """Class-level hook: LanguageModel sentinel + grlc protocol layer stash (timeout hardening)."""
    from appworld_agents.code.simplified import language_model as _lm_mod

    _lm_mod._TRACE_TRACER = _CollectorShim(collector)

    _orig_grlc = _lm_mod.get_raw_lm_caller

    def _patched_grlc(api_type, client_name, api_key=None, base_url=None):
        f = _orig_grlc(api_type, client_name, api_key, base_url)
        _c = getattr(f, "__self__", None)
        if _c is not None and hasattr(_c, "timeout"):
            _c.timeout = 600.0
            _c.max_retries = 3

        def _wrapped(**kw):
            _STASH["tools"] = kw.get("tools")
            result = f(**kw)
            _STASH["response"] = result
            return result

        return _wrapped

    _lm_mod.get_raw_lm_caller = _patched_grlc


def launch(sample: LocomoSample, env: LocomoMemory, collector) -> dict:
    """Per-question loop: LanguageModel + self-maintained message list + execute_tool backfill."""
    from appworld_agents.code.simplified.language_model import LanguageModel

    _install_hooks(collector)
    lm = LanguageModel(
        name=MAIN_MODEL_NAME, client_name="openai",
        api_type="chat_completions", base_url=MAIN_BASE_URL,
        api_key_env_name="OPENAI_API_KEY",
        temperature=0.0, seed=100, max_tokens=8192,
        tool_choice="auto", extra_body=THINKING_MODE, use_cache=False,
        max_retries=100, retry_after_n_seconds=15,
        cost_per_token={"input_cache_miss": 0, "input_cache_hit": 0,
                        "input_cache_write": 0, "output": 0},
    )

    qas = sample_questions(sample)
    system_prompt = build_system_prompt(sample)
    base_messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def step_fn(messages):
        # generate returns a message-level dict (role/content/tool_calls/standardized_usage)
        resp = lm.generate(base_messages + messages, tools=OAI_TOOL_SCHEMAS)
        out = resp.get("content") or ""
        tcs = resp.get("tool_calls") or []
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
    return {"answers": answers}


def run_one(real_tid: str, prefix: str) -> str:
    return run_one_sample(AGENT_NAME, real_tid, launch, OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="LocoMo simplified driver")
    parser.add_argument("--samples", default="", help="comma-separated sample_id")
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
