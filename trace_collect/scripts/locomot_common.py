"""LocoMo common driver layer -- collection infrastructure shared by the
per-agent drivers.

Follows the swebench_common.py pattern: skip-if-exists resume, collector/env/
checkpoint assembly, a launch callback, and save + locomo metadata enrichment.
LocoMo is a read-only benchmark (answer questions by searching the
conversation database), and the LLM recording mechanism varies by framework
(each driver handles its own; this layer does not register litellm global
callbacks).

launch contract:
    launch(sample: LocomoSample, env: LocomoEnv, collector) -> dict | None
    Returns {"answers": [...], "read_ids": [...], ...}; exceptions are caught by this layer.

Question sampling: with a fixed seed, draw LOC_QA_LIMIT (default 50)
questions from sample.qas so that traces are reproducible for the same
prompt/model.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parent.parent
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from adapter.locomot import (  # noqa: E402
    LocomoSample, LocomoMemory, load_locomo_samples,
    make_locomo_checkpoint_provider, execute_tool,
    OAI_TOOL_SCHEMAS, TOOL_DESCRIPTIONS,
)
from adapter.model_config import MAIN_MODEL_NAME  # noqa: E402
from trace_collector import OrderedTraceCollector  # noqa: E402

# API key injection (config.json → environment variables; same convention as swebench_common)
_cfg_path = ROOT / "config.json"
try:
    _key = json.loads(_cfg_path.read_text(encoding="utf-8")).get(
        "deepseek_api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
except Exception:
    _key = os.environ.get("DEEPSEEK_API_KEY", "")
if _key:
    os.environ["OPENAI_API_KEY"] = _key
    os.environ["DEEPSEEK_API_KEY"] = _key

OUTPUT_ROOT = ROOT / "traces"
# Question sampling (fixed seed for reproducibility; LOC_QA_LIMIT=0 means all questions)
QA_SEED = 100
QA_LIMIT = int(os.environ.get("LOC_QA_LIMIT", "50") or "50")
# Max tool calls per question (guards against infinite retrieval loops)
MAX_TOOL_STEPS = int(os.environ.get("LOC_MAX_TOOL_STEPS", "10") or "10")
# Refusal phrases (official cat5 wording; the system prompt teaches the model to use them)
REFUSAL_PHRASES = ["No information available", "Not mentioned"]


def make_collector(agent_name: str, sample_id: str,
                   model: str = MAIN_MODEL_NAME) -> OrderedTraceCollector:
    return OrderedTraceCollector(agent_name=agent_name, model=model,
                                 benchmark_task_id=sample_id,
                                 benchmark="Locomo")


def load_tasks_from_env() -> list[tuple[str, str]]:
    """Load TASKS (run_task_cli contract): filtered by LOC_SAMPLE_IDS, all 10 by default."""
    ids = [s for s in
           (os.environ.get("LOC_SAMPLE_IDS", "") or "").split(",") if s.strip()]
    samples = load_locomo_samples()
    return [(s.sample_id, s.sample_id) for s in samples
            if not ids or s.sample_id in set(ids)]


def sample_questions(sample: LocomoSample, limit: int = QA_LIMIT):
    """Sample questions with a fixed seed (limit<=0 or >= total means all, preserving original order)."""
    qas = sample.qas
    if limit <= 0 or limit >= len(qas):
        return list(qas)
    idxs = sorted(random.Random(QA_SEED).sample(range(len(qas)), limit))
    return [qas[i] for i in idxs]


def build_system_prompt(sample: LocomoSample) -> str:
    """English system prompt: role + conversation database description + tool usage + answering rules."""
    return (
        "You are a helpful assistant answering questions about a long "
        "multi-session conversation between two people.\n\n"
        f"The conversation involves {sample.speaker_a!r} and {sample.speaker_b!r} "
        f"across {sample.num_sessions} sessions ({sample.num_turns} messages) "
        "spanning several months. Each message has a unique id (like 'D3:5') "
        "and belongs to a session with a date. Images are described by their "
        "captions.\n\n"
        "You have read-only tools to search the conversation database:\n"
        f"{TOOL_DESCRIPTIONS}\n\n"
        "Guidelines:\n"
        "1. Use the tools to retrieve relevant messages before answering.\n"
        "2. Answer concisely, using the exact information from the conversation.\n"
        "3. If the conversation does not contain the information, answer exactly "
        f"'{REFUSAL_PHRASES[0]}' or '{REFUSAL_PHRASES[1]}'.\n"
    )


def build_question_prompt(i: int, n: int, question: str) -> str:
    return f"Question {i + 1}/{n}: {question}"


def run_question_loop(env: LocomoMemory, collector, question: str,
                      step_fn: Callable, max_tool_steps: int = MAX_TOOL_STEPS
                      ) -> tuple[str, int]:
    """Generic question-answering sub-loop (for frameworks that keep their own
    message list: simplified/legacy).

    step_fn(messages) -> (output_text, tool_calls | None)
      tool_calls = [{"id", "name", "arguments"(dict)}] (OpenAI format).
    With tool_calls, run execute_tool for each and append role=tool messages;
    without tool_calls, output_text is the answer. On exceeding the limit,
    take the last output.
    """
    sample = env.sample
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    answer = ""

    def _norm_tc(tc: dict) -> dict:
        """Compatible with the raw OpenAI format (function.arguments is a JSON string)."""
        if "function" in tc:
            fn = tc["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return {"id": tc.get("id", ""), "name": fn.get("name", ""),
                    "arguments": args or {}}
        return tc

    for _ in range(max_tool_steps):
        out, tool_calls = step_fn(messages)
        if not tool_calls:
            answer = out
            break
        # Backfilled messages keep the raw format (needed by the next generate); normalize when executing
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": tool_calls})
        for tc in tool_calls:
            ntc = _norm_tc(tc)
            result = execute_tool(ntc["name"], ntc.get("arguments") or {},
                                  sample, env, collector)
            messages.append({"role": "tool",
                             "tool_call_id": ntc.get("id", "") or tc.get("id", ""),
                             "name": ntc["name"],
                             "content": result})
        answer = out
    return answer, _


def run_one_sample(agent_name: str, sample_id: str,
                   launch: Callable[[LocomoSample, LocomoMemory,
                                     OrderedTraceCollector], dict | None],
                   output_dir: str) -> str:
    """Single-sample collection skeleton: skip → assemble → launch → save → locomo enrich.

    Returns out_path. launch exceptions are caught (the trace is still saved,
    with an error in extra).
    """
    sample = load_locomo_samples(sample_ids=[sample_id])[0]
    out_path = os.path.join(output_dir, f"{sample_id}.json")
    if os.path.exists(out_path):
        print(f"    [SKIP] already exists")
        return out_path

    collector = make_collector(agent_name, sample_id)
    env = LocomoMemory(sample)
    collector.set_checkpoint_provider(
        make_locomo_checkpoint_provider(env, sample_id))

    extra: dict[str, Any] = {}
    try:
        extra = launch(sample, env, collector) or {}
    except Exception as e:
        import traceback
        traceback.print_exc()
        extra = {"error": f"{type(e).__name__}: {e}"}

    # Sampled questions (same convention as launch; used for evaluation alignment)
    qas = sample_questions(sample)
    task_instruction = (
        f"LocoMo {sample_id}: answer questions about a "
        f"{sample.num_sessions}-session conversation between "
        f"{sample.speaker_a} and {sample.speaker_b}"
    )
    answers = extra.get("answers") or []
    data_path = collector.save(out_path, task_instruction=task_instruction,
                               outcome="unknown")
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    data["task"] = task_instruction
    data["locomo"] = {
        "sample_id": sample.sample_id,
        "speaker_a": sample.speaker_a,
        "speaker_b": sample.speaker_b,
        "num_sessions": sample.num_sessions,
        "num_messages": sample.num_turns,
        "num_questions": len(qas),
        "questions_answered": len(answers),
        "answers": answers,
        "read_ids": sorted(env.read_dia_ids),
        "max_tool_steps": MAX_TOOL_STEPS,
    }
    if extra and "answers" in extra:
        del extra["answers"]
    if extra:
        data["agent_extra"] = extra
    tmp = data_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, data_path)
    print(f"    done: {len(answers)}/{len(qas)} answered -> {data_path}")
    return out_path
