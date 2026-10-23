"""AppWorld official AutoGen agent trace collection driver.

Uses the upstream official agent implementation
(benchmarks/appworld/experiments/code/autogen/, registered as autogen_agent):
a real AutoGen AssistantAgent + the official config flow (Agent.from_dict →
solve_task, same structure as the simplified/legacy drivers).

Trace recording: the official agent source already has hook call sites
implanted (llm_hook/tool_hook); registering the collector makes them active.
The predictor's LanguageModel goes through the _TRACE_TRACER sentinel (same
pattern as the simplified driver). Zero agent logic changes.

Uniform convention: main loop MAIN_MODEL_NAME / predictor PREDICTOR_MODEL_NAME /
temperature=0 / seed=100 / THINKING_MODE.
"""
import json
import os
import sys
import time
from pathlib import Path

_cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
try:
    with open(_cfg_path, encoding="utf-8") as _f:
        _api_key = json.load(_f).get("deepseek_api_key", "") or os.environ.get("DEEPSEEK_API_KEY", "")
except Exception:
    _api_key = os.environ.get("DEEPSEEK_API_KEY", "")
os.environ["DEEPSEEK_API_KEY"] = _api_key
os.environ["OPENAI_API_KEY"] = _api_key

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld", "experiments", "code"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(os.path.join(ROOT, "benchmarks", "appworld"))

# Windows: stdout/stderr default to GBK after redirection; the emoji in rich progress bars would crash the process
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from trace_collector import OrderedTraceCollector, make_appworld_checkpoint_provider
from adapter.model_config import (THINKING_MODE, MAIN_MODEL_NAME, MAIN_BASE_URL,
                                  PREDICTOR_MODEL_NAME, PREDICTOR_BASE_URL)

TASKS = []


def _load_tasks():
    tasks_json = os.environ.get("TASK_LIST_FILE") or os.path.join(
        ROOT, "tasks", "appworld_tasks", "appworld_65_tasks.json")
    if os.path.exists(tasks_json):
        with open(tasks_json, encoding="utf-8") as f:
            task_ids = json.load(f)
        return [(tid, tid) for tid in task_ids]
    return [
        ("0d8a4ee_1", "0d8a4ee"), ("22cc237_1", "22cc237"), ("23cf851_1", "23cf851"),
        ("27e1026_1", "27e1026"), ("29caf6f_1", "29caf6f"), ("34d9492_1", "34d9492"),
        ("383cbac_1", "383cbac"), ("3c13f5a_1", "3c13f5a"), ("6104387_1", "6104387"),
        ("6c2c621_1", "6c2c621"), ("771d8fc_1", "771d8fc"), ("b0a8eae_1", "b0a8eae"),
        ("cf6abd2_1", "cf6abd2"),
    ]


TASKS = _load_tasks()

_OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or os.path.join(
    ROOT, "traces", "appworld_autogen_batch1")
# Anchor relative paths to ROOT (cwd has already been switched to benchmarks/appworld)
OUTPUT_DIR = _OUTPUT_DIR if os.path.isabs(_OUTPUT_DIR) else os.path.join(ROOT, _OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Protocol layer stash for the predictor LanguageModel (same pattern as the simplified driver)
_STASH: dict = {}


class _CollectorShim:
    """Predictor LanguageModel _TRACE_TRACER → collector."""

    def __init__(self, collector):
        self.collector = collector

    def log_llm_call(self, model, messages):
        return {"model": model, "messages": messages}

    def log_llm_response(self, completion, model="", input_tokens=0,
                         output_tokens=0, latency_ms=0.0, llm_call_span=None, **kw):
        msgs = llm_call_span.get("messages", []) if isinstance(llm_call_span, dict) else []
        tools = _STASH.pop("tools", None)
        resp = _STASH.pop("response", None)
        struct_calls = None
        reasoning = ""
        if resp is not None:
            try:
                tcs = getattr(resp.choices[0].message, "tool_calls", None)
                if tcs:
                    struct_calls = [{"id": t.id, "name": t.function.name,
                                     "arguments": t.function.arguments} for t in tcs]
                msg = resp.choices[0].message
                for attr in ("reasoning_content", "reasoning"):
                    v = getattr(msg, attr, None)
                    if isinstance(v, str) and v.strip():
                        reasoning = v
                        break
                if not reasoning:
                    extra = getattr(msg, "model_extra", None) or {}
                    reasoning = str(extra.get("reasoning_content") or
                                    extra.get("reasoning") or "")
            except Exception:
                pass
        self.collector.log_llm(
            model=model or PREDICTOR_MODEL_NAME,
            messages=[{"role": str(m.get("role", "unk")),
                       "content": str(m.get("content") or "")} for m in msgs],
            output_text=str(completion),
            input_tokens=input_tokens, output_tokens=output_tokens,
            tools=tools, tool_calls=struct_calls,
            reasoning=reasoning,
        )


def _install_predictor_hooks(collector):
    """Predictor LanguageModel: sentinel + grlc stash (timeout hardening, same as simplified)."""
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


def run_one(real_tid: str, prefix: str) -> str:
    from appworld import AppWorld
    from appworld_agents.code.autogen.agent import Agent

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}.json")
    if os.path.exists(out_path):
        print(f"    [SKIP] already exists")
        return out_path

    collector = OrderedTraceCollector(
        agent_name="AutoGen-AssistantAgent", model=MAIN_MODEL_NAME,
        benchmark_task_id=prefix,
    )
    # Hook call sites implanted in the official agent source (llm_hook/tool_hook) → collector
    from adapter.hooks import llm_hook, tool_hook
    llm_hook.register(collector)
    tool_hook.register(collector)
    # Predictor LanguageModel recording (same as the simplified driver)
    _install_predictor_hooks(collector)

    t0 = time.perf_counter()
    try:
        model_config = {
            "client_name": "openai",
            "api_type": "chat_completions",
            "base_url": MAIN_BASE_URL,
            "api_key_env_name": "OPENAI_API_KEY",
            "name": MAIN_MODEL_NAME,
            "temperature": 0.0,
            "seed": 100,
            "extra_body": THINKING_MODE,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "cost_per_token": {"input_cache_miss": 0, "input_cache_hit": 0,
                               "input_cache_write": 0, "output": 0},
            "retry_after_n_seconds": 15,
            "use_cache": False,
            "max_retries": 100,
        }
        predictor_config = dict(model_config)
        predictor_config.update({
            "name": PREDICTOR_MODEL_NAME, "base_url": PREDICTOR_BASE_URL,
            "extra_body": THINKING_MODE,
        })
        agent_config = {
            "type": "autogen_agent",
            "model_config": model_config,
            "api_predictor_config": {
                "mode": "predicted",
                "model_config": predictor_config,
                "prompt_file_path": os.path.join(
                    ROOT, "benchmarks", "appworld", "experiments", "prompts",
                    "api_predictor.txt"),
                "demo_task_ids": ["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
                "max_predicted_apis": 20,
            },
            "appworld_config": {
                "random_seed": 100,
                "raise_on_extra_parameters": True,
                "include_direct_functions": True,
                "direct_function_separator": "__",
            },
            "logger_config": {"color": False, "verbose": False},
            "usage_tracker_config": {
                "max_cost_overall": 1000,
                "max_cost_per_task": 10,
                "max_output_tokens_per_task": 100000,
            },
            "prompt_file_path": os.path.join(
                ROOT, "benchmarks", "appworld", "experiments", "prompts",
                "function_calling_agent", "instructions.txt"),
            "demo_messages_file_path": os.path.join(
                ROOT, "benchmarks", "appworld", "experiments", "prompts",
                "function_calling_agent", "demos.json"),
            "max_steps": 50,
            "log_lm_calls": False,
            "skip_if_finished": False,
        }
        agent = Agent.from_dict(agent_config)
        # Upfront step in the official flow: logger initialization (num_tasks=1, single-task driver mode)
        agent.logger.initialize(
            experiment_name=f"trace_autogen_{prefix}", num_tasks=1,
            num_processes=1, process_index=0)
        # world is created inside solve_task; lazy getter (same as the simplified driver)
        collector.set_checkpoint_provider(
            make_appworld_checkpoint_provider(lambda: getattr(agent, "world", None), real_tid))
        agent.solve_task(real_tid)
        print(f"    done in {time.perf_counter() - t0:.0f}s")
    except Exception as e:
        import traceback
        print(f"    ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        llm_hook.unregister()
        tool_hook.unregister()
        collector.save(out_path, task_instruction=real_tid, outcome="unknown")
    return out_path


def main() -> None:
    for real_tid, prefix in TASKS:
        print(f"[task] {real_tid}", flush=True)
        run_one(real_tid, prefix)


if __name__ == "__main__":
    main()
