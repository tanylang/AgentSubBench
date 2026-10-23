"""
Collect traces using the AppWorld BUILT-IN SimplifiedFunctionCallingAgent.

- Agent source: benchmarks/appworld/experiments/code/simplified/function_calling_agent.py
  (via Agent.from_dict, registered as "simplified_function_calling")
- Tasks: 13 AppWorld tasks (same as other agents)
- Tools: real AppWorld APIs via world.execute()
- Trace: hooks (sentinel shim for LLM + AppWorld.execute class patch for tools)
"""
import os, sys, time, json

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

from trace_collector import OrderedTraceCollector, make_appworld_checkpoint_provider
from adapter.tools import record_appworld_tool
from adapter.model_config import THINKING_MODE, MAIN_MODEL_NAME, MAIN_BASE_URL

TASKS = [
    # Loaded dynamically from appworld_65_tasks.json (file names use the full task_id to avoid scenario variant clashes)
]
def _load_tasks():
    tasks_json = os.environ.get("TASK_LIST_FILE") or os.path.join(ROOT, "tasks", "appworld_tasks", "appworld_65_tasks.json")
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

_OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or os.path.join(ROOT, "traces", "appworld_simplified_real_batch1")
# Anchor relative paths to ROOT: this file has chdir'd to benchmarks/appworld, otherwise traces would land in the wrong place
OUTPUT_DIR = _OUTPUT_DIR if os.path.isabs(_OUTPUT_DIR) else os.path.join(ROOT, _OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIG = {
    "client_name": "openai",
    "api_type": "chat_completions",
    "base_url": MAIN_BASE_URL,
    "api_key_env_name": "OPENAI_API_KEY",
    "name": MAIN_MODEL_NAME,
    "temperature": 0.0,
    "seed": 100,
    "cost_per_token": {"input_cache_miss": 0, "input_cache_hit": 0,
                       "input_cache_write": 0, "output": 0},
    "retry_after_n_seconds": 15,
    "use_cache": False,
    "max_retries": 100,
    # Do not set tool_choice: the API predictor's bare completion requests have
    # no tools, and vLLM rejects "tool_choice=auto with no tools set" with a 400
    # (DeepSeek tolerates it but the local vLLM validates strictly); the main
    # loop with tools defaults to auto anyway, so behavior is unchanged
    "parallel_tool_calls": True,
    "extra_body": THINKING_MODE,
}


def _build_agent():
    """Build the built-in SimplifiedFunctionCallingAgent exactly from the config template."""
    from appworld_agents.code.simplified.agent import Agent

    agent_config = {
        "type": "simplified_function_calling",
        "model_config": MODEL_CONFIG,
        "api_predictor_config": {
            "mode": "predicted",
            "model_config": MODEL_CONFIG,
            "prompt_file_path": os.path.join(
                ROOT, "benchmarks", "appworld", "experiments", "prompts", "api_predictor.txt"),
            "demo_task_ids": ["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
            "max_predicted_apis": 30,
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
    return Agent.from_dict(agent_config)


_STASH: dict = {}  # Protocol layer data stash (request tools + raw response object)


class _CollectorShim:
    """Forward the _TRACE_TRACER log_llm_call/log_llm_response interface to the collector."""

    def __init__(self, collector):
        self.collector = collector

    def log_llm_call(self, model, messages):
        return {"model": model, "messages": messages}

    def log_llm_response(self, completion, model="", input_tokens=0,
                         output_tokens=0, latency_ms=0.0, llm_call_span=None, **kw):
        msgs = llm_call_span.get("messages", []) if isinstance(llm_call_span, dict) else []
        # Protocol layer data (stashed in _STASH by the grlc patch)
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
                                     "arguments": t.function.arguments} for t in tcs]
                # Reasoning content: DeepSeek uses reasoning_content, vLLM Qwen3
                # uses reasoning, both separate from content; the openai SDK may
                # place them in model_extra / provider_specific_fields
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


def _install_hooks(collector):
    """Class-level hooks: LLM sentinel + AppWorld.execute tool tracing. Returns restore()."""
    from appworld import AppWorld
    from appworld_agents.code.simplified import language_model as lm

    shim = _CollectorShim(collector)
    orig_sentinel = lm._TRACE_TRACER
    lm._TRACE_TRACER = shim

    orig_execute = AppWorld.execute

    def traced_execute(self, code, *args, **kwargs):
        # freezegun-immune clock (perf_counter is frozen inside the AppWorld world → elapsed time would always be 0)
        from adapter.clock import monotonic_ms
        t0 = monotonic_ms()
        try:
            result = orig_execute(self, code, *args, **kwargs)
            success = True
        except Exception as e:
            result = str(e)
            success = False
        record_appworld_tool(collector, code, result, monotonic_ms() - t0, success)
        return result

    AppWorld.execute = traced_execute

    def restore():
        lm._TRACE_TRACER = orig_sentinel
        AppWorld.execute = orig_execute

    return restore


def run_one(real_tid: str, prefix: str) -> str:
    from appworld import AppWorld

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}.json")
    if os.path.exists(out_path):
        print(f"    [SKIP] already exists")
        return out_path

    collector = OrderedTraceCollector(
        agent_name="AppWorld-Simplified", model=MAIN_MODEL_NAME,
        benchmark_task_id=prefix,
    )

    t0 = time.perf_counter()
    # Protocol layer stash: wrap the caller to stash request tools and the raw
    # response for the shim to extract structured tool_calls (thinking mode goes
    # through config's extra_body, enabled by default, see MODEL_CONFIG -- no
    # longer patched here)
    from appworld_agents.code.simplified import language_model as _lm_mod
    _orig_grlc = _lm_mod.get_raw_lm_caller

    def _patched_grlc(api_type, client_name, api_key=None, base_url=None):
        f = _orig_grlc(api_type, client_name, api_key, base_url)
        # Timeout hardening: avoid the 600s default timeout wasting tasks during vLLM hiccups (LanguageModel has its own retry loop)
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
    restore = _install_hooks(collector)
    try:
        agent = _build_agent()
        # Per-span environment snapshot -- world is created inside the solve flow, resolved lazily
        collector.set_checkpoint_provider(
            make_appworld_checkpoint_provider(lambda: getattr(agent, "world", None), real_tid))
        # Official flow: solve_tasks first initializes the logger, then wraps
        # everything in AppWorld.initializer, then calls solve_task per task.
        # Replay that flow per task here.
        agent.logger.initialize(
            experiment_name=f"trace_builtin_simplified_{prefix}",
            num_tasks=1, num_processes=1, process_index=0,
        )
        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=f"trace_builtin_simplified_{prefix}",
            **agent.appworld_config,
        ):
            agent.solve_task(real_tid)
        print(f"    done in {time.perf_counter() - t0:.0f}s")
    except Exception as e:
        import traceback
        print(f"    ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        restore()

    collector.save(out_path, task_instruction="(loaded from AppWorld task)")
    print(f"    spans={collector._step} llm={collector.llm_count} "
          f"tools={collector.tool_count} tokens={collector.token_count:,}")
    return out_path


def main():
    print(f"Running {len(TASKS)} tasks with BUILT-IN SimplifiedFunctionCallingAgent\n")
    total_t0 = time.perf_counter()
    for i, (real_tid, prefix) in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] {real_tid}")
        run_one(real_tid, prefix)
    elapsed = time.perf_counter() - total_t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Traces: {OUTPUT_DIR}")

    # ── Required post-collection step: validator writes pass + event_type annotation ──
    from runners.finalize_traces import finalize_trace_dir
    finalize_trace_dir(OUTPUT_DIR, experiment_name="offline_eval_simplified")


if __name__ == "__main__":
    main()
