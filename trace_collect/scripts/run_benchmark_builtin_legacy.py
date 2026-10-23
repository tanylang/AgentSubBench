"""
Collect traces using the AppWorld BUILT-IN legacy FunctionCallingAgent.

- Agent source: benchmarks/appworld/experiments/code/legacy/plain/agents/function_calling_agent.py
  (via Agent.from_dict, registered as "function_calling_agent")
- Tasks: 13 AppWorld tasks
- Tools: real AppWorld APIs via self.world.execute() (captured via a class-level hook)
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
# openai SDK global base_url (the legacy openai_lm models read this env); the
# import order puts this before adapter.model_config, reading env directly
# (defaults to DeepSeek, overridable via MODEL_BASE_URL)
os.environ["OPENAI_BASE_URL"] = os.environ.get("MODEL_BASE_URL", "https://api.deepseek.com/v1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld", "experiments", "code"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(os.path.join(ROOT, "benchmarks", "appworld"))

from trace_collector import OrderedTraceCollector, make_appworld_checkpoint_provider
from adapter.tools import record_appworld_tool
from adapter.model_config import THINKING_MODE, MAIN_MODEL_NAME, PREDICTOR_MODEL_NAME

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

DEMO_TASK_IDS = ["82e2fac_1", "29caf6f_1", "d0b1f43_1"]
_OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or os.path.join(ROOT, "traces", "appworld_legacy_batch1")
# Anchor relative paths to ROOT: this file has chdir'd to benchmarks/appworld, otherwise traces would land in the wrong place
OUTPUT_DIR = _OUTPUT_DIR if os.path.isabs(_OUTPUT_DIR) else os.path.join(ROOT, _OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


_STASH: dict = {}  # Protocol layer data stash between the completion patch and the shim


class _CollectorShim:
    """Forward the _TRACE_TRACER log_llm_call/log_llm_response interface to the collector."""

    def __init__(self, collector):
        self.collector = collector

    def log_llm_call(self, model, messages):
        return {"model": model, "messages": messages}

    def log_llm_response(self, completion, model="", input_tokens=0,
                         output_tokens=0, latency_ms=0.0, llm_call_span=None, **kw):
        msgs = llm_call_span.get("messages", []) if isinstance(llm_call_span, dict) else []
        # Protocol layer data: the completion patch stashed the request tools and raw response in _STASH
        tools = _STASH.pop("tools", None)
        resp = _STASH.pop("response", None)
        struct_calls = None
        if resp is not None:
            try:
                msg = resp.choices[0].message
                tcs = getattr(msg, "tool_calls", None)
                if tcs:
                    struct_calls = [{"id": t.id, "name": t.function.name,
                                     "arguments": t.function.arguments} for t in tcs]
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


def _install_hooks(collector):
    """Class-level hooks: LLM sentinel (legacy openai_language_model) + AppWorld.execute tool tracing."""
    from appworld import AppWorld
    from appworld_agents.code.legacy.plain.language_models import openai_language_model as lm

    shim = _CollectorShim(collector)
    orig_sentinel = lm._TRACE_TRACER
    lm._TRACE_TRACER = shim

    # Wrap the completion function to only stash protocol layer data (request
    # tools + raw response) for the shim to extract structured tool_calls.
    # Thinking mode goes through config's extra_body (enabled by default).
    _STASH.clear()
    orig_mtf = lm.model_to_completion_function

    def _patched_mtf(model):
        f = orig_mtf(model)
        if "deepseek-v4" not in model:
            return f

        def _wrapped(*a, **kw):
            _STASH["tools"] = kw.get("tools")
            result = f(*a, **kw)
            _STASH["response"] = result
            return result

        return _wrapped

    lm.model_to_completion_function = _patched_mtf

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
        lm.model_to_completion_function = orig_mtf
        AppWorld.execute = orig_execute

    return restore


def run_one(real_tid: str, prefix: str) -> str:
    from appworld.task import Task
    from appworld_agents.code.legacy.plain.agents.agent import Agent

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}.json")
    if os.path.exists(out_path):
        print(f"    [SKIP] already exists")
        return out_path

    collector = OrderedTraceCollector(
        agent_name="AppWorld-Legacy", model=MAIN_MODEL_NAME,
        benchmark_task_id=prefix,
    )

    t0 = time.perf_counter()
    restore = _install_hooks(collector)
    try:
        # Official flow (run_agent_on_task): load train tasks → Agent.from_dict → agent.solve()
        train_tasks = [
            Task.load(task_id=tid, load_ground_truth=True, ground_truth_mode="full")
            for tid in DEMO_TASK_IDS
        ]
        Task.load(task_id=real_tid)  # Preload the test task (the official flow does the same)

        agent_config = {
            "type": "function_calling_agent",
            "test_task_id": real_tid,
            "experiment_name": f"trace_builtin_legacy_{prefix}",
            "train_tasks": train_tasks,
            "solver_config": {
                "random_seed": 100,
                "max_predicted_apis": 30,
                "demo_task_ids": DEMO_TASK_IDS,
                "oracle_first_step": False,
                # API prediction uses a cheap model; the function-calling main loop uses V4-flash
                # (larger context + stronger capability, fixes 64k context crashes and planning quality)
                "apis_language_model": {
                    "type": "openai_lm",
                    "model": PREDICTOR_MODEL_NAME,
                    "temperature": 0,
                    "seed": 100,
                    "extra_body": THINKING_MODE,  # Predictor uses the same convention as the main loop
                },
                "function_calling_language_model": {
                    "type": "openai_lm",
                    "model": MAIN_MODEL_NAME,  # Direct openai SDK connection; must use the raw API model name
                    "temperature": 0,
                    "seed": 100,
                    # Generation cap: Qwen3.8 never emits EOS for some prompts and would generate forever (8192 avoids thinking-token truncation)
                    "max_tokens": 8192,
                    "extra_body": THINKING_MODE,
                },
            },
            "max_steps": 70,
            "show_io": False,
            "skip_if_complete": False,
        }
        # Timeout hardening: legacy openai_lm uses the openai SDK global client
        # (module-level attributes); avoid the 600s default timeout wasting tasks
        # during vLLM hiccups (openai_lm has its own retry loop)
        import openai
        openai.timeout = 600.0
        openai.max_retries = 3
        # legacy's openai_lm has a fixed model registry (token limits + completion
        # type); unknown models crash outright with "No item found". Register the
        # currently configured models at runtime here (chat type + large context
        # limits), so switching models never requires changing agent source.
        # Note: do not call ModelsInformation.exists() -- the underlying _find_one
        # raises rather than returning None for missing entries (observed
        # "No item found" exception); dedupe by checking the name set directly.
        from appworld_agents.code.legacy.plain.language_models.openai_language_model import (
            ModelInformation, ModelsInformation)
        _known = {_mi.name for _mi in ModelsInformation._list}
        for _m in {MAIN_MODEL_NAME, PREDICTOR_MODEL_NAME}:
            if _m not in _known:
                ModelsInformation._list.append(ModelInformation(
                    name=_m, provider="openai", completion_type="chat",
                    token_limits={"input": 256000, "output": 8192}))
        agent = Agent.from_dict(agent_config)
        # Per-span environment snapshot (task_id/state hash/DB increments) -- world is created inside solve, resolved lazily
        collector.set_checkpoint_provider(
            make_appworld_checkpoint_provider(lambda: getattr(agent, "world", None), real_tid))
        agent.solve()
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
    print(f"Running {len(TASKS)} tasks with BUILT-IN legacy FunctionCallingAgent\n")
    total_t0 = time.perf_counter()
    for i, (real_tid, prefix) in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] {real_tid}")
        run_one(real_tid, prefix)
    elapsed = time.perf_counter() - total_t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Traces: {OUTPUT_DIR}")

    # ── Required post-collection step: validator writes pass + event_type annotation ──
    from runners.finalize_traces import finalize_trace_dir
    finalize_trace_dir(OUTPUT_DIR, experiment_name="offline_eval_legacy")


if __name__ == "__main__":
    main()
