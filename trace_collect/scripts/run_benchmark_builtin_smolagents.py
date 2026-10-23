"""
Collect traces using the AppWorld BUILT-IN smolagents ToolCallingAgent.

- Agent source: benchmarks/appworld/experiments/code/smolagents/ (the ToolCallingAgent in agents.py)
- Flow: exact replica of smolagents/run.py's run_agent_on_task (the official flow)
- Trace: pre-implanted hooks (LLM hook in models.py + tool hook in run.py)
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
# NOTE: do not insert experiments/code -- otherwise the local smolagents/ directory would shadow the PyPI smolagents package.
# Relies on the editable-installed appworld_agents package (pip install -e benchmarks/appworld/experiments).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(os.path.join(ROOT, "benchmarks", "appworld"))

from trace_collector import OrderedTraceCollector, make_appworld_checkpoint_provider
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

_OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or os.path.join(ROOT, "traces", "appworld_smolagents_batch1")
# Anchor relative paths to ROOT: this file has chdir'd to benchmarks/appworld, otherwise traces would land in the wrong place
OUTPUT_DIR = _OUTPUT_DIR if os.path.isabs(_OUTPUT_DIR) else os.path.join(ROOT, _OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIG = {
    "type": "openai",
    "base_url": MAIN_BASE_URL,
    "api_key_env_name": "OPENAI_API_KEY",
    "model_id": MAIN_MODEL_NAME,
    "temperature": 0.0,
    "seed": 100,
    "use_cache": False,
    "cost_per_token": {"input_cache_miss": 0, "input_cache_hit": 0,
                       "input_cache_write": 0, "output": 0},
}


def run_one(real_tid: str, prefix: str) -> str:
    from appworld import AppWorld
    from appworld.common.time import timeout_call
    from appworld_agents.code.common.usage_tracker import UsageTracker
    from appworld_agents.code.smolagents.agents import Agent
    from appworld_agents.code.smolagents.api_predictor import APIPredictor
    from appworld_agents.code.smolagents.models import Model
    from appworld_agents.code.smolagents.task_completer import TaskCompleter
    from appworld_agents.code.smolagents.run import get_smolagents_tools
    from adapter.hooks import llm_hook, tool_hook

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}.json")
    if os.path.exists(out_path):
        print(f"    [SKIP] already exists")
        return out_path

    collector = OrderedTraceCollector(
        agent_name="AppWorld-Smolagents", model=MAIN_MODEL_NAME,
        benchmark_task_id=prefix,
    )
    llm_hook.register(collector)
    tool_hook.register(collector)

    t0 = time.perf_counter()
    try:
        # ── Official run_agent_on_task flow ──
        from copy import deepcopy
        model_config = deepcopy(MODEL_CONFIG)
        model_config.pop("cost_per_token", {})
        model = Model.from_dict(model_config)
        # Timeout hardening: explicitly set a uniform 600s timeout and enable
        # SDK retries (APITimeoutError is a retryable error); during vLLM hiccups
        # (queuing/stalls) retries catch it instead of wasting the task outright
        if getattr(model, "_model", None) is not None and hasattr(model._model, "client"):
            model._model.client.timeout = 600.0
            model._model.client.max_retries = 3
        # Disable thinking mode: thinking conflicts with function-calling's tool_choice
        _orig_gen = model._model.generate

        def _patched_gen(*_a, **_kw):
            _kw["extra_body"] = THINKING_MODE
            # Generation cap: Qwen3.8 never emits EOS for some prompts and would
            # generate forever; the vLLM side cannot stop it → client ReadTimeout
            # wastes the task. 8192 avoids thinking-token truncation
            _kw.setdefault("max_completion_tokens", 8192)
            return _orig_gen(*_a, **_kw)

        model._model.generate = _patched_gen

        AppWorld.init_defaults.experiment_name = f"trace_builtin_smol_{prefix}"
        api_predictor = APIPredictor(
            app_api_separator="__",
            model=model,
            prompt_file_path=os.path.join(
                ROOT, "benchmarks", "appworld", "experiments", "prompts", "api_predictor.txt"),
            demo_task_ids=["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
            max_predicted_apis=30,
            mode="predicted",
        )
        task_completer = TaskCompleter(
            model=model,
            prompt_file_path=os.path.join(
                ROOT, "benchmarks", "appworld", "experiments", "prompts",
                "smolagents", "task_completer_instructions.txt"),
        )

        with AppWorld(task_id=real_tid) as world:
            collector.set_checkpoint_provider(
                make_appworld_checkpoint_provider(world, real_tid))
            tools = get_smolagents_tools(world)
            # ── Version compatibility fix (no AppWorld source changes, runtime adaptation only) ──
            # AppWorld's tool_generator defaults nullable parameters to None, but
            # the AppWorld API returns 422 for bool=None. Filter out None
            # parameters at the benchmark layer (equivalent to "not passed");
            # the forward logic is completely unchanged.
            for _t in tools:
                _cls = type(_t)
                _orig_func = _cls._function

                def _filtered_func(*_args, _orig=_orig_func, **_kwargs):
                    _kwargs = {k: v for k, v in _kwargs.items() if v is not None}
                    return _orig(*_args, **_kwargs)

                _cls._function = staticmethod(_filtered_func)
            predicted_api_names, _ = api_predictor.predict(world.task)
            tools = [t for t in tools if t.name in predicted_api_names]

            # ── requester hook (no AppWorld source changes, runtime wrapping) ──
            # task_completer goes through world.requester.request rather than
            # world.execute; wrap it here to capture complete_task calls
            # (trace completeness + outcome determination)
            _orig_request = world.requester.request

            def _traced_request(*_args, **_kwargs):
                _app = _kwargs.get("_app_name", "") or (_args[0] if _args else "")
                _api = _kwargs.get("_api_name", "") or (_args[1] if len(_args) > 1 else "")
                _t0 = time.perf_counter()
                _result = _orig_request(*_args, **_kwargs)
                _d = (time.perf_counter() - _t0) * 1000
                if _app == "supervisor" and _api == "complete_task":
                    _data = {k: v for k, v in _kwargs.items()
                             if k not in ("_app_name", "_api_name", "client",
                                          "raise_on_failure", "show", "track")}
                    collector.log_tool("supervisor.complete_task", _data, str(_result), _d, True)
                return _result

            world.requester.request = _traced_request

            # Official config: prompt_templates + max_steps (run_agent_on_task reads the YAML files)
            import yaml
            agent = Agent.from_dict({
                "type": "tool_calling",
                "tools": tools,
                "model": model._model,
                "add_base_tools": False,
                "prompt_templates": yaml.safe_load(
                    open(os.path.join(
                        ROOT, "benchmarks", "appworld", "experiments", "prompts",
                        "smolagents", "tool_calling_instructions.yaml"),
                        encoding="utf-8").read()),
                "max_steps": 50,
            })
            # NOTE: timeout_call relies on Unix signals; call synchronously on Windows
            # (the only difference from the official flow; agent internal logic is unchanged)
            answer = agent.run(instruction=world.task.instruction)
            task_completer.complete(world, answer)
        print(f"    done in {time.perf_counter() - t0:.0f}s")
    except Exception as e:
        import traceback
        print(f"    ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        llm_hook.unregister()
        tool_hook.unregister()

    collector.save(out_path, task_instruction="(loaded from AppWorld task)")
    print(f"    spans={collector._step} llm={collector.llm_count} "
          f"tools={collector.tool_count} tokens={collector.token_count:,}")
    return out_path


def main():
    print(f"Running {len(TASKS)} tasks with BUILT-IN smolagents ToolCallingAgent\n")
    total_t0 = time.perf_counter()
    for i, (real_tid, prefix) in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] {real_tid}")
        run_one(real_tid, prefix)
    elapsed = time.perf_counter() - total_t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Traces: {OUTPUT_DIR}")

    # ── Required post-collection step: validator writes pass + event_type annotation ──
    from runners.finalize_traces import finalize_trace_dir
    finalize_trace_dir(OUTPUT_DIR, experiment_name="offline_eval_smolagents")


if __name__ == "__main__":
    main()
