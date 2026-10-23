"""
Collect traces using the tau2-bench BUILT-IN LLMAgent.

- Agent source: the LLMAgent in tau2-bench/src/tau2/agent/llm_agent.py (native, unmodified)
- Tasks: 13 AppWorld tasks
- Tools: real AppWorld APIs, wrapped as tau2 Tools (as_tool)
- Driver: the adapter layer drives the loop (LLMAgent only generates messages; the environment role is played by the driver)
- Trace: the _TRACE_TRACER sentinel already present in tau2 llm_utils (LLM) + the tool hook in the driver
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
# litellm reads this env as api_base for the "openai/<model>" prefix; the
# import order puts this before adapter.model_config, reading env directly
# (defaults to DeepSeek, overridable via MODEL_BASE_URL)
os.environ["OPENAI_BASE_URL"] = os.environ.get("MODEL_BASE_URL", "https://api.deepseek.com/v1")
# Timeout hardening: litellm reads these env vars at import time; they must be set before the tau2 import chain
os.environ.setdefault("LITELLM_REQUEST_TIMEOUT", "600")
os.environ.setdefault("LITELLM_NUM_RETRIES", "3")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks", "appworld", "experiments", "code"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "source_code", "tau2-bench", "src"))
os.chdir(os.path.join(ROOT, "benchmarks", "appworld"))

from trace_collector import OrderedTraceCollector, make_appworld_checkpoint_provider
from adapter.tools import appworld_execute_tool
from adapter.model_config import (THINKING_MODE, MAIN_MODEL_NAME,
                                  PREDICTOR_MODEL_NAME, PREDICTOR_BASE_URL)

# NOTE: tau2's import chain (pandas/deepdiff and other native extensions) must be
# loaded before any litellm call; otherwise importing pandas after a litellm call
# on Windows triggers a native DLL conflict (the process is hard-killed).
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import UserMessage, ToolMessage, MultiToolMessage
from tau2.environment.tool import as_tool
from tau2.utils import llm_utils

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
_OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or os.path.join(ROOT, "traces", "appworld_tau2_batch1")
# Anchor relative paths to ROOT: this file has chdir'd to benchmarks/appworld, otherwise traces would land in the wrong place
OUTPUT_DIR = _OUTPUT_DIR if os.path.isabs(_OUTPUT_DIR) else os.path.join(ROOT, _OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CFG = {
    "name": PREDICTOR_MODEL_NAME, "client_name": "openai",
    "base_url": PREDICTOR_BASE_URL,
    "use_cache": False, "max_retries": 2, "retry_after_n_seconds": 5,
    "temperature": 0.0,  # Same convention as the other agents (previously missing on the predictor)
    "seed": 100,  # Same convention as the other agents (reproducibility)
    "extra_body": THINKING_MODE,  # Predictor uses the same convention as the main loop (consistent thinking switch)
    "api_key_env_name": "OPENAI_API_KEY",
    "cost_per_token": {"input_cache_miss": 0, "input_cache_hit": 0,
                       "input_cache_write": 0, "output": 0},
}


def _tok(u):
    if u is None: return 0, 0
    if hasattr(u, "tokens"): return u.tokens.input_cache_miss + u.tokens.input_cache_hit, u.tokens.output
    if isinstance(u, dict): return u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
    return 0, 0


class _Tau2CollectorShim:
    """Forward the _TRACE_TRACER log_llm_call/log_llm_response interface to the collector."""

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
            duration_ms=latency_ms,
        )


def _build_tau2_tools(world, allowed_apis, executor):
    """Generate a list of tau2 Tools from the real AppWorld APIs (as_tool wrappers)."""
    tools = []
    tool_map = {}
    for doc in world.task.api_docs.function_calling():
        fn = doc["function"]
        name = fn["name"]
        if name not in allowed_apis and not name.startswith("supervisor__"):
            continue
        desc = fn["description"]
        props = fn["parameters"].get("properties", {})
        required = set(fn["parameters"].get("required", []))
        app_name, api_name = name.split("__", 1)

        req_items = [(k, v) for k, v in props.items() if k in required]
        opt_items = [(k, v) for k, v in props.items() if k not in required]
        arg_parts = []
        for pname, pinfo in req_items + opt_items:
            pt = {"string": "str", "integer": "int", "number": "float",
                  "boolean": "bool"}.get(pinfo.get("type", "string"), "str")
            default = pinfo.get("default")
            if default is not None:
                arg_parts.append(f"{pname}: {pt} = {repr(default)}")
            elif pname not in required:
                arg_parts.append(f"{pname}: {pt} = None")
            else:
                arg_parts.append(f"{pname}: {pt}")

        args_str = ", ".join(arg_parts)
        src = f'''def {name}({args_str}):
    """{desc}"""
    return executor("{app_name}", "{api_name}", {{}} if False else {{
        {", ".join(f"'{p}': {p}" for p, _ in req_items + opt_items)}
    }})
'''
        # More robust: build the kwargs dict directly
        kw_lines = "\n    ".join(f"kw['{p}'] = {p}" for p, _ in req_items + opt_items)
        src = f'''def {name}({args_str}):
    """{desc}"""
    kw = {{}}
    {kw_lines}
    return executor("{app_name}", "{api_name}", kw)
'''
        ns = {"executor": executor}
        exec(src, ns)
        fn_obj = ns[name]
        fn_obj.__doc__ = desc
        tool = as_tool(fn_obj)
        tools.append(tool)
        tool_map[name] = tool
    return tools, tool_map


def run_one(real_tid: str, prefix: str) -> str:
    from appworld import AppWorld
    from appworld_agents.code.common.api_predictor import APIPredictor
    from appworld_agents.code.simplified.language_model import LanguageModel
    from appworld.common.io import dump_yaml

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}.json")
    if os.path.exists(out_path):
        print(f"    [SKIP] already exists")
        return out_path

    collector = OrderedTraceCollector(
        agent_name="Tau2-LLMAgent", model=MAIN_MODEL_NAME, benchmark_task_id=prefix,
    )

    t0 = time.perf_counter()
    try:
        with AppWorld(task_id=real_tid, experiment_name=f"trace_tau2_{prefix}") as world:
            collector.set_checkpoint_provider(
                make_appworld_checkpoint_provider(world, real_tid))
            # ── API prediction (with rate-limit retries; DeepSeek rate-limits easily in batch runs) ──
            lm = LanguageModel(**MODEL_CFG)
            api_predictor = APIPredictor(
                prompt_file_path="experiments/prompts/api_predictor.txt",
                demo_task_ids=["82e2fac_1"], max_predicted_apis=30, app_api_separator="__",
            )
            prompt_msgs = api_predictor.build_messages(world.task, include_cache_control=False)
            raw = None
            for _attempt in range(5):
                try:
                    raw = lm.generate(prompt_msgs)
                    break
                except Exception as e:
                    print(f"    API prediction retry {_attempt+1}/5: {type(e).__name__}")
                    time.sleep(15)
            if raw is None:
                raise RuntimeError("API prediction failed after 5 retries")
            predicted = api_predictor.predicted_output_to_apis(world.task, raw["content"])
            u = raw.pop("standardized_usage", None)
            i1, o1 = _tok(u)
            collector.log_llm(model=PREDICTOR_MODEL_NAME,
                              messages=[{"role": str(m.get("role", "unk")),
                                         "content": str(m.get("content") or "")}
                                        for m in prompt_msgs],
                              output_text="\n".join(predicted), input_tokens=i1, output_tokens=o1)

            # ── Tool execution (unified entry point appworld_execute_tool, canonical recording) ──
            def _executor(app, api, kwargs):
                code = f"print(apis.{app}.{api}(**{kwargs!r}))"
                return appworld_execute_tool(world, collector, code)

            allowed = set(predicted)
            allowed.add("supervisor__complete_task")
            tools, tool_map = _build_tau2_tools(world, allowed, _executor)

            # ── LLM hook: the sentinel already present in tau2 llm_utils ──
            shim = _Tau2CollectorShim(collector)
            orig_sentinel = llm_utils._TRACE_TRACER
            llm_utils._TRACE_TRACER = shim

            # ── tau2 LLMAgent (native, already imported at the top of the module) ──
            app_descs = world.task.app_descriptions
            app_descs.pop("api_docs", None)
            desc_yaml = dump_yaml(app_descs)
            domain_policy = (
                f"Available apps:\n{desc_yaml}\n\n"
                f"Task: {world.task.instruction}\n\n"
                "Call supervisor__complete_task when the task is done. "
                "Pass the answer argument only if the task explicitly asks a "
                "question (e.g., \"How many songs are in the Spotify queue?\"); "
                "in that case the answer must be concise, such as a number, a "
                "yes/no, or a specific entity. If the task is not a question "
                "(e.g., \"Start my Spotify music player.\"), leave the answer empty."
            )

            agent = LLMAgent(
                tools=tools,
                domain_policy=domain_policy,
                llm=f"openai/{MAIN_MODEL_NAME}",
                llm_args={"temperature": 0,
                          "max_tokens": 8192,  # Thinking mode + 4096 would let thinking tokens eat the whole budget and end with empty output
                          "seed": 100,  # Same convention as the other agents (previously omitted in tau2)
                          "extra_body": THINKING_MODE},
            )
            state = agent.get_init_state()
            msg = UserMessage(role="user", content=world.task.instruction)

            for _ in range(50):
                assistant_msg, state = agent.generate_next_message(msg, state)
                tcs = assistant_msg.tool_calls
                if not tcs:
                    break  # Plain text response, done
                results = []
                for tc in tcs:
                    tool = tool_map.get(tc.name)
                    try:
                        if tool is not None:
                            result = tool(**tc.arguments)
                        else:
                            result = f"Tool {tc.name} not found"
                        results.append(ToolMessage(
                            role="tool", content=str(result), id=tc.id))
                    except Exception as e:
                        # Tool call exception (missing arguments, etc.) → feed back to the agent as an error result
                        results.append(ToolMessage(
                            role="tool", content=f"Error: {e}", id=tc.id, error=True))
                    if tc.name == "supervisor__complete_task":
                        break
                msg = MultiToolMessage(role="tool", tool_messages=results)
                if world.task_completed():
                    break

            llm_utils._TRACE_TRACER = orig_sentinel

        print(f"    done in {time.perf_counter() - t0:.0f}s")
    except Exception as e:
        import traceback
        print(f"    ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

    collector.save(out_path, task_instruction="(loaded from AppWorld task)")
    print(f"    spans={collector._step} llm={collector.llm_count} "
          f"tools={collector.tool_count} tokens={collector.token_count:,}")
    return out_path


def main():
    print(f"Running {len(TASKS)} tasks with tau2-bench BUILT-IN LLMAgent\n")
    total_t0 = time.perf_counter()
    for i, (real_tid, prefix) in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] {real_tid}")
        run_one(real_tid, prefix)
    elapsed = time.perf_counter() - total_t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Traces: {OUTPUT_DIR}")

    # ── Required post-collection step: validator writes pass + event_type annotation ──
    from runners.finalize_traces import finalize_trace_dir
    finalize_trace_dir(OUTPUT_DIR, experiment_name="offline_eval_tau2")


if __name__ == "__main__":
    main()
