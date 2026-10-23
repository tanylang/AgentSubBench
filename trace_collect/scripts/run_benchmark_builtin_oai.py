"""
Collect traces using the AppWorld BUILT-IN OpenAI Agents SDK agent.

- Agent source: benchmarks/appworld/experiments/code/openai_agents/run.py (the official run_agent_on_task flow)
- MCP: AppWorld initializer starts the servers automatically (start_servers: true)
- Trace: pre-implanted hooks (stream event-flow LLM + tool hook in mcp.py)
"""
import os, sys, time, json, asyncio

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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(os.path.join(ROOT, "benchmarks", "appworld"))

from trace_collector import OrderedTraceCollector, make_appworld_checkpoint_provider
from adapter.model_config import (MAIN_MODEL_NAME, MAIN_BASE_URL,
                                  PREDICTOR_MODEL_NAME, PREDICTOR_BASE_URL)

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

_OUTPUT_DIR = os.environ.get("TRACE_OUTPUT_DIR") or os.path.join(ROOT, "traces", "appworld_openai_agents_batch1")
# Anchor relative paths to ROOT: this file has chdir'd to benchmarks/appworld, otherwise traces would land in the wrong place
OUTPUT_DIR = _OUTPUT_DIR if os.path.isabs(_OUTPUT_DIR) else os.path.join(ROOT, _OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIG = {
    "type": "openai",
    "name": MAIN_MODEL_NAME,
    "settings": {
        "type": "openai",
        "api_type": "chat_completions",
        "base_url": MAIN_BASE_URL,
        "api_key_env_name": "DEEPSEEK_API_KEY",
        "temperature": 0.0,
        "seed": 100,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        # Generation cap: Qwen3.8 never emits EOS for some prompts and would generate forever
        "max_output_tokens": 8192,
    },
    "extras": {},
}


def run_one(real_tid: str, prefix: str) -> str:
    from appworld import AppWorld
    from appworld.common.io import dump_yaml, read_file, read_json
    from appworld.common.prompts import load_prompt_to_chat_messages
    from appworld.common.text import render_template
    from agents import Agent as SDKAgent, set_default_openai_api, set_tracing_disabled
    from agents import set_default_openai_client as _set_default_openai_client
    from agents.agent import StopAtTools
    from agents.model_settings import ModelSettings
    from agents.run import RunConfig, Runner
    from openai import AsyncOpenAI
    from appworld_agents.code.openai_agents.api_predictor import APIPredictor
    from appworld_agents.code.openai_agents.mcp import AgentsMCP

    out_path = os.path.join(OUTPUT_DIR, f"{prefix}.json")
    if os.path.exists(out_path):
        print(f"    [SKIP] already exists")
        return out_path

    collector = OrderedTraceCollector(
        agent_name="AppWorld-OpenAI-Agents", model=MAIN_MODEL_NAME,
        benchmark_task_id=prefix,
    )
    # Register the collector with the hooks implanted in mcp.py
    from adapter.hooks import llm_hook, tool_hook
    llm_hook.register(collector)
    tool_hook.register(collector)
    # Request protocol context: the SDK streaming hook only sees the response;
    # the request's tools parameter is stashed at the driver-level client patch
    # and retrieved via this provider when logging the LLM call
    _OAI_STASH: dict = {}
    collector.set_request_context_provider(lambda: dict(_OAI_STASH))

    t0 = time.perf_counter()
    try:
        class _NoopLogger:
            def show_message(self, **kwargs):
                pass

        # ── MCP HTTP 307 fix (no AppWorld source changes, benchmark-layer patch) ──
        # build_mcp_config generates a /mcp URL (no trailing slash), Starlette
        # responds with 307, and the openai-agents SDK does not follow POST
        # redirects. Patch the URL to include the trailing slash.
        import appworld.serve._mcp as _aw_mcp
        _orig_build_mcp = _aw_mcp.build_mcp_config

        def _patched_build_mcp(*args, **kwargs):
            params = _orig_build_mcp(*args, **kwargs)
            if "url" in params and not params["url"].endswith("/"):
                params["url"] += "/"
            return params

        _aw_mcp.build_mcp_config = _patched_build_mcp
        # AgentsMCP's connect does `from appworld.serve._mcp import build_mcp_config`,
        # so its module-level reference needs patching too
        import appworld_agents.code.openai_agents.mcp as _oai_mcp
        _oai_mcp.build_mcp_config = _patched_build_mcp

        set_default_openai_api("chat_completions")
        set_tracing_disabled(True)

        model_settings = dict(MODEL_CONFIG["settings"])
        base_url = model_settings.pop("base_url", None)
        api_key_env_name = model_settings.pop("api_key_env_name", None)
        api_key = os.environ[api_key_env_name] if api_key_env_name else None
        api_type = model_settings.pop("api_type")
        set_default_openai_api(api_type)
        if base_url and api_key:
            # Timeout hardening: avoid the 600s default timeout wasting tasks during vLLM hiccups
            client = AsyncOpenAI(base_url=base_url, api_key=api_key,
                                 timeout=600.0, max_retries=3)
            # Disable thinking mode + protocol layer stash: the openai-agents SDK sends requests via client.chat.completions.create.
            from adapter.model_config import THINKING_MODE
            _orig_create = client.chat.completions.create
            async def _patched_create(**kwargs):
                # Inject when THINKING_MODE is non-empty (deepseek turns off thinking / qwen enable_thinking=false)
                if THINKING_MODE:
                    kwargs["extra_body"] = dict(kwargs.get("extra_body") or {}, **THINKING_MODE)
                # Protocol layer: stash the request tools and the structured tool_calls from the response
                _OAI_STASH["tools"] = kwargs.get("tools")
                _OAI_STASH["tool_calls"] = None
                result = await _orig_create(**kwargs)
                try:
                    tcs = getattr(result.choices[0].message, "tool_calls", None)
                    if tcs:
                        _OAI_STASH["tool_calls"] = [
                            {"id": t.id, "name": t.function.name,
                             "arguments": t.function.arguments} for t in tcs]
                except Exception:
                    pass
                return result
            client.chat.completions.create = _patched_create
            _set_default_openai_client(client)

        api_predictor = APIPredictor(
            app_api_separator="__",
            model_config={
                "type": "openai",
                "name": PREDICTOR_MODEL_NAME,
                "settings": {
                    "type": "openai", "api_type": "chat_completions",
                },
                # Critical fix: the official code builds the client with
                # AsyncOpenAI(**(extras or {})); when extras is empty it connects
                # to api.openai.com (designed for GPT upstream). base_url/api_key
                # must be injected here, otherwise API prediction retries forever and hangs.
                "extras": {
                    "base_url": PREDICTOR_BASE_URL,
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                },
            },
            prompt_file_path=os.path.join(
                ROOT, "benchmarks", "appworld", "experiments", "prompts", "api_predictor.txt"),
            demo_task_ids=["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
            max_predicted_apis=30,
            mode="predicted",
        )

        appworld_config = {
            "start_servers": True,
            "show_server_logs": False,
            "timeout": 240,  # Server startup timeout (the default 60s is not enough; API loading takes time)
            "remote_apis_url": "http://localhost:{port}",
            # NOTE: the initializer requires the mcp URL in localhost format;
            # AgentsMCP actually uses stdio transport (see below), avoiding the
            # HTTP 307 issue on Windows
            "remote_mcp_url": "http://localhost:{port}",
            # Server-side DB persisted to disk (jsonl change log) → remote checkpoint collection + final state available
            "apis_server_kwargs": {"on_disk": True},
            "random_seed": 100,
            "include_direct_functions": True,
            "direct_function_separator": "__",
        }

        prompt_template = read_file(os.path.join(
            ROOT, "benchmarks", "appworld", "experiments", "prompts",
            "function_calling_agent", "instructions.txt"))
        demo_messages = read_json(os.path.join(
            ROOT, "benchmarks", "appworld", "experiments", "prompts",
            "function_calling_agent", "demos.json"))

        with AppWorld.initializer(
            update_defaults=True,
            experiment_name=f"trace_builtin_oai_{prefix}",
            **appworld_config,
        ):
            remote_apis_url = AppWorld.init_defaults.remote_apis_url
            remote_mcp_url = AppWorld.init_defaults.remote_mcp_url

            async def _run():
                async with AgentsMCP(
                    remote_apis_url=remote_apis_url,
                    remote_mcp_url=remote_mcp_url, quiet=True,  # http transport (URL already patched with trailing slash)
                ) as mcp:
                    stop_at_tools = StopAtTools(
                        stop_at_tool_names=["supervisor__complete_task"])
                    sdk_agent = SDKAgent(
                        name="Assistant",
                        model=MAIN_MODEL_NAME,
                        model_settings=ModelSettings(**model_settings),
                        mcp_servers=[mcp.server],
                        tool_use_behavior=stop_at_tools,
                        reset_tool_choice=False,
                    )
                    run_config = RunConfig(tracing_disabled=True)

                    with AppWorld(task_id=real_tid) as world:
                        collector.set_checkpoint_provider(
                            make_appworld_checkpoint_provider(world, real_tid, force_remote=True))
                        app_descriptions = dict(world.task.app_descriptions)
                        app_descriptions.pop("api_docs", None)
                        app_descriptions_string = dump_yaml(app_descriptions)

                        header_content = render_template(
                            prompt_template,
                            instruction=world.task.instruction,
                            main_user=world.task.supervisor,
                            app_descriptions=app_descriptions_string,
                            max_steps=70,
                        )
                        header_messages = load_prompt_to_chat_messages(
                            header_content, skip_system_message=False, only_header=True)
                        test_input_content = render_template(
                            prompt_template,
                            instruction=world.task.instruction,
                            main_user=world.task.supervisor,
                            app_descriptions=app_descriptions_string,
                            max_steps=70,
                        )
                        test_input_messages = load_prompt_to_chat_messages(
                            test_input_content, skip_system_message=True,
                            only_body=True, end_at=1)
                        messages = header_messages + demo_messages + test_input_messages
                        # Official flow: convert_fc_demos_to_response_format converts
                        # OpenAI-format demo messages into the SDK's items format
                        # (function_call/function_call_output). Missing this
                        # conversion makes the SDK raise "Unhandled item type or structure".
                        from appworld_agents.code.openai_agents.run import convert_fc_demos_to_response_format
                        system_prompt, updated_messages = convert_fc_demos_to_response_format(messages)

                        # The API predictor's LLM call does not go through the MCP
                        # stream, so record it explicitly (aligned with the other
                        # agents). build_messages and predict internally use the same
                        # deterministic rendering; calling it again here is only to
                        # obtain the real request prompt for recording (agent logic unchanged)
                        pred_msgs = api_predictor.build_messages(
                            world.task, include_cache_control=False)
                        allowed_apis, raw_message = await api_predictor.predict(world.task)
                        collector.log_llm(
                            model=PREDICTOR_MODEL_NAME,
                            messages=[{"role": str(m.get("role", "unk")),
                                       "content": str(m.get("content") or "")}
                                      for m in pred_msgs],
                            output_text="\n".join(allowed_apis),
                            input_tokens=0, output_tokens=0,
                        )
                        mcp.set_allowed_tools(set(allowed_apis))
                        sdk_agent.instructions = system_prompt

                        used_steps = 0
                        while True:
                            left_steps = 70 - used_steps
                            streamer = Runner.run_streamed(
                                starting_agent=sdk_agent, input=updated_messages,
                                max_turns=left_steps, run_config=run_config)
                            await AgentsMCP.stream(streamer=streamer, logger=_NoopLogger())
                            world.save_state()
                            if world.task_completed():
                                break
                            used_steps += streamer.current_turn
                            if used_steps >= 70:
                                break

            asyncio.run(_run())
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
    print(f"Running {len(TASKS)} tasks with BUILT-IN OpenAI Agents SDK agent\n")
    total_t0 = time.perf_counter()
    for i, (real_tid, prefix) in enumerate(TASKS, 1):
        print(f"[{i}/{len(TASKS)}] {real_tid}")
        run_one(real_tid, prefix)
    elapsed = time.perf_counter() - total_t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Traces: {OUTPUT_DIR}")

    # ── Required post-collection step: validator writes pass + event_type annotation ──
    from runners.finalize_traces import finalize_trace_dir
    finalize_trace_dir(OUTPUT_DIR, experiment_name="offline_eval_oai")


if __name__ == "__main__":
    main()
