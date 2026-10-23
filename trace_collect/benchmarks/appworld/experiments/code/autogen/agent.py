"""AppWorld 官方 AutoGen AssistantAgent。

与 upstream `configs/autogen_assistant_agent` 对应的官方实验代码：
- 用真实 AutoGen（autogen_agentchat.AssistantAgent + OpenAIChatCompletionClient）
- 工具 = world.task.api_docs 的真实 AppWorld API（含 supervisor__complete_task）
- API 预测器预筛候选 API（与官方配置 api_predictor_config 一致）
- 官方 prompt（function_calling_agent/instructions.txt + demos.json）

trace 记录（与 hooks.py 设计一致：源码只加一行 hook 调用，未注册时 no-op）：
- LLM：client._trace_tracer → _TracerShim → adapter.hooks.llm_hook
- 工具：tool_hook（执行点记录）
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, cast

from appworld import AppWorld
from appworld.common.random import set_random_seed
from appworld.common.types import FromDict
from appworld_agents.code.autogen.api_predictor import APIPredictor
from appworld_agents.code.common.logger import Logger
from appworld_agents.code.common.usage_tracker import UsageTracker

try:
    from adapter.hooks import llm_hook, tool_hook
except ImportError:  # adapter 不可用时 no-op（官方实验代码独立可跑）
    llm_hook = tool_hook = None


class _TracerShim:
    """AutoGen _trace_tracer → adapter.hooks.llm_hook 的桥接。"""

    _TYPE_TO_ROLE = {
        "SystemMessage": "system",
        "UserMessage": "user",
        "AssistantMessage": "assistant",
        "FunctionExecutionResultMessage": "tool",
    }

    def __init__(self):
        self._span: dict[str, Any] | None = None

    @classmethod
    def _msg_to_dict(cls, m: Any) -> dict:
        if isinstance(m, dict):
            return {"role": str(m.get("role", "unk")),
                    "content": str(m.get("content") or "")}
        t = str(getattr(m, "type", ""))
        role = cls._TYPE_TO_ROLE.get(t, t or str(getattr(m, "role", "unk")))
        return {"role": role, "content": str(getattr(m, "content", None) or "")}

    def log_llm_call(self, model, messages, tools=None, tool_calls=None, **kw):
        self._span = {"model": model, "messages": messages, "tools": tools,
                      "tool_calls": tool_calls}
        return self._span

    def log_llm_response(self, completion, model="", input_tokens=0,
                         output_tokens=0, latency_ms=0.0, llm_call_span=None,
                         **kw):
        if llm_hook is None:
            return
        span = llm_call_span if isinstance(llm_call_span, dict) else self._span or {}
        # hooks 的提取器兼容 dict 形态（choices + usage），把原始值包进去
        result_obj = {
            "choices": [{"message": {"content": str(completion)}}],
            "usage": {"prompt_tokens": input_tokens or 0,
                      "completion_tokens": output_tokens or 0},
        }
        llm_hook(span.get("model") or model,
                 [self._msg_to_dict(m) for m in span.get("messages", [])],
                 result_obj,
                 duration_ms=latency_ms,
                 tools=span.get("tools"),
                 tool_calls=span.get("tool_calls"),
                 reasoning=str(kw.get("reasoning") or ""))

    def log_reasoning(self, thought: str = "", **kw):
        # reasoning 已随 log_llm_response 的 reasoning kwarg 记录；
        # 此方法仅为兼容 vendored autogen hook 协议（log_reasoning 调用点）。
        pass


class Agent(FromDict):
    """AutoGen 官方实验 Agent 基类（与 simplified 的 agent.py 同构）。"""

    def __init__(
        self,
        model_config: dict[str, Any],
        api_predictor_config: dict[str, Any] | None = None,
        appworld_config: dict[str, Any] | None = None,
        logger_config: dict[str, Any] | None = None,
        usage_tracker_config: dict[str, Any] | None = None,
        prompt_file_path: str | None = None,
        demo_messages_file_path: str | None = None,
        max_steps: int = 50,
        log_lm_calls: bool = False,
        skip_if_finished: bool = False,
    ):
        self.model_config = model_config
        self.api_predictor_config = api_predictor_config or {}
        self.appworld_config = appworld_config or {}
        self.random_seed = self.appworld_config.get("random_seed", None)
        self.prompt_file_path = prompt_file_path
        self.demo_messages_file_path = demo_messages_file_path
        self.max_steps = max_steps
        self.log_lm_calls = log_lm_calls
        self.skip_if_finished = skip_if_finished
        usage_tracker_config = usage_tracker_config or {}
        self.usage_tracker = UsageTracker(**usage_tracker_config)
        logger_config = logger_config or {}
        logger_config["usage_tracker"] = self.usage_tracker
        self.logger = Logger(**logger_config)
        self.world: AppWorld | None = None

    def initialize(self, world: AppWorld) -> None:
        self.world = world
        self.usage_tracker.reset(world.task_id)
        self.logger.start_task(world)
        set_random_seed(self.random_seed)

    def _build_client(self):
        """按官方 model_config 构造 AutoGen 客户端。"""
        from autogen_core.models import ModelFamily, ModelInfo
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model_config = dict(self.model_config)
        extra_body = model_config.pop("extra_body", None) or {}
        seed = model_config.pop("seed", None)
        temperature = model_config.pop("temperature", 0.0)
        max_tokens = model_config.pop("max_tokens", None) or \
            model_config.pop("max_completion_tokens", None)
        model_config.pop("tool_choice", None)
        model_config.pop("parallel_tool_calls", None)
        model_config.pop("cost_per_token", None)
        model_config.pop("retry_after_n_seconds", None)
        model_config.pop("use_cache", None)
        model_config.pop("max_retries", None)
        model_config.pop("client_name", None)
        model_config.pop("api_type", None)
        api_key_env = model_config.pop("api_key_env_name", None)
        client = OpenAIChatCompletionClient(
            model=model_config["name"],
            api_key=os.environ[api_key_env] if api_key_env else "",
            base_url=model_config.get("base_url"),
            model_info=ModelInfo(vision=False, function_calling=True,
                                 json_output=True, structured_output=False,
                                 family=ModelFamily.UNKNOWN),
            extra_body=extra_body,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=600.0,
            max_retries=3,
        )
        client._trace_tracer = _TracerShim()
        return client

    def _build_tools(self, world: AppWorld, predicted_apis: list[str]):
        """按预测 API 列表生成真实 AppWorld 工具（async 函数，代码生成具名签名）。"""
        import time

        from autogen_core.tools import FunctionTool

        all_apis = world.task.api_docs.function_calling()
        predicted = set(predicted_apis)
        tools = []
        for api_doc in all_apis:
            name = api_doc["function"]["name"]
            if name not in predicted and not name.startswith("supervisor__"):
                continue
            fn_name = name.replace(".", "_").replace("-", "_")
            # world.execute 只认点分形式（apis.<app>.<api>(...)），
            # 直接函数形式 apis.<app>__<api>() 会解析失败
            if "__" in name:
                app_name, api_name = name.split("__", 1)
                call_target = f"apis.{app_name}.{api_name}"
            else:
                call_target = f"apis.{name}"
            params = api_doc["function"]["parameters"]["properties"]
            sig_args = []
            call_args = []
            for pname, pspec in params.items():
                typ = "int" if pspec.get("type") == "integer" else "str"
                sig_args.append(f"{pname}: {typ}")
                call_args.append(f"{pname}={pname}")
            src = (
                f"async def {fn_name}({', '.join(sig_args)}):\n"
                f"    try:\n"
                f"        from adapter.clock import monotonic_ms as _MONO_MS\n"
                f"    except ImportError:\n"
                f"        import time\n"
                f"        _MONO_MS = lambda: time.perf_counter() * 1000\n"
                f"    code = f\"{call_target}(\" + \", \".join(f\"{{k}}={{v!r}}\" for k, v in "
                f"dict({', '.join(call_args)}).items()) + \")\"\n"
                f"    _t0 = _MONO_MS()\n"
                f"    try:\n"
                f"        result = _WORLD.execute(code)\n"
                f"        _ok = not (isinstance(result, str) and result.startswith('Execution failed'))\n"
                f"        if _TOOL_HOOK is not None: _TOOL_HOOK('{name}', dict({', '.join(call_args)}), str(result), _MONO_MS() - _t0, _ok)\n"
                f"        return str(result)\n"
                f"    except Exception as e:\n"
                f"        if _TOOL_HOOK is not None: _TOOL_HOOK('{name}', dict({', '.join(call_args)}), str(e), _MONO_MS() - _t0, False)\n"
                f"        raise\n"
            )
            ns = {"_WORLD": world, "_TOOL_HOOK": tool_hook}
            exec(src, ns)
            fn = ns[fn_name]
            fn.__name__ = name
            fn.__doc__ = api_doc["function"]["description"]
            fn.__annotations__ = {p: (int if params[p].get("type") == "integer" else str)
                                  for p in params}
            fn.__annotations__["return"] = str
            tools.append(FunctionTool(fn, name=name,
                                      description=api_doc["function"]["description"]))
        return tools

    def _build_system_prompt(self, world: AppWorld) -> str:
        """官方 prompt 流程：instructions 渲染（system 段）。"""
        from appworld.common.io import read_file
        from appworld.common.prompts import load_prompt_to_chat_messages
        from appworld.common.text import render_template
        from appworld.common.utils import dump_yaml

        app_descriptions = dict(world.task.app_descriptions)
        app_descriptions.pop("api_docs", None)
        app_descriptions_string = dump_yaml(app_descriptions)
        prompt_template = read_file(self.prompt_file_path) if self.prompt_file_path else ""
        header_content = render_template(
            prompt_template,
            instruction=world.task.instruction,
            main_user=world.task.supervisor,
            app_descriptions=app_descriptions_string,
            max_steps=self.max_steps,
        )
        header_messages = load_prompt_to_chat_messages(
            header_content, skip_system_message=False, only_header=True)
        parts = [m["content"] for m in header_messages if m["role"] == "system"]
        return "\n\n".join(parts)

    def _build_messages(self, world: AppWorld):
        """官方 prompt 流程：demos（文本 few-shot）+ 任务指令（system 走参数）。"""
        from appworld.common.io import read_json
        from autogen_agentchat.messages import TextMessage

        messages: list = []
        if self.demo_messages_file_path:
            demo_messages = read_json(self.demo_messages_file_path)
            # vendored autogen 的消息类型无 tool_calls 字段，官方 OpenAI 格式
            # demo 转成文本 few-shot（保留官方示例内容与工具调用序列）
            lines = ["Below are a few example conversations:", ""]
            for d in demo_messages:
                role = d.get("role")
                if role == "user":
                    lines.append(f"User: {d.get('content', '')}")
                elif role == "assistant":
                    tcs = d.get("tool_calls") or []
                    if tcs:
                        calls = "; ".join(
                            f"{t['function']['name']}({t['function']['arguments']})"
                            for t in tcs)
                        lines.append(f"Assistant: [tool calls: {calls}]")
                    elif d.get("content"):
                        lines.append(f"Assistant: {d.get('content', '')}")
                elif role == "tool":
                    lines.append(f"Tool result: {str(d.get('content', ''))[:300]}")
            messages.append(TextMessage(content="\n".join(lines), source="user"))
        messages.append(TextMessage(
            content=f"# Task Instruction\n{world.task.instruction}", source="user"))
        return messages

    def solve_task(self, task_id: str) -> None:
        from autogen_agentchat.agents import AssistantAgent

        self.usage_tracker.reset(task_id)
        with AppWorld(task_id=task_id) as world:
            self.initialize(world)
            predicted_apis, _ = asyncio.run(
                APIPredictor(**self.api_predictor_config).predict(world.task))
            tools = self._build_tools(world, predicted_apis)
            client = self._build_client()
            system_prompt = self._build_system_prompt(world)
            initial_messages = self._build_messages(world)
            agent = AssistantAgent(
                name="assistant",
                model_client=client,
                tools=tools,
                system_message=system_prompt,
                model_context=None,
                reflect_on_tool_use=True,
                max_tool_iterations=self.max_steps,
            )
            result = asyncio.run(agent.run(task=initial_messages))
            if world.task_completed():
                self.logger.show_message(role="termination", content="task completed")
        self.logger.complete_task()

    def solve_tasks(
        self,
        task_ids: list[str],
        experiment_name: str | None = None,
        num_processes: int = 1,
        process_index: int = 0,
    ) -> None:
        for task_id in task_ids:
            self.solve_task(task_id)


@Agent.register("autogen_agent")
class AutoGenAgent(Agent):
    """注册名 autogen_agent —— 与官方 config 的 agent.type 对应。"""
