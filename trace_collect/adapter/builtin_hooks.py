"""
Built-in agent hooks — AppWorld's built-in agents (simplified/legacy/smolagents/openai_agents)
create their own world instances and run their own loops. To avoid modifying their source,
tracing is attached via class-level monkey-patching:

1. Class-level patch of AppWorld.execute → every world instance's tool calls are recorded
   into the collector automatically
2. LanguageModel._TRACE_TRACER sentinel → a shim converts log_llm_call/log_llm_response
   into collector.log_llm (the built-in agents' LLM layer already supports this sentinel)
"""
from __future__ import annotations

import json
import time
from typing import Any

from .tools import record_appworld_tool


class _TracerShim:
    """Convert the AgentTracer interface (log_llm_call/log_llm_response) into the OrderedTraceCollector interface."""

    def __init__(self, collector: Any, model: str):
        self.collector = collector
        self.model = model

    def log_llm_call(self, model: str, messages: list) -> Any:
        """Record the start of an LLM call. Returns a placeholder span that can be passed to log_llm_response."""
        return {"model": model or self.model, "messages": messages, "t0": time.perf_counter()}

    def log_llm_response(self, completion: str, model: str = "",
                         input_tokens: int = 0, output_tokens: int = 0,
                         latency_ms: float = 0.0, llm_call_span: Any = None, **kw) -> None:
        msgs = llm_call_span.get("messages", []) if isinstance(llm_call_span, dict) else []
        self.collector.log_llm(
            model=model or self.model,
            messages=[{"role": str(m.get("role", "unk")),
                       "content": str(m.get("content") or "")[:500]} for m in msgs[-3:]],
            output_text=str(completion)[:5000],
            input_tokens=input_tokens, output_tokens=output_tokens,
        )


def patch_builtin_agent(collector: Any, model: str = "deepseek-chat"):
    """
    Class-level patch for AppWorld's built-in agents. Returns a restore() function.

    Usage:
        restore = patch_builtin_agent(collector)
        agent.solve_task(task_id)   # built-in agent's native loop
        restore()
    """
    from appworld import AppWorld
    from appworld_agents.code.simplified import language_model as lm_simplified

    shim = _TracerShim(collector, model)

    # ── 1. Class-level hook on AppWorld.execute (tool tracing) ──
    orig_execute = AppWorld.execute

    def traced_execute(self, code: str, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = orig_execute(self, code, *args, **kwargs)
            success = True
        except Exception as e:
            result = str(e)
            success = False
        d = (time.perf_counter() - t0) * 1000
        record_appworld_tool(collector, code, result, d, success)
        return result

    AppWorld.execute = traced_execute

    # ── 2. _TRACE_TRACER sentinel (LLM tracing; simplified's LanguageModel already supports it) ──
    orig_sentinels = {}
    for mod in (lm_simplified,):
        orig_sentinels[mod.__name__] = getattr(mod, "_TRACE_TRACER", None)
        mod._TRACE_TRACER = shim

    # legacy's LanguageModel may also have the sentinel (attempt it)
    try:
        from appworld_agents.code.legacy.plain.language_models import openai_language_model as lm_legacy
        orig_sentinels[lm_legacy.__name__] = getattr(lm_legacy, "_TRACE_TRACER", None)
        lm_legacy._TRACE_TRACER = shim
    except ImportError:
        pass

    def restore():
        AppWorld.execute = orig_execute
        for name, val in orig_sentinels.items():
            import sys
            mod = sys.modules.get(name)
            if mod is not None:
                mod._TRACE_TRACER = val

    return restore


class AutoGenTracerShim:
    """Convert AutoGen's source hook (client._trace_tracer) into the OrderedTraceCollector interface.

    AutoGen's OpenAIChatCompletionClient.create() already has hook points instrumented
    in the source (see autogen_ext/models/openai/_openai_client.py), calling in order:
        _tracer.log_llm_call(model, messages, tools, tool_calls)   → returns a placeholder span
        _tracer.log_llm_response(completion, model, tokens, llm_call_span)
        _tracer.log_reasoning(thought=...)
    This shim merges those calls into a single collector.log_llm(). messages are autogen
    LLMMessage objects, converted here into {role, content} (tool results additionally
    carry structured tool_results).
    """

    _TYPE_TO_ROLE = {
        "SystemMessage": "system",
        "UserMessage": "user",
        "AssistantMessage": "assistant",
        "FunctionExecutionResultMessage": "tool",
    }

    def __init__(self, collector: Any, model: str = ""):
        self.collector = collector
        self.model = model

    @classmethod
    def _msg_to_dict(cls, m: Any) -> dict:
        t = str(getattr(m, "type", ""))
        role = cls._TYPE_TO_ROLE.get(t, t or str(getattr(m, "role", "unk")))
        d = {"role": role, "content": str(getattr(m, "content", None) or "")}
        # Tool result message: also record the structured results (protocol-level
        # role=tool message, keeping call_id/name/content instead of relying on text
        # concatenation)
        if t == "FunctionExecutionResultMessage":
            results = []
            for r in (getattr(m, "content", []) or []):
                results.append({
                    "call_id": str(getattr(r, "call_id", "")),
                    "name": str(getattr(r, "name", "")),
                    "content": str(getattr(r, "content", "")),
                })
            if results:
                d["tool_results"] = results
        return d

    def log_llm_call(self, model: str, messages: Any,
                     tools: Any = None, tool_calls: Any = None, **kw) -> Any:
        """Record the start of an LLM call; returns a placeholder span to pass to log_llm_response."""
        return {
            "model": model or self.model,
            "messages": messages,
            "tools": tools,
            "tool_calls": tool_calls,
        }

    def log_llm_response(self, completion: str, model: str = "",
                         input_tokens: int = 0, output_tokens: int = 0,
                         latency_ms: float = 0.0, llm_call_span: Any = None, **kw) -> None:
        sp = llm_call_span if isinstance(llm_call_span, dict) else {}
        raw_msgs = sp.get("messages") or []
        tools = sp.get("tools")
        tool_calls = sp.get("tool_calls")
        struct_calls = None
        if tool_calls:
            struct_calls = [{"id": str(getattr(c, "id", "")),
                             "name": str(getattr(c, "name", "")),
                             "arguments": str(getattr(c, "arguments", ""))}
                            for c in tool_calls]
            out_text = json.dumps(struct_calls, ensure_ascii=False)
        else:
            out_text = str(completion)
        self.collector.log_llm(
            model=model or self.model,
            messages=[self._msg_to_dict(m) for m in raw_msgs],
            output_text=out_text,
            input_tokens=input_tokens, output_tokens=output_tokens,
            tools=tools, tool_calls=struct_calls,
            reasoning=str(kw.get("reasoning") or ""),
        )

    def log_reasoning(self, thought: str = "", plan: str = "",
                      critique: str = "", **kw) -> None:
        # reasoning is already recorded via log_llm_response's reasoning kwarg;
        # this call site is kept only for compatibility with the vendored autogen hook protocol.
        pass
