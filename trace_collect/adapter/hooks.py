"""
Global hook registry — hook call sites are instrumented into agent source code; the adapter layer provides the implementation.

Design principles:
1. Agent source code only gains hook calls (one line each), with no changes to any original logic
2. Hooks are no-ops by default (zero overhead when no collector is registered)
3. Once a collector is registered at runtime, all hooks automatically write to the unified trace
4. Same pattern as the AutoGen framework's built-in trace_tool_span:
   hook points live in the source, while the tracing implementation is injected externally

Instrumentation form in agent source code (example):

    from adapter.hooks import llm_hook, tool_hook

    # LLM call site (after the framework calls the LLM):
    result = model.generate(messages, ...)          # original logic, unchanged
    llm_hook.record_llm(model, messages, result)   # new hook line

    # Tool execution site (after the framework executes the tool):
    result = world.execute(code)                   # original logic, unchanged
    tool_hook.record_tool(code, result)            # new hook line
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional


class _Hook:
    """A single hook type: LLM call or tool execution."""

    def __init__(self, kind: str):
        self._kind = kind
        self._collector: Optional[Any] = None
        self._lock = threading.Lock()

    def register(self, collector: Any) -> None:
        """Register a collector (trace collector) at runtime."""
        with self._lock:
            self._collector = collector

    def unregister(self) -> None:
        with self._lock:
            self._collector = None

    def __call__(self, *args, **kwargs):
        """No-op with zero overhead when no collector is registered."""
        with self._lock:
            collector = self._collector
        if collector is None:
            return
        try:
            if self._kind == "llm":
                _record_llm(collector, *args, **kwargs)
            elif self._kind == "tool":
                _record_tool(collector, *args, **kwargs)
        except Exception:
            pass  # hooks never interfere with agent logic


def _record_llm(collector, model: str, messages: list, result: Any,
                duration_ms: float = 0.0, tools: Any = None,
                tool_calls: list | None = None,
                reasoning: str | None = None) -> None:
    """Extract (output, tokens, reasoning) from each framework's LLM return object and write it to the collector.

    Messages are passed in full (truncation is handled inside the collector: attributes
    store a condensed summary, while check_point.conversation stores the full conversation).
    tools: the tools argument at the request protocol layer (as-is)
    tool_calls: the structured tool_calls array from the response
    reasoning: the thinking content in the response (reasoning_content); auto-extracted from result when None
    """
    out_text = _extract_output(result)
    i_tok, o_tok = _extract_tokens(result)
    if reasoning is None:
        reasoning = _extract_reasoning(result)
    msgs = [
        {"role": str(getattr(m, "role", "unk")),
         "content": str(getattr(m, "content", None) or "")}
        for m in (messages if isinstance(messages, list) else [])
    ]
    collector.log_llm(
        model=model, messages=msgs, output_text=out_text,
        input_tokens=i_tok, output_tokens=o_tok,
        tools=tools, tool_calls=tool_calls,
        reasoning=reasoning or "",
        duration_ms=duration_ms,
    )


def _record_tool(collector, tool_name: str, arguments: dict, result: Any,
                 duration_ms: float = 0.0, success: bool = True) -> None:
    """Write a tool.execution span (tool_name/arguments are already in canonical form, independent of the benchmark)."""
    collector.log_tool(tool_name, arguments, str(result), duration_ms, success)


# ── Global hook instances (agent source code imports these two)──────────────────────
llm_hook = _Hook("llm")
tool_hook = _Hook("tool")


# ── Extraction helper functions ───────────────────────────────────
def _extract_output(result: Any) -> str:
    """Extract the output text from each framework's LLM return object (consistent with the existing trace format)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):        # AutoGen CreateResult / smolagents ChatMessage
        c = result.content
        if isinstance(c, str) and c.strip():
            return c
        # smolagents ChatMessage: content is empty during a tool call; tool_calls holds the data
        tcs = getattr(result, "tool_calls", None)
        if tcs:
            out = []
            for t in tcs:
                fn = getattr(t, "function", t)
                out.append({
                    "name": getattr(fn, "name", getattr(t, "name", "")),
                    "arguments": getattr(fn, "arguments", getattr(t, "arguments", "")),
                })
            if out:
                return json.dumps(out, ensure_ascii=False)
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return json.dumps(
                [{"name": x.name, "arguments": x.arguments} for x in c],
                ensure_ascii=False)
        return str(c)
    if hasattr(result, "choices") and result.choices:  # OpenAI format
        msg = result.choices[0].message
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            return json.dumps(
                [{"name": t.function.name, "arguments": t.function.arguments}
                 for t in tcs], ensure_ascii=False)
        return getattr(msg, "content", "") or ""
    if isinstance(result, dict):
        choices = result.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("tool_calls", msg.get("content", ""))
            if isinstance(content, str):
                return content  # already a string (avoid double JSON serialization)
            return json.dumps(content, ensure_ascii=False, default=str)
        return str(result.get("content") or "")
    return str(result)


def _extract_tokens(result: Any) -> tuple:
    """Extract (input_tokens, output_tokens), compatible with each framework's usage structure."""
    usage = None
    if hasattr(result, "usage"):
        usage = result.usage
    elif hasattr(result, "token_usage"):  # smolagents ChatMessage
        usage = result.token_usage
    elif isinstance(result, dict):
        usage = result.get("usage")
    if usage is None:
        return 0, 0
    if hasattr(usage, "prompt_tokens"):
        return usage.prompt_tokens or 0, usage.completion_tokens or 0
    if hasattr(usage, "input_tokens"):  # smolagents TokenUsage
        return getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0
    if isinstance(usage, dict):
        return (usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                usage.get("completion_tokens", 0) or usage.get("output_tokens", 0))
    return 0, 0


def _extract_reasoning(result: Any) -> str:
    """Extract the thinking content (reasoning_content / reasoning / thought) from each framework's response.

    Compatible with: litellm/openai SDK response objects and model_dump dicts (on
    choices[0].message), smolagents ChatMessage (reasoning_content attribute), and
    autogen CreateResult (thought).
    """
    if result is None:
        return ""
    # The response itself carries reasoning_content / thought (smolagents ChatMessage, autogen CreateResult)
    for attr in ("reasoning_content", "reasoning"):
        v = getattr(result, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    thought = getattr(result, "thought", None)
    if isinstance(thought, str) and thought.strip():
        return thought
    # OpenAI format: choices[0].message (object or dict)
    msg: Any = None
    if hasattr(result, "choices") and getattr(result, "choices", None):
        msg = result.choices[0].message
    elif isinstance(result, dict) and result.get("choices"):
        msg = result["choices"][0].get("message") or {}
    if msg is None:
        return ""
    if isinstance(msg, dict):
        r = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return str(r) if r else ""
    for attr in ("reasoning_content", "reasoning"):
        v = getattr(msg, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    # The openai SDK may collect unknown fields into model_extra / provider_specific_fields
    # (DeepSeek uses reasoning_content, vLLM Qwen3 uses reasoning)
    for extra_attr in ("model_extra", "provider_specific_fields"):
        extra = getattr(msg, extra_attr, None)
        if isinstance(extra, dict):
            r = extra.get("reasoning_content") or extra.get("reasoning")
            if r:
                return str(r)
    return ""


# ── Decorator form (an optional second instrumentation style in source code)────────────────────────
def traced_llm(fn: Callable, model: str = "") -> Callable:
    """Decorator: wraps an LLM call function and records automatically after the call."""
    import functools
    import asyncio

    @functools.wraps(fn)
    def sync_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        llm_hook(model, _find_messages(args, kwargs), result,
                 duration_ms=(time.perf_counter() - t0) * 1000)
        return result

    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = await fn(*args, **kwargs)
        llm_hook(model, _find_messages(args, kwargs), result,
                 duration_ms=(time.perf_counter() - t0) * 1000)
        return result

    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper


def traced_tool(fn: Callable) -> Callable:
    """Decorator: wraps a tool execution function and records automatically after execution."""
    import functools
    import asyncio

    @functools.wraps(fn)
    def sync_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            tool_hook(str(args[0]) if args else str(kwargs), result,
                      (time.perf_counter() - t0) * 1000, True)
            return result
        except Exception as e:
            tool_hook(str(args[0]) if args else str(kwargs), str(e),
                      (time.perf_counter() - t0) * 1000, False)
            raise

    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
            tool_hook(str(args[0]) if args else str(kwargs), result,
                      (time.perf_counter() - t0) * 1000, True)
            return result
        except Exception as e:
            tool_hook(str(args[0]) if args else str(kwargs), str(e),
                      (time.perf_counter() - t0) * 1000, False)
            raise

    return async_wrapper if asyncio.iscoroutinefunction(fn) else sync_wrapper


def _find_messages(args: tuple, kwargs: dict) -> list:
    for k in ("messages", "prompt", "input"):
        if k in kwargs:
            return kwargs[k]
    return list(args[:1]) if args else []
