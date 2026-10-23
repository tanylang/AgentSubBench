"""
Tool generator — generates tools in multi-framework formats from world.task.api_docs.

Single source of truth: world.task.api_docs.function_calling()
  → each API: {name: "app__api", description, parameters(JSON Schema)}

Format conversion per framework:
  oai        : native OpenAI function calling dict (used directly by simplified/legacy)
  autogen    : async Python functions (AutoGen wraps them into FunctionTool automatically)
  smolagents : Tool objects (used directly by smolagents ToolCallingAgent/CodeAgent)
  openai_sdk : native OAI dict + FunctionTool (function_tool wrapper of the openai-agents SDK)
  tau2       : tau2-bench Tool objects (as_tool wrapper)

All generated tools internally call executor(app, api, kwargs) — i.e. adapter.execute —
so they uniformly go through world.execute() and are captured by the trace automatically.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable


def _oai_tool_defs(world: Any, allowed_apis: list[str]) -> list[dict]:
    """Native OAI function calling format (passed directly to the LLM by simplified/legacy)."""
    allowed = set(allowed_apis)
    defs = []
    for doc in world.task.api_docs.function_calling():
        fn = doc["function"]
        if fn["name"] in allowed or fn["name"].startswith("supervisor__"):
            defs.append(doc)
    return defs


def _make_autogen_func(
    name: str, desc: str, params: dict, executor: Callable
) -> Any:
    """Generate an async Python function (AutoGen format).

    executor: (app_name, api_name, kwargs) -> str
    """
    props = params.get("properties", {})
    required = set(params.get("required", []))
    app_name, api_name = name.split("__", 1)

    # Parameter signature: required params first (Python syntax requires it)
    req_items = [(k, v) for k, v in props.items() if k in required]
    opt_items = [(k, v) for k, v in props.items() if k not in required]

    arg_parts = []
    for pname, pinfo in req_items + opt_items:
        pt = {"string": "str", "integer": "int", "number": "float",
              "boolean": "bool"}.get(pinfo.get("type", "string"), "str")
        default = pinfo.get("default")
        if default is not None:
            if isinstance(default, bool):
                arg_parts.append(f"{pname}: {pt} = {str(default)}")
            else:
                arg_parts.append(f"{pname}: {pt} = {json.dumps(default)}")
        elif pname not in required:
            arg_parts.append(f"{pname}: {pt} = None")
        else:
            arg_parts.append(f"{pname}: {pt}")

    conds = "\n".join(
        f'    if {p} is not None: kw["{p}"] = {p}'
        for p, _ in req_items + opt_items
    )
    args_str = ", ".join(arg_parts)

    src = f'''async def {name}({args_str}) -> str:
    """{desc}"""
    kw = {{}}
{conds}
    return executor("{app_name}", "{api_name}", kw)
'''
    ns = {"executor": executor}
    exec(src, ns)
    fn = ns[name]
    fn.__doc__ = desc
    return fn


def _make_smolagents_tool(
    name: str, desc: str, params: dict, executor: Callable
) -> Any:
    """Generate a smolagents Tool object."""
    from smolagents import Tool

    props = params.get("properties", {})
    required = set(params.get("required", []))
    app_name, api_name = name.split("__", 1)

    inputs = {}
    for pname, pinfo in props.items():
        inputs[pname] = {
            "type": pinfo.get("type", "string"),
            "description": pinfo.get("description", pname),
        }
        if "default" in pinfo:
            inputs[pname]["default"] = pinfo["default"]
        if pname in required:
            inputs[pname]["required"] = True

    class _AppWorldTool(Tool):
        name = name
        description = desc
        inputs = inputs
        output_type = "string"

        def forward(self, **kwargs) -> str:
            return executor(app_name, api_name, kwargs)

    return _AppWorldTool()


def _make_tau2_tool(
    name: str, desc: str, params: dict, executor: Callable
) -> Any:
    """Generate a tau2-bench Tool object (as_tool format)."""
    app_name, api_name = name.split("__", 1)

    def _impl(**kwargs) -> str:
        return executor(app_name, api_name, kwargs)

    # tau2's as_tool requires a pydantic model as the parameter
    from pydantic import create_model
    props = params.get("properties", {})
    fields = {}
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
    for pname, pinfo in props.items():
        ptype = type_map.get(pinfo.get("type", "string"), str)
        default = pinfo.get("default")
        if default is not None or pname not in set(params.get("required", [])):
            fields[pname] = (ptype, default)
        else:
            fields[pname] = (ptype, ...)

    model = create_model(f"{name}_args", **fields)

    from tau2.environment.tool import as_tool

    def _wrap(**kwargs) -> str:
        return _impl(**kwargs)

    tool = as_tool(_wrap)
    tool.name = name
    tool.description = desc
    return tool


def generate_tools(
    framework: str,
    world: Any,
    allowed_apis: list[str],
    executor: Callable,
) -> list:
    """Generate the tool list for a framework. executor(app, api, kwargs) -> str."""
    allowed = set(allowed_apis)

    def _iter_docs():
        for doc in world.task.api_docs.function_calling():
            fn = doc["function"]
            if fn["name"] in allowed or fn["name"].startswith("supervisor__"):
                yield fn["name"], fn["description"], fn["parameters"]

    if framework == "oai":
        return _oai_tool_defs(world, allowed_apis)

    if framework == "autogen":
        return [
            _make_autogen_func(n, d, p, executor) for n, d, p in _iter_docs()
        ]

    if framework == "smolagents":
        return [
            _make_smolagents_tool(n, d, p, executor) for n, d, p in _iter_docs()
        ]

    if framework == "tau2":
        return [
            _make_tau2_tool(n, d, p, executor) for n, d, p in _iter_docs()
        ]

    if framework == "openai_sdk":
        # the openai-agents SDK wraps callables with its native function_tool
        defs = []
        for n, d, p in _iter_docs():
            app_name, api_name = n.split("__", 1)

            def _impl(app=app_name, api=api_name, **kwargs):
                return executor(app, api, kwargs)

            from agents import function_tool
            _impl.__name__ = n
            _impl.__doc__ = d
            defs.append(function_tool(_impl))
        return defs

    raise ValueError(f"Unknown framework: {framework}")


# ── AppWorld tool execution / parsing (single tool-tracing entry point, canonical form, decoupled from the collector) ──

def _parse_dict(d: str) -> dict:
    """Parse a dict string that may use single quotes (Python repr)."""
    d = d.strip()
    try:
        return json.loads(d)
    except json.JSONDecodeError:
        pass
    try:
        fixed = re.sub(r"(?<!\\)'", '"', d)
        return json.loads(fixed)
    except json.JSONDecodeError:
        return {"raw": d}


def _parse_kwargs(args_str: str) -> dict:
    """Parse Python keyword arguments like key=value, key=\"value\" into a dict."""
    result = {}
    for m in re.finditer(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[^,)]+)', args_str):
        val = m.group(2).strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        result[m.group(1)] = val
    return result if result else {"raw": args_str[:500]}


def parse_appworld_tool_code(code_str: str) -> dict:
    """Parse an AppWorld tool code string into {tool_name, arguments} (canonical form).

    tool_name uses dot separation (app.api); arguments is a bare argument dict (not
    wrapped in kwargs). This is the only AppWorld-specific parsing logic; it is kept
    out of the generic collector.
    """
    s = str(code_str).strip()
    result = {"tool_name": "appworld.execute", "arguments": {}}

    m = re.search(r"apis\.(\w+)\.(\w+)\(\*\*(\{.*?\})\)", s, re.DOTALL)
    if m:
        result["tool_name"] = f"{m.group(1)}.{m.group(2)}"
        result["arguments"] = _parse_dict(m.group(3))
        return result

    m = re.search(r"(\w+)__(\w+)\(\*\*(\{.*?\})\)", s, re.DOTALL)
    if m:
        result["tool_name"] = f"{m.group(1)}.{m.group(2)}"
        result["arguments"] = _parse_dict(m.group(3))
        return result

    m = re.search(r"apis\.(\w+)\.(\w+)\(([^)]*)\)", s)
    if m:
        result["tool_name"] = f"{m.group(1)}.{m.group(2)}"
        args_str = m.group(3).strip()
        if args_str:
            result["arguments"] = _parse_kwargs(args_str)
        return result

    m = re.search(r"(\w+)__(\w+)\(([^)]*)\)", s)
    if m:
        result["tool_name"] = f"{m.group(1)}.{m.group(2)}"
        args_str = m.group(3).strip()
        if args_str:
            result["arguments"] = _parse_kwargs(args_str)
        return result

    return result


def _result_indicates_failure(result) -> bool:
    """AppWorld's world.execute does not raise: failures are returned as strings.

    The four paths in environment.py — unsafe syntax / syntax error / timeout / runtime
    exception — all return the "Execution failed. Traceback:..." prefix (see
    appworld/environment.py), so the prefix is used to decide whether a tool execution
    truly succeeded (otherwise tool.success would always be True).
    """
    return isinstance(result, str) and result.startswith("Execution failed")


def record_appworld_tool(collector, code, result, duration_ms: float = 0.0,
                         success: bool = True) -> None:
    """Record one AppWorld tool execution as a canonical tool.execution span (does not execute).

    Reused by agents whose execution point is not under our control (the AppWorld.execute
    patch for the builtin Simplified/Legacy agents): after execution yields a result,
    this parses the code and records it canonically.

    success is normalized uniformly: the caller's try/except can only catch real
    exceptions, while AppWorld reports execution failures as return strings (see
    _result_indicates_failure); an extra check here keeps tool.success consistent
    with the result content.
    """
    if not code or not str(code).strip():
        return
    if success and _result_indicates_failure(result):
        success = False
    p = parse_appworld_tool_code(code)
    collector.log_tool(p["tool_name"], p["arguments"], str(result), duration_ms, success)


def appworld_execute_tool(world, collector, code) -> str:
    """Execute + record: the single entry point for AppWorld tool execution (for agents
    that control their own call sites).

    For agents like AutoGen/Tau2 whose tool functions are generated by the benchmark
    itself: generate the code and call here; it executes world.execute(code) internally
    and records canonically, so there is no need to patch world.execute.
    """
    # Duration measurement uses a freezegun-immune monotonic clock (time.perf_counter is frozen inside the AppWorld world)
    from adapter.clock import monotonic_ms
    t0 = monotonic_ms()
    try:
        result = world.execute(code)
        ok = not _result_indicates_failure(result)
    except Exception as e:
        result = str(e)
        ok = False
    record_appworld_tool(collector, code, result, monotonic_ms() - t0, ok)
    return result
