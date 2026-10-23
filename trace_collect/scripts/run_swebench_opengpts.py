"""
OpenGPTs trace collection driver.

Zero agent source changes -- the opengpts source (source_code/swe_agents/opengpts/)
is untouched:
- Integration: opengpts' langgraph MessageGraph (get_tools_agent_executor:
  an agent node + a tool loop node). The harness builds the executor itself:
  ChatOpenAI points at the DeepSeek gateway (api key via OPENAI_API_KEY), and
  InMemorySaver replaces the official default Postgres checkpoint (no Postgres
  on this machine).
- LLM: ChatOpenAI goes through the OpenAI SDK (not litellm) → record the llm
  span via the langchain callback system (attached to the harness-constructed
  model instance, same approach as the deepagents driver).
- Tools: opengpts' built-in tools are all network APIs (Arxiv/DDG/Wikipedia
  etc.) with no shell/file tools -- the harness injects its own bash wrapper
  tool (runs commands in the repo directory and records tool.execution spans,
  same approach as the deepagents driver).

Run:
    python scripts/run_swebench_opengpts.py [--task ID | --tasks A,B] [--limit N]
        [--mirror PREFIX] [--out DIR]
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
OPENGPTS_ROOT = ROOT / "source_code" / "swe_agents" / "opengpts" / "backend"
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"), str(OPENGPTS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Windows: default GBK after redirection (same convention as the other drivers)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from swebench_common import record_swe_tool, run_tasks  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL, THINKING_MODE  # noqa: E402
from adapter.swebench import SWETask  # noqa: E402

SYSTEM_PROMPT = (
    "You are a helpful assistant that solves software engineering tasks. "
    "Work in the provided repository. You have a tool to run bash commands "
    "in the repository. Inspect files and make edits with bash commands. "
    "When done, stop calling tools."
)


def _shim_langgraph_tools() -> None:
    """The old langgraph ToolExecutor/ToolInvocation were removed in newer
    versions -- inject an equivalent implementation before importing opengpts
    (only touches third-party library runtime objects; opengpts is unchanged)."""
    import langgraph.prebuilt as prebuilt
    if hasattr(prebuilt, "ToolExecutor"):
        return
    from dataclasses import dataclass

    @dataclass
    class ToolInvocation:
        tool: str
        tool_input: Any

    class ToolExecutor:
        def __init__(self, tools):
            self._tools = {t.name: t for t in tools}

        async def abatch(self, invocations):
            import asyncio
            out = []
            for inv in invocations:
                tool = self._tools.get(inv.tool)
                if tool is None:
                    out.append(f"Error: tool {inv.tool} not found")
                    continue
                try:
                    if hasattr(tool, "ainvoke"):
                        r = await tool.ainvoke(inv.tool_input)
                    else:
                        r = await asyncio.to_thread(tool.invoke, inv.tool_input)
                    out.append(str(r))
                except Exception as e:
                    out.append(f"Error: {type(e).__name__}: {e}")
            return out

    prebuilt.ToolExecutor = ToolExecutor
    prebuilt.ToolInvocation = ToolInvocation


def launch(task: SWETask, repo_dir: Path, collector) -> dict | None:
    _shim_langgraph_tools()
    try:
        from app.agent_types.tools_agent import get_tools_agent_executor
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from langchain_core.tools import tool as langchain_tool
        from langchain_core.callbacks import BaseCallbackHandler
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError:
        raise RuntimeError("OpenGPTs source not ready: run scripts/fetch_swe_agents.py first")

    # LLM interception: langchain callbacks (the model instance is constructed by the harness; zero agent source changes)
    class TraceCallbackHandler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            try:
                gens = response.generations[0]
                usage = getattr(response, "llm_output", {}) or {}
                tok = (usage.get("token_usage") or {})
                collector.log_llm(
                    model=MAIN_MODEL_NAME,
                    messages=[{"role": "user", "content": "(langchain message stream)"}],
                    output_text=str(gens[0].text if hasattr(gens[0], "text") else gens[0])[:5000],
                    input_tokens=tok.get("prompt_tokens", 0),
                    output_tokens=tok.get("completion_tokens", 0),
                )
            except Exception:
                pass

    llm = ChatOpenAI(model=MAIN_MODEL_NAME,
                     base_url=MAIN_BASE_URL,
                     temperature=0.0,
                     model_kwargs={"extra_body": THINKING_MODE, "seed": 100},
                     callbacks=[TraceCallbackHandler()])

    def trace_bash(command: str) -> str:
        """Run a bash command in the repo directory and record the tool span."""
        t0 = time.perf_counter()
        try:
            p = subprocess.run(command, shell=True, cwd=repo_dir,
                               capture_output=True, text=True, timeout=300,
                               encoding="utf-8", errors="replace")
            out = (p.stdout or "") + (p.stderr or "")
            ok = p.returncode == 0
        except Exception as e:
            out = f"Error: {type(e).__name__}: {e}"
            ok = False
        record_swe_tool(collector, "bash", {"command": command},
                        out[:2000], (time.perf_counter() - t0) * 1000, ok)
        return out

    bash_tool = langchain_tool(trace_bash)
    executor = get_tools_agent_executor(
        [bash_tool], llm, SYSTEM_PROMPT, False, InMemorySaver())
    result = asyncio.run(executor.ainvoke(
        [HumanMessage(content=task.problem_statement)],  # MessageGraph state = message list
        config={"recursion_limit": 30,
                "configurable": {"thread_id": task.instance_id}},
    ))
    n = len(result) if isinstance(result, list) else len(result.get("messages", []))
    return {"messages": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenGPTs trace collection")
    parser.add_argument("--task")
    parser.add_argument("--tasks", help="comma-separated multiple tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mirror", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    run_tasks("opengpts", launch,
              ([args.task] if args.task else None)
              or (args.tasks.split(",") if args.tasks else None),
              args.limit, mirror_prefix=args.mirror, output_dir=args.out)


if __name__ == "__main__":
    main()
