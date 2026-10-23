"""
DeepAgents trace collection driver.

Zero agent source changes:
- LLM: DeepAgents (langchain ecosystem) goes through init_chat_model("deepseek-chat", ...)
  → underlying litellm → intercepted by the global callbacks
- Tools: deepagents' built-in ShellTool / file tools; the driver takes git
  checkpoints at the repository directory level (assembled by swebench_common),
  and tool-level spans are additionally recorded by injecting our wrapped
  ShellTool (create_deep_agent accepts an extra tools parameter)

Run:
    python scripts/run_swebench_deepagents.py [--task ID | --limit N] [--mirror PREFIX]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Note: deepagents uses a monorepo layout; the python package lives in libs/deepagents/ (not src/)
AGENT_SRC = ROOT / "source_code" / "swe_agents" / "deepagents" / "libs" / "deepagents"
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter"), str(AGENT_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from swebench_common import record_swe_tool, run_tasks  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL, THINKING_MODE  # noqa: E402
from adapter.swebench import SWETask  # noqa: E402


def launch(task: SWETask, repo_dir: Path, collector) -> dict | None:
    try:
        from deepagents import create_deep_agent
        from langchain.chat_models import init_chat_model
        from langchain_core.tools import tool as langchain_tool
    except ImportError:
        raise RuntimeError("DeepAgents source not ready: run scripts/fetch_swe_agents.py first")

    model = init_chat_model(MAIN_MODEL_NAME, model_provider="openai",
                            base_url=MAIN_BASE_URL,
                            model_kwargs={"extra_body": THINKING_MODE, "seed": 100},
                            temperature=0.0)  # api_key is provided by the OPENAI_API_KEY environment variable

    # langchain goes through the openai SDK (not litellm) -- use langchain's own
    # callback system, attached to **the model instance we construct**, to
    # intercept LLM calls (zero agent source changes)
    from langchain_core.callbacks import BaseCallbackHandler

    class TraceCallbackHandler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            try:
                gens = response.generations[0]
                usage = getattr(response, "llm_output", {}) or {}
                tok = (usage.get("token_usage") or {})
                collector.log_llm(
                    model=MAIN_MODEL_NAME,
                    messages=[{"role": "user", "content": "(langchain message stream; prompt in the run context)"}],
                    output_text=str(gens[0].text if hasattr(gens[0], "text") else gens[0])[:5000],
                    input_tokens=tok.get("prompt_tokens", 0),
                    output_tokens=tok.get("completion_tokens", 0),
                )
            except Exception:
                pass

    model.callbacks = [TraceCallbackHandler()]

    def trace_shell(command: str) -> str:
        """Run a bash command in the repository directory."""
        t0 = time.perf_counter()
        p = subprocess.run(command, shell=True, cwd=repo_dir,
                           capture_output=True, text=True, timeout=300,
                           encoding="utf-8", errors="replace")
        out = (p.stdout or "") + (p.stderr or "")
        record_swe_tool(collector, "bash", {"command": command},
                        out[:2000], (time.perf_counter() - t0) * 1000,
                        p.returncode == 0)
        return out

    shell_tool = langchain_tool(trace_shell)
    agent = create_deep_agent(model=model, tools=[shell_tool])
    prompt = (f"Work in the repository directory and solve the issue. "
              f"Use trace_shell to run commands.\n\nIssue:\n{task.problem_statement}")
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    out = ""
    for m in result.get("messages", []):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            out = m.content
    return {"final_message": out[:2000]}


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepAgents trace collection")
    parser.add_argument("--task")
    parser.add_argument("--tasks", help="comma-separated multiple tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mirror", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    run_tasks("deepagents", launch,
              ([args.task] if args.task else None)
              or (args.tasks.split(",") if args.tasks else None),
              args.limit, mirror_prefix=args.mirror, output_dir=args.out)


if __name__ == "__main__":
    main()
