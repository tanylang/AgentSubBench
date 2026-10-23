"""
mini-SWE-agent trace collection driver (reference implementation).

Zero agent source changes:
- Tool capture: this driver builds an **external wrapper subclass** of
  LocalEnvironment, TraceEnvironment, whose execute() records first and then
  calls the parent implementation -- the agent source (source_code/swe_agents/
  mini-swe-agent/) is untouched
- LLM capture: the litellm global callbacks from
  swebench_common.install_litellm_hook (mini-swe-agent's Model goes through litellm)

Run:
    python scripts/run_swebench_minisweagent.py [--task <id> | --limit N]
    [--mirror https://ghfast.top/] [--out traces/swebench_mini-swe-agent]

Note: LocalEnvironment runs bash directly on the host machine -- installing/
testing SWE-bench repos carries arbitrary code execution risk. For production
collection use the docker environment instead (minisweagent's DockerEnvironment
also only needs a driver-level wrapper; the interface is identical).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
AGENT_SRC = ROOT / "source_code" / "swe_agents" / "mini-swe-agent" / "minisweagent"
for p in (str(AGENT_SRC.parent), str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

from swebench_common import record_swe_tool, run_tasks  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL  # noqa: E402
from adapter.swebench import SWETask  # noqa: E402

try:
    from minisweagent.environments.local import LocalEnvironment  # noqa: E402
    from minisweagent.agents.default import DefaultAgent  # noqa: E402
    from minisweagent.models.litellm_model import LitellmModel  # noqa: E402
except ImportError as _e:
    print(f"Error: mini-swe-agent source not ready ({type(_e).__name__}: {_e})")
    print(f"  expected location: {AGENT_SRC.parent}")
    print("  fix: 1) sync the whole package (rsync -av trace_collect/ ...) to ensure source_code/ is complete;")
    print("        2) or run python scripts/fetch_swe_agents.py to re-fetch")
    sys.exit(1)

SYSTEM_TEMPLATE = (
    "You are a helpful assistant that solves software engineering tasks. "
    "Work in the provided repository. You have tools to run bash commands, "
    "view and edit files. When done, submit the final answer with "
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT."
)
INSTANCE_TEMPLATE = "Task:\n{{ task }}\n\nRepository: {{ repo }}"


class TraceEnvironment(LocalEnvironment):
    """External wrapper: record the tool.execution span first, then execute (parent logic unchanged)."""

    def __init__(self, collector, task_id, *args, **kwargs):
        self._collector = collector
        self._task_id = task_id
        super().__init__(*args, **kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        t0 = time.perf_counter()
        # Normalize the tool name: command → bash; other action keys are used directly (edit/view/create etc.)
        name, arguments = "bash", dict(action)
        if "command" in action:
            arguments = {"command": action["command"]}
        elif "action" in action:
            name = str(action["action"])
            arguments = {k: v for k, v in action.items() if k != "action"}
        try:
            result = super().execute(action, cwd=cwd, timeout=timeout)
            ok = result.get("returncode", 0) == 0
        except Exception as e:
            result = {"output": str(e), "returncode": -1}
            ok = False
        record_swe_tool(self._collector, name, arguments, result,
                        (time.perf_counter() - t0) * 1000, ok)
        return result


def launch(task: SWETask, repo_dir: Path, collector) -> dict | None:
    env = TraceEnvironment(collector, task.instance_id, cwd=str(repo_dir))
    # litellm routing: OpenAI-compatible endpoint + DeepSeek gateway (api_key injected by swebench_common)
    model = LitellmModel(model_name=f"openai/{MAIN_MODEL_NAME}",
                         model_kwargs={"api_base": MAIN_BASE_URL,
                                       "temperature": 0.0,
                                       "seed": 100})
    agent = DefaultAgent(
        model, env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        step_limit=50,
        cost_limit=5.0,
    )
    result = agent.run(
        task=task.problem_statement,
        repo=task.repo.split("/")[-1],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-SWE-agent trace collection")
    parser.add_argument("--task", help="run a single instance_id only")
    parser.add_argument("--tasks", help="comma-separated multiple tasks")
    parser.add_argument("--limit", type=int, default=0, help="run only the first N (0=all)")
    parser.add_argument("--mirror", default="", help="GitHub mirror prefix (e.g. https://ghfast.top/)")
    parser.add_argument("--out", default="", help="trace output directory")
    args = parser.parse_args()

    run_tasks(
        agent_name="mini-swe-agent",
        launch=launch,
        task_ids=([args.task] if args.task else None)
        or (args.tasks.split(",") if args.tasks else None),
        limit=args.limit,
        mirror_prefix=args.mirror,
        output_dir=args.out,
    )


if __name__ == "__main__":
    main()
