# AgentSubBench

Multi-agent control-flow trace collection and evaluation on three benchmarks:
**AppWorld** (API-calling tasks), **LocoMo** (long-term conversational memory QA),
and **SWE-bench Verified** (software engineering tasks).

Every collected trace contains environment snapshots (`env_changes` / `check_point`),
protocol-level LLM requests with `reasoning` content, tool execution records with
accurate success flags, and per-task `pass` results from the official evaluators.

Design principle: **zero modification of agent logic**. Traces are captured
passively via source-level hook points plus an externally registered collector;
prompts, message flow, tool execution, and termination conditions remain identical
to the upstream implementations.

---

## Repository Layout

```
adapter/            hook registry / tool execution & recording / model config / LocoMo env
scripts/            AppWorld agent drivers + LocoMo drivers + SWE-bench drivers + trace_collector
runners/            run_task_cli (collection entry) + offline evaluation + LocoMo validator (judge) + replay
langgraph_locomo/   LocoMo LangGraph + LangMem two-phase memory pipeline (ingestion -> QA)
autogen_locomo/     LocoMo AutoGen collection pipeline
benchmarks/         AppWorld / SWE-bench / LocoMo benchmarks and official evaluation code
tasks/              task lists and offline data (appworld_tasks / locomo_tasks / swebench_tasks)
source_code/        vendored frameworks: patched autogen (trace hook), tau2-bench, 7 SWE agent sources
papers/             LocoMo paper (ACL 2024)
calibration/        judge calibration samples and verdicts
```

> Collected traces (`traces/`), SWE-bench workspaces, and pipeline runtime data are
> excluded via `.gitignore` — collection artifacts stay local and are never committed.

## Quick Start (Local)

```bash
# 1. Configure the model API key
cp config.example.json config.json      # fill in deepseek_api_key (never commit config.json)

# 2. Install dependencies
pip install -e benchmarks/appworld/experiments
pip install litellm==1.89.7 autogen-agentchat autogen-core autogen-ext openai-agents smolagents
```

Models are injected via environment variables (defaults point to DeepSeek):

| Variable | Meaning |
|---|---|
| `MODEL_NAME` / `MODEL_BASE_URL` | main-loop model |
| `PREDICTOR_MODEL_NAME` / `PREDICTOR_BASE_URL` | API predictor model |
| `MODEL_THINKING` | thinking-mode switch (on/off) |
| `JUDGE_MODEL` / `JUDGE_BASE_URL` | LocoMo LLM-judge model |

## AppWorld

Six built-in agents, unified `temperature=0 / seed=100 / max_tokens=8192`:

| Agent | Driver | Framework |
|---|---|---|
| Simplified | `scripts/run_benchmark_builtin_simplified.py` | AppWorld SimplifiedFunctionCallingAgent |
| AutoGen | `scripts/run_benchmark_autogen.py` | AutoGen AssistantAgent (vendored, patched) |
| Smolagents | `scripts/run_benchmark_builtin_smolagents.py` | smolagents ToolCallingAgent |
| Legacy | `scripts/run_benchmark_builtin_legacy.py` | AppWorld Legacy FunctionCallingAgent |
| Tau2 | `scripts/run_benchmark_tau2_llmagent.py` | tau2-bench LLMAgent |
| OpenAI | `scripts/run_benchmark_builtin_oai.py` | openai-agents SDK + MCP |

### Collection

```bash
# single task
python runners/run_task_cli.py scripts.run_benchmark_tau2_llmagent 0

# full run, sequential / N-way parallel (resumable: existing files are skipped)
python runners/run_task_cli.py scripts.run_benchmark_tau2_llmagent all 4
```

Example with a custom model endpoint:

```bash
MODEL_NAME=Qwen3.8-27B MODEL_BASE_URL=http://<host>:8101/v1 \
PREDICTOR_MODEL_NAME=Qwen3.8-27B PREDICTOR_BASE_URL=http://<host>:8101/v1 \
python runners/run_task_cli.py scripts.run_benchmark_autogen all 4
```

### Evaluation

```bash
# offline evaluation (writes back pass + annotates event_type)
python runners/finalize_traces.py traces/appworld_tau2_batch1
```

## LocoMo (Conversational Memory QA)

### Six framework drivers (retrieval-based tool agents)

```bash
# smoke test: one sample, 3 questions
LOC_SAMPLE_IDS=conv-26 LOC_QA_LIMIT=3 \
python runners/run_task_cli.py scripts.run_locomot_simplified 0

# full run, 2-way parallel
python runners/run_task_cli.py scripts.run_locomot_simplified all 2
```

Drivers: `scripts/run_locomot_{simplified,autogen,smolagents,legacy,tau2,oai}.py`.
Output goes to `traces/locomo_<agent>_batch1/`.

### LangGraph + LangMem pipeline (two-phase memory agent)

Phase 1 — ingestion (build memories from every conversation message):

```bash
python langgraph_locomo/scripts/09_ingest_locomo_full.py \
  --data benchmarks/locomo/data/locomo10.json \
  --sample-id conv-26 --session-limit 19 \
  --base-url http://<host>:<port>/v1 --model <model-name> --api-key EMPTY \
  --output traces/langgraph_ingest/conv-26/ingestion.json \
  --snapshot traces/langgraph_snapshot/conv-26/baseline_memory.json \
  --progress traces/langgraph_snapshot/conv-26/ingestion_progress.json \
  --resume
```

Phase 2 — QA (answer every question from the memory snapshot):

```bash
python langgraph_locomo/scripts/10_run_locomo_sample.py \
  --data benchmarks/locomo/data/locomo10.json \
  --ingestion-trace traces/langgraph_ingest/conv-26/ingestion.json \
  --snapshot traces/langgraph_snapshot/conv-26/baseline_memory.json \
  --progress traces/langgraph_snapshot/conv-26/qa_progress.json \
  --partial-output traces/langgraph_samples/conv-26.partial.json \
  --output traces/langgraph_samples/conv-26.json \
  --base-url http://<host>:<port>/v1 --model <model-name> --api-key EMPTY \
  --stop-after 199 --checkpoint-every 5 --resume
```

Environment overrides for the LangGraph pipeline:

| Variable | Default | Meaning |
|---|---|---|
| `LOC_MAX_TOKENS` | 8192 | output token cap; `0` omits the cap (lets vLLM use the remaining window — use for small-window models such as Llama-3 8k) |
| `LOC_TIMEOUT` | 180/300 | request timeout in seconds (raise for slow models) |

### Evaluation

```bash
# official token-F1 (same construction as the official scorer)
python runners/finalize_locomot.py traces/locomo_simplified_batch1

# F1 + LLM-judge review for verbose answers (judge via JUDGE_MODEL/JUDGE_BASE_URL)
python runners/validate_locomot.py traces/locomo_simplified_batch1
```

## SWE-bench Verified

Seven agent drivers (`scripts/run_swebench_*.py`; agent sources under
`source_code/swe_agents/`, unmodified):

```bash
python scripts/run_swebench_sweagent.py --tasks "$(cat tasks/swebench_tasks/swe50_ids.txt)" \
    --out traces/swebench_sweagent

# evaluation (local protocol / official docker protocol)
python runners/finalize_swe_traces.py traces/swebench_sweagent --agent swe-agent
python runners/finalize_swe_traces.py traces/swebench_sweagent --agent swe-agent --mode docker
```

## Data & Privacy

- All trace artifacts live under `traces/` and are excluded by `.gitignore` —
  this repository ships collection and evaluation code only;
- `config.json` (API key) is never committed; see `config.example.json`;
- Patches to vendored sources are minimal: one `_trace_tracer` hook in
  `source_code/autogen` and one sentinel in `source_code/tau2-bench`; everything
  else matches upstream.

## License

See [LICENSE](LICENSE).
