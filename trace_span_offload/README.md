# IMPACT — Measuring Workflow Responses to Single-Span Model Substitution

> Intervention Measurement and Profiling of Agent-Call Trajectories

In a **stateful LLM agent**, replacing one physical LM invocation (span) changes
subsequent actions, state transitions, and observations. Local output agreement
cannot determine the effect of a substitution — its consequences unfold through
the live continuation. IMPACT provides a workload-agnostic intervention
methodology: starting from a common pre-invocation state, a four-condition
protocol — **C0 (frozen-replay audit) / I (recorded-output continuation) /
B (baseline resampling) / T_span (single-span substitution)** — separates replay
fidelity, continuation variation, baseline resampling, and candidate substitution
effects, and measures five response families (local, downstream, final, collateral
risk, and resources in tokens/time). The resulting conditional response atlas
supports offloading decisions under explicit quality, risk, and resource objectives.

This repository is the open-source code accompanying the paper
*IMPACT: Measuring Conditional Workflow Responses to Single-Span Model
Substitution in Stateful LLM Agents* (ICLR 2027 submission). Full experiment
inventory and all analysis numbers: `docs/EXPERIMENT_DATA_ANALYSIS_RECORD.md`.

## Repository Layout

> ⚠ Directories marked **[local]** are large or contain third-party code/data and
> are excluded via `.gitignore`; directories marked **[not included]** are kept
> private by the authors and available on request.

```
trace_span_offload/
├── README.md                        this document
├── requirements.txt                 Python dependencies
├── config.example.json              config template (base_url / models; keys via env vars;
│                                    config.json is local-only, never commit)
├── span_offload/                    core measurement package
│   ├── schemas.py / span_extraction.py / replay.py / analysis.py / allowlist.py
│   ├── appworld_adapter.py / langgraph_adapter.py / locomo_adapter.py / swebench_adapter.py
│   ├── controlflow_metrics.py / local_equivalence.py / verifiers.py / llm_client.py
│   ├── harness/                     AppWorld six-backend continuation runtime (mock reference impl)
│   └── cli.py                       extract / replay / analyze / validate CLI
├── scripts/
│   ├── analysis/                    20+ analysis scripts: main comparisons, ex-ante features,
│   │                                stratified offloadability, control flow, dynamics, DiD,
│   │                                latency calibration, ...
│   ├── runners/                     replay / C0 / offload-profile driver scripts
│   ├── manifests/                   task-manifest builders (build_task_manifest / locomo / swebench)
│   ├── utils/                       sampling, dedup, confirmation-set selection, bundle fetching
│   └── diag/                        harness diagnostic scripts
├── tests/                           offline self-tests: run_offline_test.py (33 checks),
│                                    run_agent_traces_test.py (38 checks)
├── source_code/                     vendored agent framework sources (continuation must be driven
│                                    by the real agent source code)
│   ├── agent_legacy / agent_openai_agents / agent_simplified / agent_smolagents / agent_tau2
│   ├── appworld_common
│   └── autogen / tau2-bench
├── results/                         analysis artifacts (JSON, organized by category)
│   ├── main_comparisons/            per-cohort main comparisons (17 files: quality/token/control
│   │                                flow/l67; incl. intermediate snapshots and no400 sensitivity)
│   ├── locomo/                      LoCoMo conditional analyses (10conv/4conv ex-ante, stratified
│   │                                offloadability, 999-QA task classification)
│   ├── appworld_53span/             53-span early series (control flow, edit distance, dynamics,
│   │                                A-F response classes, cost decomposition)
│   ├── stratification/              AppWorld stratified offloadability (48-point / 18-point follow-up)
│   ├── cross_agent/                 cross-agent comparison, fine taxonomy, signature DiD
│   ├── tau2/                        Tau2 cohort analyses and the final 44-point manifest
│   ├── profiles/                    per-span replay archives (autogen 23cf851_2/3) + span id mapping
│   └── aux/                         C0 gate, latency calibration, minimal metric set, dynamics plot
├── docs/                            documentation
│   ├── EXPERIMENT_DATA_ANALYSIS_RECORD.md   full experiment data & analysis record
│   │                                         (paper-reported vs. unreported comparison)
│   ├── EXPERIMENT_REPORT.md         early pilot report
│   ├── SPAN_OFFLOAD_DESIGN.md / CONTROL_FLOW_METRICS.md / input_side_offloadability_metrics.md
│   ├── preregistration_batch3.md    frozen preregistration
│   ├── tau2_interim_findings.md     Tau2 interim findings
│   └── DEVELOPMENT_NOTES.md         historical development notes (env setup, adapter/runtime
│                                    details, run commands)
├── data/                            task manifests (raw datasets are not committed)
│   ├── appworld/task_manifest.json
│   ├── locomo/task_manifest.json    [local] locomo10.json, msc_personas_all.json,
│   │                                multimodal_dialog/ fetched from the upstream repo
│   └── swebench/task_manifest.json  (58 tasks, pilot/heldout split)
├── paper/                           [not included] paper sources and arXiv submission guide
├── slurm/                           [not included] cluster run scripts
├── benchmarks/                      [local] third-party benchmark clones: appworld/, locomo/,
│                                    swebench/ (mini-swe-agent)
├── check_point_traces/              [local] raw checkpoint traces (2 GB)
└── out_span_offload/                [local] experiment replay artifacts (4 GB)
```

## Quick Start (offline smoke test, no API needed)

```bash
pip install -r requirements.txt
python -m span_offload.cli demo --offline --out data_demo
```

For a real SLM endpoint (vLLM / SGLang / Ollama, any OpenAI-compatible server):
`cp config.example.json config.json`, fill in the models and API keys (or use the
`STAGE10_SLM_*` / `STAGE10_BASELINE_*` environment variables).
**config.json contains keys — never commit it.**

## Reproducing the Experiments

1. Environment: `pip install -r requirements.txt`; for AppWorld, clone the upstream
   repo and run `appworld install` as described in `docs/DEVELOPMENT_NOTES.md` §1.
2. Data: raw traces and replay artifacts are large; the authors can provide them on
   request (`out_span_offload/`, `check_point_traces/`). LoCoMo data can be obtained
   from the [official LoCoMo repository](https://github.com/snap-research/LoCoMo) and
   placed under `data/locomo/`.
3. Running: locally `scripts/runners/run_live_replay.py`; cluster (slurm) scripts are
   not included in the public repository and are available from the authors.
4. Analysis: all analysis scripts are in `scripts/analysis/`, with outputs in
   `results/`; the frozen preregistration is `docs/preregistration_batch3.md`.
5. Complete experiment inventory and the audit of data not yet reported in the paper:
   `docs/EXPERIMENT_DATA_ANALYSIS_RECORD.md`.

## Security

The repository contains no API keys (hardcoded keys were removed from the cluster
scripts; keys are injected via environment variables only). If you have handled
keys from earlier versions, revoke and rotate them in the provider console.
