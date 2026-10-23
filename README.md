# AgentSubBench

This repository contains two experimental pipelines for measuring and
evaluating stateful LLM agent workflows, organized as two self-contained
project folders:

```
AgentSubBench/
├── trace_collect/         Input side: multi-agent control-flow trace collection
│                          and evaluation
├── trace_span_offload/    Intervention side: single-span model-substitution
│                          measurement
└── README.md              this file
```

## The two experiments

**`trace_collect/` — input side.** Collects complete execution traces of multiple
agent families on three benchmarks — **AppWorld** (API-calling tasks), **LoCoMo**
(long-term conversational memory QA), and **SWE-bench Verified** (software
engineering) — without modifying agent logic: source-level hook points plus an
externally registered collector record every physical LM invocation together
with its request, output, tool events, environment snapshots (`env_changes` /
`check_point`), and evaluator provenance. These checkpoint traces are the input
that the offload pipeline consumes. See `trace_collect/README_EN.md`.

**`trace_span_offload/` — intervention side.** Starting from a recorded trace and
one physical LM invocation (span), restores the pre-invocation state and replays
matched arms under a four-condition protocol: **C0** (frozen-replay audit),
**I** (recorded-output continuation), **B** (baseline-model resampling), and
**T_span** (single-span substitution with a smaller candidate model). Every arm
continues live with the baseline model, and the pipeline measures local validity,
downstream propagation, final quality, collateral risk, tokens, and completion
time, producing a conditional response atlas for offloading decisions. See
`trace_span_offload/README.md`.

## Data

Input traces and replay artifacts are too large for the repository; they are
released as compressed archives on the GitHub release page of this repository
(`check_point_traces.tar.gz` for collected traces, `out_span_offload.tar.gz`
for replay outputs). The `results/` folders inside each project contain the
analysis artifacts.

## License

See the LICENSE file in each project folder.
