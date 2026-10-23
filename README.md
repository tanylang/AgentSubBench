# AgentSubBench

Official code release for the paper
**IMPACT: Measuring Conditional Workflow Responses to Single-Span Model
Substitution in Stateful LLM Agents** (ICLR 2027 submission; arXiv preprint).

This repository contains the two experimental pipelines behind the paper,
organized as two self-contained project folders:

```
AgentSubBench/
├── trace_collect/         Input side: multi-agent control-flow trace collection
│                          and evaluation
├── trace_span_offload/    Intervention side: single-span model-substitution
│                          measurement (IMPACT)
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

**`trace_span_offload/` — intervention side (IMPACT).** Starting from a recorded
trace and one physical LM invocation (span), restores the pre-invocation state
and replays matched arms under the four-condition protocol:
**C0** (frozen-replay audit), **I** (recorded-output continuation),
**B** (baseline-model resampling), and **T_span** (single-span substitution with
a smaller candidate model). Every arm continues live with the baseline model,
and the pipeline measures local validity, downstream propagation, final quality,
collateral risk, tokens, and completion time, producing the conditional response
atlas used for offloading decisions. See `trace_span_offload/README.md`.

## Data

Input traces and replay artifacts are too large for the repository; they are
released as compressed archives on the GitHub release page of this repository
(`check_point_traces.tar.gz` for collected traces, `out_span_offload.tar.gz`
for replay outputs). The `results/` folders inside each project contain the
analysis artifacts used in the paper.

## Paper

- arXiv preprint: https://arxiv.org/abs/XXXX.XXXXX (link updated upon posting)
- ICLR 2027 submission (anonymized)

## Citation

```bibtex
@article{agentsubbench2026,
  title={IMPACT: Measuring Conditional Workflow Responses to Single-Span Model Substitution in Stateful LLM Agents},
  author={Anonymous},
  year={2026}
}
```

## License

See the LICENSE file in each project folder.
