# AgentSubBench

This repository studies a central question for stateful LLM agents: **how does an
agent workflow respond when one physical model invocation is replaced by another
(smaller) model?**

Replacing a single call can change later actions, state transitions, and
observations, so local output agreement does not determine the substitution's
effect. A recorded trajectory is also only one realization of the baseline
workflow: regenerating the same call with the original model can already change
the suffix. This repository therefore measures substitution as a controlled
intervention — separating the substitution effect from ordinary baseline
variability — and organizes the resulting evidence into a conditional response
atlas for offloading decisions.

## Approach

A span is one physical LM invocation. From a common restored pre-invocation
state, a four-condition protocol separates the sources of workflow variation:

| Condition | What it does |
|---|---|
| **C0** | Frozen-replay audit: injects the recorded output and checks replay-fidelity predicates before any live trial |
| **I** | Recorded-output continuation: fixes the recorded target output and measures live suffix variability |
| **B** | Baseline-model resampling: regenerates the target with the baseline model and continues live (the matched baseline distribution) |
| **T_span** | Single-span substitution: generates the target with a candidate (smaller) model; every later call returns to the baseline model |

The primary contrast is **T_span − B**, with native recovery retained in both
arms. Each run records five response families: local validity, downstream
propagation, final quality, collateral risk, and resources (tokens and
completion time). For offloading, a population is supported only when explicit
bounds are met: the lower confidence bound on the quality contrast stays within
a non-inferiority margin, the upper bound on collateral risk stays within a
budget, and the lower bound on net token saving is positive.
Net saving decomposes as direct target saving minus downstream compensation
(continuation cost can amplify or erase target-level gains).

## Experiments

Two pipelines, released as two project folders:

```
AgentSubBench/
├── trace_collect/         Input side: collects checkpoint traces of multiple
│                          agent families on AppWorld, LoCoMo, and SWE-bench
├── trace_span_offload/    Intervention side: restores pre-invocation state and
│                          replays the C0/I/B/T_span arms with live continuation
└── README.md              this file
```

The measured study spans six cohorts — AppWorld Simplified (48 intervention
points), an outcome-selected 18-point follow-up, AppWorld Tau2 (44 points), a
disjoint re-collected AppWorld screen (84 points), LoCoMo with a LangGraph +
LangMem agent (2,623 spans over 1,056 questions), and SWE-bench with
mini-SWE-agent (134 points over four Django instances) — for 17,100 live
continuations. The baseline endpoint is Qwen3.8-27B and the candidate endpoint
is Qwen3.5-9B; all arms use native recovery only.

## Key Findings

- **The native trace is not the counterfactual baseline.** Repeated baseline
  resampling yields both successes and failures at 1/48, 6/18, 11/44, 23/84,
  384/2,623, and 3/134 sampled points across the six cohorts; recorded-output
  continuation (I) and baseline resampling (B) differ materially.
- **Local agreement does not determine final success.** Treatment local
  equivalence ranges 13.6–44.0% across cohorts, yet locally non-equivalent runs
  frequently still finish successfully, and equivalence-passing runs can fail.
- **A local substitution induces a signed workflow response.** On the LoCoMo
  cohort, substitution lowers mean strict success by 9.8 points while reducing
  model tokens by 2.86k and completion time by 0.84 s (7.2%) on average;
  downstream compensation can amplify or erase target-level savings.
- **Coarse labels do not define a universal offloading rule.** All 2,623 LoCoMo
  spans are declared read-only, yet the aggregate quality contrast is negative;
  a target-size heuristic correlates 0.668 with direct saving on one AppWorld
  agent but 0.068 on another.
- **Pre-decision conditions structure the response.** Upstream execution load,
  in-loop status, position, and execution phase are the most informative
  pre-decision features; prospective routing must use only information available
  before the target invocation.
- **Screening conclusions are unstable.** The five-repeat screen and fifteen-
  repeat follow-up agree poorly at the same 18 checkpoints (Pearson −0.302,
  four direct sign reversals), supporting population-conditioned interpretation
  rather than per-checkpoint verdicts.

The measured populations are sampled cohorts, not workload-wide estimates;
evidence is organized in `trace_span_offload/results/` with uncertainty
preserving task/trace dependence.

## Data

Input traces and replay artifacts are too large for the repository; they are
released as compressed archives on the GitHub release page of this repository
(`check_point_traces.tar.gz`, `out_span_offload.tar.gz`).

## License

See the LICENSE file in each project folder.
