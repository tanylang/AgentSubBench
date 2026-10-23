"""Stage 10: Conditional Span Offloadability Pilot (AgentSubBench).

Implements the single-span counterfactual replay measurement pipeline from the
Stage 10 spec (sections 14.1-14.9):

  - span extraction from native traces with three-layer conditioning
    (span intrinsic descriptor / task condition / agent condition, 14.2)
  - single-span SLM substitution replay against a frozen representative agent
    (Prefix -> SLM_s -> Baseline LLM for later spans -> Final Evaluator, 14.4)
  - strict-success conditional offloadability profiles O_m^strict(phi) (14.6)
  - task / agent / architecture-conditioned analyses and the descriptor
    completeness test (14.7)
  - lightweight static-profile allowlist utility validation (14.8)

This stage intentionally does NOT implement a full online multi-span routing
runtime; that boundary belongs to TraceOffload (14.9).
"""

__version__ = "0.1.0"
