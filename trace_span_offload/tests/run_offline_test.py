"""Offline end-to-end test for the Stage 10 pipeline (no pytest dependency).

Runs the whole demo pipeline with the deterministic scripted LLM and asserts:
  - native traces are produced and baseline-successful
  - span extraction + eligibility work (14.1)
  - single-span replay records carry all outcome layers (14.5)
  - strict success is the I[final && !collateral] indicator (14.5)
  - profiles / conditioned analyses / completeness test produce cells (14.6-14.7)
  - the static allowlist + multi-span validation comparison runs (14.8)

Usage:  python tests/run_offline_test.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    print("%s %s %s" % ("PASS" if condition else "FAIL", name, detail))


def main() -> int:
    from span_offload.config import Stage10Config
    from span_offload.demo import run_pipeline
    from span_offload.replay import load_replays
    from span_offload.span_extraction import load_spans
    from span_offload.schemas import derive_strict_success, from_dict, FinalOutcome

    tmp = tempfile.mkdtemp(prefix="span_offload_offline_test_")
    try:
        cfg = Stage10Config()
        cfg.offline = True
        cfg.baseline_llm.model = "mock-baseline"
        cfg.baseline_llm.price_per_1k_input = 3.0
        cfg.baseline_llm.price_per_1k_output = 15.0
        cfg.slm.models = ["mock-slm-1b"]
        cfg.slm.price_per_1k_input = 0.05
        cfg.slm.price_per_1k_output = 0.30
        cfg.replay.fallback_mode = "recorded_baseline"
        cfg.replay.min_cell_samples = 2  # tiny pilot so cells have support
        cfg.allowlist.min_cell_samples = 2
        cfg.allowlist.lcb_threshold = 0.8

        report_path = run_pipeline(cfg, out_dir=tmp)
        check("report written", os.path.exists(report_path), report_path)

        spans_path = os.path.join(tmp, "spans.jsonl")
        replays_path = os.path.join(tmp, "replays.jsonl")
        check("spans.jsonl exists", os.path.exists(spans_path))
        check("replays.jsonl exists", os.path.exists(replays_path))

        spans = load_spans(spans_path)
        replays = load_replays(replays_path)
        check("spans extracted", len(spans) > 0, "n=%d" % len(spans))

        eligible = [s for s in spans if s.get("eligible")]
        check("eligible spans exist", len(eligible) > 0, "n=%d" % len(eligible))
        check("no baseline-failed spans (scripted always succeeds)",
              all(not s.get("baseline_failed") for s in spans))
        check("every span has task+agent conditions",
              all(s.get("task_condition") and s.get("agent_condition") for s in spans))
        check("every span has replay input refs",
              all(s.get("replay_input", {}).get("state_snapshot", {}).get("environment_snapshot_ref")
                  for s in spans))
        check("pilot + heldout splits present",
              {s.get("experiment_split") for s in spans} == {"pilot", "heldout"})

        # descriptors must not leak labels (14.2.5)
        leaks = []
        for s in spans:
            d = json.dumps(s.get("span_descriptor"))
            for forbidden in ("final_task_success", "collateral_damage", "strict_success",
                              "candidate_model", "local_verifier_passed"):
                if forbidden in d:
                    leaks.append(forbidden)
        check("no label leakage in descriptors (14.2.5)", not leaks, str(sorted(set(leaks))))

        check("replay records exist", len(replays) > 0, "n=%d" % len(replays))
        complete = all(
            r.get("local_outcome") and r.get("downstream_outcome") and r.get("final_outcome")
            and r.get("fallback") is not None and r.get("resource") for r in replays)
        check("every replay has all 5 outcome layers (14.5)", complete)
        check("strict_success in {0,1} and equals I[final && !collateral]",
              all(r.get("strict_success") in (True, False) and
                  r.get("strict_success") == (bool(r["final_outcome"]["final_task_success"])
                                              and not bool(r["final_outcome"]["collateral_damage"]))
                  for r in replays))
        check("snapshot hash matches recorded at restore",
              all((not r.get("replay_meta", {}).get("error", "") or
                   "snapshot hash mismatch" not in r["replay_meta"].get("error", ""))
                  for r in replays))
        n_err = sum(1 for r in replays if r.get("replay_meta", {}).get("error"))
        check("no replay errors", n_err == 0, "errors=%d" % n_err)

        # offline: identical scripted SLM => strict success should be very high
        n_strict = sum(1 for r in replays if r.get("strict_success"))
        check("offline strict success high", n_strict / len(replays) >= 0.9,
              "%d/%d" % (n_strict, len(replays)))

        # resources populated
        check("resource accounting populated",
              all(r["resource"].get("baseline_span_cost", 0) > 0 for r in replays),
              "baseline span costs from mock price table")

        # analysis artifacts
        analysis_path = os.path.join(tmp, "report", "analysis.json")
        check("analysis.json exists", os.path.exists(analysis_path))
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        check("profiles have cells", any(p.get("cells") for p in analysis.get("profiles", {}).values()))
        check("task dimension tables exist", bool(analysis.get("task_tables")))
        check("matched analysis has both agents",
              set(analysis.get("within_agent", {}).keys()) == {"repA", "repB"},
              str(list(analysis.get("within_agent", {}).keys())))
        check("completeness test ran",
              analysis.get("completeness", {}).get("n_units_tested", 0) > 0)

        # validation comparison (14.8)
        validation_path = os.path.join(tmp, "report", "validation.json")
        check("validation.json exists", os.path.exists(validation_path))
        with open(validation_path, "r", encoding="utf-8") as f:
            validation = json.load(f)
        configs = validation.get("configs", {})
        check("4 configs compared",
              set(configs.keys()) == {"llm_only", "slm_only", "allowlist", "allowlist_fallback"},
              str(sorted(configs.keys())))
        check("llm_only succeeds on held-out (offline)",
              configs.get("llm_only", {}).get("final_task_success_rate", 0) == 1.0)
        check("llm_only offloads nothing",
              configs.get("llm_only", {}).get("n_offloaded", -1) == 0)
        check("slm_only offloads everything",
              configs.get("slm_only", {}).get("n_offloaded", 0) > 0)
        check("allowlist offloads a strict subset",
              0 < configs.get("allowlist", {}).get("n_offloaded", 0)
              < configs.get("slm_only", {}).get("n_offloaded", 10 ** 9))
        check("allowlist costs no more than llm_only (offline)",
              configs.get("allowlist", {}).get("total_cost", 0)
              <= configs.get("llm_only", {}).get("total_cost", 0) + 1e-9)

        # oracle bound
        oracle_path = os.path.join(tmp, "report", "oracle.json")
        check("oracle.json exists", os.path.exists(oracle_path))
        with open(oracle_path, "r", encoding="utf-8") as f:
            oracle = json.load(f)
        check("oracle bound >= 0", oracle.get("oracle_opportunity_cost_bound", -1) >= 0)

        # census
        census_path = os.path.join(tmp, "report", "census.json")
        check("census.json exists", os.path.exists(census_path))
        with open(census_path, "r", encoding="utf-8") as f:
            census = json.load(f)
        check("census has node patterns", census.get("node_pattern_counts", {}).get("planning", 0) > 0)

        print("")
        print("artifacts kept at %s" % tmp)
    finally:
        # keep artifacts for debugging on failure; remove on success
        failed = [name for name, ok, _ in CHECKS if not ok]
        if not failed:
            shutil.rmtree(tmp, ignore_errors=True)

    failed = [name for name, ok, _ in CHECKS if not ok]
    print("")
    print("%d/%d checks passed" % (len(CHECKS) - len(failed), len(CHECKS)))
    if failed:
        for name, _, detail in CHECKS:
            if name in failed:
                print("  FAILED: %s (%s)" % (name, detail))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
