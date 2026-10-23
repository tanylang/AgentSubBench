#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_one(sample: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    qa_count = len(sample["qa"])
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "qa_count": qa_count,
        "trace_path": str(trace_path),
        "complete": False,
        "errors": [],
    }
    if not trace_path.exists():
        result["errors"].append("trace_missing")
        return result
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        root = trace["spans"][0]
        attrs = root["attributes"]
        answers = attrs.get("locomo.answers") or []
        error_spans = [
            span for span in trace["spans"][1:] if span["status"] != "OK"
        ]
        checks = {
            "root_status_ok": root["status"] == "OK",
            "sample_id_matches": attrs.get("locomo.sample_id") == sample_id,
            "phase_complete": attrs.get("locomo.phase") == "complete_sample",
            "qa_count_matches": attrs.get("locomo.qa_count") == qa_count,
            "answers_complete": len(answers) == qa_count,
            "answer_indices_complete": [
                item.get("question_index") for item in answers
            ] == list(range(qa_count)),
            "step_count_matches": trace.get("total_steps") == len(trace["spans"]) - 1,
            "single_trace_id": len({span["trace_id"] for span in trace["spans"]}) == 1,
        }
        llm_spans = [span for span in trace["spans"] if span["type"] == "llm.call"]
        checks["all_llm_have_reasoning"] = bool(llm_spans) and all(
            (span["attributes"].get("gen_ai.output.reasoning") or "")
            for span in llm_spans
        )
        result.update(
            {
                "checks": checks,
                "questions_answered": attrs.get("locomo.questions_answered"),
                "span_count": len(trace["spans"]),
                "total_steps": trace.get("total_steps"),
                "total_llm_calls": trace.get("total_llm_calls"),
                "total_tool_calls": trace.get("total_tool_calls"),
                "input_tokens": trace.get("input_tokens"),
                "output_tokens": trace.get("output_tokens"),
                "processed_turns": attrs.get(
                    "locomo.processed_turns", attrs.get("locomo.ingested_turns")
                ),
                "ingested_turns": attrs.get("locomo.ingested_turns"),
                "failed_turn_count": attrs.get("locomo.failed_turn_count", 0),
                "failed_turns": attrs.get("locomo.failed_turns", []),
                "file_size_bytes": trace_path.stat().st_size,
                "trace_id": trace.get("trace_id"),
                "episode_id": trace.get("episode_id"),
                "error_span_count": len(error_spans),
                "error_spans": [
                    {
                        "type": span.get("type"),
                        "step": span.get("attributes", {}).get("agent.step"),
                        "tool": span.get("attributes", {}).get("gen_ai.tool.name"),
                        "error_type": span.get("attributes", {}).get("error.type"),
                        "error_message": span.get("attributes", {}).get("error.message"),
                    }
                    for span in error_spans
                ],
            }
        )
        result["errors"] = [name for name, passed in checks.items() if not passed]
        result["complete"] = not result["errors"]
    except Exception as error:
        result["errors"].append(f"validation_exception:{type(error).__name__}:{error}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = json.loads(args.data.read_text(encoding="utf-8"))
    results = [
        validate_one(sample, args.trace_dir / f"{sample['sample_id']}.json")
        for sample in samples
    ]
    complete_count = sum(bool(item["complete"]) for item in results)
    manifest = {
        "schema_version": 1,
        "benchmark": "locomo",
        "agent_name": "LoCoMo-LangGraph-ReAct-LangMem",
        "sample_count": len(results),
        "complete_count": complete_count,
        "all_complete": complete_count == len(results),
        "total_trace_bytes": sum(
            int(item.get("file_size_bytes") or 0) for item in results
        ),
        "samples": results,
    }
    atomic_json_write(args.output, manifest)

    print("sample_count:", len(results))
    print("complete_count:", complete_count)
    print("all_complete:", manifest["all_complete"])
    for item in results:
        print(
            f"sample_id={item['sample_id']} complete={item['complete']} "
            f"errors={item['errors']}"
        )
    print("manifest:", args.output)
    if not manifest["all_complete"]:
        raise SystemExit(1)
    print("LOCOMO_DATASET_VALID")


if __name__ == "__main__":
    main()

