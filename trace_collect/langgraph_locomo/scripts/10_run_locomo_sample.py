#!/usr/bin/env python3
"""Continue a completed LoCoMo ingestion trace with stateful sequential QA.

The final deliverable is one trace per LoCoMo sample.  Ground-truth answers are
recorded only after inference and are never included in the agent's messages or
memory.
"""

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from locomo_trace_core import (
    DIMS,
    SYSTEM_PROMPT,
    ToolTraceHandler,
    TraceCollector,
    TracedReasoningChatOpenAI,
    embed_texts,
    jsonable,
    load_sample,
    memory_records,
    now_ns,
    semantic_memory_hash,
)


def atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def configuration_hash(
    sample_id: str,
    model: str,
    ingestion_trace_id: str,
    baseline_hash: str,
) -> str:
    value = {
        "schema_version": 1,
        "sample_id": sample_id,
        "model": model,
        "ingestion_trace_id": ingestion_trace_id,
        "baseline_hash": baseline_hash,
        "system_prompt": SYSTEM_PROMPT,
        "qa_history": "previous_question_and_predicted_answer_only_v1",
        "ground_truth_policy": "trace_only_never_agent_input",
        "embedding": "deterministic_hashing_vectorizer_v1_384",
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def restore_records(
    store: InMemoryStore,
    namespace: tuple[str, ...],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for record in records:
        store.put(namespace, record["key"], record["value"])
    return memory_records(store, namespace)


def compact_history_messages(history: list[dict[str, Any]]) -> list[Any]:
    """Keep prior Q/A continuity without carrying old tool payloads."""
    messages: list[Any] = []
    for item in history:
        messages.append(
            HumanMessage(
                content=(
                    f"LoCoMo question {item['question_index']}:\n"
                    f"{item['question']}"
                )
            )
        )
        messages.append(AIMessage(content=item["predicted_answer"]))
    return messages


def annotate_new_spans(
    spans: list[dict[str, Any]],
    start_index: int,
    *,
    sample_id: str,
    question_index: int,
    category: Any,
) -> None:
    for span in spans[start_index:]:
        attributes = span["attributes"]
        attributes["locomo.phase"] = "qa"
        attributes["locomo.sample_id"] = sample_id
        attributes["locomo.question_index"] = question_index
        attributes["locomo.category"] = jsonable(category)
        checkpoint = span.get("check_point")
        if isinstance(checkpoint, dict):
            locomo = checkpoint.setdefault("locomo", {})
            locomo["phase"] = "qa"
            locomo["sample_id"] = sample_id
            locomo["question_index"] = question_index
            locomo["category"] = jsonable(category)


def finalize_spans(spans: list[dict[str, Any]]) -> None:
    spans.sort(key=lambda span: (span["start_time"], span["end_time"]))
    previous_end: int | None = None
    for step, span in enumerate(spans, start=1):
        span["attributes"]["agent.step"] = step
        if previous_end is None:
            span["gap_from_prev_ms"] = 0.0
        else:
            span["gap_from_prev_ms"] = (
                span["start_time"] - previous_end
            ) / 1_000_000
        previous_end = max(previous_end or span["end_time"], span["end_time"])


def token_totals(spans: list[dict[str, Any]]) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for span in spans:
        if span["type"] != "llm.call":
            continue
        attributes = span["attributes"]
        input_tokens += int(attributes.get("gen_ai.usage.input_tokens") or 0)
        output_tokens += int(attributes.get("gen_ai.usage.output_tokens") or 0)
    return input_tokens, output_tokens


def save_progress(
    path: Path,
    *,
    config_hash: str,
    sample_id: str,
    namespace: tuple[str, ...],
    records: list[dict[str, Any]],
    qa_children: list[dict[str, Any]],
    history: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    qa_wall_time_ms: float,
    qa_attempts: int,
    status: str,
) -> None:
    atomic_json_write(
        path,
        {
            "schema_version": 1,
            "status": status,
            "config_hash": config_hash,
            "sample_id": sample_id,
            "namespace": list(namespace),
            "completed_questions": len(answers),
            "memory_hash": semantic_memory_hash(records),
            "records": records,
            "qa_children": qa_children,
            "history": history,
            "answers": answers,
            "qa_wall_time_ms": qa_wall_time_ms,
            "qa_attempts": qa_attempts,
        },
    )


def build_sample_trace(
    *,
    ingestion_trace: dict[str, Any],
    ingestion_children: list[dict[str, Any]],
    qa_children: list[dict[str, Any]],
    sample_id: str,
    model_name: str,
    answers: list[dict[str, Any]],
    qa_count: int,
    baseline_snapshot: Path,
    baseline_count: int,
    baseline_hash: str,
    final_records: list[dict[str, Any]],
    qa_wall_time_ms: float,
    qa_attempts: int,
    complete: bool,
) -> dict[str, Any]:
    children = copy.deepcopy(ingestion_children) + copy.deepcopy(qa_children)
    finalize_spans(children)
    llm_spans = [span for span in children if span["type"] == "llm.call"]
    tool_spans = [span for span in children if span["type"] == "tool.execution"]
    input_tokens, output_tokens = token_totals(children)
    error_spans = [span for span in children if span["status"] != "OK"]
    final_hash = semantic_memory_hash(final_records)
    ingestion_root = ingestion_trace["spans"][0]
    ingestion_wall_time_ms = float(
        ingestion_root["attributes"].get("agent.wall_time_ms") or 0.0
    )
    ingestion_attempts = int(
        ingestion_root["attributes"].get("agent.attempts") or 1
    )
    root_end_time = now_ns()
    root_span = {
        "type": "invoke_agent",
        "span_id": ingestion_root["span_id"],
        "trace_id": ingestion_trace["trace_id"],
        "parent_id": None,
        "start_time": ingestion_root["start_time"],
        "end_time": root_end_time,
        "attributes": {
            "agent.episode.id": ingestion_trace["episode_id"],
            "agent.outcome": (
                "sample_complete"
                if complete
                else f"qa_partial_{len(answers)}_of_{qa_count}"
            ),
            "agent.total_llm_calls": len(llm_spans),
            "agent.total_steps": len(children),
            "agent.total_tool_calls": len(tool_spans),
            "agent.wall_time_ms": ingestion_wall_time_ms + qa_wall_time_ms,
            "agent.attempts": ingestion_attempts + qa_attempts,
            "agent.ingestion_attempts": ingestion_attempts,
            "agent.qa_attempts": qa_attempts,
            "agent.error_span_count": len(error_spans),
            "agent.recovered_error_count": len(error_spans),
            "benchmark": "locomo",
            "benchmark_task_id": f"locomo:{sample_id}",
            "gen_ai.agent.name": "LoCoMo-LangGraph-ReAct-LangMem",
            "model": model_name,
            "locomo.sample_id": sample_id,
            "locomo.phase": "complete_sample" if complete else "qa_partial",
            "locomo.ingested_sessions": ingestion_root["attributes"].get(
                "locomo.ingested_sessions"
            ),
            "locomo.ingested_turns": ingestion_root["attributes"].get(
                "locomo.ingested_turns"
            ),
            "locomo.processed_turns": ingestion_root["attributes"].get(
                "locomo.processed_turns",
                ingestion_root["attributes"].get("locomo.ingested_turns"),
            ),
            "locomo.failed_turn_count": ingestion_root["attributes"].get(
                "locomo.failed_turn_count", 0
            ),
            "locomo.failed_turns": ingestion_root["attributes"].get(
                "locomo.failed_turns", []
            ),
            "locomo.qa_count": qa_count,
            "locomo.questions_answered": len(answers),
            "locomo.answers": jsonable(answers),
            "memory.snapshot_path": str(baseline_snapshot),
            "memory.baseline_count": baseline_count,
            "memory.baseline_hash": baseline_hash,
            "memory.final_count": len(final_records),
            "memory.final_hash": final_hash,
        },
        # Child errors remain visible in the trace.  Reaching this point means
        # the sample (or requested partial run) completed despite them.
        "status": "OK",
        "kind": "SERVER",
    }
    return {
        "agent_name": "LoCoMo-LangGraph-ReAct-LangMem",
        "episode_id": ingestion_trace["episode_id"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "pass": None,
        "pass_details": None,
        "spans": [root_span, *children],
        "task": f"LoCoMo sample {sample_id}",
        "total_llm_calls": len(llm_spans),
        "total_steps": len(children),
        "total_tool_calls": len(tool_spans),
        "trace_id": ingestion_trace["trace_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ingestion-trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--partial-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=3,
        help="Stop once this absolute number of questions has been completed.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ingestion_trace = json.loads(args.ingestion_trace.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    sample_id = snapshot["sample_id"]
    sample = load_sample(args.data, sample_id)
    qa_records = sample["qa"]
    qa_count = len(qa_records)
    target_count = min(max(args.stop_after, 0), qa_count)
    namespace = tuple(snapshot["namespace"])
    baseline_hash = snapshot["semantic_hash"]
    baseline_count = len(snapshot["records"])

    ingestion_root = ingestion_trace["spans"][0]
    trace_hash = ingestion_root["attributes"].get("memory.state_hash")
    if trace_hash != baseline_hash:
        raise RuntimeError(
            "Ingestion trace and baseline snapshot hashes differ: "
            f"trace={trace_hash} snapshot={baseline_hash}"
        )
    if ingestion_root["attributes"].get("locomo.sample_id") != sample_id:
        raise RuntimeError("Ingestion trace and snapshot sample IDs differ")
    if ingestion_root["status"] != "OK":
        ingestion_errors = [
            span
            for span in ingestion_trace["spans"][1:]
            if span["status"] != "OK"
        ]
        recoverable_errors = bool(ingestion_errors) and all(
            (
                span["type"] == "tool.execution"
                and span["attributes"].get("gen_ai.tool.name") == "manage_memory"
                and span["attributes"].get("error.type") == "ValidationError"
            )
            or (
                span["type"] == "llm.call"
                and span["attributes"].get("error.type")
                in {
                    "APIConnectionError",
                    "APITimeoutError",
                    "InternalServerError",
                    "RateLimitError",
                }
            )
            for span in ingestion_errors
        )
        if not recoverable_errors:
            raise RuntimeError(
                "Ingestion root status is ERROR and contains "
                "non-recoverable child errors"
            )
        print(
            f"INGESTION_RECOVERED_ERRORS_OK count={len(ingestion_errors)}",
            flush=True,
        )

    expected_config_hash = configuration_hash(
        sample_id,
        args.model,
        ingestion_trace["trace_id"],
        baseline_hash,
    )
    progress: dict[str, Any] | None = None
    if args.resume and args.progress.exists():
        progress = json.loads(args.progress.read_text(encoding="utf-8"))
        if progress.get("config_hash") != expected_config_hash:
            raise RuntimeError("QA progress configuration does not match this run")

    store = InMemoryStore(index={"dims": DIMS, "embed": embed_texts})
    collector = TraceCollector(task_id=f"locomo:{sample_id}", model=args.model)
    collector.trace_id = ingestion_trace["trace_id"]
    collector.root_span_id = ingestion_root["span_id"]
    collector.episode_id = ingestion_trace["episode_id"]
    history: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    qa_wall_before_ms = 0.0
    qa_attempts = 1

    if progress is None:
        restored_records = restore_records(store, namespace, snapshot["records"])
        restored_hash = semantic_memory_hash(restored_records)
        if restored_hash != baseline_hash:
            raise RuntimeError(
                "Baseline restore hash mismatch: "
                f"expected={baseline_hash} actual={restored_hash}"
            )
        print(
            f"BASELINE_RESTORE_OK memory_count={len(restored_records)} "
            f"memory_hash={restored_hash}",
            flush=True,
        )
    else:
        if progress["sample_id"] != sample_id:
            raise RuntimeError("QA progress sample ID does not match")
        restored_records = restore_records(store, namespace, progress["records"])
        restored_hash = semantic_memory_hash(restored_records)
        if restored_hash != progress["memory_hash"]:
            raise RuntimeError(
                "QA progress restore hash mismatch: "
                f"expected={progress['memory_hash']} actual={restored_hash}"
            )
        collector.children = progress["qa_children"]
        history = progress["history"]
        answers = progress["answers"]
        qa_wall_before_ms = float(progress.get("qa_wall_time_ms", 0.0))
        qa_attempts = int(progress.get("qa_attempts", 0)) + 1
        if len(history) != len(answers):
            raise RuntimeError("QA progress history and answers have different lengths")
        print(
            f"QA_RESUME_OK completed_questions={len(answers)} "
            f"memory_count={len(restored_records)} attempts={qa_attempts}",
            flush=True,
        )

    if len(answers) > target_count:
        raise RuntimeError(
            f"Progress already has {len(answers)} questions, exceeding stop-after "
            f"target {target_count}; use a target >= {len(answers)}"
        )

    collector.state_provider = lambda: memory_records(store, namespace)
    handler = ToolTraceHandler(collector)
    model = TracedReasoningChatOpenAI(
        collector=collector,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0,
        max_tokens=int(os.environ.get("LOC_MAX_TOKENS", "8192")) or None,  # LOC_MAX_TOKENS=0 omits the cap (vLLM uses the remaining window)
        timeout=float(os.environ.get("LOC_TIMEOUT", "300")),
        max_retries=2,
    )
    tools = [
        create_manage_memory_tool(namespace=namespace),
        create_search_memory_tool(namespace=namespace),
    ]
    agent = create_react_agent(
        model=model,
        tools=tools,
        store=store,
        prompt=SYSTEM_PROMPT,
    )
    attempt_start_mono = time.perf_counter_ns()

    for question_index in range(len(answers), target_count):
        question_record = qa_records[question_index]
        question = question_record["question"]
        collector.set_context(
            phase="qa",
            sample_id=sample_id,
            question_index=question_index,
            category=question_record.get("category"),
        )
        messages = compact_history_messages(history)
        messages.append(
            HumanMessage(
                content=(
                    f"LoCoMo question {question_index}:\n{question}\n\n"
                    "Answer using the memory tools when relevant. Do not assume "
                    "that earlier model answers are correct; verify against memory."
                )
            )
        )
        child_start = len(collector.children)
        question_start_ns = now_ns()
        question_start_mono = time.perf_counter_ns()
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": 12, "callbacks": [handler]},
        )
        question_end_mono = time.perf_counter_ns()
        question_end_ns = now_ns()
        predicted_answer = str(result["messages"][-1].content).strip()
        annotate_new_spans(
            collector.children,
            child_start,
            sample_id=sample_id,
            question_index=question_index,
            category=question_record.get("category"),
        )

        # Only the model's prediction enters future conversational history.
        # Ground truth and evidence are appended to trace metadata afterwards.
        history.append(
            {
                "question_index": question_index,
                "question": question,
                "predicted_answer": predicted_answer,
            }
        )
        answers.append(
            {
                "question_index": question_index,
                "question": question,
                "predicted_answer": predicted_answer,
                "ground_truth": jsonable(question_record.get("answer")),
                "evidence": jsonable(question_record.get("evidence", [])),
                "category": jsonable(question_record.get("category")),
                "status": "success",
                "start_time": question_start_ns,
                "end_time": question_end_ns,
                "duration_ms": (
                    question_end_mono - question_start_mono
                ) / 1_000_000,
            }
        )
        records = collector.state()
        active_wall_ms = qa_wall_before_ms + (
            time.perf_counter_ns() - attempt_start_mono
        ) / 1_000_000
        checkpointed = (
            len(answers) % args.checkpoint_every == 0
            or len(answers) == target_count
            or len(answers) == qa_count
        )
        if checkpointed:
            save_progress(
                args.progress,
                config_hash=expected_config_hash,
                sample_id=sample_id,
                namespace=namespace,
                records=records,
                qa_children=collector.children,
                history=history,
                answers=answers,
                qa_wall_time_ms=active_wall_ms,
                qa_attempts=qa_attempts,
                status="complete" if len(answers) == qa_count else "running",
            )
        print(
            f"answered={question_index}/{qa_count - 1} "
            f"progress={len(answers)}/{qa_count} "
            f"new_spans={len(collector.children) - child_start} "
            f"memory_count={len(records)} checkpointed={checkpointed}",
            flush=True,
        )
        print("predicted_answer:", predicted_answer, flush=True)

    qa_wall_time_ms = qa_wall_before_ms + (
        time.perf_counter_ns() - attempt_start_mono
    ) / 1_000_000
    final_records = collector.state()
    complete = len(answers) == qa_count
    sample_trace = build_sample_trace(
        ingestion_trace=ingestion_trace,
        ingestion_children=ingestion_trace["spans"][1:],
        qa_children=collector.children,
        sample_id=sample_id,
        model_name=args.model,
        answers=answers,
        qa_count=qa_count,
        baseline_snapshot=args.snapshot,
        baseline_count=baseline_count,
        baseline_hash=baseline_hash,
        final_records=final_records,
        qa_wall_time_ms=qa_wall_time_ms,
        qa_attempts=qa_attempts,
        complete=complete,
    )

    if complete:
        atomic_json_write(args.output, sample_trace)
        args.partial_output.unlink(missing_ok=True)
        save_progress(
            args.progress,
            config_hash=expected_config_hash,
            sample_id=sample_id,
            namespace=namespace,
            records=final_records,
            qa_children=collector.children,
            history=history,
            answers=answers,
            qa_wall_time_ms=qa_wall_time_ms,
            qa_attempts=qa_attempts,
            status="complete",
        )
        print("questions_answered:", len(answers), flush=True)
        print("final_trace:", args.output, flush=True)
        print("LOCOMO_SAMPLE_COMPLETE", flush=True)
    else:
        atomic_json_write(args.partial_output, sample_trace)
        print("questions_answered:", len(answers), flush=True)
        print("partial_trace:", args.partial_output, flush=True)
        print("LOCOMO_SAMPLE_PARTIAL_OK", flush=True)


if __name__ == "__main__":
    main()

