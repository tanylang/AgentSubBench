#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from langchain_core.messages import HumanMessage
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
    load_sample,
    memory_records,
    now_ns,
    semantic_memory_hash,
    session_numbers,
)


def atomic_json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def configuration_hash(sample_id: str, model: str, session_limit: int) -> str:
    value = {
        "sample_id": sample_id,
        "model": model,
        "session_limit": session_limit,
        "system_prompt": SYSTEM_PROMPT,
        "embedding": "deterministic_hashing_vectorizer_v1_384",
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def restore_records(store: InMemoryStore, namespace: tuple[str, ...], records: list[dict]) -> None:
    for record in records:
        store.put(namespace, record["key"], record["value"])


def save_progress(
    path: Path,
    *,
    config_hash: str,
    sample_id: str,
    namespace: tuple[str, ...],
    collector: TraceCollector,
    records: list[dict],
    completed_dia_ids: set[str],
    failed_turns: list[dict],
    root_start_time: int,
    accumulated_wall_time_ms: float,
    attempts: int,
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
            "semantic_hash": semantic_memory_hash(records),
            "records": records,
            "completed_dia_ids": sorted(completed_dia_ids),
            "failed_turns": failed_turns,
            "collector": {
                "trace_id": collector.trace_id,
                "root_span_id": collector.root_span_id,
                "episode_id": collector.episode_id,
                "children": collector.children,
            },
            "root_start_time": root_start_time,
            "accumulated_wall_time_ms": accumulated_wall_time_ms,
            "attempts": attempts,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--session-limit", type=int, default=19)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    expected_config_hash = configuration_hash(
        args.sample_id, args.model, args.session_limit
    )
    if args.resume and args.progress.exists():
        existing = json.loads(args.progress.read_text(encoding="utf-8"))
        if existing.get("config_hash") != expected_config_hash:
            raise RuntimeError("Progress file configuration does not match this run")
        if (
            existing.get("status") == "complete"
            and args.output.exists()
            and args.snapshot.exists()
        ):
            print("Ingestion is already complete; existing outputs were preserved.")
            print("trace:", args.output)
            print("snapshot:", args.snapshot)
            print("LOCOMO_FULL_INGESTION_ALREADY_COMPLETE")
            return
    else:
        existing = None

    sample = load_sample(args.data, args.sample_id)
    conversation = sample["conversation"]
    selected_sessions = session_numbers(conversation)[: args.session_limit]
    if not selected_sessions:
        raise ValueError("No conversation sessions selected")

    task_id = f"locomo:{args.sample_id}:ingestion"
    namespace = ("locomo", args.sample_id, "memories")
    store = InMemoryStore(index={"dims": DIMS, "embed": embed_texts})
    collector = TraceCollector(task_id=task_id, model=args.model)
    completed_dia_ids: set[str] = set()
    failed_turns: list[dict] = []
    accumulated_before_ms = 0.0
    attempts = 1
    root_start_time = now_ns()

    if existing is not None:
        restore_records(store, namespace, existing["records"])
        restored_hash = semantic_memory_hash(memory_records(store, namespace))
        if restored_hash != existing["semantic_hash"]:
            raise RuntimeError(
                "Progress restore hash mismatch: "
                f"expected={existing['semantic_hash']} actual={restored_hash}"
            )
        collector.trace_id = existing["collector"]["trace_id"]
        collector.root_span_id = existing["collector"]["root_span_id"]
        collector.episode_id = existing["collector"]["episode_id"]
        collector.children = existing["collector"]["children"]
        completed_dia_ids = set(existing["completed_dia_ids"])
        failed_turns = list(existing.get("failed_turns", []))
        accumulated_before_ms = float(existing.get("accumulated_wall_time_ms", 0.0))
        attempts = int(existing.get("attempts", 0)) + 1
        root_start_time = int(existing["root_start_time"])
        print(
            f"RESUME_OK completed_turns={len(completed_dia_ids)} "
            f"memory_count={len(existing['records'])} attempts={attempts}",
            flush=True,
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
        timeout=float(os.environ.get("LOC_TIMEOUT", "180")),
        max_retries=1,
    )
    tools = [
        create_manage_memory_tool(namespace=namespace),
        create_search_memory_tool(namespace=namespace),
    ]
    agent = create_react_agent(model=model, tools=tools, store=store, prompt=SYSTEM_PROMPT)
    attempt_start_mono = time.perf_counter_ns()

    total_selected_turns = sum(
        len(conversation[f"session_{number}"]) for number in selected_sessions
    )
    for session_number in selected_sessions:
        date = conversation.get(f"session_{session_number}_date_time")
        turns = conversation[f"session_{session_number}"]
        print(
            f"session={session_number} date={date!r} turn_count={len(turns)}",
            flush=True,
        )
        for turn_index, turn in enumerate(turns):
            dia_id = turn.get("dia_id")
            if dia_id in completed_dia_ids:
                continue
            collector.set_context(
                phase="ingestion",
                sample_id=args.sample_id,
                session_number=session_number,
                turn_index=turn_index,
                dia_id=dia_id,
            )
            message = (
                f"Session date and time: {date}\n"
                f"Dialogue ID: {dia_id}\n"
                f"Speaker: {turn.get('speaker')}\n"
                f"Message: {turn.get('text', '')}\n"
                "Process this message for long-term memory."
            )
            try:
                agent.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    config={"recursion_limit": 12, "callbacks": [handler]},
                )
            except Exception as error:
                error_type = type(error).__name__
                recoverable_api_errors = {
                    "APIConnectionError",
                    "APITimeoutError",
                    "InternalServerError",
                    "RateLimitError",
                }
                if error_type not in recoverable_api_errors:
                    raise
                failed_turns.append(
                    {
                        "session_number": session_number,
                        "turn_index": turn_index,
                        "dia_id": dia_id,
                        "error_type": error_type,
                        "error_message": str(error),
                    }
                )
                completed_dia_ids.add(dia_id)
                records = collector.state()
                active_ms = accumulated_before_ms + (
                    time.perf_counter_ns() - attempt_start_mono
                ) / 1_000_000
                save_progress(
                    args.progress,
                    config_hash=expected_config_hash,
                    sample_id=args.sample_id,
                    namespace=namespace,
                    collector=collector,
                    records=records,
                    completed_dia_ids=completed_dia_ids,
                    failed_turns=failed_turns,
                    root_start_time=root_start_time,
                    accumulated_wall_time_ms=active_ms,
                    attempts=attempts,
                    status="running",
                )
                print(
                    f"ingestion_failed={dia_id} "
                    f"progress={len(completed_dia_ids)}/{total_selected_turns} "
                    f"error_type={error_type} checkpointed=True",
                    flush=True,
                )
                continue
            completed_dia_ids.add(dia_id)
            records = collector.state()
            active_ms = accumulated_before_ms + (
                time.perf_counter_ns() - attempt_start_mono
            ) / 1_000_000
            checkpointed = (
                len(completed_dia_ids) % 10 == 0
                or turn_index == len(turns) - 1
            )
            if checkpointed:
                save_progress(
                    args.progress,
                    config_hash=expected_config_hash,
                    sample_id=args.sample_id,
                    namespace=namespace,
                    collector=collector,
                    records=records,
                    completed_dia_ids=completed_dia_ids,
                    failed_turns=failed_turns,
                    root_start_time=root_start_time,
                    accumulated_wall_time_ms=active_ms,
                    attempts=attempts,
                    status="running",
                )
            print(
                f"ingested={dia_id} progress={len(completed_dia_ids)}/"
                f"{total_selected_turns} memory_count={len(records)} "
                f"checkpointed={checkpointed}",
                flush=True,
            )

    active_wall_time_ms = accumulated_before_ms + (
        time.perf_counter_ns() - attempt_start_mono
    ) / 1_000_000
    root_end_time = now_ns()
    collector.finalize_children()
    records = collector.state()
    state_hash = semantic_memory_hash(records)
    successful_turn_count = len(completed_dia_ids) - len(failed_turns)

    snapshot = {
        "schema_version": 1,
        "sample_id": args.sample_id,
        "namespace": list(namespace),
        "semantic_hash": state_hash,
        "embedding": {
            "type": "deterministic_hashing_vectorizer",
            "dimensions": DIMS,
        },
        "ingestion": {
            "session_numbers": selected_sessions,
            "turn_count": len(completed_dia_ids),
            "successful_turn_count": successful_turn_count,
            "failed_turns": failed_turns,
            "attempts": attempts,
        },
        "records": records,
    }
    atomic_json_write(args.snapshot, snapshot)

    llm_spans = [span for span in collector.children if span["type"] == "llm.call"]
    tool_spans = [span for span in collector.children if span["type"] == "tool.execution"]
    input_tokens = sum(span["attributes"]["gen_ai.usage.input_tokens"] for span in llm_spans)
    output_tokens = sum(span["attributes"]["gen_ai.usage.output_tokens"] for span in llm_spans)
    failed = any(span["status"] != "OK" for span in collector.children)
    root_span = {
        "type": "invoke_agent",
        "span_id": collector.root_span_id,
        "trace_id": collector.trace_id,
        "parent_id": None,
        "start_time": root_start_time,
        "end_time": root_end_time,
        "attributes": {
            "agent.episode.id": collector.episode_id,
            "agent.outcome": "ingestion_complete",
            "agent.total_llm_calls": len(llm_spans),
            "agent.total_steps": len(collector.children),
            "agent.total_tool_calls": len(tool_spans),
            "agent.wall_time_ms": active_wall_time_ms,
            "agent.attempts": attempts,
            "benchmark": "locomo",
            "benchmark_task_id": task_id,
            "gen_ai.agent.name": "LoCoMo-LangGraph-ReAct-LangMem",
            "model": args.model,
            "locomo.sample_id": args.sample_id,
            "locomo.phase": "ingestion",
            "locomo.ingested_sessions": len(selected_sessions),
            "locomo.processed_turns": len(completed_dia_ids),
            "locomo.ingested_turns": successful_turn_count,
            "locomo.failed_turn_count": len(failed_turns),
            "locomo.failed_turns": failed_turns,
            "memory.count": len(records),
            "memory.state_hash": state_hash,
            "memory.snapshot_path": str(args.snapshot),
            "ingestion.progress_path": str(args.progress),
        },
        "status": "ERROR" if failed else "OK",
        "kind": "SERVER",
    }
    trace = {
        "agent_name": "LoCoMo-LangGraph-ReAct-LangMem",
        "episode_id": collector.episode_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "pass": None,
        "pass_details": None,
        "spans": [root_span, *collector.children],
        "task": f"Ingest LoCoMo memory for {args.sample_id}",
        "total_llm_calls": len(llm_spans),
        "total_steps": len(collector.children),
        "total_tool_calls": len(tool_spans),
        "trace_id": collector.trace_id,
    }
    atomic_json_write(args.output, trace)
    save_progress(
        args.progress,
        config_hash=expected_config_hash,
        sample_id=args.sample_id,
        namespace=namespace,
        collector=collector,
        records=records,
        completed_dia_ids=completed_dia_ids,
        failed_turns=failed_turns,
        root_start_time=root_start_time,
        accumulated_wall_time_ms=active_wall_time_ms,
        attempts=attempts,
        status="complete",
    )

    print("ingested_sessions:", len(selected_sessions), flush=True)
    print("processed_turns:", len(completed_dia_ids), flush=True)
    print("ingested_turns:", successful_turn_count, flush=True)
    print("failed_turn_count:", len(failed_turns), flush=True)
    print("memory_count:", len(records), flush=True)
    print("memory_hash:", state_hash, flush=True)
    print("attempts:", attempts, flush=True)
    print("total_llm_calls:", len(llm_spans), flush=True)
    print("total_tool_calls:", len(tool_spans), flush=True)
    print("trace:", args.output, flush=True)
    print("snapshot:", args.snapshot, flush=True)
    print("progress_file:", args.progress, flush=True)
    print("LOCOMO_FULL_INGESTION_OK", flush=True)


if __name__ == "__main__":
    main()

