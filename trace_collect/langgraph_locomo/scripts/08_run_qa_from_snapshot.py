#!/usr/bin/env python3
import argparse
import json
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
    json_text,
    jsonable,
    load_sample,
    memory_records,
    now_ns,
    semantic_memory_hash,
)


def restore_store(snapshot: dict, store: InMemoryStore) -> tuple[tuple[str, ...], list[dict]]:
    namespace = tuple(snapshot["namespace"])
    for record in snapshot["records"]:
        store.put(namespace, record["key"], record["value"])
    restored_records = memory_records(store, namespace)
    expected_hash = snapshot["semantic_hash"]
    actual_hash = semantic_memory_hash(restored_records)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Snapshot hash mismatch: expected={expected_hash} actual={actual_hash}"
        )
    return namespace, restored_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--question-index", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    sample_id = snapshot["sample_id"]
    sample = load_sample(args.data, sample_id)
    question_record = sample["qa"][args.question_index]
    task_id = f"locomo:{sample_id}:q{args.question_index:03d}"

    store = InMemoryStore(index={"dims": DIMS, "embed": embed_texts})
    namespace, restored_records = restore_store(snapshot, store)
    restored_hash = semantic_memory_hash(restored_records)
    print("snapshot_hash:", snapshot["semantic_hash"], flush=True)
    print("restored_hash:", restored_hash, flush=True)
    print("restored_memory_count:", len(restored_records), flush=True)
    print("SNAPSHOT_RESTORE_OK", flush=True)

    collector = TraceCollector(task_id=task_id, model=args.model)
    collector.state_provider = lambda: memory_records(store, namespace)
    collector.set_context(
        phase="qa",
        sample_id=sample_id,
        question_index=args.question_index,
        restored_from=str(args.snapshot),
    )
    handler = ToolTraceHandler(collector)
    model = TracedReasoningChatOpenAI(
        collector=collector,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0,
        max_tokens=8192,  # Reasoning mode + a small budget truncates output (same spec as the other drivers)
        timeout=180,
        max_retries=0,
    )
    tools = [
        create_manage_memory_tool(namespace=namespace),
        create_search_memory_tool(namespace=namespace),
    ]
    agent = create_react_agent(model=model, tools=tools, store=store, prompt=SYSTEM_PROMPT)

    root_start_time = now_ns()
    root_start_mono = time.perf_counter_ns()
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Answer this LoCoMo question using the memory tools when relevant:\n"
                        + question_record["question"]
                    )
                )
            ]
        },
        config={"recursion_limit": 12, "callbacks": [handler]},
    )
    root_end_mono = time.perf_counter_ns()
    root_end_time = now_ns()

    predicted_answer = str(result["messages"][-1].content).strip()
    collector.finalize_children()
    final_records = collector.state()
    final_hash = semantic_memory_hash(final_records)
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
            "agent.outcome": predicted_answer,
            "agent.total_llm_calls": len(llm_spans),
            "agent.total_steps": len(collector.children),
            "agent.total_tool_calls": len(tool_spans),
            "agent.wall_time_ms": (root_end_mono - root_start_mono) / 1_000_000,
            "benchmark": "locomo",
            "benchmark_task_id": task_id,
            "gen_ai.agent.name": "LoCoMo-LangGraph-ReAct-LangMem",
            "model": args.model,
            "locomo.sample_id": sample_id,
            "locomo.phase": "qa",
            "locomo.question_index": args.question_index,
            "locomo.ground_truth": jsonable(question_record.get("answer")),
            "locomo.evidence": json_text(question_record.get("evidence", [])),
            "locomo.category": question_record.get("category"),
            "memory.snapshot_path": str(args.snapshot),
            "memory.baseline_count": len(restored_records),
            "memory.baseline_hash": snapshot["semantic_hash"],
            "memory.restored_hash": restored_hash,
            "memory.final_count": len(final_records),
            "memory.final_hash": final_hash,
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
        "task": question_record["question"],
        "total_llm_calls": len(llm_spans),
        "total_steps": len(collector.children),
        "total_tool_calls": len(tool_spans),
        "trace_id": collector.trace_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("question:", question_record["question"], flush=True)
    print("predicted_answer:", predicted_answer, flush=True)
    print("ground_truth:", question_record.get("answer"), flush=True)
    print("total_llm_calls:", len(llm_spans), flush=True)
    print("total_tool_calls:", len(tool_spans), flush=True)
    print("trace:", args.output, flush=True)
    print("LOCOMO_QA_FROM_SNAPSHOT_OK", flush=True)


if __name__ == "__main__":
    main()

