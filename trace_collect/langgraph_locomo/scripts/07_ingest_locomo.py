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
    load_sample,
    memory_records,
    now_ns,
    semantic_memory_hash,
    session_numbers,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--session-limit", type=int, default=1)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    sample = load_sample(args.data, args.sample_id)
    conversation = sample["conversation"]
    available_sessions = session_numbers(conversation)
    selected_sessions = available_sessions[: args.session_limit]
    if not selected_sessions:
        raise ValueError("No conversation sessions selected")

    task_id = f"locomo:{args.sample_id}:ingestion"
    namespace = ("locomo", args.sample_id, "memories")
    store = InMemoryStore(index={"dims": DIMS, "embed": embed_texts})
    collector = TraceCollector(task_id=task_id, model=args.model)
    collector.state_provider = lambda: memory_records(store, namespace)
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
    ingested_turns = 0

    for session_number in selected_sessions:
        date = conversation.get(f"session_{session_number}_date_time")
        turns = conversation[f"session_{session_number}"]
        print(
            f"session={session_number} date={date!r} turn_count={len(turns)}",
            flush=True,
        )
        for turn_index, turn in enumerate(turns):
            collector.set_context(
                phase="ingestion",
                sample_id=args.sample_id,
                session_number=session_number,
                turn_index=turn_index,
                dia_id=turn.get("dia_id"),
            )
            message = (
                f"Session date and time: {date}\n"
                f"Dialogue ID: {turn.get('dia_id')}\n"
                f"Speaker: {turn.get('speaker')}\n"
                f"Message: {turn.get('text', '')}\n"
                "Process this message for long-term memory."
            )
            agent.invoke(
                {"messages": [HumanMessage(content=message)]},
                config={"recursion_limit": 12, "callbacks": [handler]},
            )
            ingested_turns += 1
            print(
                f"ingested={turn.get('dia_id')} "
                f"progress={ingested_turns} memory_count={len(collector.state())}",
                flush=True,
            )

    root_end_mono = time.perf_counter_ns()
    root_end_time = now_ns()
    collector.finalize_children()
    records = collector.state()
    state_hash = semantic_memory_hash(records)

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
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
            "turn_count": ingested_turns,
        },
        "records": records,
    }
    args.snapshot.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

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
            "agent.wall_time_ms": (root_end_mono - root_start_mono) / 1_000_000,
            "benchmark": "locomo",
            "benchmark_task_id": task_id,
            "gen_ai.agent.name": "LoCoMo-LangGraph-ReAct-LangMem",
            "model": args.model,
            "locomo.sample_id": args.sample_id,
            "locomo.phase": "ingestion",
            "locomo.ingested_sessions": len(selected_sessions),
            "locomo.ingested_turns": ingested_turns,
            "memory.count": len(records),
            "memory.state_hash": state_hash,
            "memory.snapshot_path": str(args.snapshot),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("ingested_sessions:", len(selected_sessions), flush=True)
    print("ingested_turns:", ingested_turns, flush=True)
    print("memory_count:", len(records), flush=True)
    print("memory_hash:", state_hash, flush=True)
    print("total_llm_calls:", len(llm_spans), flush=True)
    print("total_tool_calls:", len(tool_spans), flush=True)
    print("trace:", args.output, flush=True)
    print("snapshot:", args.snapshot, flush=True)
    print("LOCOMO_INGESTION_OK", flush=True)


if __name__ == "__main__":
    main()

