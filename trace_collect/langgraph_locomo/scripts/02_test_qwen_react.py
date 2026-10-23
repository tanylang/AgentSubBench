#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from openai import OpenAI


def now_ns() -> int:
    return time.time_ns()


def reasoning_from_mapping(value: dict[str, Any]) -> Any:
    for key in ("reasoning_content", "reasoning", "output_reasoning"):
        if value.get(key) not in (None, ""):
            return value[key]
    for key in ("additional_kwargs", "response_metadata"):
        nested = value.get(key)
        if isinstance(nested, dict):
            found = reasoning_from_mapping(nested)
            if found not in (None, ""):
                return found
    return None


def message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        data = message.model_dump(mode="json")
    else:
        data = {"repr": repr(message)}
    data["python_class"] = type(message).__name__
    data["captured_reasoning"] = reasoning_from_mapping(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "running",
        "base_url": args.base_url,
        "model": args.model,
        "started_at_ns": now_ns(),
    }

    print("===== 1. Raw OpenAI response probe =====", flush=True)
    raw_client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=120.0,
        max_retries=0,
    )
    raw_start_ns = now_ns()
    raw_start_mono = time.perf_counter_ns()
    raw_response = raw_client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": "Think briefly, then reply with exactly RAW_OK.",
            }
        ],
        temperature=0,
    )
    raw_end_mono = time.perf_counter_ns()
    raw_end_ns = now_ns()
    raw_client.close()

    raw_dict = raw_response.model_dump(mode="json")
    raw_message = raw_dict["choices"][0]["message"]
    raw_reasoning = reasoning_from_mapping(raw_message)
    report["raw_probe"] = {
        "start_time_ns": raw_start_ns,
        "end_time_ns": raw_end_ns,
        "duration_ms": (raw_end_mono - raw_start_mono) / 1_000_000,
        "response": raw_dict,
        "reasoning": raw_reasoning,
    }
    print("raw_content:", raw_message.get("content"), flush=True)
    print("raw_message_keys:", sorted(raw_message), flush=True)
    print("raw_reasoning_chars:", len(raw_reasoning or ""), flush=True)
    print("raw_usage:", raw_dict.get("usage"), flush=True)

    print("===== 2. LangGraph ReAct tool probe =====", flush=True)
    tool_executions: list[dict[str, Any]] = []

    @tool
    def lookup_fact(topic: str) -> str:
        """Return one fixed fact for a named topic."""
        start_ns = now_ns()
        start_mono = time.perf_counter_ns()
        result = f"The lookup for {topic} succeeded."
        end_mono = time.perf_counter_ns()
        end_ns = now_ns()
        tool_executions.append(
            {
                "tool_name": "lookup_fact",
                "arguments": {"topic": topic},
                "result": result,
                "start_time_ns": start_ns,
                "end_time_ns": end_ns,
                "duration_ms": (end_mono - start_mono) / 1_000_000,
                "success": True,
            }
        )
        return result

    model = ChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    agent = create_react_agent(model=model, tools=[lookup_fact])
    react_start_ns = now_ns()
    react_start_mono = time.perf_counter_ns()
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "You must call lookup_fact exactly once with "
                        "topic='Caroline'. After the tool result, answer "
                        "with exactly REACT_OK."
                    )
                )
            ]
        },
        config={"recursion_limit": 10},
    )
    react_end_mono = time.perf_counter_ns()
    react_end_ns = now_ns()

    messages = [message_to_dict(message) for message in result["messages"]]
    reasoning_messages = [
        item for item in messages if item.get("captured_reasoning") not in (None, "")
    ]
    report["react_probe"] = {
        "start_time_ns": react_start_ns,
        "end_time_ns": react_end_ns,
        "duration_ms": (react_end_mono - react_start_mono) / 1_000_000,
        "messages": messages,
        "tool_executions": tool_executions,
        "reasoning_message_count": len(reasoning_messages),
    }
    report["status"] = "success"
    report["ended_at_ns"] = now_ns()

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final_content = messages[-1].get("content") if messages else None
    print("message_types:", [item["python_class"] for item in messages], flush=True)
    print("tool_execution_count:", len(tool_executions), flush=True)
    print("reasoning_message_count:", len(reasoning_messages), flush=True)
    print("final_content:", final_content, flush=True)
    print("probe_saved:", args.output, flush=True)

    if not tool_executions:
        raise RuntimeError("The ReAct agent did not execute lookup_fact")
    print("QWEN_REACT_PROBE_OK", flush=True)


if __name__ == "__main__":
    main()

