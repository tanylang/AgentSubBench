#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool


class ReasoningChatOpenAI(ChatOpenAI):
    """Preserve reasoning fields returned by the vLLM server."""

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> Any:
        if isinstance(response, dict):
            raw_response = response
        elif hasattr(response, "model_dump"):
            raw_response = response.model_dump(mode="json")
        else:
            raw_response = {}

        result = super()._create_chat_result(response, generation_info)
        choices = raw_response.get("choices") or []
        for generation, choice in zip(result.generations, choices):
            raw_message = choice.get("message") or {}
            for key in ("reasoning", "reasoning_content", "output_reasoning"):
                reasoning = raw_message.get(key)
                if reasoning not in (None, ""):
                    generation.message.additional_kwargs["reasoning"] = reasoning
                    generation.message.additional_kwargs["reasoning_source_field"] = key
                    break
        return result


DIMS = 384


def embed_one(text: str) -> list[float]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    features = list(words)
    for word in words:
        padded = f"^{word}$"
        features.extend(padded[i : i + 3] for i in range(max(1, len(padded) - 2)))
    vector = [0.0] * DIMS
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        vector[number % DIMS] += 1.0 if ((number >> 9) & 1) == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_one(text) for text in texts]


def message_to_dict(message: Any) -> dict[str, Any]:
    data = message.model_dump(mode="json")
    data["python_class"] = type(message).__name__
    return data


def json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def memory_to_dict(item: Any) -> dict[str, Any]:
    return {
        "namespace": list(item.namespace),
        "key": item.key,
        "value": item.value,
        "created_at": json_value(getattr(item, "created_at", None)),
        "updated_at": json_value(getattr(item, "updated_at", None)),
        "score": getattr(item, "score", None),
    }


def tool_names(messages: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call.get("name"):
                names.append(call["name"])
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    namespace = ("locomo", "langmem-smoke", "memories")
    store = InMemoryStore(index={"dims": DIMS, "embed": embed_texts})
    tools = [
        create_manage_memory_tool(namespace=namespace),
        create_search_memory_tool(namespace=namespace),
    ]
    model = ReasoningChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    agent = create_react_agent(model=model, tools=tools, store=store)

    print("===== 1. Ingestion: create memory =====", flush=True)
    ingestion = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "This is a memory-ingestion test. You must call manage_memory "
                        "to remember this fact: Caroline attended the LGBTQ support "
                        "group on 7 May 2023. After storing it, reply INGEST_OK."
                    )
                )
            ]
        },
        config={"recursion_limit": 10},
    )
    ingestion_messages = [message_to_dict(item) for item in ingestion["messages"]]
    memories = [memory_to_dict(item) for item in store.search(namespace, limit=100)]
    print("ingestion_tools:", tool_names(ingestion_messages), flush=True)
    print("memory_count:", len(memories), flush=True)

    if not memories:
        raise RuntimeError("manage_memory did not create a memory")

    print("===== 2. Fresh QA context: search memory =====", flush=True)
    qa = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use search_memory before answering. When did Caroline attend "
                        "the LGBTQ support group? Reply with the date only."
                    )
                )
            ]
        },
        config={"recursion_limit": 10},
    )
    qa_messages = [message_to_dict(item) for item in qa["messages"]]
    qa_tools = tool_names(qa_messages)
    final_answer = str(qa_messages[-1].get("content", "")).strip()

    output = {
        "status": "success",
        "model": args.model,
        "namespace": list(namespace),
        "embedding": {
            "type": "deterministic_hashing_vectorizer",
            "dimensions": DIMS,
        },
        "ingestion": {
            "messages": ingestion_messages,
            "tool_names": tool_names(ingestion_messages),
        },
        "memory_snapshot": memories,
        "qa": {
            "messages": qa_messages,
            "tool_names": qa_tools,
            "final_answer": final_answer,
        },
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("qa_tools:", qa_tools, flush=True)
    print("final_answer:", final_answer, flush=True)
    print("output:", args.output, flush=True)

    if "manage_memory" not in output["ingestion"]["tool_names"]:
        raise RuntimeError("manage_memory was not called")
    if "search_memory" not in qa_tools:
        raise RuntimeError("search_memory was not called")
    if "7 May 2023" not in final_answer:
        raise RuntimeError(f"Unexpected QA answer: {final_answer!r}")
    print("LANGMEM_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()

