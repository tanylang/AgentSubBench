#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


class ReasoningChatOpenAI(ChatOpenAI):
    """Preserve reasoning fields returned by an OpenAI-compatible server."""

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
            reasoning = (
                raw_message.get("reasoning")
                or raw_message.get("reasoning_content")
                or raw_message.get("output_reasoning")
            )
            if reasoning not in (None, ""):
                generation.message.additional_kwargs["reasoning"] = reasoning
                generation.message.additional_kwargs["reasoning_source_field"] = next(
                    key
                    for key in ("reasoning", "reasoning_content", "output_reasoning")
                    if raw_message.get(key) not in (None, "")
                )
        return result


def message_to_dict(message: Any) -> dict[str, Any]:
    data = message.model_dump(mode="json")
    data["python_class"] = type(message).__name__
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tool_calls: list[dict[str, Any]] = []

    @tool
    def lookup_fact(topic: str) -> str:
        """Return one fixed fact for a named topic."""
        result = f"The lookup for {topic} succeeded."
        tool_calls.append(
            {
                "name": "lookup_fact",
                "arguments": {"topic": topic},
                "result": result,
            }
        )
        return result

    model = ReasoningChatOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0,
        timeout=120,
        max_retries=0,
    )
    agent = create_react_agent(model=model, tools=[lookup_fact])
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "You must call lookup_fact exactly once with "
                        "topic='Caroline'. After receiving the tool result, "
                        "answer with exactly PASSTHROUGH_OK."
                    )
                )
            ]
        },
        config={"recursion_limit": 10},
    )

    messages = [message_to_dict(message) for message in result["messages"]]
    ai_messages = [item for item in messages if item["python_class"] == "AIMessage"]
    reasoning_messages = [
        item
        for item in ai_messages
        if item.get("additional_kwargs", {}).get("reasoning") not in (None, "")
    ]

    output = {
        "status": "success",
        "model": args.model,
        "messages": messages,
        "tool_calls": tool_calls,
        "ai_message_count": len(ai_messages),
        "reasoning_message_count": len(reasoning_messages),
    }
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for index, message in enumerate(ai_messages):
        additional = message.get("additional_kwargs", {})
        reasoning = additional.get("reasoning") or ""
        print(
            f"ai_message_{index}: reasoning_chars={len(reasoning)} "
            f"source={additional.get('reasoning_source_field')} "
            f"tool_calls={len(message.get('tool_calls') or [])}",
            flush=True,
        )
    print("tool_execution_count:", len(tool_calls), flush=True)
    print("reasoning_message_count:", len(reasoning_messages), flush=True)
    print("final_content:", messages[-1].get("content"), flush=True)
    print("output:", args.output, flush=True)

    if not tool_calls:
        raise RuntimeError("lookup_fact was not executed")
    if not reasoning_messages:
        raise RuntimeError("reasoning was not preserved in any AIMessage")
    print("REASONING_PASSTHROUGH_OK", flush=True)


if __name__ == "__main__":
    main()

