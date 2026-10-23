import argparse
import asyncio
import dataclasses
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient


def make_json_serializable(value: Any) -> Any:
    """Convert an AutoGen event into an object that can be written to JSON."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if hasattr(value, "model_dump"):
        return make_json_serializable(
            value.model_dump(mode="json")
        )

    if dataclasses.is_dataclass(value):
        return {
            field.name: make_json_serializable(
                getattr(value, field.name)
            )
            for field in dataclasses.fields(value)
        }

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    return str(value)


def write_event(
    output_file,
    sequence: int,
    event_type: str,
    payload: Any,
) -> None:
    """Write one event to the JSONL file."""

    record = {
        "sequence": sequence,
        "timestamp_ns": time.time_ns(),
        "event_type": event_type,
        "payload": make_json_serializable(payload),
    }

    output_file.write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )
    output_file.flush()


async def run_test(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # vLLM exposes an OpenAI-compatible API.
    model_client = OpenAIChatCompletionClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=0,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    agent = AssistantAgent(
        name="locomo_autogen_test",
        model_client=model_client,
        system_message=(
            "You are a test agent. "
            "Follow the user's instruction exactly and answer concisely."
        ),
        model_client_stream=False,
    )

    task = (
        "This is a connectivity test. "
        "Please answer with only AUTOGEN_OK and nothing else. "
    )

    sequence = 0
    final_answer = None

    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            write_event(
                output_file,
                sequence,
                "run_start",
                {
                    "model": args.model,
                    "base_url": args.base_url,
                    "task": task,
                },
            )
            sequence += 1

            async for event in agent.run_stream(task=task):
                event_type = type(event).__name__

                write_event(
                    output_file,
                    sequence,
                    event_type,
                    event,
                )
                sequence += 1

                print(f"event: {event_type}")

                if isinstance(event, TaskResult):
                    for message in reversed(event.messages):
                        content = getattr(message, "content", None)
                        if isinstance(content, str) and content.strip():
                            final_answer = content.strip()
                            break

            write_event(
                output_file,
                sequence,
                "run_end",
                {
                    "status": "success",
                    "final_answer": final_answer,
                },
            )

        print(f"final_answer: {final_answer}")
        print(f"trace_saved: {output_path}")

        if final_answer is None:
            raise RuntimeError("AutoGen returned no text answer")

    except Exception as exc:
        # If the run fails before the trace file is created, still print the full exception.
        print(f"run_failed: {type(exc).__name__}: {exc}")
        raise

    finally:
        await model_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://192.168.10.25:8201/v1",
    )
    parser.add_argument(
        "--model",
        default="Qwen3.8-27B",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
    )
    parser.add_argument(
        "--output",
        default=(
            "/share/home/tanylang/trace_collect/"
            "autogen_locomo/traces/model_smoke.jsonl"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_test(parse_args()))
