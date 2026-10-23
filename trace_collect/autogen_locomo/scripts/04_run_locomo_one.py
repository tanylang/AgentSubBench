import argparse
import asyncio
import dataclasses
import hashlib
import json
import re
import time
import traceback
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient


AGENT_NAME = "AutoGen-AssistantAgent"
BENCHMARK_NAME = "LocoMo"


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def json_text(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False)


def event_time_ns(event: Any, fallback: int | None = None) -> int:
    created_at = getattr(event, "created_at", None)
    if isinstance(created_at, datetime):
        return int(created_at.timestamp() * 1_000_000_000)
    return fallback if fallback is not None else time.time_ns()


def state_hash(messages: list[dict[str, Any]], read_ids: set[str]) -> str:
    encoded = json.dumps(
        {"messages": messages, "read_ids": sorted(read_ids)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def load_sample(data_path: Path, sample_id: str) -> dict[str, Any]:
    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for sample in data:
        if sample.get("sample_id") == sample_id:
            return sample
    raise ValueError(f"sample_id not found: {sample_id}")


def flatten_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    sessions: list[tuple[int, str, list[dict[str, Any]]]] = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            sessions.append((int(match.group(1)), key, value))
    sessions.sort(key=lambda item: item[0])

    for session_number, session_name, session_messages in sessions:
        session_date = conversation.get(f"{session_name}_date_time", "")
        for message in session_messages:
            text = str(message.get("text", "")).strip()
            caption = str(message.get("blip_caption", "")).strip()
            if caption:
                suffix = f"[Image description: {caption}]"
                text = f"{text}\n{suffix}" if text else suffix
            flattened.append(
                {
                    "dia_id": message.get("dia_id"),
                    "session": session_name,
                    "session_number": session_number,
                    "date": session_date,
                    "speaker": message.get("speaker"),
                    "text": text,
                    "img_url": message.get("img_url"),
                }
            )
    return flattened


def make_tools(
    messages: list[dict[str, Any]],
    speaker_a: str,
    speaker_b: str,
    read_ids: set[str],
):
    speakers = {
        speaker_a.casefold(): speaker_a,
        speaker_b.casefold(): speaker_b,
    }

    def get_person_id(name: str) -> dict[str, Any]:
        """Resolve a person name to the canonical speaker name.

        Args:
            name: Person name, such as Caroline or Melanie.
        """
        matched = speakers.get(name.strip().casefold())
        if matched is None:
            return {
                "matched": False,
                "person_id": None,
                "available_people": [speaker_a, speaker_b],
            }
        return {"matched": True, "person_id": matched}

    def search_messages(
        query: str,
        person_id: str = "",
        session: str = "",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search conversation messages using case-insensitive text matching.

        Args:
            query: Text or phrase to search for.
            person_id: Optional canonical speaker name.
            session: Optional session name, such as session_1.
            top_k: Maximum number of messages to return.
        """
        normalized_query = query.strip().casefold()
        normalized_person = person_id.strip().casefold()
        normalized_session = session.strip().casefold()
        if not normalized_query:
            return {"error": "query must not be empty", "results": []}

        safe_top_k = max(1, min(int(top_k), 50))
        results: list[dict[str, Any]] = []
        for message in messages:
            if normalized_query not in message["text"].casefold():
                continue
            if (
                normalized_person
                and str(message["speaker"]).casefold() != normalized_person
            ):
                continue
            if (
                normalized_session
                and message["session"].casefold() != normalized_session
            ):
                continue
            results.append(
                {
                    "dia_id": message["dia_id"],
                    "session": message["session"],
                    "date": message["date"],
                    "speaker": message["speaker"],
                    "text": message["text"],
                }
            )
            if message["dia_id"]:
                read_ids.add(str(message["dia_id"]))
            if len(results) >= safe_top_k:
                break

        return {
            "query": query,
            "person_id": person_id or None,
            "session": session or None,
            "count": len(results),
            "results": results,
        }

    return get_person_id, search_messages


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_person_id",
                "description": "Resolve a person name to the canonical speaker name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Person name."}
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_messages",
                "description": (
                    "Search conversation messages using case-insensitive text matching."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "person_id": {"type": "string", "default": ""},
                        "session": {"type": "string", "default": ""},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
        },
    ]


class SpanCollector:
    def __init__(
        self,
        *,
        trace_id: str,
        root_span_id: str,
        task_id: str,
        model: str,
        system_message: str,
        task_message: str,
        read_ids: set[str],
    ) -> None:
        self.trace_id = trace_id
        self.root_span_id = root_span_id
        self.task_id = task_id
        self.model = model
        self.read_ids = read_ids
        self.spans: list[dict[str, Any]] = []
        self.step = 0
        self.total_llm_calls = 0
        self.total_tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.first_llm = True
        self.pending_calls: dict[str, dict[str, Any]] = {}
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": task_message},
        ]
        self.last_time_ns = time.time_ns()

    def _base_span(
        self,
        span_type: str,
        start_time: int,
        end_time: int,
        kind: str,
        status: str = "OK",
    ) -> dict[str, Any]:
        return {
            "type": span_type,
            "span_id": uuid.uuid4().hex[:16],
            "trace_id": self.trace_id,
            "parent_id": self.root_span_id,
            "start_time": start_time,
            "end_time": max(start_time, end_time),
            "attributes": {},
            "status": status,
            "kind": kind,
        }

    @staticmethod
    def _usage(event: Any) -> tuple[int, int]:
        usage = getattr(event, "models_usage", None)
        return (
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _checkpoint(
        self,
        output_text: str,
        tool_calls: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        conversation: dict[str, Any] = {
            "has_full_conversation": True,
            "messages": jsonable(self.messages),
            "output_text": output_text,
        }
        if tool_calls:
            conversation["tool_calls"] = tool_calls
        checkpoint: dict[str, Any] = {
            "task_id": self.task_id,
            "application_state_hash": state_hash(self.messages, self.read_ids),
            "env_changes": {},
            "conversation": conversation,
        }
        if self.first_llm:
            checkpoint["env_changes_baseline"] = {}
        return checkpoint

    def add_llm_event(
        self,
        event: Any,
        output_text: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        end_time = event_time_ns(event)
        start_time = min(self.last_time_ns, end_time)
        prompt_tokens, completion_tokens = self._usage(event)
        self.step += 1
        self.total_llm_calls += 1
        self.input_tokens += prompt_tokens
        self.output_tokens += completion_tokens

        span = self._base_span("llm.call", start_time, end_time, "CLIENT")
        span["attributes"] = {
            "agent.step": self.step,
            "gen_ai.request.model": self.model,
            "gen_ai.system": self.model.split("-")[0],
            "gen_ai.input.messages": json_text(self.messages),
            "gen_ai.output.messages": output_text,
            "gen_ai.response.model": self.model,
            "gen_ai.usage.input_tokens": prompt_tokens,
            "gen_ai.usage.output_tokens": completion_tokens,
            "gen_ai.request.tools": json_text(tool_schemas()),
        }
        span["check_point"] = self._checkpoint(output_text, tool_calls)
        self.spans.append(span)
        self.first_llm = False
        self.last_time_ns = end_time

        if tool_calls:
            self.messages.append(
                {"role": "assistant", "content": "", "tool_calls": tool_calls}
            )
            for call in tool_calls:
                self.pending_calls[call["id"]] = call
        else:
            self.messages.append({"role": "assistant", "content": output_text})

    def add_tool_results(self, event: Any) -> None:
        end_time = event_time_ns(event)
        start_time = min(self.last_time_ns, end_time)
        for result in list(getattr(event, "content", []) or []):
            call_id = str(getattr(result, "call_id", ""))
            name = str(getattr(result, "name", ""))
            content = str(getattr(result, "content", ""))
            is_error = bool(getattr(result, "is_error", False))
            call = self.pending_calls.pop(call_id, {})

            self.step += 1
            self.total_tool_calls += 1
            span = self._base_span(
                "tool.execution",
                start_time,
                end_time,
                "INTERNAL",
                "ERROR" if is_error else "OK",
            )
            span["attributes"] = {
                "agent.step": self.step,
                "gen_ai.tool.name": name,
                "gen_ai.tool.call.arguments": str(call.get("arguments", "{}")),
                "gen_ai.tool.call.result": content,
                "tool.duration_ms": max(0.0, (end_time - start_time) / 1_000_000),
                "tool.success": not is_error,
            }
            span["check_point"] = {
                "task_id": self.task_id,
                "application_state_hash": state_hash(self.messages, self.read_ids),
                "env_changes": {},
            }
            self.spans.append(span)
            self.messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        self.last_time_ns = end_time


def make_trace_document(
    *,
    trace_id: str,
    episode_id: str,
    root_span_id: str,
    root_start_ns: int,
    root_end_ns: int,
    sample_id: str,
    question_index: int,
    question: str,
    model: str,
    collector: SpanCollector,
    outcome: str,
    pass_details: dict[str, Any],
) -> dict[str, Any]:
    total_steps = collector.total_llm_calls + collector.total_tool_calls
    task_id = f"{sample_id}:q{question_index:03d}"
    root_span = {
        "type": "invoke_agent",
        "span_id": root_span_id,
        "trace_id": trace_id,
        "parent_id": None,
        "start_time": root_start_ns,
        "end_time": root_end_ns,
        "attributes": {
            "agent.episode.id": episode_id,
            "gen_ai.agent.name": AGENT_NAME,
            "model": model,
            "benchmark": BENCHMARK_NAME,
            "benchmark_task_id": task_id,
            "agent.outcome": outcome,
            "agent.total_steps": total_steps,
            "agent.total_llm_calls": collector.total_llm_calls,
            "agent.total_tool_calls": collector.total_tool_calls,
            "agent.wall_time_ms": max(
                0.0, (root_end_ns - root_start_ns) / 1_000_000
            ),
        },
        "status": "OK" if outcome == "success" else "ERROR",
        "kind": "SERVER",
    }
    return {
        "trace_id": trace_id,
        "episode_id": episode_id,
        "agent_name": AGENT_NAME,
        "task": f"LocoMo {task_id}: {question}",
        "total_steps": total_steps,
        "total_llm_calls": collector.total_llm_calls,
        "total_tool_calls": collector.total_tool_calls,
        "input_tokens": collector.input_tokens,
        "output_tokens": collector.output_tokens,
        "spans": [root_span, *collector.spans],
        "pass": None,
        "pass_details": pass_details,
    }


async def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    sample = load_sample(Path(args.data), args.sample_id)
    conversation = sample["conversation"]
    questions = sample["qa"]
    if not 0 <= args.question_index < len(questions):
        raise ValueError(
            f"question-index {args.question_index} is out of range; "
            f"sample has {len(questions)} questions"
        )

    qa = questions[args.question_index]
    messages = flatten_messages(conversation)
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    read_ids: set[str] = set()
    get_person_id, search_messages = make_tools(
        messages, speaker_a, speaker_b, read_ids
    )
    system_message = (
        "You answer questions about a long conversation between "
        f"{speaker_a} and {speaker_b}. You do not have the conversation in your "
        "prompt. Use the available tools to find relevant messages before "
        "answering. Session dates are important for questions that use words "
        "such as yesterday, last week, or last year. Give a concise final "
        "answer. Do not mention tool names."
    )
    task_message = f"Question: {qa['question']}"
    trace_id = uuid.uuid4().hex
    episode_id = str(uuid.uuid4())
    root_span_id = uuid.uuid4().hex[:16]
    task_id = f"{args.sample_id}:q{args.question_index:03d}"
    root_start_ns = time.time_ns()
    collector = SpanCollector(
        trace_id=trace_id,
        root_span_id=root_span_id,
        task_id=task_id,
        model=args.model,
        system_message=system_message,
        task_message=task_message,
        read_ids=read_ids,
    )

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
        name="locomo_autogen",
        model_client=model_client,
        tools=[get_person_id, search_messages],
        system_message=system_message,
        reflect_on_tool_use=True,
        max_tool_iterations=args.max_tool_iterations,
        model_client_stream=False,
    )
    final_answer: str | None = None
    task_result: TaskResult | None = None

    try:
        async for event in agent.run_stream(task=task_message):
            event_type = type(event).__name__
            print(f"event: {event_type}")
            if event_type == "ToolCallRequestEvent":
                calls = [
                    {
                        "id": str(getattr(call, "id", "")),
                        "name": str(getattr(call, "name", "")),
                        "arguments": str(getattr(call, "arguments", "{}")),
                    }
                    for call in (getattr(event, "content", []) or [])
                ]
                collector.add_llm_event(event, json_text(calls), calls)
            elif event_type == "ToolCallExecutionEvent":
                collector.add_tool_results(event)
            elif event_type == "TextMessage" and getattr(
                event, "models_usage", None
            ) is not None:
                collector.add_llm_event(
                    event, str(getattr(event, "content", ""))
                )
            elif isinstance(event, TaskResult):
                task_result = event

        if task_result is not None:
            for message in reversed(task_result.messages):
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    final_answer = content.strip()
                    break
        if final_answer is None:
            raise RuntimeError("AutoGen returned no final text answer")

        details = {
            "benchmark": BENCHMARK_NAME,
            "sample_id": args.sample_id,
            "question_index": args.question_index,
            "question": qa["question"],
            "predicted_answer": final_answer,
            "ground_truth": qa.get("answer"),
            "evidence": qa.get("evidence", []),
            "category": qa.get("category"),
            "read_ids": sorted(read_ids),
        }
        document = make_trace_document(
            trace_id=trace_id,
            episode_id=episode_id,
            root_span_id=root_span_id,
            root_start_ns=root_start_ns,
            root_end_ns=time.time_ns(),
            sample_id=args.sample_id,
            question_index=args.question_index,
            question=qa["question"],
            model=args.model,
            collector=collector,
            outcome="success",
            pass_details=details,
        )
        write_json_atomic(output_path, document)
        print(f"question: {qa['question']}")
        print(f"predicted_answer: {final_answer}")
        print(f"ground_truth: {qa.get('answer')}")
        print(f"read_ids: {sorted(read_ids)}")
        print(f"trace_saved: {output_path}")
    except Exception as exc:
        details = {
            "benchmark": BENCHMARK_NAME,
            "sample_id": args.sample_id,
            "question_index": args.question_index,
            "question": qa["question"],
            "ground_truth": qa.get("answer"),
            "evidence": qa.get("evidence", []),
            "category": qa.get("category"),
            "read_ids": sorted(read_ids),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        document = make_trace_document(
            trace_id=trace_id,
            episode_id=episode_id,
            root_span_id=root_span_id,
            root_start_ns=root_start_ns,
            root_end_ns=time.time_ns(),
            sample_id=args.sample_id,
            question_index=args.question_index,
            question=qa["question"],
            model=args.model,
            collector=collector,
            outcome="error",
            pass_details=details,
        )
        write_json_atomic(output_path, document)
        raise
    finally:
        await model_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=(
            "/share/home/tanylang/trace_collect/"
            "benchmarks/locomo/data/locomo10.json"
        ),
    )
    parser.add_argument("--sample-id", default="conv-26")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument(
        "--base-url", default="http://192.168.10.25:8201/v1"
    )
    parser.add_argument("--model", default="Qwen3.8-27B")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-tool-iterations", type=int, default=10)
    parser.add_argument(
        "--output",
        default=(
            "/share/home/tanylang/trace_collect/"
            "autogen_locomo/traces/conv-26_q000.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
