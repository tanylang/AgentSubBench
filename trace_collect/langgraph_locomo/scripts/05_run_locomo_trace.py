#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import PrivateAttr


DIMS = 384
SYSTEM_PROMPT = """You are a memory agent processing a chronological conversation for later question answering.
During ingestion, store only durable facts, events, preferences, relationships, plans, and dated information that may matter later. Resolve relative dates using the supplied session date when possible. Use manage_memory to create, update, or delete memories and search_memory when existing memory may need checking. Do not role-play the speakers. After processing an ingestion message, reply briefly with ACK.
During QA, use the memory tools when relevant and answer only from stored memory. Give a concise answer. Never invent missing facts."""


def now_ns() -> int:
    return time.time_ns()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    return repr(value)


def json_text(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
def result_text(value: Any) -> str:
    """Keep textual tool output as text; serialize structured output once."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json_text(value)

def embed_one(text: str) -> list[float]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    features = list(words)
    for word in words:
        padded = f"^{word}$"
        features.extend(padded[index : index + 3] for index in range(max(1, len(padded) - 2)))
    vector = [0.0] * DIMS
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        vector[number % DIMS] += 1.0 if ((number >> 9) & 1) == 0 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_one(text) for text in texts]


def trace_message(message: BaseMessage) -> dict[str, Any]:
    if message.type == "human":
        role = "user"
    elif message.type == "ai":
        role = "assistant"
    elif message.type == "system":
        role = "system"
    elif message.type == "tool":
        role = "tool"
    else:
        role = message.type

    output: dict[str, Any] = {"role": role, "content": jsonable(message.content)}
    if isinstance(message, AIMessage) and message.tool_calls:
        output["tool_calls"] = jsonable(message.tool_calls)
    if getattr(message, "name", None):
        output["name"] = message.name
    if getattr(message, "tool_call_id", None):
        output["tool_call_id"] = message.tool_call_id
    return output


def memory_records(store: InMemoryStore, namespace: tuple[str, ...]) -> list[dict[str, Any]]:
    records = []
    for item in store.search(namespace, limit=10000):
        records.append(
            {
                "namespace": list(item.namespace),
                "key": item.key,
                "value": jsonable(item.value),
                "created_at": jsonable(getattr(item, "created_at", None)),
                "updated_at": jsonable(getattr(item, "updated_at", None)),
            }
        )
    return sorted(records, key=lambda item: item["key"])


def semantic_memory_hash(records: list[dict[str, Any]]) -> str:
    semantic = [{"namespace": item["namespace"], "key": item["key"], "value": item["value"]} for item in records]
    encoded = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def memory_diff(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    before_map = {item["key"]: item for item in before}
    after_map = {item["key"]: item for item in after}
    changes = []
    for key in sorted(before_map.keys() | after_map.keys()):
        if key not in before_map:
            changes.append({"operation": "create", "key": key, "after": after_map[key]["value"]})
        elif key not in after_map:
            changes.append({"operation": "delete", "key": key, "before": before_map[key]["value"]})
        elif before_map[key]["value"] != after_map[key]["value"]:
            changes.append(
                {
                    "operation": "update",
                    "key": key,
                    "before": before_map[key]["value"],
                    "after": after_map[key]["value"],
                }
            )
    return changes


class TraceCollector:
    def __init__(self, task_id: str, model: str) -> None:
        self.trace_id = uuid.uuid4().hex
        self.root_span_id = uuid.uuid4().hex[:16]
        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        self.model = model
        self.children: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self.state_provider: Callable[[], list[dict[str, Any]]] = lambda: []
        self.pending_tools: dict[str, dict[str, Any]] = {}

    def set_context(self, **values: Any) -> None:
        self.context = jsonable(values)

    def state(self) -> list[dict[str, Any]]:
        return self.state_provider()

    def checkpoint(
        self,
        records: list[dict[str, Any]],
        env_changes: list[dict[str, Any]],
        conversation: dict[str, Any] | None = None,
        before_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        checkpoint: dict[str, Any] = {
            "task_id": self.task_id,
            "application_state_hash": semantic_memory_hash(records),
            "env_changes": env_changes,
            "locomo": self.context,
            "memory_count": len(records),
        }
        if conversation is not None:
            checkpoint["conversation"] = conversation
        if before_records is not None:
            checkpoint["env_changes_baseline"] = {
                "memory_count": len(before_records),
                "memory_hash": semantic_memory_hash(before_records),
            }
        return checkpoint

    def add_llm(
        self,
        messages: list[BaseMessage],
        result: Any,
        request_kwargs: dict[str, Any],
        start_time: int,
        end_time: int,
        duration_ms: float,
        before_records: list[dict[str, Any]],
        error: Exception | None = None,
    ) -> None:
        after_records = self.state()
        output_message = None if result is None else result.generations[0].message
        output_text = "" if output_message is None else jsonable(output_message.content)
        reasoning = ""
        tool_calls: list[dict[str, Any]] = []
        response_model = self.model
        input_tokens = 0
        output_tokens = 0
        if output_message is not None:
            reasoning = output_message.additional_kwargs.get("reasoning") or ""
            tool_calls = jsonable(getattr(output_message, "tool_calls", []) or [])
            response_model = (
                output_message.response_metadata.get("model_name")
                or output_message.response_metadata.get("model")
                or self.model
            )
            usage = output_message.usage_metadata or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)

        input_messages = [trace_message(message) for message in messages]
        output_messages = [] if output_message is None else [trace_message(output_message)]
        attributes: dict[str, Any] = {
            "agent.step": 0,
            "gen_ai.input.messages": json_text(input_messages),
            "gen_ai.output.messages": json_text(output_messages),
            "gen_ai.output.reasoning": reasoning,
            "gen_ai.request.duration_ms": duration_ms,
            "gen_ai.request.model": self.model,
            "gen_ai.response.model": response_model,
            "gen_ai.system": SYSTEM_PROMPT,
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "locomo.phase": self.context.get("phase"),
        }
        if request_kwargs.get("tools") is not None:
            attributes["gen_ai.request.tools"] = json_text(request_kwargs["tools"])
        if error is not None:
            attributes["error.type"] = type(error).__name__
            attributes["error.message"] = str(error)

        conversation = {
            "has_full_conversation": True,
            "messages": input_messages,
            "output_text": output_text,
            "output_reasoning": reasoning,
            "tool_calls": tool_calls,
        }
        event_type = "action" if tool_calls else "terminal"
        self.children.append(
            {
                "type": "llm.call",
                "span_id": uuid.uuid4().hex[:16],
                "trace_id": self.trace_id,
                "parent_id": self.root_span_id,
                "start_time": start_time,
                "end_time": end_time,
                "attributes": attributes,
                "status": "ERROR" if error else "OK",
                "kind": "CLIENT",
                "check_point": self.checkpoint(
                    after_records,
                    [],
                    conversation=conversation,
                    before_records=before_records,
                ),
                "duration_ms": duration_ms,
                "event_type": event_type,
                "gap_from_prev_ms": 0.0,
            }
        )

    def begin_tool(self, run_id: Any, name: str, arguments: Any) -> None:
        before_records = self.state()
        self.pending_tools[str(run_id)] = {
            "name": name,
            "arguments": jsonable(arguments),
            "start_time": now_ns(),
            "start_mono": time.perf_counter_ns(),
            "before_records": before_records,
            "context": dict(self.context),
        }

    def end_tool(self, run_id: Any, output: Any = None, error: Exception | None = None) -> None:
        pending = self.pending_tools.pop(str(run_id), None)
        if pending is None:
            return
        end_time = now_ns()
        end_mono = time.perf_counter_ns()
        after_records = self.state()
        changes = memory_diff(pending["before_records"], after_records)
        if hasattr(output, "content"):
            result_value = output.content
        else:
            result_value = output
        attributes: dict[str, Any] = {
            "agent.step": 0,
            "gen_ai.tool.call.arguments": json_text(pending["arguments"]),
            "gen_ai.tool.call.result": result_text(result_value),
            "gen_ai.tool.name": pending["name"],
            "tool.duration_ms": (end_mono - pending["start_mono"]) / 1_000_000,
            "tool.success": error is None,
            "locomo.phase": pending["context"].get("phase"),
        }
        if error is not None:
            attributes["error.type"] = type(error).__name__
            attributes["error.message"] = str(error)
        self.children.append(
            {
                "type": "tool.execution",
                "span_id": uuid.uuid4().hex[:16],
                "trace_id": self.trace_id,
                "parent_id": self.root_span_id,
                "start_time": pending["start_time"],
                "end_time": end_time,
                "attributes": attributes,
                "status": "ERROR" if error else "OK",
                "kind": "INTERNAL",
                "check_point": self.checkpoint(
                    after_records,
                    changes,
                    before_records=pending["before_records"],
                ),
                "duration_ms": (end_mono - pending["start_mono"]) / 1_000_000,
                "event_type": "observation",
                "gap_from_prev_ms": 0.0,
            }
        )

    def finalize_children(self) -> None:
        self.children.sort(key=lambda span: (span["start_time"], span["end_time"]))
        previous_end = None
        for step, span in enumerate(self.children, start=1):
            span["attributes"]["agent.step"] = step
            if previous_end is None:
                span["gap_from_prev_ms"] = 0.0
            else:
                span["gap_from_prev_ms"] = (span["start_time"] - previous_end) / 1_000_000
            previous_end = max(previous_end or span["end_time"], span["end_time"])


class ToolTraceHandler(BaseCallbackHandler):
    def __init__(self, collector: TraceCollector) -> None:
        self.collector = collector

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.collector.begin_tool(
            run_id,
            serialized.get("name") or "unknown_tool",
            inputs if inputs is not None else input_str,
        )

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self.collector.end_tool(run_id, output=output)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        if isinstance(error, Exception):
            exception = error
        else:
            exception = RuntimeError(str(error))
        self.collector.end_tool(run_id, error=exception)


class TracedReasoningChatOpenAI(ChatOpenAI):
    _collector: TraceCollector = PrivateAttr()

    def __init__(self, *, collector: TraceCollector, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._collector = collector

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
        for generation, choice in zip(result.generations, raw_response.get("choices") or []):
            raw_message = choice.get("message") or {}
            for key in ("reasoning", "reasoning_content", "output_reasoning"):
                reasoning = raw_message.get(key)
                if reasoning not in (None, ""):
                    generation.message.additional_kwargs["reasoning"] = reasoning
                    generation.message.additional_kwargs["reasoning_source_field"] = key
                    break
        return result

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        start_time = now_ns()
        start_mono = time.perf_counter_ns()
        before_records = self._collector.state()
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except Exception as error:
            end_mono = time.perf_counter_ns()
            end_time = now_ns()
            self._collector.add_llm(
                messages,
                None,
                kwargs,
                start_time,
                end_time,
                (end_mono - start_mono) / 1_000_000,
                before_records,
                error=error,
            )
            raise
        end_mono = time.perf_counter_ns()
        end_time = now_ns()
        self._collector.add_llm(
            messages,
            result,
            kwargs,
            start_time,
            end_time,
            (end_mono - start_mono) / 1_000_000,
            before_records,
        )
        return result


def load_sample(data_path: Path, sample_id: str) -> dict[str, Any]:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    for sample in data:
        if sample.get("sample_id") == sample_id:
            return sample
    raise ValueError(f"sample_id not found: {sample_id}")


def session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", key)
        if match and isinstance(value, list):
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def invoke_agent(agent: Any, text: str, handler: ToolTraceHandler) -> dict[str, Any]:
    return agent.invoke(
        {"messages": [HumanMessage(content=text)]},
        config={"recursion_limit": 12, "callbacks": [handler]},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--question-index", type=int, required=True)
    parser.add_argument("--session-limit", type=int, default=1)
    parser.add_argument("--turn-limit", type=int, default=3)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()

    sample = load_sample(args.data, args.sample_id)
    question_record = sample["qa"][args.question_index]
    task_id = f"locomo:{args.sample_id}:q{args.question_index:03d}"
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
    conversation = sample["conversation"]
    selected_sessions = session_numbers(conversation)[: args.session_limit]
    ingested_turns = 0

    print("===== Phase A: ingestion =====", flush=True)
    for session_number in selected_sessions:
        date = conversation.get(f"session_{session_number}_date_time")
        turns = conversation[f"session_{session_number}"][: args.turn_limit]
        for turn_index, turn in enumerate(turns):
            collector.set_context(
                phase="ingestion",
                sample_id=args.sample_id,
                session_number=session_number,
                turn_index=turn_index,
                dia_id=turn.get("dia_id"),
            )
            text = (
                f"Session date and time: {date}\n"
                f"Dialogue ID: {turn.get('dia_id')}\n"
                f"Speaker: {turn.get('speaker')}\n"
                f"Message: {turn.get('text', '')}\n"
                "Process this message for long-term memory."
            )
            invoke_agent(agent, text, handler)
            ingested_turns += 1
            print(
                f"ingested dia_id={turn.get('dia_id')} memory_count={len(collector.state())}",
                flush=True,
            )

    pre_qa_snapshot = collector.state()
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "namespace": list(namespace),
                "semantic_hash": semantic_memory_hash(pre_qa_snapshot),
                "records": pre_qa_snapshot,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== Phase B: QA =====", flush=True)
    collector.set_context(
        phase="qa",
        sample_id=args.sample_id,
        question_index=args.question_index,
    )
    qa_result = invoke_agent(
        agent,
        "Answer this LoCoMo question using the memory tools when relevant:\n"
        + question_record["question"],
        handler,
    )
    final_message = qa_result["messages"][-1]
    predicted_answer = str(final_message.content).strip()

    root_end_mono = time.perf_counter_ns()
    root_end_time = now_ns()
    collector.finalize_children()
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
            "locomo.sample_id": args.sample_id,
            "locomo.question_index": args.question_index,
            "locomo.ground_truth": jsonable(question_record.get("answer")),
            "locomo.evidence": json_text(question_record.get("evidence", [])),
            "locomo.category": question_record.get("category"),
            "locomo.ingested_sessions": len(selected_sessions),
            "locomo.ingested_turns": ingested_turns,
            "memory.pre_qa_count": len(pre_qa_snapshot),
            "memory.pre_qa_hash": semantic_memory_hash(pre_qa_snapshot),
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
        "task": question_record["question"],
        "total_llm_calls": len(llm_spans),
        "total_steps": len(collector.children),
        "total_tool_calls": len(tool_spans),
        "trace_id": collector.trace_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("question:", question_record["question"], flush=True)
    print("predicted_answer:", predicted_answer, flush=True)
    print("ground_truth:", question_record.get("answer"), flush=True)
    print("memory_count:", len(pre_qa_snapshot), flush=True)
    print("total_llm_calls:", len(llm_spans), flush=True)
    print("total_tool_calls:", len(tool_spans), flush=True)
    print("total_steps:", len(collector.children), flush=True)
    print("trace:", args.output, flush=True)
    print("snapshot:", args.snapshot, flush=True)
    print("LOCOMO_TRACE_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()

