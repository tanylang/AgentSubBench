import json
import time
from pathlib import Path

from openai import OpenAI


DATA_PATH = Path("/share/home/tanylang/locomo/data/locomo10.json")
OUTPUT_PATH = Path("/share/home/tanylang/locomo/outputs/one_result.json")

BASE_URL = "http://192.168.10.25:8201/v1"
MODEL = "Qwen3.8-27B"


def session_number(name):
    return int(name.split("_")[1])


def build_conversation(conversation):
    session_names = sorted(
        (
            key
            for key in conversation
            if key.startswith("session_")
            and not key.endswith("_date_time")
        ),
        key=session_number,
    )

    lines = []

    for session_name in session_names:
        date = conversation.get(session_name + "_date_time", "unknown date")
        lines.append(f"\n### {session_name} ({date})")

        for turn in conversation[session_name]:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            dia_id = turn.get("dia_id", "")
            lines.append(f"[{dia_id}] {speaker}: {text}")

    return "\n".join(lines)


def main():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[0]
    qa = sample["qa"][0]
    conversation_text = build_conversation(sample["conversation"])

    system_prompt = (
        "You answer questions using only the supplied conversation history. "
        "Do not use outside knowledge. Give a short, direct answer. "
        "If the answer cannot be determined, say 'Unknown'."
    )

    user_prompt = (
        "Conversation history:\n"
        f"{conversation_text}\n\n"
        "Question:\n"
        f"{qa['question']}\n\n"
        "Answer:"
    )

    client = OpenAI(
        base_url=BASE_URL,
        api_key="EMPTY",
    )

    print("sample_id:", sample["sample_id"], flush=True)
    print("question:", qa["question"], flush=True)
    print("ground_truth:", qa["answer"], flush=True)
    print("calling model:", MODEL, flush=True)

    start_time = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    duration = time.time() - start_time
    message = response.choices[0].message
    predicted_answer = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)

    trace = {
        "event": "llm.call",
        "benchmark": "locomo",
        "sample_id": sample["sample_id"],
        "question_index": 0,
        "question": qa["question"],
        "ground_truth": qa["answer"],
        "evidence": qa.get("evidence", []),
        "category": qa.get("category"),
        "model": MODEL,
        "base_url": BASE_URL,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "predicted_answer": predicted_answer,
        "reasoning_content": reasoning,
        "duration_seconds": duration,
        "usage": (
            response.usage.model_dump()
            if response.usage is not None
            else None
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("predicted_answer:", predicted_answer, flush=True)
    print("duration_seconds:", round(duration, 2), flush=True)
    print("trace_saved:", OUTPUT_PATH, flush=True)


if __name__ == "__main__":
    main()
