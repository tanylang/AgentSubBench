"""LLM clients for Stage 10.

ChatModelClient speaks the OpenAI-compatible /chat/completions protocol, which
is what virtually all self-hosted SLM deployments expose (vLLM, SGLang, Ollama,
TGI...).  Point `base_url` at the deployed SLM's API root (e.g.
http://host:8000/v1) and set the candidate `models` list; the same client class
serves the baseline LLM.

Every response carries usage + wall-clock latency so replay resource outcomes
(14.5) can be computed; costs use configurable per-model price tables.
"""
from __future__ import annotations

import json
import time

from .timing_util import monotonic_ms
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class LMResponse:
    raw_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    error: str = ""
    #: native function-calling output, when the request carried `tools`
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    #: The thinking content of a large model (27B etc.) in thinking mode, recorded as
    #: a separate part of the output (same structure as the original trace's
    #: output_reasoning); not merged into raw_text, not part of the injected rc chain
    reasoning_content: str = ""
    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def estimate_tokens(text: str) -> int:
    """Very rough fallback when the server reports no usage."""
    return max(1, len(text) // 4)


class ChatModelClient:
    """OpenAI-compatible chat completions client with retry + cost accounting."""

    def __init__(self, base_url: str, api_key: str = "", model: str = "",
                 temperature: float = 0.0, max_output_tokens: int = 8192,
                 price_per_1k_input: float = 0.0, price_per_1k_output: float = 0.0,
                 timeout_s: float = 120.0, max_retries: int = 3,
                 extra_body: Optional[Dict[str, Any]] = None):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.price_per_1k_input = price_per_1k_input
        self.price_per_1k_output = price_per_1k_output
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self.session = requests.Session()

    @property
    def name(self) -> str:
        return self.model or "unknown-model"

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1000.0) * self.price_per_1k_input + \
               (completion_tokens / 1000.0) * self.price_per_1k_output

    def complete(self, messages: List[Dict[str, Any]],
                 tools: Optional[List[Dict[str, Any]]] = None,
                 response_format: Optional[Dict[str, Any]] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 tool_choice: Optional[str] = None,
                 ctx: Any = None) -> LMResponse:
        """One chat completion.  Never raises on HTTP/transport errors; the
        returned LMResponse carries `error` instead.  `ctx` (SpanContext) is
        ignored here; the mock client uses it for script lookup.

        tool_choice ("required"/"auto"/...): legacy / smolagents' native protocol forces
        "required"; other agents do not pass it (server default)."""
        if not self.base_url:
            return LMResponse(error="no base_url configured for model %r" % self.model)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        # max_tokens=None -> omit the field (server default, same as the collection
        # side which has no such key); the constructor arg max_output_tokens is no
        # longer implicitly injected — callers that want a cap must pass max_tokens
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        if self.extra_body:
            payload.update(self.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer %s" % self.api_key

        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            # Freeze-clock immunity: target-layer generation happens in the freeze leak region
            # after restore, perf_counter's two samples are fake epochs with a 0 difference
            # (measured: target_gen always 0); monotonic_ms uses ctypes syscalls, unaffected by freezegun
            t0 = monotonic_ms()
            try:
                resp = self.session.post(
                    self.base_url + "/chat/completions",
                    json=payload, headers=headers,
                    timeout=self.timeout_s,
                )
                latency_ms = monotonic_ms() - t0
            except requests.exceptions.RequestException as exc:
                last_error = "transport error: %s" % exc
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return LMResponse(error=last_error)

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    choice = data["choices"][0]
                    message = choice.get("message") or {}
                    text = message.get("content") or ""
                    tool_calls: List[Dict[str, Any]] = []
                    for tc in (message.get("tool_calls") or []):
                        fn = tc.get("function") or {}
                        try:
                            arguments = json.loads(fn.get("arguments") or "{}")
                        except ValueError:
                            arguments = {"_raw": fn.get("arguments")}
                        tool_calls.append({
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": arguments,
                        })
                    usage = data.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(json_dumps(messages)))
                    completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(text))
                    # A large model's (27B etc.) thinking is in a separate field (reasoning/
                    # reasoning_content); capture it as a separate part of the output
                    # (same structure as the original trace's output_reasoning);
                    # not merged into raw_text
                    reasoning = (message.get("reasoning_content")
                                 or message.get("reasoning") or "")
                    return LMResponse(
                        raw_text=text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        latency_ms=latency_ms,
                        finish_reason=choice.get("finish_reason", ""),
                        tool_calls=tool_calls,
                        reasoning_content=reasoning,
                    )
                except (ValueError, KeyError, IndexError) as exc:
                    last_error = "bad response payload: %s" % exc
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    return LMResponse(error=last_error)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = "HTTP %d: %s" % (resp.status_code, resp.text[:300])
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                return LMResponse(error=last_error)
            return LMResponse(error="HTTP %d: %s" % (resp.status_code, resp.text[:300]))

        return LMResponse(error=last_error)


def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
