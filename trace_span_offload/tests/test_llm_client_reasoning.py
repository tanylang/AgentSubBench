#!/usr/bin/env python
"""llm_client data-flavor smoke tests (client layer for the 2026-09-20 .1/.2 changes):

  - .1: the LLM thinking-mode reasoning_content/reasoning is captured as a separate output
    into LMResponse.reasoning_content, not merged into raw_text;
  - .2: when the caller explicitly passes max_tokens, the outbound payload carries that value (=8192 flavor);
    with max_tokens=None the field is omitted (server default, relied on by the diag probe).

Verified with an in-thread mock HTTP server (no real LLM endpoint needed, runs locally on Windows):
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONPATH=. \
    .venv_appworld/Scripts/python.exe -m pytest tests/test_llm_client_reasoning.py -v
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from span_offload.llm_client import ChatModelClient


class _MockHandler(BaseHTTPRequestHandler):
    captured: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(
            int(self.headers.get("Content-Length") or 0)))
        _MockHandler.captured.append(body)
        reasoning = body.get("messages", [{}])[0].get(
            "x_reasoning", "")  # echo the reasoning back if the first message carries that key
        resp = {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "final content",
                    "reasoning_content": reasoning or None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "phone__search_contacts",
                                     "arguments": "{\"query\": \"x\"}"},
                    }],
                },
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        payload = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence access logs
        pass


@pytest.fixture(scope="module")
def mock_server():
    _MockHandler.captured = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()
    t.join(timeout=5)


def _client(mock_server, max_output_tokens=8192):
    return ChatModelClient(base_url=mock_server, api_key="k", model="m",
                           temperature=0.0, max_output_tokens=max_output_tokens)


def test_max_tokens_8192_emitted(mock_server):
    """Explicitly passing max_tokens=8192 -> the outbound payload carries that cap (.2 flavor)."""
    _MockHandler.captured = []
    c = _client(mock_server)
    c.complete([{"role": "user", "content": "hi"}], max_tokens=8192)
    assert _MockHandler.captured, "no request captured"
    assert _MockHandler.captured[0]["max_tokens"] == 8192


def test_max_tokens_none_omits_field(mock_server):
    """max_tokens=None -> the field is omitted (diag probe / server-default flavor preserved)."""
    _MockHandler.captured = []
    c = _client(mock_server)
    c.complete([{"role": "user", "content": "hi"}], max_tokens=None)
    assert _MockHandler.captured, "no request captured"
    assert "max_tokens" not in _MockHandler.captured[0]


def test_reasoning_captured_separately(mock_server):
    """reasoning_content goes into LMResponse.reasoning_content, not merged into raw_text (.1 flavor)."""
    _MockHandler.captured = []
    c = _client(mock_server)
    resp = c.complete(
        [{"role": "user", "content": "hi", "x_reasoning": "I think..."}])
    assert resp.ok
    assert resp.reasoning_content == "I think..."
    assert resp.raw_text == "final content"  # the body text contains no reasoning
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["name"] == "phone__search_contacts"
    assert resp.tool_calls[0]["arguments"] == {"query": "x"}


def test_reasoning_empty_when_absent(mock_server):
    """Without a reasoning field (small models), reasoning_content is "" and the output is all in the body text."""
    _MockHandler.captured = []
    c = _client(mock_server)
    resp = c.complete([{"role": "user", "content": "hi"}])
    assert resp.ok
    assert resp.reasoning_content == ""
    assert resp.raw_text == "final content"


def test_reasoning_key_fallback(mock_server):
    """Also captured when newer vLLM Qwen3 returns the reasoning key name."""
    _MockHandler.captured = []

    class _AltHandler(_MockHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(
                int(self.headers.get("Content-Length") or 0)))
            _MockHandler.captured.append(body)
            resp = {"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "answer",
                "reasoning": "thinking-aloud"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            payload = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = HTTPServer(("127.0.0.1", 0), _AltHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = _client("http://127.0.0.1:%d" % srv.server_address[1])
        resp = c.complete([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
        t.join(timeout=5)
    assert resp.reasoning_content == "thinking-aloud"
    assert resp.raw_text == "answer"
