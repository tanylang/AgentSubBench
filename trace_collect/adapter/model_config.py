"""Shared model request configuration.

DeepSeek v4 enables thinking mode by default, but under function calling,
tool_choice="required" is rejected by the API (which the legacy FC flow requires),
so thinking mode is disabled by default. Each agent injects extra_body through its
own framework's config; monkey-patching is no longer used.

To enable thinking mode, set THINKING_MODE to {"thinking": {"type": "enabled"}}.

Model switching: the MODEL_NAME / MODEL_BASE_URL environment variables (main-loop
model) and PREDICTOR_MODEL_NAME / PREDICTOR_BASE_URL (API predictor model) can
override the default DeepSeek configuration and point to any OpenAI-compatible
service (e.g. a local vLLM).
"""

import os

# ── Main-loop model (agent reasoning), API predictor model (pre-filtering candidate APIs), validator judge model ──
MAIN_MODEL_NAME: str = os.environ.get("MODEL_NAME", "deepseek-v4-flash")
MAIN_BASE_URL: str = os.environ.get("MODEL_BASE_URL", "https://api.deepseek.com/v1")
PREDICTOR_MODEL_NAME: str = os.environ.get("PREDICTOR_MODEL_NAME", "deepseek-chat")
PREDICTOR_BASE_URL: str = os.environ.get("PREDICTOR_BASE_URL", "https://api.deepseek.com/v1")
# Validator LLM judge: independent of the main-loop model, hard-defaults to flash (does not follow the MODEL_NAME env var)
JUDGE_MODEL_NAME: str = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
JUDGE_BASE_URL: str = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
JUDGE_TEMPERATURE: float = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_SEED: int = int(os.environ.get("JUDGE_SEED", "100"))
JUDGE_MAX_TOKENS: int = int(os.environ.get("JUDGE_MAX_TOKENS", "4096"))
# Judge thinking switch (JUDGE_THINKING=on/off, default on): temperature/seed stay
# fixed in thinking mode, so verdict reproducibility is unaffected (thinking content
# does not affect the determinism of the final verdict)
_JUDGE_THINKING = os.environ.get("JUDGE_THINKING", "on").lower()
if _JUDGE_THINKING == "on":
    if JUDGE_MODEL_NAME.lower().startswith("qwen"):
        JUDGE_THINKING_MODE: dict = {"enable_thinking": True}
    elif JUDGE_MODEL_NAME.lower().startswith("deepseek"):
        JUDGE_THINKING_MODE = {"thinking": {"type": "enabled"}}
    else:
        JUDGE_THINKING_MODE = {}
else:
    if JUDGE_MODEL_NAME.lower().startswith("deepseek"):
        JUDGE_THINKING_MODE = {"thinking": {"type": "disabled"}}
    elif JUDGE_MODEL_NAME.lower().startswith("qwen"):
        JUDGE_THINKING_MODE = {"enable_thinking": False}
    else:
        JUDGE_THINKING_MODE = {}

# Thinking mode switch (MODEL_THINKING=on/off/auto, default on):
# - on (default): inject "thinking on" params per model
#   · Qwen (vLLM deployment): enable_thinking=true — vLLM disables thinking for
#     Qwen3 by default (the template appends /no_think), so it must be explicitly
#     enabled; the server also needs the --enable-reasoning startup flag
#   · deepseek: no injection (the server enables thinking by default)
# - off / auto: inject "thinking off" params per model
#   · deepseek: thinking.type=disabled (official field)
#   · Qwen (vLLM deployment): enable_thinking=false (Qwen3 thinking switch)
#   · other models: empty extra_body (no thinking params)
_MODEL_THINKING = os.environ.get("MODEL_THINKING", "on").lower()
if _MODEL_THINKING == "on":
    if MAIN_MODEL_NAME.lower().startswith("qwen"):
        THINKING_MODE = {"enable_thinking": True}
    else:
        THINKING_MODE = {}
else:
    if MAIN_MODEL_NAME.lower().startswith("deepseek"):
        THINKING_MODE = {"thinking": {"type": "disabled"}}
    elif MAIN_MODEL_NAME.lower().startswith("qwen"):
        THINKING_MODE = {"enable_thinking": False}
    else:
        THINKING_MODE = {}
