"""Configuration loading for Stage 10.

Config file is JSON (see config.example.json).  Credentials and endpoints can
also be supplied through environment variables, which is the recommended way
to inject the deployed SLM API address:

    STAGE10_SLM_BASE_URL       e.g. http://host:8000/v1
    STAGE10_SLM_API_KEY
    STAGE10_SLM_MODELS         comma-separated candidate model names
    STAGE10_BASELINE_BASE_URL
    STAGE10_BASELINE_API_KEY
    STAGE10_BASELINE_MODEL
    STAGE10_JUDGE_BASE_URL     verifier LLM judge endpoint (collection side / offload side separate,
    STAGE10_JUDGE_API_KEY      the offload side holds its own judge config, does not read $TRACE_COLLECT)
    STAGE10_JUDGE_MODEL
    STAGE10_OFFLINE            1 -> use the deterministic mock LLM everywhere
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_DEFAULT_MODEL = {
    "base_url": "",
    "api_key": "",
    "api_key_env": "",
    "model": "",
    "models": [],
    "temperature": 0.0,
    "max_output_tokens": 8192,
    "price_per_1k_input": 0.0,
    "price_per_1k_output": 0.0,
    "timeout_s": 120.0,
    "max_retries": 3,
}

_ENV_OVERRIDES = {
    "STAGE10_SLM_BASE_URL": ("slm", "base_url"),
    "STAGE10_SLM_API_KEY": ("slm", "api_key"),
    "STAGE10_SLM_MODEL": ("slm", "model"),
    "STAGE10_SLM_MODELS": ("slm", "models", "csv"),
    "STAGE10_BASELINE_BASE_URL": ("baseline_llm", "base_url"),
    "STAGE10_BASELINE_API_KEY": ("baseline_llm", "api_key"),
    "STAGE10_BASELINE_MODEL": ("baseline_llm", "model"),
    "STAGE10_JUDGE_BASE_URL": ("judge", "base_url"),
    "STAGE10_JUDGE_API_KEY": ("judge", "api_key"),
    "STAGE10_JUDGE_MODEL": ("judge", "model"),
}


@dataclass
class ModelConfig:
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""
    model: str = ""
    models: List[str] = field(default_factory=list)
    temperature: float = 0.0
    max_output_tokens: int = 8192
    price_per_1k_input: float = 0.0
    price_per_1k_output: float = 0.0
    timeout_s: float = 120.0
    max_retries: int = 3
    extra_body: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    @property
    def candidate_models(self) -> List[str]:
        if self.models:
            return [m for m in self.models if m]
        if self.model:
            return [self.model]
        return []


@dataclass
class ReplayConfig:
    fallback_mode: str = "recorded_baseline"  # none | recorded_baseline | live_baseline
    max_remaining_steps_margin: int = 10
    timeout_s: float = 600.0
    budget_usd: float = 10.0
    min_cell_samples: int = 10


@dataclass
class AllowlistConfig:
    lcb_threshold: float = 0.8
    min_cell_samples: int = 10
    rules: Dict[str, bool] = field(default_factory=lambda: {
        "read_only": True,
        "structured_output": True,
        "strong_pre_commit_verifier": True,
        "lcb_above_threshold": True,
    })


@dataclass
class PathsConfig:
    traces_dir: str = "data/traces"
    spans_path: str = "data/spans.jsonl"
    replays_path: str = "data/replays.jsonl"
    report_dir: str = "data/report"
    manifest_path: str = ""


@dataclass
class Stage10Config:
    baseline_llm: ModelConfig = field(default_factory=ModelConfig)
    slm: ModelConfig = field(default_factory=ModelConfig)
    judge: ModelConfig = field(default_factory=ModelConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    allowlist: AllowlistConfig = field(default_factory=AllowlistConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    offline: bool = False
    pilot_tasks: List[str] = field(default_factory=list)
    heldout_tasks: List[str] = field(default_factory=list)


def _apply_env(raw: Dict) -> Dict:
    for env_name, spec in _ENV_OVERRIDES.items():
        value = os.environ.get(env_name)
        if not value:
            continue
        section, key = spec[0], spec[1]
        kind = spec[2] if len(spec) > 2 else "str"
        raw.setdefault(section, {})
        if kind == "csv":
            raw[section][key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            raw[section][key] = value
    return raw


def load_config(path: Optional[str] = None) -> Stage10Config:
    raw: Dict = {}
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    raw = _apply_env(raw)

    def model_cfg(section: str) -> ModelConfig:
        d = {**_DEFAULT_MODEL, **(raw.get(section) or {})}
        d["extra_body"] = dict(d.get("extra_body") or {})
        return ModelConfig(**d)

    replay = raw.get("replay") or {}
    allowlist = raw.get("allowlist") or {}
    paths = raw.get("paths") or {}
    cfg = Stage10Config(
        baseline_llm=model_cfg("baseline_llm"),
        slm=model_cfg("slm"),
        judge=model_cfg("judge"),
        replay=ReplayConfig(
            fallback_mode=replay.get("fallback_mode", "recorded_baseline"),
            max_remaining_steps_margin=int(replay.get("max_remaining_steps_margin", 10)),
            timeout_s=float(replay.get("timeout_s", 600.0)),
            budget_usd=float(replay.get("budget_usd", 10.0)),
            min_cell_samples=int(replay.get("min_cell_samples", 10)),
        ),
        allowlist=AllowlistConfig(
            lcb_threshold=float(allowlist.get("lcb_threshold", 0.8)),
            min_cell_samples=int(allowlist.get("min_cell_samples", 10)),
            rules=dict(allowlist.get("rules", AllowlistConfig().rules)),
        ),
        paths=PathsConfig(
            traces_dir=paths.get("traces_dir", "data/traces"),
            spans_path=paths.get("spans_path", "data/spans.jsonl"),
            replays_path=paths.get("replays_path", "data/replays.jsonl"),
            report_dir=paths.get("report_dir", "data/report"),
            manifest_path=paths.get("manifest_path", ""),
        ),
        offline=bool(raw.get("offline", False)) or os.environ.get("STAGE10_OFFLINE") == "1",
        pilot_tasks=list(raw.get("experiment", {}).get("pilot_tasks", [])) if isinstance(raw.get("experiment", {}), dict) else [],
        heldout_tasks=list(raw.get("experiment", {}).get("heldout_tasks", [])) if isinstance(raw.get("experiment", {}), dict) else [],
    )
    return cfg
