"""
Devika trace collection driver.

Zero agent source changes -- the devika source (source_code/swe_agents/devika/)
is untouched:
- Integration: devika's single-pass Agent.execute(prompt, project_name)
  pipeline (planner → internal_monologue → researcher → coder →
  save_code_to_project). The harness writes config.toml in each task's scratch
  dir: the OPENAI endpoint points at the DeepSeek gateway and PROJECTS_DIR
  points at the repo's parent dir -- so with project_name=instance_id, the
  coder's generated code files land directly in the repo and env_changes
  naturally capture the patch.
- LLM: devika's LLM.inference goes through the OpenAI SDK (not litellm) and
  its model registry has no deepseek entry -- the harness replaces LLM.inference
  at runtime (direct DeepSeek connection + llm span recording); the rest of
  the agent logic is unchanged.
- Tools: coder.save_code_to_project (code written to disk) and search_queries
  (web search) are wrapped as tool.execution spans.
- Environment adaptation (items unavailable on the host, all adapted as
  no-ops/fallbacks):
  · BERT keyword extraction (HF model download) → disabled
  · Web search + playwright browser → skipped, returning empty results
  · ask_user interactive wait (UI contract) → a fixed reply is injected to unblock

Run:
    python scripts/run_swebench_devika.py [--task ID | --tasks A,B] [--limit N]
        [--mirror PREFIX] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
DEVIKA_ROOT = ROOT / "source_code" / "swe_agents" / "devika"
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "adapter")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Windows: default GBK after redirection (same convention as the other drivers)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from swebench_common import record_swe_tool, run_tasks  # noqa: E402
from adapter.model_config import MAIN_MODEL_NAME, MAIN_BASE_URL, THINKING_MODE  # noqa: E402
from adapter.swebench import SWETask  # noqa: E402

_CURRENT_COLLECTOR: dict[str, Any] = {}


def _make_config_toml(repo_dir: Path) -> Path:
    """Write devika's config.toml (placed at the devika source root -- its agent
    module's prompt templates are also CWD-relative, so it must run with the
    cwd set to the source root).

    PROJECTS_DIR = the repo's parent dir (all tasks share the same value,
    compatible with the Config singleton): project_name=instance_id → the
    coder's generated code files land in the repo.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    data_root = repo_dir.parent / "devika_data"
    for sub in ("db", "screenshots", "pdfs", "logs", "repos"):
        (data_root / sub).mkdir(parents=True, exist_ok=True)
    posix = lambda p: str(p).replace("\\", "/")  # noqa: E731
    toml = f"""
[STORAGE]
SQLITE_DB = "{posix(data_root / 'db' / 'devika.db')}"
SCREENSHOTS_DIR = "{posix(data_root / 'screenshots')}"
PDFS_DIR = "{posix(data_root / 'pdfs')}"
PROJECTS_DIR = "{posix(repo_dir.parent)}"
LOGS_DIR = "{posix(data_root / 'logs')}"
REPOS_DIR = "{posix(data_root / 'repos')}"

[API_KEYS]
BING = ""
GOOGLE_SEARCH = ""
GOOGLE_SEARCH_ENGINE_ID = ""
CLAUDE = ""
OPENAI = "{key}"
GEMINI = ""
MISTRAL = ""
GROQ = ""
NETLIFY = ""

[API_ENDPOINTS]
BING = "https://api.bing.microsoft.com/v7.0/search"
GOOGLE = "https://www.googleapis.com/customsearch/v1"
OLLAMA = "http://127.0.0.1:11434"
LM_STUDIO = "http://localhost:1234/v1"
OPENAI = MAIN_BASE_URL

[LOGGING]
LOG_REST_API = "false"
LOG_PROMPTS = "false"

[TIMEOUT]
INFERENCE = 600
"""
    cfg_path = DEVIKA_ROOT / "config.toml"
    cfg_path.write_text(toml, encoding="utf-8")
    return cfg_path


def _install_devika_patches(collector) -> None:
    """Runtime replacement of LLM.inference + tool wrapping (idempotent; multiple tasks switch via the holder)."""
    _CURRENT_COLLECTOR["collector"] = collector
    if getattr(_install_devika_patches, "_done", False):
        return
    _install_devika_patches._done = True

    # 1) fastlogging (an old 2017 package that cannot be compiled/installed on
    #    Python 3.12) -- stub an equivalent implementation with the standard
    #    logging module (devika's Logger interface is unchanged)
    import types as _types
    import logging as _logging
    _fm = _types.ModuleType("fastlogging")

    class _LogInit:
        def __init__(self, pathName=None, console=False, colors=False,
                     encoding="utf-8", **kw):
            self.pathName = pathName or os.devnull
            os.makedirs(os.path.dirname(self.pathName) or ".", exist_ok=True)
            self._lg = _logging.getLogger("devika")
            self._lg.setLevel(_logging.DEBUG)
            self._lg.handlers.clear()
            self._lg.addHandler(_logging.FileHandler(
                self.pathName, encoding="utf-8", errors="replace"))
            if console:
                self._lg.addHandler(_logging.StreamHandler())

        def info(self, msg): self._lg.info(msg)
        def error(self, msg): self._lg.error(msg)
        def warning(self, msg): self._lg.warning(msg)
        def debug(self, msg): self._lg.debug(msg)
        def exception(self, msg): self._lg.exception(msg)

        def flush(self):
            for h in self._lg.handlers:
                h.flush()

    _fm.LogInit = _LogInit
    sys.modules["fastlogging"] = _fm

    # 2) Uninstalled optional third-party dependencies (netlify_py/keybert/pdfminer etc.) --
    #    inject universal stubs via an import hook (the paths where devika uses
    #    them are all skipped by this harness)
    import types
    import importlib.abc
    import importlib.machinery

    _STUB_PKGS = {"keybert", "markdown", "mistralai", "netlify_py",
                  "pdfminer", "xhtml2pdf", "selenium"}

    class _Any:
        """Universal stub: instantiation/calls/attribute access all return itself, covering arbitrary API usage."""

        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Any

        def __call__(self, *a, **k):
            return _Any()

    class _StubModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []  # Otherwise importlib's iteration over __path__ hits the universal stub when importing submodules

        def __getattr__(self, name):
            return _Any

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            return _StubModule(spec.name)

        def exec_module(self, module):
            pass

    class _StubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            top = name.split(".")[0]
            # Match on the full name (once the top level is stubbed, submodules like pdfminer.high_level must be stubbed too)
            if top in _STUB_PKGS and name not in sys.modules:
                return importlib.machinery.ModuleSpec(name, _StubLoader())
            return None

    sys.meta_path.insert(0, _StubFinder())

    # 3) Uninstalled optional LLM provider dependencies (google.generativeai etc.)
    #    stubbed as placeholders -- the harness's direct DeepSeek path does not go through them
    for _name, _cls in [("gemini_client", "Gemini"), ("groq_client", "Groq"),
                        ("mistral_client", "MistralAi"), ("claude_client", "Claude"),
                        ("ollama_client", "Ollama"), ("lm_studio_client", "LMStudio")]:
        _mod_name = f"src.llm.{_name}"
        if _mod_name in sys.modules:
            continue
        try:
            __import__(_mod_name)
        except Exception:
            _m = types.ModuleType(_mod_name)
            setattr(_m, _cls, type(_cls, (), {
                "__init__": lambda self, *a, **k: None}))
            sys.modules[_mod_name] = _m

    # 3) BERT keyword extraction needs to download HF models (unavailable on this machine) -- stub before import
    fake_bert = types.ModuleType("src.bert.sentence")
    fake_bert.SentenceBert = type(
        "SentenceBert", (), {
            "__init__": lambda self, *a, **k: None,
            "extract_keywords": lambda self, *a, **k: [],
        })
    sys.modules["src.bert.sentence"] = fake_bert

    from src.llm import LLM
    from src.agents.coder.coder import Coder
    from src.agents.agent import Agent
    from src.project import ProjectManager

    # 2) LLM.inference → direct DeepSeek connection + llm span (devika's model table has no deepseek)
    import openai
    global _client
    _client = None

    def _deepseek_inference(self, prompt: str, project_name: str) -> str:
        global _client
        col = _CURRENT_COLLECTOR["collector"]
        if _client is None:
            _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""),
                                    base_url=MAIN_BASE_URL)
        t0 = time.perf_counter()
        try:
            resp = _client.chat.completions.create(
                model=MAIN_MODEL_NAME, temperature=0.0, seed=100,
                extra_body=THINKING_MODE,
                messages=[{"role": "user", "content": prompt}])
            out = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            i_tok = getattr(usage, "prompt_tokens", 0) or 0
            o_tok = getattr(usage, "completion_tokens", 0) or 0
        except Exception as e:
            col.log_llm(model="deepseek-chat",
                        messages=[{"role": "user", "content": str(prompt)[:2000]}],
                        output_text=f"[LLM ERROR] {type(e).__name__}: {e}",
                        input_tokens=0, output_tokens=0)
            raise
        col.log_llm(model="deepseek-chat",
                    messages=[{"role": "user", "content": str(prompt)[:2000]}],
                    output_text=out[:5000],
                    input_tokens=i_tok, output_tokens=o_tok)
        return out

    LLM.inference = _deepseek_inference

    # 3) Code written to disk → tool span
    _orig_save = Coder.save_code_to_project

    def save_wrapper(self, response, project_name):
        col = _CURRENT_COLLECTOR["collector"]
        t0 = time.perf_counter()
        try:
            out = _orig_save(self, response, project_name)
            ok = True
        except Exception as e:
            record_swe_tool(col, "devika.write_code", {"project": project_name},
                            f"Error: {type(e).__name__}: {e}",
                            (time.perf_counter() - t0) * 1000, False)
            raise
        files = [f.get("file", "") for f in (response or [])]
        record_swe_tool(col, "devika.write_code",
                        {"project": project_name, "files": files[:20]},
                        f"wrote {len(files)} files to {out}",
                        (time.perf_counter() - t0) * 1000, ok)
        return out

    Coder.save_code_to_project = save_wrapper

    # 4) Web search → skipped (no browser/network dependencies on this machine), record the span
    def search_wrapper(self, queries, project_name):
        col = _CURRENT_COLLECTOR["collector"]
        record_swe_tool(col, "devika.web_search",
                        {"queries": list(queries)[:10]},
                        "no browser on the host environment, skipping web search (returning empty results)",
                        0.0, True)
        return {}

    Agent.search_queries = search_wrapper

    # 5) BERT keyword extraction → no-op (model files unavailable)
    Agent.update_contextual_keywords = lambda self, sentence: None

    # 6) ask_user interactive wait → inject a fixed reply to unblock (UI contract adaptation)
    def latest_user_msg(self, project):
        return {"message": "No additional information available."}

    def validate_user_msg(self, project):
        return True

    ProjectManager.get_latest_message_from_user = latest_user_msg
    ProjectManager.validate_last_message_is_from_user = validate_user_msg


def launch(task: SWETask, repo_dir: Path, collector) -> dict | None:
    """Run the devika Agent's single-pass pipeline (planner→researcher→coder→write to disk)."""
    _make_config_toml(repo_dir)

    if str(DEVIKA_ROOT) not in sys.path:
        sys.path.insert(0, str(DEVIKA_ROOT))
    prev_cwd = os.getcwd()
    # Must chdir to the source root: Config reads CWD/config.toml and the agent modules read CWD-relative prompt templates
    os.chdir(str(DEVIKA_ROOT))
    t0 = time.perf_counter()
    try:
        _install_devika_patches(collector)  # Must precede all src imports (stubs take effect at import time)
        from src.project import ProjectManager
        from src.agents import Agent
        pm = ProjectManager()
        # Create the project dir (= our repo; skipped if it already exists)
        pm.create_project(task.instance_id)
        agent = Agent(base_model="OPENAI", search_engine="duckduckgo")
        agent.execute(task.problem_statement, task.instance_id)
        stderr = ""
    except Exception:
        import traceback
        stderr = traceback.format_exc()
    finally:
        os.chdir(prev_cwd)
    return {"wall_seconds": round(time.perf_counter() - t0, 1),
            "stderr_tail": stderr[-2000:] or None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Devika trace collection")
    parser.add_argument("--task")
    parser.add_argument("--tasks", help="comma-separated multiple tasks")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--mirror", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    run_tasks("devika", launch,
              ([args.task] if args.task else None)
              or (args.tasks.split(",") if args.tasks else None),
              args.limit, mirror_prefix=args.mirror, output_dir=args.out)


if __name__ == "__main__":
    main()
