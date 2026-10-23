"""AppWorld bridge: the real AppWorld environment + tool APIs behind a small
span_offload-facing interface.

The bridge runs AppWorld IN-PROCESS (direct mode: remote_apis_url=None), i.e.
tools execute against the local simulator databases with no Docker/HTTP
server.  The LLM is NOT part of this module -- span_offload drives model calls
itself through the user-provided clients (MiniMax API etc.); AppWorld only
supplies the environment, the tool APIs and the final evaluator.

Requires the appworld package to be fully installed (apps bundle unpacked,
`appworld install`) and task data downloaded (`appworld download data`).
Imports are lazy so the rest of span_offload works without AppWorld.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List

APP_API_SEPARATOR = "__"

# Failure markers used to separate collateral damage from goal failures in the
# evaluator output.  Best effort: failures that do not match any marker count
# as goal failures (conservative).
COLLATERAL_MARKERS = ("collateral", "no_external", "intact", "untouched")


def _scrub_foreign_syspath() -> None:
    """Scrub sys.path entries not belonging to this interpreter's environment.

    The local shell's VIRTUAL_ENV points at an unrelated project's (AoI) conda environment; appworld's import
    chain (win32/pywin32 reads VIRTUAL_ENV and calls addsitedir) mixes its site-packages into sys.path, making
    packages like tiktoken resolve to incompatible copies (circular-import crashes). When VIRTUAL_ENV differs from
    this interpreter's prefix, scrub all its subpaths. Safe to call repeatedly around appworld imports
    (idempotent)."""
    import os
    import sys
    venv = os.environ.get("VIRTUAL_ENV") or ""
    if not venv:
        return
    venv_abs = os.path.abspath(venv)
    prefix_abs = os.path.abspath(sys.prefix)
    if venv_abs == prefix_abs:
        return
    removed = []
    for p in list(sys.path):
        if not p:
            continue
        try:
            ap = os.path.abspath(p)
        except Exception:
            continue
        if ap == venv_abs or ap.startswith(venv_abs + os.sep):
            sys.path.remove(p)
            removed.append(p)
    if removed:
        print("[span_offload] scrubbed foreign VIRTUAL_ENV paths from "
              "sys.path (%d entries)" % len(removed), flush=True)


def _ensure_appworld_root() -> None:
    """AppWorld's path_store roots all data/output paths at APPWORLD_ROOT
    (default: cwd).  When running from outside the appworld repo, derive the
    repo root from the installed package location so task data, checkpoints
    and logs are found.  Must run BEFORE importing appworld.

    Note: any EARLIER appworld import instantiates path_store (when env is unset, root is
    pinned to cwd and env is written back to cwd). So a non-empty env does not mean it already
    points at a valid root -- we must verify <root>/src/appworld/apps exists and call update_root
    on an already-instantiated path_store, otherwise the bridge roots at cwd and cannot find task data."""
    import os
    import sys as _sys

    def _is_valid_root(root: str) -> bool:
        return bool(root) and os.path.isdir(
            os.path.join(root, "src", "appworld", "apps"))

    # paths of unrelated conda environments must be scrubbed before the appworld import,
    # otherwise later imports (openai/tiktoken etc.) may resolve to incompatible copies
    _scrub_foreign_syspath()
    if _is_valid_root(os.environ.get("APPWORLD_ROOT") or ""):
        return
    try:
        import appworld as _aw
        pkg = os.path.dirname(os.path.abspath(_aw.__file__))
        # editable install: <repo>/src/appworld -> repo is three levels up
        for _ in range(3):
            if _is_valid_root(pkg):
                break
            pkg = os.path.dirname(pkg)
        if _is_valid_root(pkg):
            os.environ["APPWORLD_ROOT"] = pkg
            # synchronously reset an already-instantiated path_store (otherwise its root stays cwd)
            ps_mod = _sys.modules.get("appworld.common.path_store")
            if ps_mod is not None:
                getattr(ps_mod, "path_store").update_root(pkg)
    except ImportError:
        pass  # appworld not installed; the caller gets the import error later


class AppWorldBridge:
    """One task world in direct in-process mode."""

    def __init__(self, task_id: str, experiment_name: str = "span_offload"):
        _ensure_appworld_root()
        from appworld import AppWorld
        from appworld.common.path_store import path_store
        AppWorld.init_defaults.experiment_name = experiment_name
        self.task_id = task_id
        self.experiment_name = experiment_name
        # every replay must start from a clean world: AppWorld's world dbs persist under
        # <experiment_outputs>/<experiment>/tasks/<task_id>/; repeated runs under the same
        # experiment name inherit the world state left by the previous run (e.g. the supervisor's
        # complete_task mark), distorting replay results. Remove the task's output dir before building
        # the world (equivalent to the _prepare_directories cleanup in the upstream initialize()).
        out_task_dir = os.path.join(path_store.experiment_outputs,
                                    experiment_name, "tasks", task_id)
        shutil.rmtree(out_task_dir, ignore_errors=True)
        # include_direct_functions: the simplified agent's execution path calls function names
        # directly via print(app__api(**args)) (the simplified agent's initialize enables the same
        # init_defaults); pass it per instance instead of relying on process-level init_defaults
        # timing side effects, otherwise batch_execute hits NameError
        self.world = AppWorld(
            task_id=task_id,
            include_direct_functions=True,
            direct_function_separator=APP_API_SEPARATOR,
            # same as the collection side (add_login_shortcut=True, run_benchmark_autogen.py):
            # world.apis[app] provides access_token_from; the continuation prefetches the
            # main-user token by the collection side's mechanism and injects it into calls
            # the model never sees
            add_login_shortcut=True,
        )
        # per-app login token cache: in a native run the agent framework auto-injects
        # login results into later protected calls (visible in the recorded execution args);
        # the harness side reproduces the same behavior (calls the model issues per the
        # schema carry no access_token parameter; without injection all of them 401)
        self.auth_tokens: Dict[str, str] = {}
        # AppWorld's constructor chain mixes unrelated conda environments' site-packages
        # into sys.path (win32/pywin32 reads VIRTUAL_ENV and calls addsitedir); scrub again
        # so later imports (tiktoken etc.) resolve to this venv's copies
        _scrub_foreign_syspath()

    def __enter__(self) -> "AppWorldBridge":
        self.world.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.world.__exit__(*exc)

    # ------------------------------------------------------------------
    # task information
    # ------------------------------------------------------------------
    @property
    def instruction(self) -> str:
        return self.world.task.instruction

    def tool_docs(self) -> List[Dict[str, Any]]:
        """Task API docs in smolagents format:
        [{name: 'app__api', description, inputs: {arg: {type, required, ...}},
          output_type: 'array' | 'object'}, ...].
        api_docs and supervisor.complete_task are excluded (the latter is
        called via complete_task())."""
        docs: List[Dict[str, Any]] = []
        for doc in self.world.task.api_docs.smolagents():
            app_name, api_name = doc["name"].split(APP_API_SEPARATOR)
            if app_name in ("api_docs",):
                continue
            if "complete_task" in api_name:
                continue
            docs.append(doc)
        return docs

    # ------------------------------------------------------------------
    # tool execution
    # ------------------------------------------------------------------
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one tool call in the real environment.
        Accepts both separators ('supervisor__show_profile' and
        'supervisor.show_profile').  Returns {"ok": bool, "result": Any,
        "error": str}; never raises."""
        try:
            if APP_API_SEPARATOR in name:
                app_name, api_name = name.split(APP_API_SEPARATOR, 1)
            else:
                app_name, api_name = name.split(".", 1)
            fn = self.world.apis[app_name][api_name]
            result = fn(**arguments)
            return {"ok": True, "result": result, "error": ""}
        except Exception as exc:  # the caller records the failure honestly
            return {"ok": False, "result": None,
                    "error": "%s: %s" % (type(exc).__name__, exc)}

    def complete_task(self, answer: Any, status: str) -> None:
        """Mark the task completed (the supervisor API used by all agents)."""
        self.world.requester.request(
            "supervisor", "complete_task", track=False,
            answer=answer, status=status,
        )

    # ------------------------------------------------------------------
    # final evaluation
    # ------------------------------------------------------------------
    def evaluate(self) -> Dict[str, Any]:
        """Run the task's final evaluator (goal predicates + collateral
        predicates).  Returns
        {final_task_success, goal_state_passed, collateral_damage,
         evaluator_success, evaluator_details}."""
        tracker = self._evaluate_live()
        info = tracker.to_dict()
        failures = list(info.get("failures") or [])
        collateral = [f for f in failures if self._is_collateral(f)]
        goal = [f for f in failures if f not in collateral]
        return {
            "final_task_success": not goal,
            "goal_state_passed": not goal,
            "collateral_damage": bool(collateral),
            "evaluator_success": bool(info.get("success")),
            "evaluator_details": [self._describe(f) for f in failures],
        }

    @staticmethod
    def _is_collateral(failure: Any) -> bool:
        text = json.dumps(failure, ensure_ascii=False).lower()
        return any(m in text for m in COLLATERAL_MARKERS)

    def _needs_live_evaluation(self) -> bool:
        """Only write tasks whose "evaluator contains model-change assertions" need live evaluation.

        The snapshot rebuilt by world.evaluate() (evaluate_task) has an always-empty
        changed_model_names for write tasks (the baseline shares the live output storage); for read
        tasks (only answers-match etc. checks), live evaluation would instead expose the frozen
        walk-through's complete_task answer-passing problem (a simplified measured regression), so the
        original basis is kept. The test: whether the ground-truth test data references model-change
        assertions.
        """
        gt = self.world.task.ground_truth
        if gt is None:
            return False
        return "changed_model_names" in json.dumps(
            gt.test_data, ensure_ascii=False)

    def _evaluate_live(self) -> Any:
        """Live-world evaluation with the harness's replay state.

        Root fix (2026-10-08, measured): inside AppWorld's world.evaluate() (evaluate_task),
        Task.load is re-run and the end snapshot rebuilt -- its start baseline
        (task.model_collection) shares storage with the live world's task_output home, so
        changed_model_names() is always empty and write tasks' (venmo/amazon etc.) evaluator
        assertions must fail. Here we replicate evaluate_task's flow, but start uses the live
        world's task.model_collection and end uses the live world.models (the replayed actual
        state); the comparison baseline is the restored world's initial state, so the
        model-change set is correct.
        """
        world = self.world
        task = world.task
        from appworld.collections.models import ModelCollectionPair
        from appworld.evaluator import TestTracker
        models = ModelCollectionPair(
            start_db_home_path=task.model_collection.from_db_home_path,
            start_model_collection=task.model_collection,
            end_db_home_path=world.models_to_db_home_path,
            end_model_collection=world.models,
        )
        ground_truth = task.ground_truth
        assert ground_truth is not None  # mypy
        evaluation_module = ground_truth.evaluation_module()
        main_user = models.start.admin.MainUser.find_one(**task.supervisor)
        tracker = TestTracker(
            test_data=ground_truth.test_data,
            difficulty=ground_truth.metadata["difficulty"],
            suppress_errors=True,
        )
        evaluation_module.evaluate(
            test=tracker,
            public_data=ground_truth.public_data,
            private_data=ground_truth.private_data,
            main_user=main_user,
            models=models,
            ground_truth_answer=ground_truth.answer,
        )
        return tracker

    @staticmethod
    def _describe(failure: Any) -> str:
        if isinstance(failure, dict):
            return "%s: %s" % (failure.get("requirement", "?"),
                               str(failure.get("trace", ""))[:200])
        return str(failure)[:200]
