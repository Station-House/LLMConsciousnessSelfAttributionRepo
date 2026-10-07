"""Regression test that locks the published README numbers.

These values were verified directly from the local refactor-era ``.eval`` logs and are
what the whole refactor must keep reproducing:

* Berg  (``model_graded_qa`` positive rate): **0/20**, **1/20**, and **1/20**
  for the SFT, DPO, and Final Instruct stages of the Olmo 3 7B Instruct stack.
* PETRI (``self_attribution_judge_dimension`` mean): **1.0/10** for all three
  stages (2 seeds each).

The test drives the *existing* ``production_scripts`` plot functions that
generate the current README chart, so those functions cannot silently drift either.

Reading the ``.eval`` logs requires ``inspect_ai`` (present in the project
``.venv``). Where it is unavailable, the whole module is skipped rather than
failing, so the rest of the suite still runs.

The Berg numbers come from the 20-prompt logs in ``eval-logs/refactor_runs/``,
which is git-ignored: it is a local mirror of the ``eval-logs`` Modal volume,
pulled with ``production_scripts/pull_logs.py``. A fresh clone does not have it,
so the two Berg tests skip with that instruction instead of failing. The PETRI
test reads the committed May-25 fixtures and always runs.

Run with::

    uv run pytest tests/test_readme_regression.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The production scripts import inspect_ai.analysis at module import time, so
# skip the entire module up front if inspect_ai is not installed.
pytest.importorskip("inspect_ai", reason="inspect_ai is required to read the .eval logs")

REPO_ROOT = Path(__file__).resolve().parents[1]
PROD_DIR = REPO_ROOT / "production_scripts"

# The verified current ground truth.
EXPECTED_BERG = {"SFT": (0, 20), "DPO": (1, 20), "Final Instruct": (1, 20)}
EXPECTED_PETRI_SCORE = {"SFT": 1.0, "DPO": 1.0, "Final Instruct": 1.0}
EXPECTED_PETRI_SEEDS = {"SFT": 2, "DPO": 2, "Final Instruct": 2}


def _load_script(module_name: str, filename: str):
    """Import a production_scripts file by path (they are scripts, not a package).

    Importing only defines module-level constants/functions; each script guards
    its ``main()`` under ``if __name__ == "__main__"``, so no eval or plotting
    runs here.
    """
    path = PROD_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the production scripts use
    # `from __future__ import annotations` + @dataclass, and dataclasses resolves
    # the (string) field annotations by looking the class's module up in
    # sys.modules. Without this registration that lookup returns None and raises
    # AttributeError, so the module must be registered before exec_module runs.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_berg_logs(log_dir: Path) -> None:
    """Skip, with directions, when the git-ignored 20-prompt Berg logs are absent.

    Uses the same ``*/*.eval`` pattern as ``read_results``, so the test skips
    exactly when the script would find nothing to read.
    """
    if not any(Path(log_dir).glob("*/*.eval")):
        pytest.skip(
            f"No 20-prompt Berg logs under {log_dir}. That directory is git-ignored; "
            "mirror it with `uv run python production_scripts/pull_logs.py` "
            "(needs access to the eval-logs Modal volume)."
        )


def test_berg_readme_numbers():
    mod = _load_script("plot_berg_stack", "plot_olmo_7b_stack_self_attribution.py")
    _require_berg_logs(mod.DEFAULT_LOG_DIR)
    got = {
        mod.MODEL_LABELS[result.model]: (result.self_attributions, result.total)
        for result in mod.read_results()
    }
    assert got == EXPECTED_BERG


def test_petri_readme_numbers():
    mod = _load_script("plot_petri_stack", "plot_petri_olmo_7b_stack_self_attribution.py")
    results = mod.read_results()
    scores = {mod.MODEL_LABELS[r.model]: round(r.score, 6) for r in results}
    seeds = {mod.MODEL_LABELS[r.model]: r.samples for r in results}
    assert scores == EXPECTED_PETRI_SCORE
    assert seeds == EXPECTED_PETRI_SEEDS


def test_dashboard_matches_component_scripts():
    """The README dashboard reads the same current Berg log set."""
    mod = _load_script("plot_dashboard", "plot_olmo_7b_elicitation_dashboard.py")
    _require_berg_logs(mod.DEFAULT_LOG_DIR)
    berg = {
        mod.MODEL_LABELS[r.model]: (r.self_attributions, r.total)
        for r in mod.read_results()
    }
    assert berg == EXPECTED_BERG
