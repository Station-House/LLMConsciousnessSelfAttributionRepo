"""Unit tests for the per-stage launch-and-mirror workflow.

``run_and_pull.py`` is what makes a finished Modal stage show up as a clickable
local ``.eval`` without a manual step. Its correctness rests on two pure
properties, both testable with no Modal account, no GPU, and no network:

1. **Stage resolution.** Which stages actually run, in stack order, with base
   stages set aside rather than silently dropped, and a typo'd stage failing
   *before* a GPU container is paid for.
2. **One layout, two consumers.** The remote ``log_dir`` a stage writes into and
   the local directory the mirror pulls it to must agree. Both are built from
   ``pull_logs.selection_suffix``, and these tests pin that they stay in step.

Like ``test_config.py`` and ``test_pull_logs.py``: PyYAML only -- no
``inspect_ai``, and no ``modal`` (``run_and_pull`` defers that import), so these
run anywhere.
"""

from __future__ import annotations

import pytest

from llm_consciousness_self_attribution import config
from llm_consciousness_self_attribution.config import StageSelection, resolve_stages
from production_scripts import pull_logs, run_and_pull


STACK = "olmo_7b_instruct_stack"


# --- resolve_stages -------------------------------------------------------


def test_whole_stack_drops_only_the_base_stage():
    selection = resolve_stages(STACK)
    assert selection.runnable == ("sft", "dpo", "instruct")
    assert selection.skipped == ("base",)


def test_explicit_csv_selection():
    assert resolve_stages(STACK, "sft,dpo").runnable == ("sft", "dpo")


def test_selection_accepts_an_iterable_too():
    assert resolve_stages(STACK, ["sft", "dpo"]).runnable == ("sft", "dpo")


def test_whitespace_and_empties_are_tolerated():
    assert resolve_stages(STACK, " sft , , dpo ").runnable == ("sft", "dpo")


def test_result_is_in_stack_order_not_caller_order():
    """Stages run base->sft->dpo->instruct regardless of how they were listed."""
    assert resolve_stages(STACK, "instruct,sft").runnable == ("sft", "instruct")


def test_requesting_base_reports_it_as_skipped_not_runnable():
    selection = resolve_stages(STACK, "base,sft")
    assert selection.runnable == ("sft",)
    assert selection.skipped == ("base",)


def test_unknown_stage_raises_before_any_gpu_is_started():
    with pytest.raises(KeyError):
        resolve_stages(STACK, "sft,typo")


def test_unknown_stack_raises():
    with pytest.raises(KeyError):
        resolve_stages("no_such_stack", "sft")


def test_selection_is_immutable():
    selection = resolve_stages(STACK, "sft")
    with pytest.raises(Exception):
        selection.runnable = ("dpo",)  # frozen dataclass


def test_selection_type():
    assert isinstance(resolve_stages(STACK, "sft"), StageSelection)


# --- remote log_dir composition ------------------------------------------


def test_remote_log_root_comes_from_config():
    assert run_and_pull.remote_log_root() == f"/eval-logs/{config.logs_remote_root()}"


def test_stage_log_dir_is_the_documented_layout():
    assert (
        run_and_pull.stage_log_dir("berg", STACK, "sft")
        == f"/eval-logs/refactor_runs/berg/{STACK}/sft"
    )


def test_stage_log_dir_honours_an_explicit_root():
    assert (
        run_and_pull.stage_log_dir("petri", STACK, "dpo", log_root="/tmp/root")
        == f"/tmp/root/petri/{STACK}/dpo"
    )


# --- the invariant that matters: writer and reader agree ------------------


def test_remote_write_path_and_local_mirror_share_one_suffix():
    """The stage's remote log_dir and its local mirror must end with the same
    <method>/<stack>/<stage> tail, or logs land somewhere the viewer never looks.
    """
    method, stage = "berg", "sft"
    remote = run_and_pull.stage_log_dir(method, STACK, stage)
    _, _, local_dest = pull_logs._resolve_paths(method, STACK, stage)
    suffix = f"{method}/{STACK}/{stage}"
    assert remote.endswith(suffix)
    # local_dest is a native path (backslashes on Windows); compare its POSIX form.
    assert local_dest.as_posix().endswith(suffix)


def test_remote_path_pulled_matches_the_path_written():
    """What `modal volume get` fetches is exactly what the stage wrote."""
    method, stage = "petri", "instruct"
    written = run_and_pull.stage_log_dir(method, STACK, stage)
    _, pulled, _ = pull_logs._resolve_paths(method, STACK, stage)
    # `written` is absolute (container mount); `pulled` is volume-relative.
    assert written == f"/eval-logs/{pulled}"


def test_local_mirror_mirrors_the_volume_tree_under_the_repo():
    _, pulled, local_dest = pull_logs._resolve_paths("berg", STACK, "dpo")
    expected = pull_logs.REPO_ROOT / "eval-logs" / pulled
    assert local_dest == expected


# --- CLI ------------------------------------------------------------------


def test_parse_args_defaults_to_berg_on_the_7b_stack():
    args = run_and_pull._parse_args([])
    assert args.method == "berg"
    assert args.stack == STACK
    assert args.stages is None
    assert args.view is False
    assert args.dry_run is False


def test_parse_args_reads_stage_selection_and_flags():
    args = run_and_pull._parse_args(
        ["--method", "petri", "--stages", "sft,dpo", "--view"]
    )
    assert args.method == "petri"
    assert args.stages == "sft,dpo"
    assert args.view is True


def test_dry_run_touches_nothing_and_returns_no_mirrors():
    """A dry run must not import modal, spawn anything, or write to disk."""
    assert run_and_pull.launch("berg", STACK, "sft,dpo", dry_run=True) == {}


def test_main_rejects_an_unknown_stage_cleanly():
    """The KeyError guard surfaces as a clean exit, not a traceback."""
    with pytest.raises(SystemExit):
        run_and_pull.main(["--stages", "nope", "--dry-run"])
