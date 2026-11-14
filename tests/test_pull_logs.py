"""Unit tests for the log-pull tool's pure command builders.

``production_scripts/pull_logs.py`` replaces the manual "download the ``.eval``
from the Modal web UI, copy it into a folder, open it in VS Code" loop with one
command. Its value depends on two things being right, and both are pure
functions that need no Modal account, no network, and no ``inspect_ai``:

1. **The selection is unambiguous.** ``method``/``stack``/``stage`` are
   hierarchical, so a narrower selection requires the broader ones. Without the
   guard, ``--stage sft`` alone would silently mirror the wrong subtree.
2. **Reader and writer agree on where logs live.** The paths are derived from
   ``run_defaults.yaml`` ``logs:`` -- the same config the Modal launcher writes
   with -- so these tests are the regression gate on that DRY link.

Like ``test_config.py``: no model calls and no ``inspect_ai`` (only PyYAML), so
these run anywhere. The commands are only *built* here, never executed, so no
subprocess is spawned and nothing touches the Modal volume.
"""

from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath

import pytest

from llm_consciousness_self_attribution import config
from production_scripts import pull_logs

cli_dest_of = pull_logs.cli_destination


# --- selection_suffix: the valid combinations -----------------------------


def test_no_selection_mirrors_everything():
    """No narrowing flags -> empty suffix, i.e. the whole remote root."""
    assert pull_logs.selection_suffix(None, None, None) == ""


def test_method_only():
    assert pull_logs.selection_suffix("berg", None, None) == "berg"


def test_method_and_stack():
    assert (
        pull_logs.selection_suffix("berg", "olmo_7b_instruct_stack", None)
        == "berg/olmo_7b_instruct_stack"
    )


def test_method_stack_and_stage():
    assert (
        pull_logs.selection_suffix("petri", "olmo_7b_instruct_stack", "sft")
        == "petri/olmo_7b_instruct_stack/sft"
    )


def test_suffix_order_matches_the_documented_layout():
    """The layout is <method>/<stack>/<stage>, as documented in run_defaults.yaml."""
    parts = pull_logs.selection_suffix("berg", "some_stack", "dpo").split("/")
    assert parts == ["berg", "some_stack", "dpo"]


# --- selection_suffix: the ordering guard ---------------------------------


def test_stage_without_stack_raises():
    with pytest.raises(ValueError):
        pull_logs.selection_suffix("berg", None, "sft")


def test_stack_without_method_raises():
    with pytest.raises(ValueError):
        pull_logs.selection_suffix(None, "olmo_7b_instruct_stack", None)


def test_stage_without_method_or_stack_raises():
    with pytest.raises(ValueError):
        pull_logs.selection_suffix(None, None, "sft")


def test_stage_with_stack_but_no_method_raises():
    with pytest.raises(ValueError):
        pull_logs.selection_suffix(None, "olmo_7b_instruct_stack", "sft")


# --- command builders -----------------------------------------------------


# The local destination is a native path, so on Windows str() gives "\\tmp\\mirror".
# Expectations use str(MIRROR) for that reason; the remote path stays a POSIX
# string because it is a path inside the Modal volume, not on this machine.
MIRROR = Path("/tmp/mirror")


def test_get_command_is_modal_volume_get_with_force():
    cmd = pull_logs.get_command("eval-logs", "refactor_runs/berg", MIRROR)
    assert cmd == [
        "modal",
        "volume",
        "get",
        "eval-logs",
        "refactor_runs/berg",
        str(MIRROR),
        "--force",
    ]


def test_get_command_without_force_omits_the_flag():
    cmd = pull_logs.get_command(
        "eval-logs", "refactor_runs/berg", Path("/tmp/mirror"), force=False
    )
    assert "--force" not in cmd
    assert cmd[:3] == ["modal", "volume", "get"]


def test_view_command_is_inspect_view_with_log_dir():
    assert pull_logs.view_command(MIRROR) == [
        "inspect",
        "view",
        "--log-dir",
        str(MIRROR),
    ]


def test_builders_reuse_offtheshelf_tools_not_a_custom_syncer():
    """Guard the design decision: shell out to `modal volume get` + `inspect view`."""
    assert pull_logs.get_command("v", "r", Path("/d"))[0] == "modal"
    assert pull_logs.view_command(Path("/d"))[0] == "inspect"


# --- path resolution: the DRY link to run_defaults.yaml -------------------


def test_configured_log_convention_matches_the_published_paths():
    """Pin the *values* of the convention, not just that config is consulted.

    ``README.md``, ``REPLICATION.md``, and ``modal_app``'s default ``log_root``
    all quote these exact strings, and the Modal volume already holds runs under
    them. The other tests here derive their expectations from ``logs_config()``,
    so they would happily pass if someone edited ``run_defaults.yaml`` -- this
    one fails instead, forcing the docs to be updated in the same change.
    """
    assert config.logs_config() == {
        "volume": "eval-logs",
        "remote_root": "refactor_runs",
        "local_dir": "eval-logs/refactor_runs",
    }


def test_resolve_paths_uses_the_configured_volume_and_roots():
    volume, remote_path, local_dest = pull_logs._resolve_paths(None, None, None)
    logs = config.logs_config()
    assert volume == logs["volume"]
    assert remote_path == logs["remote_root"]
    assert local_dest == pull_logs.REPO_ROOT / logs["local_dir"]


def test_resolve_paths_appends_the_selection_to_both_sides():
    """A narrowed selection must extend the remote path and the local mirror
    identically, or the mirror silently lands in the wrong directory."""
    logs = config.logs_config()
    volume, remote_path, local_dest = pull_logs._resolve_paths(
        "berg", "olmo_7b_instruct_stack", "sft"
    )
    suffix = "berg/olmo_7b_instruct_stack/sft"
    assert volume == logs["volume"]
    assert remote_path == f"{logs['remote_root']}/{suffix}"
    assert local_dest == pull_logs.REPO_ROOT / logs["local_dir"] / suffix


def test_resolve_paths_propagates_the_ordering_guard():
    with pytest.raises(ValueError):
        pull_logs._resolve_paths("berg", None, "sft")


def test_local_mirror_stays_inside_the_repo():
    """The mirror is git-ignored inside the repo; never an absolute path elsewhere."""
    _, _, local_dest = pull_logs._resolve_paths("berg", None, None)
    assert pull_logs.REPO_ROOT in local_dest.parents


# --- mirror(): the public "get these logs onto my disk" entry point -------


def test_mirror_dry_run_returns_the_destination():
    dest = pull_logs.mirror("berg", "olmo_7b_instruct_stack", "sft", dry_run=True)
    assert dest == (
        pull_logs.REPO_ROOT
        / config.logs_config()["local_dir"]
        / "berg/olmo_7b_instruct_stack/sft"
    )


def test_mirror_dry_run_creates_nothing_on_disk():
    """Hermetic: point the repo root at an empty temp tree so the assertion is
    about what the code *does*, not about whether a real mirror happens to exist.
    (Asserting ``not dest.exists()`` against the real repo passes on a clean
    checkout and fails the moment you have actually pulled logs.)
    """
    original_root = pull_logs.REPO_ROOT
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pull_logs.REPO_ROOT = Path(tmp)
            dest = pull_logs.mirror("berg", "olmo_7b_instruct_stack", "sft", dry_run=True)
            assert Path(tmp) in dest.parents
            assert not dest.exists()
            assert not dest.parent.exists()  # nor the CLI destination
            assert list(Path(tmp).iterdir()) == []
    finally:
        pull_logs.REPO_ROOT = original_root


def test_mirror_defaults_to_the_whole_remote_root():
    dest = pull_logs.mirror(dry_run=True)
    assert dest == pull_logs.REPO_ROOT / config.logs_config()["local_dir"]


def test_mirror_propagates_the_ordering_guard():
    with pytest.raises(ValueError):
        pull_logs.mirror("berg", None, "sft", dry_run=True)


# --- cli_destination: the measured `modal volume get` contract ------------


def test_cli_destination_is_the_parent_not_the_target():
    """Measured behaviour: given an existing dir, the CLI appends the remote
    basename. So to land logs at <...>/sft we must hand it <...>/ as the target.
    """
    local_dest = Path("/repo/eval-logs/refactor_runs/berg/stack/sft")
    assert cli_dest_of("refactor_runs/berg/stack/sft", local_dest) == local_dest.parent


def test_cli_destination_for_the_whole_root():
    local_dest = Path("/repo/eval-logs/refactor_runs")
    assert cli_dest_of("refactor_runs", local_dest) == Path("/repo/eval-logs")


def test_cli_destination_rejects_a_basename_mismatch():
    """If the two ever drift, handing over the parent would silently misplace the
    tree -- so refuse rather than create <parent>/<remote basename>.
    """
    with pytest.raises(ValueError):
        cli_dest_of("refactor_runs/berg/stack/sft", Path("/repo/mirror/dpo"))


def test_every_selection_keeps_basenames_aligned():
    """The invariant that makes the parent trick safe, across all selections."""
    for selection in [
        (None, None, None),
        ("berg", None, None),
        ("berg", "olmo_7b_instruct_stack", None),
        ("berg", "olmo_7b_instruct_stack", "sft"),
    ]:
        _, remote_path, local_dest = pull_logs._resolve_paths(*selection)
        assert PurePosixPath(remote_path).name == local_dest.name
        assert pull_logs.cli_destination(remote_path, local_dest) == local_dest.parent


# --- verify_mirror: the post-condition that a green CLI does not guarantee --


def test_verify_mirror_rejects_double_nesting():
    """The other way it can go wrong: logs at <dest>/<dest name>/*.eval."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "sft"
        (dest / "sft").mkdir(parents=True)
        (dest / "sft" / "run.eval").write_bytes(b"PK\x03\x04")
        with pytest.raises(SystemExit):
            pull_logs.verify_mirror(dest)


def test_verify_mirror_accepts_a_directory_of_eval_logs():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "sft"
        dest.mkdir()
        (dest / "run_a.eval").write_bytes(b"PK\x03\x04")
        (dest / "run_b.eval").write_bytes(b"PK\x03\x04")
        found = pull_logs.verify_mirror(dest)
        assert [p.name for p in found] == ["run_a.eval", "run_b.eval"]


def test_verify_mirror_finds_logs_in_nested_stage_directories():
    """A method- or stack-level pull nests one directory per stage."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "olmo_7b_instruct_stack"
        (dest / "sft").mkdir(parents=True)
        (dest / "sft" / "run.eval").write_bytes(b"PK\x03\x04")
        assert len(pull_logs.verify_mirror(dest)) == 1


def test_verify_mirror_rejects_the_collapsed_single_file_case():
    """The real failure: `modal volume get` wrote the lone .eval AS `.../sft`.

    The CLI exits 0 and the tool used to print "Mirrored to ...", but the path is
    a binary file, so `inspect view` shows nothing and VS Code refuses to open it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        collapsed = Path(tmp) / "sft"
        collapsed.write_bytes(b"PK\x03\x04binary-eval-contents")
        with pytest.raises(SystemExit):
            pull_logs.verify_mirror(collapsed)


def test_verify_mirror_rejects_an_empty_directory():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "sft"
        dest.mkdir()
        with pytest.raises(SystemExit):
            pull_logs.verify_mirror(dest)


def test_verify_mirror_ignores_non_eval_files():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "sft"
        dest.mkdir()
        (dest / "notes.txt").write_text("not a log")
        with pytest.raises(SystemExit):
            pull_logs.verify_mirror(dest)


def test_eval_files_on_a_missing_path_is_empty_not_an_error():
    assert pull_logs.eval_files(Path("/definitely/not/here")) == []


# --- argument parsing -----------------------------------------------------


def test_parse_args_defaults_to_mirroring_everything():
    args = pull_logs._parse_args([])
    assert args.method is None and args.stack is None and args.stage is None
    assert args.view is False
    assert args.dry_run is False


def test_parse_args_reads_the_narrowing_flags():
    args = pull_logs._parse_args(
        ["--method", "berg", "--stack", "olmo_7b_instruct_stack", "--view"]
    )
    assert args.method == "berg"
    assert args.stack == "olmo_7b_instruct_stack"
    assert args.view is True


def test_main_rejects_an_ambiguous_selection():
    """The ValueError guard surfaces as a clean exit, not a traceback."""
    with pytest.raises(SystemExit):
        pull_logs.main(["--stage", "sft"])
