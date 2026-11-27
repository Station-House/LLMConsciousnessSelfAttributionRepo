"""Launch a Modal eval run and mirror each stage's logs the moment it finishes.

The last manual step in the log workflow. ``modal_app`` fans stages out with
``.spawn()`` and returns immediately, so nothing local knows when a stage is
done; you had to notice it yourself and then run ``pull_logs.py``. This script
closes that gap: it submits every stage up front, then waits on them one at a
time and mirrors each stage's ``.eval`` logs into the repo as soon as *that*
stage lands -- so ``sft``'s transcripts are clickable in the VS Code Logs pane
while ``dpo`` is still on the GPU.

The local tree mirrors the Volume's exactly::

    volume: refactor_runs/<method>/<stack>/<stage>/*.eval
    local:  eval-logs/refactor_runs/<method>/<stack>/<stage>/*.eval

That is not a coincidence to be maintained by hand: the remote ``log_dir`` this
script hands to Modal and the local mirror path are both composed from
``pull_logs.selection_suffix()``, so writer and reader cannot disagree about the
layout.

Design notes:
- **Reuse, don't reinvent.** Stage resolution comes from
  ``config.resolve_stages()``; pulling is ``pull_logs.mirror()`` (itself a wrapper
  over the off-the-shelf ``modal volume get``); viewing is ``inspect view``.
  This module only supplies the *waiting*, which is the one thing that did not
  exist.
- **Attended by design.** Stages run inside ``app.run()``, so closing the
  terminal ends the app. For long unattended runs use the detached canonical
  command and pull afterwards::

      uv run modal run --detach -m llm_consciousness_self_attribution.modal_app ...
      uv run python production_scripts/pull_logs.py --method berg --stack ... --view

- ``modal`` is imported lazily inside ``launch()`` so the pure helpers here stay
  importable (and testable) without Modal installed, matching how ``run.py``
  defers ``inspect_ai``.

Examples::

    # Berg across two stages; each stage's logs appear as it finishes:
    uv run python production_scripts/run_and_pull.py \\
        --method berg --stack olmo_7b_instruct_stack --stages sft,dpo

    # ...and open the Inspect viewer at the end:
    uv run python production_scripts/run_and_pull.py \\
        --method petri --stack olmo_7b_instruct_stack --stages sft,dpo --view

    # Show what would run, touching nothing:
    uv run python production_scripts/run_and_pull.py --method berg --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

from llm_consciousness_self_attribution import config
from llm_consciousness_self_attribution.config import resolve_stages

from production_scripts import pull_logs

# Mount point of the eval-logs Volume inside the Modal container. Mirrors
# modal_app.EVAL_LOGS_MOUNT, imported lazily there to avoid requiring `modal`
# just to compute a path.
EVAL_LOGS_MOUNT = "/eval-logs"


def remote_log_root() -> str:
    """The in-container directory runs are written under (``/eval-logs/refactor_runs``)."""
    return f"{EVAL_LOGS_MOUNT}/{config.logs_remote_root()}"


def stage_log_dir(method: str, stack: str, stage: str, log_root: str | None = None) -> str:
    """The remote ``log_dir`` for one stage.

    Composed from ``pull_logs.selection_suffix`` -- the same function that builds
    the local mirror path -- so the directory a stage writes into and the
    directory the mirror pulls from are the same string by construction.
    """
    root = remote_log_root() if log_root is None else log_root
    return f"{root}/{pull_logs.selection_suffix(method, stack, stage)}"


def launch(
    method: str,
    stack: str,
    stages: str | None = None,
    *,
    sample_id: str | None = None,
    judge: str | None = None,
    auditor: str | None = None,
    view: bool = False,
    dry_run: bool = False,
) -> dict[str, str]:
    """Run ``method`` across ``stages`` of ``stack``, mirroring each as it lands.

    Returns ``{stage: local_directory}`` for the stages that completed. A stage
    that fails on Modal is reported and skipped rather than aborting the run, so
    one bad stage does not cost you the logs of the stages that did work.

    ``sample_id`` restricts each stage to the named samples. For PETRI a sample id
    is the seed's filename stem, so this is how you run some seeds and not others.

    ``judge`` / ``auditor`` override the grader models (any Inspect model id);
    unset, they come from ``$CSA_JUDGE_MODEL`` / ``$CSA_AUDITOR_MODEL`` or
    run_defaults.yaml.
    """
    selection = resolve_stages(stack, stages)
    sample_ids = config.parse_sample_ids(sample_id)
    if judge:
        os.environ[config.JUDGE_MODEL_ENV] = judge
    if auditor:
        os.environ[config.AUDITOR_MODEL_ENV] = auditor
    # Resolved (and validated) here; passed explicitly because the local
    # environment does not reach the Modal container.
    judge, auditor = config.judge_model(), config.auditor_model()
    if method.startswith("petri"):
        # Every seed must declare the target's system prompt. Checked here, before
        # anything is spawned, for the same reason a typo'd stage name is: the
        # alternative is discovering it minutes into a paid GPU container.
        config.validate_seed_bank()
    for stage in selection.skipped:
        print(f"Skipping base stage {stack}:{stage} (no chat template yet)")
    if not selection.runnable:
        raise SystemExit(f"error: no runnable stages selected for stack {stack!r}")

    if dry_run:
        print("# would run:")
        print(f"# judge={judge} auditor={auditor}")
        if sample_ids is not None:
            print(f"# restricted to sample id(s): {', '.join(sample_ids)}")
        for stage in selection.runnable:
            print(f"$ run_stage({method}, {stack}, {stage}) -> {stage_log_dir(method, stack, stage)}")
            pull_logs.mirror(method, stack, stage, dry_run=True)
        return {}

    # Imported here so this module stays importable without Modal installed.
    from llm_consciousness_self_attribution.modal_app import app, run_stage

    mirrored: dict[str, str] = {}
    with app.run():
        # Submit every stage up front, so a slow first stage does not delay the
        # queueing of the others (and so the set of work is fixed before any
        # waiting begins).
        calls = [
            (
                stage,
                run_stage.spawn(
                    method,
                    stack,
                    stage,
                    stage_log_dir(method, stack, stage),
                    sample_ids,
                    judge=judge,
                    auditor=auditor,
                ),
            )
            for stage in selection.runnable
        ]
        for stage, call in calls:
            print(f"\nWaiting for {method}:{stack}:{stage} ...")
            try:
                # Waited in submission order rather than completion order: Modal
                # queues these behind a single GPU anyway, and it avoids
                # depending on the exact timeout-exception type of `.get()`.
                result = call.get()
            except Exception as exc:  # noqa: BLE001 -- report and keep going
                print(f"  {stage} FAILED on Modal: {exc!r}")
                print("  (its logs, if any, can still be pulled with pull_logs.py)")
                continue
            print(f"  {stage} finished: {result}")
            local_dest = pull_logs.mirror(method, stack, stage)
            mirrored[stage] = str(local_dest)
            print(f"  mirrored to {local_dest}")

    if not mirrored:
        print("\nNo stages completed; nothing mirrored.")
        return mirrored

    print("\nDone. Local logs:")
    for stage, path in mirrored.items():
        print(f"  {stage}: {path}")
    print("Click them in the Inspect VS Code extension's Logs pane.")

    if view:
        # One viewer over the whole run (the stack directory), not per stage.
        _, _, run_dir = pull_logs._resolve_paths(method, stack, None)
        pull_logs._run(pull_logs.view_command(run_dir))

    return mirrored


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", default="berg", help="Elicitation method (berg, petri).")
    parser.add_argument(
        "--stack", default="olmo_7b_instruct_stack", help="Model stack to evaluate."
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages (e.g. sft,dpo). Default: the whole stack.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help=(
            "Comma-separated sample ids to run, instead of the whole dataset. For "
            "PETRI a sample id is the seed's filename stem (e.g. admit_direct). "
            "fnmatch patterns work, so 'admit_*' is valid. Default: every sample."
        ),
    )
    parser.add_argument(
        "--judge",
        default=None,
        help="Judge model, any Inspect id (e.g. anthropic/claude-opus-4-6). Default: run_defaults.yaml.",
    )
    parser.add_argument(
        "--auditor",
        default=None,
        help="PETRI auditor model, any Inspect id. Default: run_defaults.yaml.",
    )
    parser.add_argument(
        "--view", action="store_true", help="Open `inspect view` on the run when it finishes."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run without touching Modal."
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        launch(
            args.method,
            args.stack,
            args.stages,
            sample_id=args.sample_id,
            judge=args.judge,
            auditor=args.auditor,
            view=args.view,
            dry_run=args.dry_run,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
