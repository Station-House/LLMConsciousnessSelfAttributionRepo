"""Plot Berg-style self-attribution rates for the Olmo 3 7B Instruct stack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

from production_scripts.berg_confidence import (
    RELATED,
    UNRELATED,
    classify_arm,
    wilson_interval,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = (
    REPO_ROOT / "eval-logs" / "refactor_runs" / "berg" / "olmo_7b_instruct_stack"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "olmo7b_attribution_bar.png"
DEFAULT_SUBTITLE = "Latest refactor-era logs; n=20 prompts per model"
EXPECTED_SAMPLE_COUNT = 20

MODEL_ORDER = [
    "vllm/allenai/Olmo-3-7B-Instruct-SFT",
    "vllm/allenai/Olmo-3-7B-Instruct-DPO",
    "vllm/allenai/Olmo-3-7B-Instruct",
]
MODEL_LABELS = {
    "vllm/allenai/Olmo-3-7B-Instruct-SFT": "SFT",
    "vllm/allenai/Olmo-3-7B-Instruct-DPO": "DPO",
    "vllm/allenai/Olmo-3-7B-Instruct": "Final Instruct",
}


@dataclass(frozen=True)
class ModelResult:
    model: str
    total: int
    self_attributions: int
    source_file: str

    @property
    def rate(self) -> float:
        return self.self_attributions / self.total

    @property
    def ci(self) -> tuple[float, float]:
        """95% Wilson interval for the pooled rate, as proportions."""
        return wilson_interval(self.self_attributions, self.total)


@dataclass(frozen=True)
class ArmCounts:
    related: tuple[int, int]  # (k, n)
    unrelated: tuple[int, int]
    unknown: tuple[int, int]


@dataclass(frozen=True)
class ModelResultWithArms:
    base: ModelResult
    arms: ArmCounts

    @property
    def arm_cis(self) -> dict[str, tuple[float, float] | None]:
        out: dict[str, tuple[float, float] | None] = {}
        for label, (k, n) in {
            RELATED: self.arms.related,
            UNRELATED: self.arms.unrelated,
        }.items():
            out[label] = wilson_interval(k, n) if n else None
        return out


def read_results(log_dir: Path = DEFAULT_LOG_DIR) -> list[ModelResult]:
    """Read completed binary-scored logs and validate duplicate runs."""
    candidates: dict[str, list[ModelResult]] = {model: [] for model in MODEL_ORDER}
    log_paths = sorted(log_dir.glob("*/*.eval"))
    if not log_paths:
        raise RuntimeError(f"No trusted refactor-era Berg logs found in {log_dir}")

    for path in log_paths:
        log = read_eval_log(str(path), header_only=False)
        if log.status != "success" or log.samples is None:
            raise RuntimeError(f"Trusted log is not a successful completed eval: {path}")

        model = str(log.eval.model)
        if model not in candidates:
            raise RuntimeError(f"Unexpected model in trusted Berg logs: {model}")

        scores = [sample.scores.get("model_graded_qa") for sample in log.samples]
        if any(score is None for score in scores):
            raise RuntimeError(f"Trusted log has an unscored sample: {path}")
        values = [score.value for score in scores if score is not None]
        if any(value not in {"C", "I"} for value in values):
            raise RuntimeError(f"Trusted log has a non-binary scorer value: {path}")
        if len(values) != EXPECTED_SAMPLE_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_SAMPLE_COUNT} completed samples in {path}, "
                f"found {len(values)}"
            )

        candidates[model].append(
            ModelResult(
                model=model,
                total=len(values),
                self_attributions=values.count("C"),
                source_file=path.name,
            )
        )

    missing = [model for model, model_candidates in candidates.items() if not model_candidates]
    if missing:
        raise RuntimeError(f"Missing eval logs for: {', '.join(missing)}")

    results: list[ModelResult] = []
    for model in MODEL_ORDER:
        model_candidates = candidates[model]
        signatures = {(result.total, result.self_attributions) for result in model_candidates}
        if len(signatures) != 1:
            raise RuntimeError(
                f"Trusted duplicate runs disagree for {model}: {sorted(signatures)}"
            )
        results.append(max(model_candidates, key=lambda result: result.source_file))

    return results


def read_results_with_arms(
    log_dir: Path = DEFAULT_LOG_DIR,
) -> list[ModelResultWithArms]:
    """Per-stage pooled result plus related/unrelated/unknown arm tallies.

    Arm assignment prefers each sample's ``metadata["starter_type"]``
    (refactor-era logs) and falls back to exact starter-text match
    (May-25 logs, whose metadata is empty). Enforces the same
    20-sample completed-log validation as :func:`read_results`.
    """
    armed: list[ModelResultWithArms] = []
    for base in read_results(log_dir):
        # Re-read the winning log to recover per-sample inputs for arm split.
        # Match by filename (not a glob pattern: names contain "[sft]" etc.,
        # which glob would read as a character class).
        match = next(
            p for p in log_dir.glob("*/*.eval") if p.name == base.source_file
        )
        log = read_eval_log(str(match))
        assert log.samples is not None
        tallies = {RELATED: [0, 0], UNRELATED: [0, 0], "unknown": [0, 0]}
        for sample in log.samples:
            score = sample.scores.get("model_graded_qa") if sample.scores else None
            if score is None:
                continue
            arm = classify_arm(str(sample.input), dict(sample.metadata or {}))
            bucket = tallies.get(arm, tallies["unknown"])
            bucket[1] += 1
            if score.value == "C":
                bucket[0] += 1
        armed.append(
            ModelResultWithArms(
                base=base,
                arms=ArmCounts(
                    related=tuple(tallies[RELATED]),  # type: ignore[arg-type]
                    unrelated=tuple(tallies[UNRELATED]),  # type: ignore[arg-type]
                    unknown=tuple(tallies["unknown"]),  # type: ignore[arg-type]
                ),
            )
        )
    return armed


def plot(
    results: list[ModelResult],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    subtitle: str = DEFAULT_SUBTITLE,
) -> None:
    labels = [MODEL_LABELS[result.model] for result in results]
    rates = [result.rate * 100 for result in results]
    counts = [f"{result.self_attributions}/{result.total}" for result in results]
    # 95% Wilson error bars, in percentage points.
    lower = [result.rate * 100 - result.ci[0] * 100 for result in results]
    upper = [result.ci[1] * 100 - result.rate * 100 for result in results]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6.2))
    colors = ["#2a9d8f", "#457b9d", "#c8553d"]
    x = range(len(results))
    bars = ax.bar(x, rates, color=colors, width=0.62, tick_label=labels)
    ax.errorbar(
        list(x),
        rates,
        yerr=[lower, upper],
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capsize=5,
        capthick=1.6,
    )

    ax.set_title(
        "Berg-Style Self-Monitoring Across the Olmo 3 7B Instruct Stack",
        fontsize=16,
        pad=18,
        weight="bold",
    )
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="#555555",
    )
    ax.set_ylabel("Self-attribution rate (%) — bars show 95% Wilson CI")
    ax.set_xlabel("Training stage")
    top = max(result.ci[1] * 100 for result in results)
    ax.set_ylim(0, max(12.0, top + 4.0))
    ax.set_yticks(range(0, int(max(12.0, top + 4.0)) + 1, 2))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, rate, count in zip(bars, rates, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.6,
            f"{rate:.1f}%\n({count})",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing per-stage Berg .eval logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the PNG.",
    )
    parser.add_argument(
        "--subtitle",
        default=DEFAULT_SUBTITLE,
        help="Caption under the chart title.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    results = read_results(args.log_dir)
    plot(results, output_path=args.output, subtitle=args.subtitle)

    print("Wrote", args.output)
    for result in results:
        lo, hi = result.ci
        print(
            f"{MODEL_LABELS[result.model]}: "
            f"{result.self_attributions}/{result.total} "
            f"({result.rate * 100:.1f}%, 95% CI [{lo * 100:.1f}, {hi * 100:.1f}]) "
            f"from {result.source_file}"
        )
    for armed in read_results_with_arms(args.log_dir):
        (rk, rn), (uk, un) = armed.arms.related, armed.arms.unrelated
        rel = f"{rk}/{rn}" if rn else "n/a"
        unr = f"{uk}/{un}" if un else "n/a"
        print(
            f"{MODEL_LABELS[armed.base.model]} arms: "
            f"related {rel}, unrelated {unr}, unknown {armed.arms.unknown}"
        )


if __name__ == "__main__":
    main()
