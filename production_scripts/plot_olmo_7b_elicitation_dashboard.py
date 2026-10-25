"""Plot the current Berg-style self-attribution dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from production_scripts.plot_olmo_7b_stack_self_attribution import (
    DEFAULT_LOG_DIR,
    MODEL_LABELS,
    ModelResult,
    read_results,
)
from production_scripts.berg_confidence import wilson_interval


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "olmo7b_elicitation_dashboard.png"
DEFAULT_SUBTITLE = "Latest refactor-era Berg-style logs; n=20 prompts per model"


def plot(
    results: list[ModelResult],
    output_path: Path = DEFAULT_OUTPUT_PATH,
    subtitle: str = DEFAULT_SUBTITLE,
) -> None:
    labels = [MODEL_LABELS[result.model] for result in results]
    rates = [result.rate * 100 for result in results]
    counts = [f"{result.self_attributions}/{result.total}" for result in results]
    lower = [
        result.rate * 100 - wilson_interval(result.self_attributions, result.total)[0] * 100
        for result in results
    ]
    upper = [
        wilson_interval(result.self_attributions, result.total)[1] * 100 - result.rate * 100
        for result in results
    ]

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.24)
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

    fig.suptitle(
        "Olmo 3 7B Instruct Stack: Trusted Self-Attribution Results",
        fontsize=18,
        weight="bold",
        y=0.95,
    )
    fig.text(
        0.5,
        0.895,
        subtitle,
        ha="center",
        va="top",
        fontsize=11,
        color="#555555",
    )

    ax.set_ylabel("Berg-style self-attribution rate (%) — bars show 95% Wilson CI")
    ax.set_xlabel("Training stage")
    top = max(
        wilson_interval(r.self_attributions, r.total)[1] * 100 for r in results
    )
    ax.set_ylim(0, max(10.0, top + 4.0))
    ax.set_yticks(range(0, int(max(10.0, top + 4.0)) + 1, 2))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, rate, count in zip(bars, rates, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.35,
            f"{rate:.1f}%\n({count})",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    fig.text(
        0.5,
        0.075,
        "PETRI is omitted because its 1-10 judge score is not directly comparable "
        "to this binary rate and has not been re-run on the corrected prompt bank.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

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
        help="Caption under the dashboard title.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    results = read_results(args.log_dir)
    plot(results, output_path=args.output, subtitle=args.subtitle)

    print("Wrote", args.output)
    for result in results:
        lo, hi = wilson_interval(result.self_attributions, result.total)
        print(
            f"{MODEL_LABELS[result.model]}: "
            f"{result.self_attributions}/{result.total} "
            f"({result.rate * 100:.1f}%, 95% CI [{lo * 100:.1f}, {hi * 100:.1f}]) "
            f"from {result.source_file}"
        )


if __name__ == "__main__":
    main()
