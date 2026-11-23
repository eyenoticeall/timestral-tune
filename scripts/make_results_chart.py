"""Draw the README results chart, ``assets/results.png``, from the committed benchmark results.

One panel per benchmark dataset, three bars each: the best classical statistical model, Timestral-28M
as released, and Timestral-28M after full fine-tuning. Bars start at zero, so their lengths are in
true proportion. A dashed line marks the model as released and the gap to the fine-tuned bar is
shaded. Under the fine-tuned bar the chart states how many series improved and the 95% confidence
interval of the gain, so that a reader can see the gain is not noise.

Every number is read from the committed results; nothing is typed in by hand. Retail demand comes from
``docs/benchmarks/sig_v2.json`` and German power, a test set from after the models' release, from
``docs/benchmarks/wp3_unfamiliar_domains.json``. The tuned gradient-boosted baseline of that second
file is deliberately not a bar: the README states that comparison in words next to the chart.

matplotlib is not a project dependency. Run from the repository root:

    uv run --with matplotlib==3.11.2 python scripts/make_results_chart.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
SIGNIFICANCE = ROOT / "docs" / "benchmarks" / "sig_v2.json"
POST_RELEASE = ROOT / "docs" / "benchmarks" / "wp3_unfamiliar_domains.json"
OUT = ROOT / "assets" / "results.png"

# Colours follow the banner: near-black navy surface, cream text. The fine-tuned bar is blue and
# everything it is compared with is a neutral grey, so the fine-tuned result is what stands out.
SURFACE = "#0e0e18"
TEXT = "#f3f2ed"
TEXT_SOFT = "#a3adc6"
TUNED = "#3b98f0"
OTHER = "#6f7690"

MODEL = "Timestral-28M"
MODEL_KEY = "28M"
TUNED_MODE = "full_ft"
# The classical baseline drawn for the post-release dataset. Named explicitly, because the smallest
# baseline error in that file belongs to the tuned gradient-boosted model, which is not classical.
POST_RELEASE_CLASSICAL = "classical_domain"
# (results file, key in it, title, subtitle, what one series is called)
PANELS = (
    (SIGNIFICANCE, "retail_demand", "Retail demand", "hourly sales of 64 store-product series", "series"),
    (POST_RELEASE, "de_power_15m", "German wind and solar power", "15-minute output of 10 regional series", "series"),
)
X_MAX = 0.5
BAR_HEIGHT = 0.56


def load(source: Path, dataset: str) -> dict[str, float]:
    results = json.loads(source.read_text(encoding="utf-8"))
    if source == SIGNIFICANCE:
        data = results["datasets"][dataset]
        released = data["zero_shot"]["wql"]
        mode = data["modes"][TUNED_MODE]
        by_series = mode["paired_by_series"]
        classical = min(baseline["wql"] for baseline in data["baselines"].values())
        tuned = mode["wql_mean"]
        low, high = (100.0 * bound / released for bound in by_series["ci95_delta_wql"])
        improved = by_series["clusters_improved_wql"]
    else:
        model = results["models"][MODEL_KEY][dataset]
        released = model["zero_shot"]["raw"]["wql"]
        mode = model["modes"][TUNED_MODE]
        by_series = mode["paired_vs_zero_shot"]["raw"]["by_series"]["wql"]
        classical = results["baselines"][dataset][POST_RELEASE_CLASSICAL]["metrics"]["raw"]["wql"]
        tuned = mode["summary"]["raw"]["wql_mean"]
        low, high = (100.0 * bound for bound in by_series["ci95_rel_delta"])
        improved = by_series["clusters_improved"]
    return {
        "classical": classical,
        "released": released,
        "tuned": tuned,
        "change": 100.0 * (tuned / released - 1.0),
        "ci_small": min(abs(low), abs(high)),
        "ci_large": max(abs(low), abs(high)),
        "improved": improved,
        "series": by_series["n_clusters"],
    }


def draw_bar(ax, y: float, value: float, colour: str, alpha: float = 1.0) -> None:
    """A bar from zero to `value`, square at the zero line and rounded at the data end."""
    box = FancyBboxPatch(
        (-0.02, y - BAR_HEIGHT / 2),
        value + 0.02,  # starts left of zero; the clip below squares it off at the baseline
        BAR_HEIGHT,
        boxstyle="round,pad=0,rounding_size=0.0035",
        mutation_aspect=52,
        linewidth=0,
        facecolor=colour,
        alpha=alpha,
    )
    ax.add_patch(box)
    box.set_clip_path(Rectangle((0, y - 1.0), 1.0, 2.0, transform=ax.transData))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig = plt.figure(figsize=(14, 7.2), dpi=100, facecolor=SURFACE)
    fig.text(0.035, 0.925, "Fine-tuning lowers forecast error on every series", color=TEXT, fontsize=22, weight="bold")
    fig.text(
        0.035,
        0.868,
        "Forecast error on later data the model had never seen.  Shorter bars are better.",
        color=TEXT_SOFT,
        fontsize=14,
    )

    for (source, dataset, title, subtitle, unit), bottom in zip(PANELS, (0.47, 0.05), strict=True):
        ax = fig.add_axes((0.27, bottom, 0.695, 0.275), facecolor=SURFACE)
        stats = load(source, dataset)
        rows = (
            ("Classical statistical model", stats["classical"], OTHER),
            (f"{MODEL}, as released", stats["released"], OTHER),
            (f"{MODEL}, fine-tuned", stats["tuned"], TUNED),
        )
        ax.set_xlim(0.0, X_MAX)
        ax.set_ylim(len(rows) + 0.05, -0.5)
        ax.axis("off")
        ax.plot([0, 0], [-0.5, len(rows) - 0.5], color=TEXT_SOFT, linewidth=1.2, zorder=3)

        # The gap that fine-tuning closed, shaded at true scale, and a marker for the model as released.
        released = stats["released"]
        draw_bar(ax, len(rows) - 1, released, TUNED, alpha=0.22)
        ax.plot([released, released], [0.58, len(rows) - 0.58], color=TEXT, linewidth=1.1, dashes=(4, 3), zorder=5)

        for y, (label, value, colour) in enumerate(rows):
            tuned_row = colour == TUNED
            draw_bar(ax, y, value, colour)
            ax.text(
                -0.012,
                y,
                label,
                color=TEXT if tuned_row else TEXT_SOFT,
                fontsize=14,
                weight="bold" if tuned_row else "normal",
                ha="right",
                va="center",
            )
            edge = released if tuned_row else value
            ax.text(edge + 0.006, y, f"{value:.3f}", color=TEXT, fontsize=14, weight="bold", va="center")
            if tuned_row:
                ax.text(
                    edge + 0.044,
                    y,
                    f"{abs(stats['change']):.0f}% less error than as released",
                    color=TEXT,
                    fontsize=14,
                    weight="bold",
                    va="center",
                )
                ax.text(
                    edge + 0.006,
                    y + 0.62,
                    f"better on {stats['improved']} of {stats['series']} {unit}   ·   "
                    f"95% interval: {stats['ci_small']:.0f}% to {stats['ci_large']:.0f}% less",
                    color=TEXT_SOFT,
                    fontsize=12.5,
                    va="center",
                )

        heading = fig.text(0.035, bottom + 0.322, title, color=TEXT, fontsize=17, weight="bold", va="center")
        right = heading.get_window_extent(fig.canvas.get_renderer()).x1 / (fig.get_figwidth() * fig.dpi)
        fig.text(right + 0.014, bottom + 0.32, subtitle, color=TEXT_SOFT, fontsize=13, va="center")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output} ({fig.get_size_inches()[0] * fig.dpi:.0f} x {fig.get_size_inches()[1] * fig.dpi:.0f})")
    for source, dataset, _, _, _ in PANELS:
        print(dataset, {key: round(value, 4) for key, value in load(source, dataset).items()})


if __name__ == "__main__":
    main()
