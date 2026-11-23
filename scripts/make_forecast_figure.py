"""Draw the README results figure, ``assets/forecast_example.png``.

Two German power series, one row each: a week of day-ahead forecasts from Timestral-28M as released
(left) and after full fine-tuning (right), against what happened. Both panels of a row share one axis
that starts at zero. The range width and how often it held, printed above each panel, are for all 25
test days, so a panel cannot flatter the week it happens to show.

Every number is read from ``docs/benchmarks/forecast_example_de_power.json``, which
``scripts/benchmarks/export_forecast_example.py`` writes; that script also chooses the two series by
rule. Nothing is typed in by hand.

matplotlib is not a project dependency. Run from the repository root:

    uv run --with matplotlib==3.11.2 python scripts/make_forecast_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "benchmarks" / "forecast_example_de_power.json"
OUT = ROOT / "assets" / "forecast_example.png"

# Colours follow the banner: near-black navy surface, cream text. The fine-tuned model is blue and the
# released model a neutral grey; what happened is drawn in the text colour on top of both.
SURFACE = "#0e0e18"
TEXT = "#f3f2ed"
TEXT_SOFT = "#a3adc6"
TUNED = "#3b98f0"
OTHER = "#9aa1b8"
GRID = "#262a3c"
SPINE = "#3a3f55"

OPERATOR = {"50HZ": "50Hertz", "AMP": "Amprion", "TBW": "TransnetBW", "TTG": "TenneT"}
SOURCE_KIND = {"solar": "solar", "wind_on": "onshore wind", "wind_off": "offshore wind"}
# (key in the results file, colour, heading above the panel)
MODELS = (("released", OTHER, "As released"), ("fine_tuned", TUNED, "Fine-tuned"))


def series_title(series_id: str) -> str:
    operator, kind = series_id.split("_", 1)
    return f"{OPERATOR[operator]} {SOURCE_KIND[kind]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    data = json.loads(SOURCE.read_text(encoding="utf-8"))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})
    fig = plt.figure(figsize=(14, 7.2), dpi=100, facecolor=SURFACE)
    fig.text(0.035, 0.925, "Fine-tuned forecasts commit to a narrower range", color=TEXT, fontsize=22, weight="bold")
    fig.text(
        0.035,
        0.872,
        "Timestral-28M day-ahead forecasts of German wind power against what happened.  "
        "Each day is a separate forecast.",
        color=TEXT_SOFT,
        fontsize=14,
    )

    handles = {}
    for row, (series_id, shown) in enumerate(data["shown"].items()):
        bottom = 0.475 - 0.36 * row
        fig.text(0.035, bottom + 0.268, series_title(series_id), color=TEXT, fontsize=17, weight="bold", va="center")
        actual = np.asarray(shown["actual_mw"])
        start = np.datetime64(shown["start_utc"], "m")
        times = (start + np.timedelta64(shown["step_minutes"], "m") * np.arange(actual.size)).astype(object)
        per_day = shown["steps_per_forecast"]
        top = 1.06 * max(actual.max(), *(max(shown[model]["q90_mw"]) for model, _, _ in MODELS))

        for column, (model, colour, heading) in enumerate(MODELS):
            ax = fig.add_axes((0.075 + 0.475 * column, bottom, 0.435, 0.185), facecolor=SURFACE)
            forecast = shown[model]
            band = ax.fill_between(times, forecast["q10_mw"], forecast["q90_mw"], color=colour, alpha=0.30, linewidth=0)
            (line,) = ax.plot(times, forecast["q50_mw"], color=colour, linewidth=2.0)
            (happened,) = ax.plot(times, actual, color=TEXT, linewidth=1.4)
            handles[model] = (line, band)
            handles["actual"] = happened
            for day in range(1, actual.size // per_day):
                ax.axvline(times[per_day * day], color=TEXT_SOFT, linewidth=0.5, alpha=0.3)
            ax.set_ylim(0, top)
            ax.set_xlim(times[0], times[-1])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(SPINE)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(colors=TEXT_SOFT, labelsize=12, length=3)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:,.0f}"))
            if column == 0:
                ax.set_ylabel("MW", color=TEXT_SOFT, fontsize=12)

            tuned = model == "fine_tuned"
            stats = data["per_series"][series_id][model]
            ax.text(
                0.0,
                1.06,
                heading,
                transform=ax.transAxes,
                color=colour if tuned else TEXT_SOFT,
                fontsize=13.5,
                weight="bold",
                va="bottom",
            )
            ax.text(
                1.0,
                1.06,
                f"80% range {stats['width_80_mw']:,.0f} MW wide, held {100 * stats['coverage_80']:.0f}%",
                transform=ax.transAxes,
                color=TEXT if tuned else TEXT_SOFT,
                fontsize=13,
                va="bottom",
                ha="right",
            )

    fig.legend(
        [
            handles["actual"],
            handles["released"][0],
            handles["released"][1],
            handles["fine_tuned"][0],
            handles["fine_tuned"][1],
        ],
        ["what happened", "forecast, as released", "its 80% range", "forecast, fine-tuned", "its 80% range"],
        loc="upper left",
        bbox_to_anchor=(0.027, 0.848),
        ncol=5,
        frameon=False,
        fontsize=12.5,
        labelcolor=TEXT_SOFT,
        handlelength=1.6,
        columnspacing=1.6,
    )
    fig.text(
        0.035,
        0.028,
        "Range widths and how often they held are for all 25 test days; the week shown is the first.",
        color=TEXT_SOFT,
        fontsize=12,
    )
    fig.text(
        0.985,
        0.028,
        f"Data: {data['source'].split('@')[0]} (ENTSO-E), {data['licence']}",
        color=TEXT_SOFT,
        fontsize=12,
        ha="right",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=SURFACE)
    print(f"wrote {args.output} ({fig.get_size_inches()[0] * fig.dpi:.0f} x {fig.get_size_inches()[1] * fig.dpi:.0f})")
    for series_id in data["shown"]:
        print(
            series_id, {m: {k: round(v, 3) for k, v in data["per_series"][series_id][m].items()} for m, _, _ in MODELS}
        )
    print("overall", {m: {k: round(v, 3) for k, v in data["overall"][m].items()} for m, _, _ in MODELS})


if __name__ == "__main__":
    main()
