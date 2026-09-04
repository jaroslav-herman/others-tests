"""Plot EIS 1/R2 versus current and fitted y-intercept versus Time."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

import wepy.basics as we


SAMPLE_FILES = {
    "467": Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\467_fit_parameters.csv"),
    "457": Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\457_eisfit_export.csv"),
    '453': Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\453_eisfit_export.csv"),
    '455': Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\455_eisfit_export.csv"),
    '468': Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\468_fit_parameters.csv"),
}
CURRENT_MIN_MA = 50.0
CURRENT_MAX_MA = 500.0
OUTPUT = Path("results") / "eis_fit_1_over_R2_vs_current_457_467.png"
INTERCEPT_OUTPUT = Path("results") / "eis_fit_y_intercept_time_evolution_457_467.png"


def load_sample(sample: str):
    """Load a fit-parameter CSV through wepy's pandas reader."""
    path = SAMPLE_FILES[sample]
    if not path.is_file():
        raise FileNotFoundError(path)
    data = we.read_file(str(path), skiprows=0, delimiter=",")
    required = {"Time", "I_mA", "R2"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    data = data[["Time", "I_mA", "R2"]].copy()
    data["I_mA"] = np.asarray(data["I_mA"], dtype=float)
    data["R2"] = np.asarray(data["R2"], dtype=float)
    data["one_over_R2"] = 1.0 / data["R2"]
    return path, data.replace([np.inf, -np.inf], np.nan).dropna()


def fit_by_time(data):
    """Return fitted (slope, intercept) values for each Time series."""
    fits = {}
    for time in sorted(data["Time"].unique()):
        series = data[(data["Time"] == time) & (data["I_mA"] >= CURRENT_MIN_MA) & (data["I_mA"] <= CURRENT_MAX_MA)].sort_values("I_mA")
        if len(series) < 2:
            print(f"Warning: skipping Time {time}: fewer than 2 points")
            continue
        fits[float(time)] = (series, *np.polyfit(series["I_mA"], series["one_over_R2"], 1))
    return fits


def main() -> None:
    samples = {sample: load_sample(sample) for sample in ("457", "467","455", "453", "468")}
    fits = {sample: fit_by_time(data) for sample, (_, data) in samples.items()}
    time_values = sorted({time for sample_fits in fits.values() for time in sample_fits})
    time_colors = we.get_colors(len(time_values))
    color_by_time = dict(zip(time_values, time_colors))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, sample in zip(axes, samples):
        for time, (series, slope, intercept) in fits[sample].items():
            fit_current = np.linspace(CURRENT_MIN_MA, CURRENT_MAX_MA, 100)
            color = color_by_time[time]
            axis.scatter(series["I_mA"], series["one_over_R2"], color=color, s=18, alpha=0.65)
            axis.plot(fit_current, slope * fit_current + intercept, color=color)
        axis.set_title(f"Sample {sample}")
        axis.set_xlabel("Current (mA)")
        axis.set_xlim(CURRENT_MIN_MA, CURRENT_MAX_MA)
        axis.grid(False)

    axes[0].set_ylabel(r"$1/R_2$ ($\Omega^{-1}$)")
    figure.suptitle("EIS fit parameter: $1/R_2$ versus current")
    time_cmap = ListedColormap(time_colors)
    time_norm = BoundaryNorm(np.arange(len(time_values) + 1) - 0.5, len(time_values))
    scalar_map = plt.cm.ScalarMappable(norm=time_norm, cmap=time_cmap)
    scalar_map.set_array([])
    colorbar = figure.colorbar(scalar_map, ax=axes, ticks=np.arange(len(time_values)), pad=0.02, fraction=0.04)
    colorbar.ax.set_yticklabels([f"{time:g}" for time in time_values])
    colorbar.set_label("Time")
    figure.subplots_adjust(left=0.08, right=0.86, bottom=0.14, top=0.88, wspace=0.12)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(figure)

    sample_colors = we.get_colors(len(samples))
    intercept_figure, intercept_axis = plt.subplots(figsize=(7.5, 5))
    for (sample, sample_fits), color in zip(fits.items(), sample_colors):
        times = sorted(sample_fits)
        if times:
            intercept_axis.plot(times, [sample_fits[time][2] for time in times], "o-", color=color, label=f"Sample {sample}")
    intercept_axis.set_xlabel("Time")
    intercept_axis.set_ylabel(r"Fitted y-intercept, $1/R_2$ ($\Omega^{-1}$)")
    intercept_axis.set_title("EIS fit $1/R_2$ y-intercept time evolution")
    intercept_axis.grid(False)
    intercept_axis.legend(frameon=False)
    intercept_figure.tight_layout()
    intercept_figure.savefig(INTERCEPT_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(intercept_figure)

    print(f"467 source: {samples['467'][0]}")
    print(f"457 source: {samples['457'][0]}")
    print(f"Saved: {OUTPUT.resolve()}")
    print(f"Saved: {INTERCEPT_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
