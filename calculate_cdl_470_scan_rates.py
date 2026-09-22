"""Calculate double-layer capacitance from the 470 scan-rate CV batches.

For each upper-potential batch, the script discovers the CV ``.mpr`` files,
uses the last CV cycle in each file, and separates its forward and reverse
branches.  At every potential in 0.9--1.2 V it calculates

    delta_I = abs(I_forward - I_reverse) / 2

and fits delta_I versus scan rate.  The slope is the double-layer capacitance
in A/V (F).  Results are written as a CSV table.  Since the measurement
folder does not contain an electrode area, the script reports total
capacitance and, optionally, area-normalised capacitance when
``ELECTRODE_AREA_CM2`` is set below.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

import wepy.basics as we

# Some local Matplotlib configurations enable LaTeX during later imports.
# Keep these standalone plots independent of a system TeX installation.
matplotlib.rcParams["text.usetex"] = False


ROOT = Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVs\470 CVs 50 sccm H2")
OUT = Path("results")
VOLTAGE_MIN = 0.9
VOLTAGE_MAX = 1.2
N_VOLTAGE_POINTS = 61

# Set this to the geometric electrode area to add area-normalised output.
# Leave as None when the area is not known.
ELECTRODE_AREA_CM2: float | None = None

BATCH_PATTERNS = {
    "1.4 V": re.compile(r"different_scan_rates_(?P<index>\d+)_CV_.*\.mpr$", re.I),
    "1.5 V": re.compile(r"different_scan_rates_1,5V_(?P<index>\d+)_CV_.*\.mpr$", re.I),
    "1.6 V": re.compile(r"different_scan_rates_1,6V_(?P<index>\d+)_CV_.*\.mpr$", re.I),
}


def discover_cv_files() -> dict[str, list[Path]]:
    """Return naturally ordered CV files for the three upper potentials."""
    files = list(ROOT.glob("*.mpr"))
    batches: dict[str, list[Path]] = {}
    for label, pattern in BATCH_PATTERNS.items():
        matched = [(int(m.group("index")), path) for path in files if (m := pattern.search(path.name))]
        batches[label] = [path for _, path in sorted(matched)]
        if not batches[label]:
            raise FileNotFoundError(f"No CV files found for {label} in {ROOT}")
    return batches


def load_last_cycle(path: Path) -> pd.DataFrame:
    """Read an MPR file and return its last CV cycle with clean voltage/current."""
    data = we.read_file_safe(path, error_on_unknown_column=False, on_error="raise")
    if data is None or data.empty:
        raise ValueError(f"No data in {path}")
    required = {"Ewe/V", "<I>/mA", "cycle number", "time/s"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")

    cycle = data["cycle number"].max()
    result = data.loc[data["cycle number"] == cycle, ["time/s", "Ewe/V", "<I>/mA"]].copy()
    result.columns = ["time_s", "voltage", "current_mA"]
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    result = result.sort_index()
    if len(result) < 10:
        raise ValueError(f"Last cycle of {path.name} has too few points")
    return result


def branch_currents(data: pd.DataFrame, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate forward/reverse current onto *grid* and return scan rate."""
    voltage = data["voltage"].to_numpy(float)
    current = data["current_mA"].to_numpy(float)
    direction = np.diff(voltage)
    direction = direction[np.isfinite(direction) & (np.abs(direction) > 1e-8)]
    if len(direction) == 0:
        raise ValueError("Cannot determine CV sweep direction")
    forward_sign = 1.0 if np.nanmedian(direction) > 0 else -1.0
    dv = np.diff(voltage)
    forward = np.r_[False, dv * forward_sign >= 0]
    reverse = np.r_[False, dv * forward_sign < 0]

    # Include the point immediately before each direction transition.  This
    # avoids losing the turning point while keeping each branch monotonic.
    forward[:-1] |= reverse[1:] & ((forward_sign * dv) >= 0)
    reverse[:-1] |= forward[1:] & ((forward_sign * dv) < 0)

    def interpolate(mask: np.ndarray) -> np.ndarray:
        x, y = voltage[mask], current[mask]
        order = np.argsort(x)
        x, y = x[order], y[order]
        x, unique = np.unique(x, return_index=True)
        y = y[unique]
        if len(x) < 2 or grid.min() < x.min() or grid.max() > x.max():
            raise ValueError("CV branch does not cover the requested voltage range")
        return np.interp(grid, x, y)

    # Scan rate is the median absolute dE/dt, converted to V/s.
    if "time_s" in data:
        dt = np.diff(data["time_s"].to_numpy(float))
        valid = np.isfinite(dt) & (dt > 0) & (np.abs(dv) > 1e-8)
        scan_rate = float(np.nanmedian(np.abs(dv[valid] / dt[valid])))
    else:
        scan_rate = float("nan")
    return interpolate(forward), interpolate(reverse), scan_rate


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, and R-squared for a linear fit."""
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else np.nan
    return float(slope), float(intercept), r_squared


def main() -> None:
    OUT.mkdir(exist_ok=True)
    grid = np.linspace(VOLTAGE_MIN, VOLTAGE_MAX, N_VOLTAGE_POINTS)
    rows: list[dict[str, float | str]] = []
    summary: list[dict[str, float | str]] = []
    plot_data: dict[str, dict[str, np.ndarray | list[Path]]] = {}

    for batch, paths in discover_cv_files().items():
        scan_rates: list[float] = []
        delta_currents: list[np.ndarray] = []
        for path in paths:
            data = load_last_cycle(path)
            forward, reverse, scan_rate = branch_currents(data, grid)
            if not np.isfinite(scan_rate) or scan_rate <= 0:
                raise ValueError(f"Invalid scan rate inferred from {path.name}: {scan_rate}")
            scan_rates.append(scan_rate)
            # The sign depends on the potentiostat current convention and on
            # which sweep direction is labelled forward.  Capacitance uses
            # the positive branch separation, so take the absolute value.
            delta_currents.append(np.abs(forward - reverse) / 2.0)  # mA

        x = np.asarray(scan_rates)
        y = np.asarray(delta_currents)
        if len(x) < 3:
            raise ValueError(f"At least three scan rates are required for {batch}")
        order = np.argsort(x)
        x, y = x[order], y[order]
        plot_data[batch] = {"scan_rates": x, "delta_currents": y, "paths": paths}

        for voltage_index, voltage in enumerate(grid):
            # mA / (V/s) = mF. Convert to F only for the CSV's total_F column.
            slope_mF, intercept_mA, r_squared = linear_fit(x, y[:, voltage_index])
            row: dict[str, float | str] = {
                "upper_potential_V": batch,
                "voltage_V": voltage,
                "capacitance_mF": slope_mF,
                "capacitance_F": slope_mF / 1000.0,
                "fit_intercept_mA": intercept_mA,
                "r_squared": r_squared,
                "n_scan_rates": len(x),
            }
            if ELECTRODE_AREA_CM2:
                row["capacitance_mF_cm2"] = slope_mF / ELECTRODE_AREA_CM2
            rows.append(row)

        mean_delta = np.mean(y, axis=1)
        slope, intercept, r_squared = linear_fit(x, mean_delta)
        summary.append({
            "upper_potential_V": batch,
            "voltage_range_V": f"{VOLTAGE_MIN:.2f}-{VOLTAGE_MAX:.2f}",
            "capacitance_mF": slope,
            "capacitance_F": slope / 1000.0,
            "fit_intercept_mA": intercept,
            "r_squared": r_squared,
            "n_scan_rates": len(x),
            "scan_rates_V_s": ";".join(f"{rate:.6g}" for rate in x),
        })
        print(f"{batch}: scan rates = {', '.join(f'{rate:.5g}' for rate in x)} V/s; "
              f"Cdl = {slope:.6g} mF, R² = {r_squared:.5f}")

    pd.DataFrame(rows).to_csv(OUT / "cdl_470_by_potential.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "cdl_470_summary.csv", index=False)
    make_cv_plot(plot_data)
    make_fit_plot(plot_data)
    print(f"Wrote {OUT / 'cdl_470_by_potential.csv'}")
    print(f"Wrote {OUT / 'cdl_470_summary.csv'}")
    print(f"Wrote {OUT / 'cdl_470_cv_region_0p9_1p2V.png'}")
    print(f"Wrote {OUT / 'cdl_470_current_vs_scan_rate_fits.png'}")


def make_cv_plot(plot_data: dict[str, dict[str, np.ndarray | list[Path]]]) -> None:
    """Plot the last cycle of every scan-rate CV in the voltage range."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, (batch, values) in zip(axes, plot_data.items()):
        paths = values["paths"]
        scan_rates = np.asarray(values["scan_rates"])
        colors = we.get_colors(len(paths))
        for path, scan_rate, color in zip(paths, scan_rates, colors):
            data = load_last_cycle(path)
            selected = data[data["voltage"].between(VOLTAGE_MIN, VOLTAGE_MAX)]
            ax.plot(selected["voltage"], selected["current_mA"], color=color, lw=1.0,
                    label=f"{scan_rate:g} V/s")
        ax.set_title(f"Upper potential {batch}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, title="Scan rate", title_fontsize=8)
        ax.set_xlabel("Cell voltage (V)")
    axes[0].set_ylabel("Current (mA)")
    fig.suptitle("470 CVs in the double-layer-capacitance region")
    fig.tight_layout()
    fig.savefig(OUT / "cdl_470_cv_region_0p9_1p2V.png", dpi=300)
    plt.close(fig)


def make_fit_plot(plot_data: dict[str, dict[str, np.ndarray | list[Path]]]) -> None:
    """Plot the current-separation versus scan-rate fits used for Cdl."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = we.get_colors(len(plot_data))
    for (batch, values), color in zip(plot_data.items(), colors):
        scan_rates = np.asarray(values["scan_rates"])
        delta_currents = np.asarray(values["delta_currents"])
        mean_delta = np.mean(delta_currents, axis=1)
        slope, intercept, r_squared = linear_fit(scan_rates, mean_delta)
        fit_x = np.linspace(scan_rates.min(), scan_rates.max(), 100)
        ax.scatter(scan_rates, mean_delta, color=color, s=35, label=f"{batch} data")
        ax.plot(fit_x, slope * fit_x + intercept, color=color, lw=1.5,
                label=f"{batch} fit: {slope:.2f} mF, R²={r_squared:.4f}")
    ax.set_xlabel("Scan rate (V/s)")
    ax.set_ylabel("Half branch-current separation (mA)")
    ax.set_title("Double-layer-capacitance fits, averaged over 0.9–1.2 V")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "cdl_470_current_vs_scan_rate_fits.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))
    main()
