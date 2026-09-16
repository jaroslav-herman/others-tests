"""Compare EIS fit parameters for samples 462, 465, and 466.

The input exports are expected to contain one row per EIS spectrum, with
columns such as Time, I_mA, Ecell_V, R0-R3, C1-C3, and/or tau1-tau3.
Run from the repository root with the project's Python environment.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wepy.basics as we

# Keep the script portable: no external TeX installation or writable TeX
# cache is required for these diagnostic plots.
matplotlib.rcParams["text.usetex"] = False


SAMPLE_FILES = {
    "462": Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\462_export.csv"),
    "465": Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\465_export.csv"),
    "466": Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\466_export.csv"),
}
OUTPUT_DIR = Path("results") / "eis_parameters_462_465_466"


def load_sample(sample: str, path: Path) -> pd.DataFrame:
    """Read one export through wepy and standardize numeric columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Sample {sample}: {path}")
    # These are fit-export CSVs, not raw EC-Lab files; use wepy's explicit
    # delimited-file reader rather than the raw-file autodetector.
    data = we.read_file(str(path), skiprows=0, delimiter=",")
    if data is None or data.empty:
        raise ValueError(f"Sample {sample}: no tabular data could be read from {path}")
    data.columns = [str(column).strip() for column in data.columns]
    for column in data.columns:
        if column not in {"Time", "Cycle mod 15"}:
            converted = pd.to_numeric(data[column], errors="coerce")
            if converted.notna().sum() >= max(2, len(data) // 10):
                data[column] = converted
    data["Sample"] = sample
    return data


def parameter_columns(data: pd.DataFrame) -> list[str]:
    """Select fitted EIS parameters, excluding identifiers and operating values."""
    excluded = {"Sample", "Time", "Cycle mod 15", "Cycle mod", "I_mA", "Ecell_V", "Ewe_V", "Ewe-ce"}
    return [
        column for column in data.columns
        if column not in excluded
        and not column.endswith("_e")
        and re.fullmatch(r"(?:R|L|C|Q|a|tau)\d+", column) is not None
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if len(values) else np.nan


def make_summary(all_data: pd.DataFrame, parameters: list[str]) -> pd.DataFrame:
    rows = []
    for sample, group in all_data.groupby("Sample", sort=True):
        for parameter in parameters:
            values = pd.to_numeric(group[parameter], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) == 0:
                continue
            rows.append({
                "Sample": sample,
                "Parameter": parameter,
                "N": len(values),
                "Coverage": len(values) / len(group),
                "Median": values.median(),
                "Mean": values.mean(),
                "Std": values.std(),
                "Minimum": values.min(),
                "Maximum": values.max(),
            })
    return pd.DataFrame(rows)


def plot_dependency(all_data: pd.DataFrame, parameters: list[str], x_column: str, filename: str, xlabel: str) -> None:
    """Plot raw parameter dependencies, connecting points within each Time series.

    No pooled average is used. The color identifies the export's Time value,
    while separate panels preserve the identity of each sample.
    """
    ncols = 2
    nrows = int(np.ceil(len(parameters) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(13, max(4, 3.6 * nrows)), squeeze=False)
    sample_names = sorted(all_data["Sample"].unique())
    time_values = sorted(pd.to_numeric(all_data["Time"], errors="coerce").dropna().unique())
    time_colors = dict(zip(time_values, we.get_colors(max(1, len(time_values)))))
    for axis, parameter in zip(axes.flat, parameters):
        for sample in sample_names:
            subset = all_data[all_data["Sample"] == sample]
            for time, group in subset.groupby("Time", sort=True):
                valid = group[[x_column, parameter]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(x_column)
                if not valid.empty:
                    axis.plot(valid[x_column], valid[parameter], "o-", ms=2.2, lw=0.8,
                              alpha=0.7, color=time_colors.get(time, "0.4"))
        axis.set_title(parameter)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(parameter)
        if parameter.startswith(("R", "C", "Q", "tau", "L")):
            positive = all_data[parameter].dropna() > 0
            if len(positive) and positive.all():
                axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(parameters):]:
        axis.remove()
    figure.suptitle(f"EIS-fit parameter dependencies on {xlabel}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=250)
    plt.close(figure)


def plot_time_evolution(all_data: pd.DataFrame, parameters: list[str]) -> None:
    """Plot raw fitted values against the exported ``Time`` index."""
    time_column = "Time"
    figure, axes = plt.subplots(len(parameters), 1, figsize=(11, max(5, 2.5 * len(parameters))), squeeze=False)
    sample_colors = dict(zip(sorted(all_data["Sample"].unique()), we.get_colors(all_data["Sample"].nunique())))
    for axis, parameter in zip(axes[:, 0], parameters):
        for sample, group in all_data.groupby("Sample", sort=True):
            valid = group[[time_column, parameter]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(time_column)
            if not valid.empty:
                axis.plot(valid[time_column], valid[parameter], "o-", ms=2.5, lw=1,
                          alpha=0.7, color=sample_colors[sample], label=sample)
        axis.set_ylabel(parameter)
        if (all_data[parameter].dropna() > 0).all():
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time")
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.suptitle("Raw EIS-parameter time evolution")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "parameter_time_evolution.png", dpi=250)
    plt.close(figure)


def plot_single_sample_dependencies(all_data: pd.DataFrame, sample: str, parameters: list[str]) -> None:
    """Write unpooled current, voltage, and Time plots for one sample."""
    sample_data = all_data[all_data["Sample"] == sample].copy()
    for x_column, filename, xlabel in (
        ("I_mA", f"sample_{sample}_vs_current.png", "Current (mA)"),
        ("Ecell_V", f"sample_{sample}_vs_ecell.png", "Cell voltage (V)"),
    ):
        plot_dependency(sample_data, parameters, x_column, filename, xlabel)

    ncols = 2
    nrows = int(np.ceil(len(parameters) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(13, max(4, 3.6 * nrows)), squeeze=False)
    for axis, parameter in zip(axes.flat, parameters):
        valid = sample_data[["Time", parameter]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("Time")
        axis.plot(valid["Time"], valid[parameter], "o-", ms=2.5, lw=0.9, alpha=0.75)
        axis.set_title(parameter)
        axis.set_xlabel("Time")
        axis.set_ylabel(parameter)
        if len(valid) and (valid[parameter] > 0).all():
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(parameters):]:
        axis.remove()
    figure.suptitle(f"Sample {sample}: raw EIS-parameter evolution with Time")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"sample_{sample}_time_evolution.png", dpi=250)
    plt.close(figure)


def plot_median_heatmap(summary: pd.DataFrame) -> None:
    """Plot log10 median ratios to expose the strongest sample differences."""
    pivot = summary.pivot(index="Parameter", columns="Sample", values="Median")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    # Log scale is useful for positive EIS values, while signed/zero values
    # remain visible as blank cells rather than generating warnings.
    matrix = np.log10(pivot.where(pivot > 0))
    figure, axis = plt.subplots(figsize=(7, max(4, 0.38 * len(matrix))))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Median EIS parameters (log10 scale)")
    figure.colorbar(image, ax=axis, label="log10(parameter value)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "median_parameter_heatmap.png", dpi=250)
    plt.close(figure)


def plot_resistance_summary(summary: pd.DataFrame) -> None:
    resistances = [parameter for parameter in summary["Parameter"].unique() if parameter.startswith("R")]
    if not resistances:
        return
    table = summary[summary["Parameter"].isin(resistances)].pivot(index="Parameter", columns="Sample", values="Median")
    table = table.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if table.empty:
        return
    axis = table.plot(kind="bar", logy=True, figsize=(8, 5), color=we.get_colors(len(table.columns)))
    axis.set_ylabel("Median resistance (ohm; log scale)")
    axis.set_title("Median fitted resistances")
    axis.grid(axis="y", alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUT_DIR / "median_resistances.png", dpi=250)
    plt.close(axis.figure)


def print_interpretation(summary: pd.DataFrame) -> None:
    medians = summary.pivot(index="Parameter", columns="Sample", values="Median")
    print("\nData overview")
    for sample in sorted(medians.columns):
        print(f"  Sample {sample}: {int(summary.loc[summary.Sample == sample, 'N'].max())} valid values per parameter")
    print("\nLargest median separations")
    for parameter, row in medians.iterrows():
        values = row.dropna()
        if len(values) >= 2 and (values > 0).all():
            ratio = values.max() / values.min()
            if ratio >= 1.5:
                high, low = values.idxmax(), values.idxmin()
                coverage = summary[(summary["Parameter"] == parameter) & summary["Sample"].isin(values.index)].set_index("Sample")["Coverage"]
                coverage_text = ", ".join(f"{sample} {coverage.get(sample, np.nan):.0%}" for sample in (high, low))
                print(f"  {parameter}: {high}/{low} = {ratio:.2g} ({values[high]:.4g} vs {values[low]:.4g}; coverage {coverage_text})")
    print("\nInterpretation guide: parameters with similar medians and overlapping point clouds are shared features; large median ratios or separated current trends are sample-specific differences.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = [load_sample(sample, path) for sample, path in SAMPLE_FILES.items()]
    all_data = pd.concat(samples, ignore_index=True)
    parameters = sorted(set.intersection(*(set(parameter_columns(data)) for data in samples)))
    if not parameters:
        raise ValueError("No common numeric EIS parameter columns were found.")
    # Charge-transfer conductance is often more directly comparable than R1.
    # Keep it as a derived coordinate without changing the source R1 values.
    all_data["inv_R1"] = 1.0 / pd.to_numeric(all_data["R1"], errors="coerce")
    parameters_with_inverse = [*parameters, "inv_R1"]
    summary = make_summary(all_data, parameters_with_inverse)
    # This table is retained as a diagnostic reference only. The primary
    # analysis is the unpooled dependency plots below.
    summary.to_csv(OUTPUT_DIR / "parameter_summary_diagnostic.csv", index=False)
    plot_dependency(all_data, parameters_with_inverse, "I_mA", "parameter_vs_current.png", "Current (mA)")
    plot_dependency(all_data, parameters_with_inverse, "Ecell_V", "parameter_vs_ecell.png", "Cell voltage (V)")
    plot_time_evolution(all_data, ["R0", "R1", "inv_R1", "C1", "Q1", "a1", "L0"])
    for sample in sorted(all_data["Sample"].unique()):
        plot_single_sample_dependencies(all_data, sample, parameters_with_inverse)
    print_interpretation(summary)
    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
