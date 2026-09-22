"""Compare exported FTacV harmonic envelopes for harmonics 1--12."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(
    r"\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode_etching_series_GDE"
    r"\FTacV\parameter range\50 Hz, 300 mV amp"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "ftacv_envelope_comparison_50Hz_300mV"
FUNDAMENTAL_HZ = 50.0
TRIM_CYCLES = 10.0
EDGE_GUARD_CYCLES = 10.0
HARMONIC_PATTERN = re.compile(r"^harmonic_(\d+)_envelope/mA$")


def load_comparison(input_dir: Path) -> tuple[pd.DataFrame, list[int], list[str]]:
    envelope_files = sorted(input_dir.glob("*_harmonic_envelopes.csv"))
    if not envelope_files:
        raise FileNotFoundError(f"No exported harmonic envelopes found under {input_dir}")

    records: list[pd.DataFrame] = []
    harmonic_numbers: set[int] = set()
    file_labels: list[str] = []
    for envelope_path in envelope_files:
        stem = envelope_path.name.removesuffix("_harmonic_envelopes.csv")
        envelope = pd.read_csv(envelope_path)
        if "dc_potential/V" not in envelope.columns:
            raise ValueError(f"{envelope_path.name} has no dc_potential/V column")
        columns = [column for column in envelope.columns if HARMONIC_PATTERN.match(column)]
        if not columns:
            raise ValueError(f"{envelope_path.name} has no harmonic envelope columns")
        harmonic_numbers.update(int(HARMONIC_PATTERN.match(column).group(1)) for column in columns)

        envelope.insert(0, "file", stem)
        records.append(envelope)
        file_labels.append(stem)

    comparison = pd.concat(records, ignore_index=True)
    return comparison, sorted(harmonic_numbers), file_labels


def plot_comparison(comparison: pd.DataFrame, harmonics: list[int], file_labels: list[str], output: Path) -> None:
    components = [f"harmonic_{number:02d}_envelope/mA" for number in harmonics]
    titles = [f"H{number} envelope ({number * FUNDAMENTAL_HZ:g} Hz)" for number in harmonics]
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(file_labels)))
    color_by_file = dict(zip(file_labels, colors))

    ncols = 3
    nrows = int(np.ceil(len(components) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()
    for axis, component, title in zip(axes_flat, components, titles):
        for label in file_labels:
            subset = comparison[comparison["file"] == label].sort_values("dc_potential/V")
            axis.plot(subset["dc_potential/V"], subset[component], color=color_by_file[label], lw=1.25, label=label)
        axis.set_title(title)
        axis.set_xlabel("DC potential / V")
        axis.set_ylabel("Current / mA")
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(components):]:
        axis.set_visible(False)
    axes_flat[0].legend(fontsize=8, ncol=2)
    figure.suptitle("FTacV envelope comparison: 50 Hz, 300 mV amplitude", fontsize=15)
    figure.savefig(output / "ftacv_harmonic_envelopes_comparison.png", dpi=220)
    plt.close(figure)


def write_summary(comparison: pd.DataFrame, harmonics: list[int], output: Path) -> None:
    components = [f"harmonic_{number:02d}_envelope/mA" for number in harmonics]
    rows = []
    for file_label, subset in comparison.groupby("file", sort=False):
        for component in components:
            values = subset[component].to_numpy(dtype=float)
            rows.append({"file": file_label, "component": component, "max_mA": np.nanmax(values), "mean_mA": np.nanmean(values)})
    pd.DataFrame(rows).to_csv(output / "ftacv_envelopes_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparison, harmonics, file_labels = load_comparison(args.input)
    comparison.to_csv(args.output / "ftacv_envelopes_comparison.csv", index=False)
    write_summary(comparison, harmonics, args.output)
    plot_comparison(comparison, harmonics, file_labels, args.output)
    print(f"Compared {len(file_labels)} files: {', '.join(file_labels)}")
    print(f"Harmonics: {', '.join(str(number) for number in harmonics)}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
