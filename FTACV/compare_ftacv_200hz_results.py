"""Compare exported FTacV envelopes, filtered dc current, and spectra."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paste the complete results-folder path between the quotes, then run this file
# with VS Code's Run Python File button.  Keep the r prefix for Windows paths.
INPUT_FOLDER = r'\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode_etching_series_GDE\FTacV\parameter range\50 Hz, 300 mV amp'

DEFAULT_INPUT = Path(
    INPUT_FOLDER
)
# Save the generated comparison plots and CSV directly in the selected folder.
DEFAULT_OUTPUT = DEFAULT_INPUT
HARMONIC_PATTERN = re.compile(r"^harmonic_(\d+)_envelope/mA$")
FREQUENCY_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*Hz", re.IGNORECASE)
AMPLITUDE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*mV", re.IGNORECASE)


def output_prefix(input_dir: Path) -> str:
    """Create a filesystem-safe filename prefix from the selected folder name."""
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", input_dir.name).strip("_.")
    return prefix or "ftacv_results"


def folder_description(input_dir: Path) -> tuple[float | None, str]:
    """Return the excitation frequency and a human-readable folder label."""
    folder_name = input_dir.name
    frequency_match = FREQUENCY_PATTERN.search(folder_name)
    amplitude_match = AMPLITUDE_PATTERN.search(folder_name)
    frequency = float(frequency_match.group(1)) if frequency_match else None
    if frequency_match and amplitude_match:
        amplitude = float(amplitude_match.group(1))
        label = f"{frequency:g} Hz, {amplitude:g} mV amplitude"
    else:
        label = folder_name
    return frequency, label


def discover_stems(input_dir: Path) -> list[str]:
    stems = []
    for path in sorted(input_dir.glob("*_harmonic_envelopes.csv")):
        stem = path.name.removesuffix("_harmonic_envelopes.csv")
        required = [
            input_dir / f"{stem}_filtered_dc_current.csv",
            input_dir / f"{stem}_power_spectrum.csv",
        ]
        if all(path.exists() for path in required):
            stems.append(stem)
    if not stems:
        raise FileNotFoundError("No complete envelope/DC-current/spectrum result sets were found")
    return stems


def load_envelopes(input_dir: Path, stems: list[str]) -> tuple[pd.DataFrame, list[int]]:
    frames = []
    harmonics: set[int] = set()
    for stem in stems:
        frame = pd.read_csv(input_dir / f"{stem}_harmonic_envelopes.csv")
        columns = [column for column in frame.columns if HARMONIC_PATTERN.match(column)]
        harmonics.update(int(HARMONIC_PATTERN.match(column).group(1)) for column in columns)
        frame.insert(0, "file", stem)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), sorted(harmonics)


def plot_results(input_dir: Path, output: Path, stems: list[str], envelopes: pd.DataFrame, harmonics: list[int]) -> None:
    prefix = output_prefix(input_dir)
    frequency, folder_label = folder_description(input_dir)
    frequency_for_labels = frequency if frequency is not None else 1.0
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(stems)))
    color_by_stem = dict(zip(stems, colors))
    components = [f"harmonic_{number:02d}_envelope/mA" for number in harmonics]
    ncols = 3
    nrows = int(np.ceil(len(components) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.1 * nrows), squeeze=False, constrained_layout=True)
    for axis, harmonic, component in zip(axes.ravel(), harmonics, components):
        for stem in stems:
            # Preserve acquisition order, as in ftacv_fourier_harmonic_filter.py.
            subset = envelopes[envelopes["file"] == stem]
            axis.plot(
                subset["dc_potential/V"],
                subset[component],
                lw=1.0,
                marker=".",
                ms=2,
                color=color_by_stem[stem],
                label=stem,
            )
        axis.set_title(f"H{harmonic} envelope ({harmonic * frequency_for_labels:g} Hz)")
        axis.set_xlabel("DC potential / V")
        axis.set_ylabel("Envelope current / mA")
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(components):]:
        axis.set_visible(False)
    axes.ravel()[0].legend(fontsize=8, ncol=2)
    fig.suptitle(f"FTacV harmonic-envelope comparison: {folder_label}", fontsize=15)
    fig.savefig(output / f"{prefix}_harmonic_envelopes.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for stem in stems:
        data = pd.read_csv(input_dir / f"{stem}_filtered_dc_current.csv")
        # Preserve acquisition order, as in the ten-panel filtered-dc plot.
        axis.plot(data["dc_potential_V"], data["filtered_dc_current_mA"], lw=0.8, color=color_by_stem[stem], label=stem)
    axis.set_title(f"Filtered dc-current comparison: {folder_label}")
    axis.set_xlabel("DC potential / V")
    axis.set_ylabel("Filtered dc current / mA")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9, ncol=2)
    fig.savefig(output / f"{prefix}_filtered_dc_current.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for stem in stems:
        data = pd.read_csv(input_dir / f"{stem}_power_spectrum.csv")
        data = data[data["frequency_Hz"] > 0]
        axis.plot(data["frequency_Hz"], data["power_mA2"], lw=1.0, color=color_by_stem[stem], label=stem)
    axis.set_yscale("log")
    axis.set_xlim(left=0)
    axis.set_title(f"Current power-spectrum comparison: {folder_label}")
    axis.set_xlabel("Frequency / Hz")
    axis.set_ylabel("Power / mA² (log scale)")
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=9, ncol=2)
    fig.savefig(output / f"{prefix}_power_spectra.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stems = discover_stems(args.input)
    envelopes, harmonics = load_envelopes(args.input, stems)
    prefix = output_prefix(args.input)
    envelopes.to_csv(args.output / f"{prefix}_harmonic_envelopes_comparison.csv", index=False)
    plot_results(args.input, args.output, stems, envelopes, harmonics)
    print(f"Compared {len(stems)} files: {', '.join(stems)}")
    print(f"Harmonics present in exports: {', '.join(str(number) for number in harmonics)}")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
