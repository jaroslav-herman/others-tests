"""Compare FTacV results across frequency, amplitude, and scan-rate groups.

The input root contains folders named like ``10 Hz, 100 mV amp``.  Within
each folder, ``FTACV_01_ACV`` through ``FTACV_05_ACV`` are treated as scan
rates, as requested.  Only PNG graphs are written, in three subfolders of the
input root.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_ROOT = Path(
    r"\\ELECTROLYZER\PEM-WE_measurements\2026\485_I_PTL_no_Pt_higher_Ir_loading_N115_none_none"
    r"\FTacV\parameter range 2"
)

FOLDER_PATTERN = re.compile(
    r"^(?P<frequency>[0-9]+(?:\.[0-9]+)?)\s*Hz,\s*"
    r"(?P<amplitude>[0-9]+(?:\.[0-9]+)?)\s*mV\s*amp\s*$",
    re.IGNORECASE,
)
FILE_PATTERN = re.compile(r"^FTACV_(?P<scan>[0-9]+)_ACV$")
HARMONIC_PATTERN = re.compile(r"^harmonic_(\d+)_envelope/mA$")


@dataclass(frozen=True)
class Result:
    folder: Path
    frequency: float
    amplitude: float
    scan_rate: int
    stem: str
    envelope_path: Path
    dc_path: Path


def safe_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def discover_results(root: Path) -> list[Result]:
    results: list[Result] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        folder_match = FOLDER_PATTERN.match(folder.name.strip())
        if not folder_match:
            continue
        frequency = float(folder_match.group("frequency"))
        amplitude = float(folder_match.group("amplitude"))
        for envelope_path in sorted(folder.glob("*_harmonic_envelopes.csv")):
            stem = envelope_path.name.removesuffix("_harmonic_envelopes.csv")
            file_match = FILE_PATTERN.match(stem)
            dc_path = folder / f"{stem}_filtered_dc_current.csv"
            if file_match and dc_path.exists():
                results.append(
                    Result(
                        folder=folder,
                        frequency=frequency,
                        amplitude=amplitude,
                        scan_rate=int(file_match.group("scan")),
                        stem=stem,
                        envelope_path=envelope_path,
                        dc_path=dc_path,
                    )
                )
    if not results:
        raise FileNotFoundError(f"No complete FTacV result sets found under {root}")
    return results


def read_envelope(result: Result) -> pd.DataFrame:
    return pd.read_csv(result.envelope_path)


def read_dc_current(result: Result) -> pd.DataFrame:
    data = pd.read_csv(result.dc_path, usecols=["dc_potential_V", "filtered_dc_current_mA"])
    # Keep acquisition order, matching the ten-panel plots. Limit only the
    # rendering density for very large filtered-current exports.
    max_points = 12000
    step = max(1, int(np.ceil(len(data) / max_points)))
    return data.iloc[::step]


def result_label(result: Result) -> str:
    return f"{result.frequency:g} Hz, {result.amplitude:g} mV, scan {result.scan_rate:02d}"


def group_title(kind: str, group: list[Result]) -> str:
    first = group[0]
    if kind == "frequency":
        return f"Frequency comparison | {first.amplitude:g} mV amplitude | scan rate {first.scan_rate:02d}"
    if kind == "amplitude":
        return f"Amplitude comparison | {first.frequency:g} Hz | scan rate {first.scan_rate:02d}"
    return f"Scan-rate comparison | {first.frequency:g} Hz | {first.amplitude:g} mV amplitude"


def group_filename(kind: str, group: list[Result]) -> str:
    first = group[0]
    if kind == "frequency":
        return f"amplitude_{safe_number(first.amplitude)}mV_scan_{first.scan_rate:02d}"
    if kind == "amplitude":
        return f"frequency_{safe_number(first.frequency)}Hz_scan_{first.scan_rate:02d}"
    return f"frequency_{safe_number(first.frequency)}Hz_amplitude_{safe_number(first.amplitude)}mV"


def plot_harmonics(group: list[Result], output_dir: Path, kind: str) -> None:
    harmonic_numbers: set[int] = set()
    envelopes: dict[Result, pd.DataFrame] = {}
    for result in group:
        data = read_envelope(result)
        envelopes[result] = data
        harmonic_numbers.update(
            int(match.group(1))
            for column in data.columns
            if (match := HARMONIC_PATTERN.match(column))
        )
    harmonics = sorted(harmonic_numbers)
    ncols = 3
    nrows = int(np.ceil(len(harmonics) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 4.1 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(group)))
    for axis, harmonic in zip(axes.ravel(), harmonics):
        column = f"harmonic_{harmonic:02d}_envelope/mA"
        for result, color in zip(group, colors):
            data = envelopes[result]
            if column not in data.columns:
                continue
            axis.plot(
                data["dc_potential/V"],
                data[column],
                lw=1.0,
                marker=".",
                ms=1.8,
                color=color,
                label=result_label(result),
            )
        axis.set_title(f"H{harmonic} envelope ({harmonic * group[0].frequency:g} Hz reference)")
        axis.set_xlabel("DC potential / V")
        axis.set_ylabel("Envelope current / mA")
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(harmonics):]:
        axis.set_visible(False)
    axes.ravel()[0].legend(fontsize=8, ncol=2)
    figure.suptitle(f"FTacV harmonic envelopes — {group_title(kind, group)}", fontsize=15)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{group_filename(kind, group)}_harmonic_envelopes.png", dpi=220)
    plt.close(figure)


def plot_dc_current(group: list[Result], output_dir: Path, kind: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(group)))
    for result, color in zip(group, colors):
        data = read_dc_current(result)
        axis.plot(
            data["dc_potential_V"],
            data["filtered_dc_current_mA"],
            lw=0.8,
            color=color,
            label=result_label(result),
        )
    axis.set_title(f"Filtered dc-current comparison — {group_title(kind, group)}")
    axis.set_xlabel("DC potential / V")
    axis.set_ylabel("Filtered dc current / mA")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9, ncol=2)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{group_filename(kind, group)}_filtered_dc_current.png", dpi=220)
    plt.close(figure)


def make_groups(results: list[Result]) -> dict[str, list[list[Result]]]:
    definitions = {
        "frequency": lambda result: (result.amplitude, result.scan_rate),
        "amplitude": lambda result: (result.frequency, result.scan_rate),
        "scan_rate": lambda result: (result.frequency, result.amplitude),
    }
    groups: dict[str, list[list[Result]]] = {}
    for kind, key_function in definitions.items():
        grouped: dict[tuple[float, int | float], list[Result]] = {}
        for result in results:
            grouped.setdefault(key_function(result), []).append(result)
        groups[kind] = [
            sorted(group, key=lambda result: (result.frequency, result.amplitude, result.scan_rate))
            for group in grouped.values()
            if len(group) > 1
        ]
    return groups


def main() -> None:
    results = discover_results(INPUT_ROOT)
    groups = make_groups(results)
    output_dirs = {
        "frequency": INPUT_ROOT / "comparisons_by_frequency",
        "amplitude": INPUT_ROOT / "comparisons_by_amplitude",
        "scan_rate": INPUT_ROOT / "comparisons_by_scan_rate",
    }
    graph_count = 0
    for kind, kind_groups in groups.items():
        for group in kind_groups:
            plot_harmonics(group, output_dirs[kind], kind)
            plot_dc_current(group, output_dirs[kind], kind)
            graph_count += 2
    print(f"Discovered {len(results)} complete result sets")
    print(f"Created {graph_count} graphs in {len(output_dirs)} folders under {INPUT_ROOT}")
    for kind, kind_groups in groups.items():
        print(f"{kind}: {len(kind_groups)} comparison groups -> {output_dirs[kind]}")


if __name__ == "__main__":
    main()
