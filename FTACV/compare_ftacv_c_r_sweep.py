"""Compare Cdl and Rs FTACV sweep results and write PNG plots only."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HARMONIC_PATTERN = re.compile(r"^harmonic_(\d+)_envelope/mA$")
FOLDER_PATTERN = re.compile(
    r"^C_(?P<capacitance>[0-9]+(?:p[0-9]+)?)F_R_(?P<resistance>[0-9]+(?:p[0-9]+)?)ohm$"
)


@dataclass(frozen=True)
class Result:
    folder: Path
    capacitance_F: float
    resistance_ohm: float

    @property
    def envelope_path(self) -> Path:
        return self.folder / "simulation_harmonic_envelopes.csv"

    @property
    def dc_path(self) -> Path:
        return self.folder / "simulation_filtered_dc_current.csv"


def parse_number(value: str) -> float:
    return float(value.replace("p", "."))


def discover_results(root: Path) -> list[Result]:
    results = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        match = FOLDER_PATTERN.fullmatch(folder.name)
        if not match:
            continue
        result = Result(
            folder=folder,
            capacitance_F=parse_number(match.group("capacitance")),
            resistance_ohm=parse_number(match.group("resistance")),
        )
        if result.envelope_path.exists() and result.dc_path.exists():
            results.append(result)
    if len(results) != 16:
        raise FileNotFoundError(f"Expected 16 complete sweep results, found {len(results)}")
    return results


def label(result: Result) -> str:
    return f"C={result.capacitance_F:g} F, R={result.resistance_ohm:g} Ω"


def harmonics(result: Result) -> tuple[pd.DataFrame, list[int]]:
    data = pd.read_csv(result.envelope_path)
    numbers = sorted(
        int(match.group(1))
        for column in data.columns
        if (match := HARMONIC_PATTERN.match(column))
    )
    return data, numbers


def plot_harmonics(group: list[Result], output: Path, title: str, filename: str) -> None:
    loaded = [(result, *harmonics(result)) for result in group]
    harmonic_numbers = loaded[0][2]
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False, constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(group)))
    for axis, harmonic in zip(axes.ravel(), harmonic_numbers):
        column = f"harmonic_{harmonic:02d}_envelope/mA"
        for (result, data, _), color in zip(loaded, colors):
            axis.plot(data["dc_potential/V"], data[column], lw=1.0, color=color, label=label(result))
        axis.set_title(f"H{harmonic} ({harmonic * 20:g} Hz)")
        axis.set_xlabel("DC potential / V")
        axis.set_ylabel("Envelope current / mA")
        axis.grid(alpha=0.25)
    for axis in axes.ravel()[len(harmonic_numbers):]:
        axis.set_visible(False)
    axes.ravel()[0].legend(fontsize=8, ncol=2)
    figure.suptitle(title)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / filename, dpi=220)
    plt.close(figure)


def plot_dc_current(group: list[Result], output: Path, title: str, filename: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(group)))
    for result, color in zip(group, colors):
        data = pd.read_csv(result.dc_path)
        axis.plot(
            data["dc_potential_V"],
            data["filtered_dc_current_mA"],
            lw=0.8,
            color=color,
            label=label(result),
        )
    axis.set_title(title)
    axis.set_xlabel("DC potential / V")
    axis.set_ylabel("Filtered DC current / mA")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9, ncol=2)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / filename, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results" / "ftacv_c_r_sweep")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.input / "comparisons"
    results = discover_results(args.input)

    by_resistance = output / "by_resistance"
    for resistance in sorted({result.resistance_ohm for result in results}):
        group = sorted(
            [result for result in results if result.resistance_ohm == resistance],
            key=lambda result: result.capacitance_F,
        )
        token = str(resistance).replace(".", "p")
        plot_harmonics(
            group,
            by_resistance,
            f"Harmonic comparison: Cdl at R={resistance:g} Ω",
            f"harmonics_at_R_{token}ohm.png",
        )
        plot_dc_current(
            group,
            by_resistance,
            f"Filtered DC current: Cdl at R={resistance:g} Ω",
            f"filtered_dc_at_R_{token}ohm.png",
        )

    by_capacitance = output / "by_capacitance"
    for capacitance in sorted({result.capacitance_F for result in results}):
        group = sorted(
            [result for result in results if result.capacitance_F == capacitance],
            key=lambda result: result.resistance_ohm,
        )
        token = str(capacitance).replace(".", "p")
        plot_harmonics(
            group,
            by_capacitance,
            f"Harmonic comparison: Rs at Cdl={capacitance:g} F",
            f"harmonics_at_C_{token}F.png",
        )
        plot_dc_current(
            group,
            by_capacitance,
            f"Filtered DC current: Rs at Cdl={capacitance:g} F",
            f"filtered_dc_at_C_{token}F.png",
        )

    files = [path for path in output.rglob("*") if path.is_file()]
    if any(path.suffix.lower() != ".png" for path in files):
        raise RuntimeError("Comparison output contains non-PNG files")
    print(f"Created {len(files)} PNG comparison plots in {output}")


if __name__ == "__main__":
    main()

