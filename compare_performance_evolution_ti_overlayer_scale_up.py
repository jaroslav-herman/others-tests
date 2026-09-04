"""Compare performance evolution for the 2026 Ti overlayer scale-up series."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["text.usetex"] = False

import wepy.basics as we
import wepy.iv_curve as weiv
from measurement_paths import candidate_measurement_roots


YEAR = 2026
SAMPLE_IDS = ("456", "462", "465", "466")
SAMPLE_TYPES = {sample_id: "PEM" for sample_id in SAMPLE_IDS}
SAMPLE_NAMES_FROM_GOOGLE_SHEET = {
    "456": "Ti in 17Ar 0,1O2 on Ir in Big",
    "462": "Ti_overlayer_400nm_in_Red_with_Ir_in_Big",
    "465": "Ti_in_17Ar_2O2_on_Ir_in_Big",
    "466": "Ti_in_17Ar_angled_on_Ir_in_Big",
}
CELL_VOLTAGES = (1.6, 1.8, 2.0)
OUTPUT_DIR = Path("results")


def find_sample_folders() -> dict[str, Path]:
    """Find one folder per sample using its Google Sheet ``Type``."""
    found: dict[str, Path] = {}
    for sample_id in SAMPLE_IDS:
        matches: list[Path] = []
        for root in candidate_measurement_roots(YEAR, SAMPLE_TYPES.get(sample_id)):
            folders = we.load_folders(
                str(root), contains_string=sample_id, natural_sort=True, mode="any"
            )
            if not isinstance(folders, str):
                matches.extend(
                    Path(folder)
                    for folder in folders
                    if Path(folder).name.startswith(f"{sample_id}_")
                    or Path(folder).name == sample_id
                )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one folder for {sample_id}, found {matches}"
            )
        found[sample_id] = matches[0]
    return found


def current_evolution(folder: Path) -> dict[float, list[float]]:
    """Return current nearest each target voltage for valid SV MPR files."""
    files = we.load_files(
        str(folder), contains_string="SV", extension=".mpr", natural_sort=True
    )
    result = {voltage: [] for voltage in CELL_VOLTAGES}
    if isinstance(files, str):
        print(f"Warning: {files}")
        return result

    for file in files:
        data = we.read_file_safe(file)
        if data is None:
            continue
        voltages, currents = weiv.IV_curves_data(data)
        for voltage_curve, current_curve in zip(voltages, currents):
            for target_voltage in CELL_VOLTAGES:
                index = int(np.abs(voltage_curve - target_voltage).argmin())
                result[target_voltage].append(float(current_curve[index]))
    return result


def main() -> None:
    folders = find_sample_folders()
    colors = we.get_colors(len(SAMPLE_IDS))
    evolution = {
        sample_id: current_evolution(folder)
        for sample_id, folder in folders.items()
    }
    if not any(values for sample in evolution.values() for values in sample.values()):
        raise RuntimeError("No valid IV curves were found for the Ti overlayer series")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target_voltage in CELL_VOLTAGES:
        fig, axis = plt.subplots(figsize=(12, 4.8))
        for sample_id, color in zip(SAMPLE_IDS, colors):
            currents = evolution[sample_id][target_voltage]
            if currents:
                axis.plot(
                    range(1, len(currents) + 1),
                    currents,
                    "o-",
                    color=color,
                    label=f"{sample_id} — {SAMPLE_NAMES_FROM_GOOGLE_SHEET[sample_id]}",
                )
        axis.set_title(f"Ti overlayer scale-up — {target_voltage:g} V")
        axis.set_xlabel("Measurement sequence")
        axis.set_ylabel("Current (mA)")
        axis.grid(False)
        axis.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.subplots_adjust(right=0.56)
        output = OUTPUT_DIR / (
            f"performance_evolution_ti_overlayer_scale_up_{target_voltage:g}V.png"
        )
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
