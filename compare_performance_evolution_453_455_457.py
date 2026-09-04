"""Compare current evolution at selected cell voltages for samples 453, 455, 457."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import wepy.basics as we
import wepy.iv_curve as weiv
from measurement_paths import candidate_measurement_roots


YEAR = 2026
SAMPLE_IDS = ("453", "455", "457","467","468", "470")
CELL_VOLTAGES = (1.6, 1.8, 2.0)

OUTPUT_DIR = Path("results")


def find_sample_folders() -> dict[str, Path]:
    """Find one 2026 measurement folder for each requested sample."""
    found: dict[str, Path] = {}
    for sample_id in SAMPLE_IDS:
        matches = []
        for root in candidate_measurement_roots(YEAR):
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
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple folders found for sample {sample_id}: {matches}"
            )
        if matches:
            found[sample_id] = matches[0]

    missing = [sample_id for sample_id in SAMPLE_IDS if sample_id not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not find folders for samples {missing} in the year and AEM-WE folders"
        )
    return found


def current_evolution(folder: Path) -> dict[float, list[float]]:
    """Return current values nearest each requested voltage, in file order."""
    files = we.load_files(
        str(folder),
        contains_string="SV",
        extension=".mpr",
        natural_sort=True,
    )
    result = {voltage: [] for voltage in CELL_VOLTAGES}

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
    sample_names = {
        sample_id: we.get_sample_name(
            sample_id, folders[sample_id] / "sample_log.csv"
        )
        for sample_id in SAMPLE_IDS
    }
    colors = we.get_colors(len(SAMPLE_IDS))
    evolution = {
        sample_id: current_evolution(folder)
        for sample_id, folder in folders.items()
    }

    if not any(values for sample in evolution.values() for values in sample.values()):
        raise RuntimeError("No valid IV curves were found for the requested samples")

    for target_voltage in CELL_VOLTAGES:
        fig, axis = plt.subplots(figsize=(6.5, 4.5))
        for sample_id, color in zip(SAMPLE_IDS, colors):
            currents = evolution[sample_id][target_voltage]
            if currents:
                axis.plot(
                    range(1, len(currents) + 1),
                    currents,
                    "o-",
                    color=color,
                    label=f"{sample_id} — {sample_names[sample_id]}",
                )
        axis.set_title(f"{target_voltage:g} V")
        axis.set_xlabel("Measurement sequence")
        axis.set_ylabel("Current (mA)")
        axis.grid(False)
        axis.legend(frameon=False)
        fig.suptitle(
            f"Performance evolution at {target_voltage:g} V"
        )
        fig.tight_layout()

        output = OUTPUT_DIR / (
            f"performance_evolution_etching_series_{target_voltage:g}V.png"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved: {output.resolve()}")
        plt.show()


if __name__ == "__main__":
    main()
