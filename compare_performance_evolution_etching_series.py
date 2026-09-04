"""Compare performance evolution for every sample in the 2026 ``etching series`` project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import wepy.basics as we
import wepy.iv_curve as weiv
from measurement_paths import candidate_measurement_roots


YEAR = 2026
SAMPLE_NAMES_FROM_GOOGLE_SHEET = {
    "453": "cathode etching 140 min",
    "455": "cathode_etching_series_35min",
    "457": "cathode_etching_series_0min",
    "467": "cathode_etching_series_20min",
    "468": "cathode_etching_series_210min",
    "470": "cathode_etching_series_GDE",
}
SAMPLE_IDS = tuple(SAMPLE_NAMES_FROM_GOOGLE_SHEET)
CELL_VOLTAGES = (1.6, 1.8, 2.0)
OUTPUT_DIR = Path("results")


def find_sample_folders() -> dict[str, Path]:
    """Find the measurement folder for each sample using ``load_folders``."""
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
            raise RuntimeError(f"Multiple folders found for {sample_id}: {matches}")
        if matches:
            found[sample_id] = matches[0]

    missing = [sample_id for sample_id in SAMPLE_IDS if sample_id not in found]
    if missing:
        raise FileNotFoundError(
            f"Could not find folders for samples {missing} in the year and AEM-WE folders"
        )
    return found


def current_evolution(folder: Path) -> dict[float, list[float]]:
    """Return current nearest each target voltage for valid SV MPR files."""
    files = we.load_files(
        str(folder),
        contains_string="SV",
        extension=".mpr",
        natural_sort=True,
    )
    if isinstance(files, str):
        print(f"Warning: {files}")
        return {voltage: [] for voltage in CELL_VOLTAGES}

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
    colors = we.get_colors(len(SAMPLE_IDS))
    evolution = {
        sample_id: current_evolution(folder)
        for sample_id, folder in folders.items()
    }

    if not any(values for sample in evolution.values() for values in sample.values()):
        raise RuntimeError("No valid IV curves were found for the etching series")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target_voltage in CELL_VOLTAGES:
        fig, axis = plt.subplots(figsize=(6.5, 4.5))
        for sample_id, color in zip(SAMPLE_IDS, colors):
            currents = evolution[sample_id][target_voltage]
            if not currents:
                print(f"Warning: no valid curves for sample {sample_id}")
                continue
            axis.plot(
                range(1, len(currents) + 1),
                currents,
                "o-",
                color=color,
                label=f"{sample_id} — {SAMPLE_NAMES_FROM_GOOGLE_SHEET[sample_id]}",
            )

        axis.set_title(f"{target_voltage:g} V")
        axis.set_xlabel("Measurement sequence")
        axis.set_ylabel("Current (mA)")
        axis.grid(False)
        axis.legend(frameon=False)
        fig.suptitle(f"Etching-series performance evolution at {target_voltage:g} V")
        fig.tight_layout()

        output = OUTPUT_DIR / (
            f"performance_evolution_etching_series_{target_voltage:g}V.png"
        )
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
