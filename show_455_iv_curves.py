"""Plot IV-curve time evolution for sample 455."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import wepy.basics as we
import wepy.iv_curve as weiv
from measurement_paths import candidate_measurement_roots


SAMPLE_ID = "455"
OUTPUT = Path("results") / f"{SAMPLE_ID}_IV_time_evolution.png"


def main() -> None:
    data_folder = next(
        (
            root / "455_IV_cathode_etching_series_35min"
            for root in candidate_measurement_roots(2026, sample_type=None)
            if (root / "455_IV_cathode_etching_series_35min").is_dir()
        ),
        None,
    )
    if data_folder is None:
        raise FileNotFoundError("Could not find the sample 455 measurement folder")
    if not data_folder.is_dir():
        raise FileNotFoundError(f"Measurement folder is not accessible: {data_folder}")

    all_files = we.load_files(
        str(data_folder),
        contains_string="SV",
        extension="all",
        natural_sort=True,
    )
    files = [
        Path(file)
        for file in all_files
        if Path(file).suffix.lower() == ".mpr"
    ]
    if not files:
        raise FileNotFoundError(f"No SV measurement files found in {data_folder}")

    measurements = []
    for file in files:
        print(file)

        data = we.read_file_safe(str(file))
        if data is None:
            continue
        voltages, currents = weiv.IV_curves_data(data)
        for cycle, (voltage, current) in enumerate(
            zip(voltages, currents), start=1
        ):
            measurements.append((file, cycle, voltage, current))


    if not measurements:
        raise RuntimeError("No IV curves could be extracted from the MPR files")

    colors = we.get_colors(len(measurements))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for (file, cycle, voltage, current), color in zip(measurements, colors):
        ax.plot(current, voltage, color=color, label=f"{file.stem}, cycle {cycle}")

    ax.set_xlabel("Current (mA)")
    ax.set_ylabel("Cell voltage (V)")
    ax.set_title(f"Sample {SAMPLE_ID} — IV-curve time evolution")
    ax.grid(False)

    if len(measurements) <= 12:
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
