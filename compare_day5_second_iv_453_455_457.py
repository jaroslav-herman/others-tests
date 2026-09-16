"""Compare the second Day 5 IV curve for samples 453, 455, and 457."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

import wepy.basics as we
import wepy.iv_curve as weiv
from measurement_paths import find_measurements_root


YEAR = 2026
SAMPLE_IDS = ("453", "455", "457","467", "468", '470')
OUTPUT = Path("results") / "day4_second_iv_comparison_453_455_457.png"

# Names read from the shared sample table. Empty values intentionally fall
# back to the sample number in plot labels.
SAMPLE_NAMES = {
    "453": "cathode etching 140 min",
    "455": "cathode_etching_series_35min",
    "457": "cathode_etching_series_0min",
    "467": "cathode_etching_series_20min",
    "468": "cathode_etching_series_210min",
    "470": "cathode_etching_series_GDE",
}



def find_sample_folders() -> dict[str, Path]:
    """Find the AEM-WE folder for each requested sample."""
    data_root = find_measurements_root(YEAR)
    folders = we.load_folders(
        str(data_root),
        contains_string=list(SAMPLE_IDS),
        natural_sort=True,
        mode="any",
    )
    if isinstance(folders, str):
        raise FileNotFoundError(folders)

    found = {}
    for sample_id in SAMPLE_IDS:
        matches = [
            Path(folder)
            for folder in folders
            if Path(folder).name.startswith(f"{sample_id}_")
            or Path(folder).name == sample_id
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one folder for sample {sample_id}, found: {matches}"
            )
        found[sample_id] = matches[0]
    return found


def second_iv(folder: Path):
    """Read cycle 2 from the first valid Day 7 SV MPR file."""
    files = we.load_files(
        str(folder),
        contains_string=["Day7", "SV"],
        extension=".mpr",
        natural_sort=True,
        mode="all",
    )
    for file in files:
        data = we.read_file_safe(file)
        if data is None:
            continue

        voltages, currents = weiv.IV_curves_data(data)
        if len(voltages) >= 1:
            voltage, current = voltages[0], currents[0]
            return voltage, current, Path(file)

    raise RuntimeError(f"No second IV curve found in {folder}")


def label_for(sample_id: str) -> str:
    name = SAMPLE_NAMES.get(sample_id, "").strip()
    return f"{sample_id} — {name}" if name else sample_id


def main() -> None:
    folders = find_sample_folders()
    colors = we.get_colors(len(SAMPLE_IDS))

    fig, axis = plt.subplots(figsize=(7, 5))
    for sample_id, color in zip(SAMPLE_IDS, colors):
        voltage, current, source = second_iv(folders[sample_id])
        axis.plot(current, voltage, color=color, label=label_for(sample_id))
        print(f"{sample_id}: {source}")

    axis.set_xlabel("Current (mA)")
    axis.set_ylabel("Cell voltage (V)")
    axis.set_title("Day 7 — second IV curve")
    axis.grid(False)
    axis.legend(frameon=False)
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
