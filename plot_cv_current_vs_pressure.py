"""Plot CV current at 2.0 V against the pressure encoded in each filename."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = False
import numpy as np

import wepy.basics as we
import wepy.iv_curve as weiv

plt.rcParams["text.usetex"] = False


MEASUREMENT_FOLDERS = {
    "432": Path(
        r"C:\Users\Herman\OneDrive - Univerzita Karlova\Letní projekt\Tlaky\432_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_Pressureincrease-0-5bar"
    ),
    "433": Path(
        r"C:\Users\Herman\OneDrive - Univerzita Karlova\Letní projekt\Tlaky\433_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_withscrewedbolts-N115_none_BDC929"
    ),
    "438": Path(
        r'C:\Users\Herman\OneDrive - Univerzita Karlova\Letní projekt\Tlaky\438_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_none_BDC929'
    ),
    "439": Path(
        r"C:\Users\Herman\OneDrive - Univerzita Karlova\Letní projekt\Tlaky\439_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_etchedcathode_BDC929"
    ),
}
TARGET_VOLTAGES = (1.6, 1.8, 2.0)
OUTPUT = Path("results") / "cv_current_at_2V_vs_pressure.png"
CSV_OUTPUT = Path("results") / "cv_current_vs_pressure_data.csv"

PRESSURE_PATTERNS = (
    # Original form and variants with a descriptive suffix, for example:
    # _3,5alebo4_bar, _0alebo0-druhykrat_bar, _2,5alebo3--druhzkrat_bar
    re.compile(
        r"_(?P<first>[+-]?\d+(?:[.,]\d+)?)alebo"
        r"(?P<second>[+-]?\d+(?:[.,]\d+)?)(?:-[^_]*)?_bar",
        re.IGNORECASE,
    ),
    # Screwed-bolts form, for example: 2baralebo2,7bar
    re.compile(
        r"(?P<first>\d+(?:[.,]\d+)?)baralebo"
        r"(?P<second>[+-]?\d+(?:[.,]\d+)?)bar",
        re.IGNORECASE,
    ),
)

SINGLE_PRESSURE_PATTERN = re.compile(
    r"(?:^|[_-])(?P<single>\d+(?:[.,]\d+)?)(?:_bar|bar)(?=[_.-]|$)",
    re.IGNORECASE,
)


def extract_pressures(filename: str) -> tuple[float, float]:
    """Extract pressures from either supported filename convention."""
    for pattern in PRESSURE_PATTERNS:
        match = pattern.search(filename)
        if match is not None:
            return (
                float(match.group("first").replace(",", ".")),
                float(match.group("second").replace(",", ".")),
            )

    single_match = SINGLE_PRESSURE_PATTERN.search(filename)
    if single_match is not None:
        single_pressure = float(single_match.group("single").replace(",", "."))
        return single_pressure, single_pressure

    raise ValueError(f"Could not extract pressure from filename: {filename}")


def basename(value: object) -> str:
    """Return a filename basename for either slash convention."""
    return str(value).strip().strip('"\'').replace("\\", "/").rsplit(
        "/", maxsplit=1
    )[-1]


def modified_date_for_source(source_file: Path, mps_files: list[Path]) -> datetime:
    """Use the matching .mps date, or the source .mpr date if no .mps exists."""
    source_name = basename(source_file.name).casefold()
    matches = [
        path
        for path in mps_files
        if source_name == path.name.casefold()
        or source_name.startswith(f"{path.stem.casefold()}_")
    ]
    if not matches:
        print(
            f"No matching .mps file for {source_file.name}; "
            "using the .mpr modified date."
        )
        return datetime.fromtimestamp(source_file.stat().st_mtime)

    match = max(matches, key=lambda path: len(path.stem))
    return datetime.fromtimestamp(match.stat().st_mtime)


def currents_at_target_voltages(
    data, target_voltages: tuple[float, ...]
) -> dict[float, float]:
    """Return mean current nearest each target across extracted CV curves."""
    voltages, currents = weiv.IV_curves_data(data)
    values_by_voltage = {voltage: [] for voltage in target_voltages}
    for voltage_curve, current_curve in zip(voltages, currents):
        if len(voltage_curve) == 0 or len(current_curve) == 0:
            continue
        for target_voltage in target_voltages:
            index = int(np.abs(voltage_curve - target_voltage).argmin())
            values_by_voltage[target_voltage].append(float(current_curve[index]))

    if not any(values_by_voltage.values()):
        raise ValueError("No valid CV curves were found")

    return {
        voltage: float(np.mean(values)) if values else np.nan
        for voltage, values in values_by_voltage.items()
    }


def main() -> None:
    results = []
    for sample, measurement_folder in MEASUREMENT_FOLDERS.items():
        if not measurement_folder.is_dir():
            raise FileNotFoundError(
                f"Measurement folder is not accessible: {measurement_folder}"
            )

        loaded_files = we.load_files(
            str(measurement_folder),
            contains_string="CV",
            extension=".mpr",
            natural_sort=True,
        )
        files = [
            Path(file)
            for file in loaded_files
            if Path(file).suffix.lower() == ".mpr"
        ]
        if not files:
            raise FileNotFoundError(f"No CV .mpr files found in {measurement_folder}")
        mps_files = list(measurement_folder.glob("*.mps"))

        for file in files:
            try:
                before_pressure, after_pressure = extract_pressures(file.name)
                modified_date = modified_date_for_source(file, mps_files)
            except (ValueError, FileNotFoundError) as error:
                print(f"Skipping {file.name}: {error}")
                continue

            data = we.read_file_safe(
                str(file), error_on_unknown_column=False, on_error="raise"
            )
            if data is None:
                continue

            try:
                currents = currents_at_target_voltages(data, TARGET_VOLTAGES)
            except ValueError as error:
                print(f"Skipping {file.name}: {error}")
                continue

            results.append(
                (
                    modified_date,
                    sample,
                    before_pressure,
                    after_pressure,
                    currents,
                    file.name,
                )
            )

    if not results:
        raise RuntimeError("No valid CV current/pressure points were extracted")

    results.sort(key=lambda row: (row[0], row[1], row[5]))

    csv_fields = [
        "Modified date",
        "Sample",
        "Filename",
        "Pressure before alebo (bar)",
        "Pressure after alebo (bar)",
        *[f"Current at {voltage:g} V (mA)" for voltage in TARGET_VOLTAGES],
    ]
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=csv_fields)
        writer.writeheader()
        for date, sample, before, after, currents, filename in results:
            writer.writerow(
                {
                    "Modified date": date.strftime("%Y-%m-%d %H:%M:%S"),
                    "Sample": sample,
                    "Filename": filename,
                    "Pressure before alebo (bar)": before,
                    "Pressure after alebo (bar)": after,
                    **{
                        f"Current at {voltage:g} V (mA)": currents[voltage]
                        for voltage in TARGET_VOLTAGES
                    },
                }
            )
    print(f"Saved data: {CSV_OUTPUT.resolve()}")

    plot_voltage = 2.0
    fig, axis = plt.subplots(figsize=(7, 5))
    colors = {
        sample: color
        for sample, color in zip(
            MEASUREMENT_FOLDERS,
            ("tab:blue", "tab:orange", "tab:green", "tab:red"),
        )
    }
    for sample in MEASUREMENT_FOLDERS:
        sample_results = [row for row in results if row[1] == sample]
        if not sample_results:
            continue
        axis.plot(
            [row[3] for row in sample_results],
            [row[4][plot_voltage] for row in sample_results],
            "o-",
            color=colors[sample],
            label=sample,
        )

    scatter = axis.scatter(
        [row[3] for row in results],
        [row[4][plot_voltage] for row in results],
        c=[row[2] for row in results],
        cmap="viridis",
        alpha=0,
    )
    axis.set_xlabel("Pressure after alebo (bar)")
    axis.set_ylabel(f"Current at {plot_voltage:g} V (mA)")
    axis.set_title("CV current at 2.0 V versus pressure")
    axis.grid(False)
    axis.legend(title="Sample")
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Pressure before alebo (bar)")
    fig.tight_layout()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT.resolve()}")
    for date, sample, before, after, value, filename in results:
        print(
            f"{date:%Y-%m-%d %H:%M:%S} | {sample} | {filename}: "
            f"{before:g} -> {after:g} bar, "
            + ", ".join(
                f"{voltage:g} V = {value[voltage]:g} mA"
                for voltage in TARGET_VOLTAGES
            )
        )
    plt.show()


if __name__ == "__main__":
    main()
