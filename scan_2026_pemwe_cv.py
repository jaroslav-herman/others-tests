"""Find and plot CV measurements from all 2026 PEM-WE samples.

Matching files must contain ``Procedure``, ``Day`` and ``CV`` in the file
name, without regard to case.  MPR files are preferred for each sample.  If a
sample has no matching MPR files, matching MPT files are used instead.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import wepy.basics as we
from measurement_paths import find_measurements_root


YEAR = 2026
REQUIRED_TOKENS = ("procedure", "day", "cv")
STATIONS = ("VIII", "VII", "VI", "V", "IV", "III", "II", "I")


def station_from_folder(folder_name: str) -> str:
    """Extract the station token from names such as ``439_III_...``."""
    pattern = rf"^(?:\d+[_ -])?({'|'.join(STATIONS)})(?:[_ -]|$)"
    match = re.match(pattern, folder_name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "unknown"


def matching_measurement_files(sample_folder: Path) -> list[Path]:
    """Return matching files, preferring MPR over MPT for this sample."""
    candidates = sorted(
        (
            path
            for path in sample_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".mpr", ".mpt"}
            and all(token in path.name.lower() for token in REQUIRED_TOKENS)
        ),
        key=lambda path: str(path).lower(),
    )
    mpr = [path for path in candidates if path.suffix.lower() == ".mpr"]
    return mpr or [path for path in candidates if path.suffix.lower() == ".mpt"]


def sample_folders(year_root: Path) -> list[Path]:
    """Return top-level sample folders in the year and optional AEM-WE roots."""
    roots = [year_root]
    aem_root = year_root / "AEM-WE"
    if aem_root.is_dir():
        roots.append(aem_root)

    folders: list[Path] = []
    for root in roots:
        folders.extend(path for path in root.iterdir() if path.is_dir())
    return sorted(set(folders), key=lambda path: str(path).lower())


def pick_column(data: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in data.columns:
            return name
    raise KeyError(f"None of {names!r} found; columns are {list(data.columns)!r}")


def plot_sample(sample_folder: Path, files: list[Path], output_root: Path) -> Path:
    colors = we.get_colors(len(files))
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    plotted = 0

    for file, color in zip(files, colors):
        data = we.read_file_safe(str(file), warn=True)
        if data is None or data.empty:
            continue
        try:
            ewe_col = pick_column(data, ("Ewe/V", "<Ewe>/V", "Voltage/V"))
            current_col = pick_column(data, ("<I>/mA", "I/mA", "Current/mA"))
        except KeyError as error:
            print(f"Skipping {file}: {error}")
            continue

        ece_col = next(
            (name for name in ("Ece/V", "<Ece>/V") if name in data.columns),
            None,
        )
        source_columns = [ewe_col, current_col]
        if ece_col is not None:
            source_columns.append(ece_col)
        numeric = data[source_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna()
        if ece_col is not None:
            numeric["voltage"] = numeric[ewe_col] - numeric[ece_col]
        else:
            numeric["voltage"] = numeric[ewe_col]
        numeric = numeric[numeric["voltage"] <= 1.4]
        if numeric.empty:
            continue

        cycle_col = next((name for name in ("cycle number", "cycle", "loop") if name in data.columns), None)
        if cycle_col is None:
            groups = [(None, numeric)]
        else:
            numeric[cycle_col] = pd.to_numeric(data.loc[numeric.index, cycle_col], errors="coerce")
            groups = [(cycle, group) for cycle, group in numeric.groupby(cycle_col, dropna=False)]

        for cycle, group in groups:
            label = file.stem if cycle is None else f"{file.stem}, cycle {cycle:g}"
            ax.plot(group["voltage"], group[current_col], color=color, linewidth=1.1, label=label)
            plotted += 1

    if not plotted:
        plt.close(fig)
        raise RuntimeError("No readable CV curves with voltage <= 1.4 V")

    station = station_from_folder(sample_folder.name)
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_xlim(right=1.4)
    ax.set_title(f"{sample_folder.name} — station {station} — CV")
    ax.grid(False)
    if plotted <= 12:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()

    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{sample_folder.name}_CV.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results") / "cv_2026")
    args = parser.parse_args()

    try:
        year_root = find_measurements_root(YEAR)
    except FileNotFoundError:
        # Some Windows installations expose the share as PEM-WE\_measurements
        # rather than the PEM-WE_measurements share used by older workflows.
        alternate_root = Path(r"\\ELECTROLYZER\PEM-WE\_measurements") / str(YEAR)
        if not alternate_root.is_dir():
            raise FileNotFoundError(
                "Could not find the 2026 PEM-WE measurement root. Checked:\n"
                f"- {alternate_root}\n"
                "- the standard roots handled by measurement_paths.py"
            ) from None
        year_root = alternate_root
    rows: list[dict[str, str | int]] = []
    for folder in sample_folders(year_root):
        files = matching_measurement_files(folder)
        if not files:
            continue
        row: dict[str, str | int] = {
            "sample_folder": folder.name,
            "station": station_from_folder(folder.name),
            "n_files": len(files),
            "files": "; ".join(str(file) for file in files),
        }
        try:
            output = plot_sample(folder, files, args.output)
            row["plot"] = str(output)
            print(f"{folder.name}\tstation {row['station']}\t{len(files)} file(s)\t{output}")
        except RuntimeError as error:
            row["plot"] = f"ERROR: {error}"
            print(f"{folder.name}\tstation {row['station']}\t{error}")
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "cv_samples_2026.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_folder", "station", "n_files", "plot", "files"))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} sample(s) with matching CV file(s).")
    print(f"Report: {report.resolve()}")


if __name__ == "__main__":
    main()
