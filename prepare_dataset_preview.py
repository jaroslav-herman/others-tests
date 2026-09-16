"""Create a reviewable filename/order preview for the electrochemical dataset.

This script is intentionally preview-only: it never renames, moves, copies, or
changes the measured files.  It writes a CSV mapping and a README containing
the proposed dataset filename and a human-readable content label.

Run from any directory, for example:
    python prepare_dataset_preview.py

The metadata in DATASET_CONTEXT can be edited if the article uses a different
MEA/day/procedure description.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


DATA_ROOT = Path(__file__).resolve().parent / "Used data for dataset"
PREVIEW_DIR = DATA_ROOT / "_dataset_preview"
DATASET_DIR = DATA_ROOT / "dataset"
DATASET_PREFIX = "VZ2_062_021_UK_D"
DATASET_SUFFIX = "v2"
DATASET_CONTEXT = "MEA 7.5 µg v2 from day 5 procedure 1"

CONVENTIONAL = "conventional_PEIS__1,46V_OCV180s_open_cell"
OPEN_TOP = "TRPEIS__1,46V_OCV180s_open_cell"
N2_TOP = "TRPEIS__1,46V_OCV180s_N2_3"
H2_TOP = "TRPEIS__1,46V_OCV180s_H2"
CV_TOP = "TRPEIS_CV__1,9V_OCV180s"
CLOSED_DIR = "1,46 V different time gaps closed cell"
OPEN_DIR = "1,46 V different time gaps open cell"


def natural_key(path: Path) -> tuple:
    """Sort names naturally, so L_02 and _10 have numerical order."""
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def is_dataset_file(path: Path) -> bool:
    """Only measured MPT and CSV files belong in the dataset preview."""
    return path.is_file() and path.suffix.casefold() in {".mpt", ".csv"}


def technique_from_name(name: str) -> str:
    # Use the final technique token.  This is important because names such as
    # conventional_PEIS_* and TRPEIS_* contain PEIS as part of the procedure
    # name, while the final token identifies the actual file technique.
    matches = re.findall(r"_(PEIS|CA|OCV|CV)(?=_|\.|$)", name.upper())
    if matches:
        return matches[-1]
    return "unknown technique"


def content_label(source: Path) -> str:
    name = source.name
    technique = technique_from_name(name)
    if name.startswith(CONVENTIONAL):
        return (
            f"Electrochemical data from technique {technique} in conventional PEIS "
            "approach for PEM-WE in open cathode configuration with OCV time gap 180 s."
        )
    if name.startswith(OPEN_TOP):
        return (
            f"Electrochemical data from technique {technique} in TR-PEIS procedure "
            "for PEM-WE in open cathode configuration with OCV time gap 180 s."
        )
    if name.startswith(N2_TOP) or name.startswith(H2_TOP):
        flow = "H2" if name.startswith(H2_TOP) else "N2"
        return (
            f"Electrochemical data from technique {technique} in TR-PEIS procedure "
            f"for PEM-WE in {flow} flow with OCV time gap 180 s."
        )
    if name.startswith(CV_TOP):
        return (
            f"Electrochemical data from technique {technique} in TR-PEIS procedure "
            "with CV for AEM-WE with OCV time gap 180 s."
        )
    if source.parent.name in (CLOSED_DIR, OPEN_DIR):
        configuration = "closed" if "closed" in source.parent.name.casefold() else "open"
        gap_match = re.search(r"_(30|45|60|90|120|180)s(?:_|$)", name, re.IGNORECASE)
        gap = gap_match.group(1) if gap_match else "unknown"
        if "initial_loops" in name.casefold():
            return (
                f"Electrochemical data from technique {technique} in TR-PEIS procedure "
                f"for PEM-WE in {configuration} cathode configuration with OCV time gap {gap} s, "
                "initial loops."
            )

        frequency_match = re.search(
            r"from\s+([0-9]+(?:\.[0-9]+)?(?:kHz|Hz))\s+to\s+"
            r"([0-9]+(?:\.[0-9]+)?(?:kHz|Hz))",
            name,
            re.IGNORECASE,
        )
        if frequency_match:
            high, low = frequency_match.groups()
            return (
                f"Electrochemical data from technique {technique} in TR-PEIS procedure "
                f"for PEM-WE in {configuration} cathode configuration with OCV time gap {gap} s, "
                f"with PEIS frequency in between {high} and {low}."
            )

        return (
            f"Electrochemical data from technique {technique} in TR-PEIS procedure "
            f"for PEM-WE in {configuration} cathode configuration with OCV time gap {gap} s."
        )
    return f"Electrochemical data of {DATASET_CONTEXT}, technique {technique}."


def top_group(path: Path) -> int:
    name = path.name
    if name.startswith(CONVENTIONAL):
        return 10
    if name.startswith(OPEN_TOP):
        return 20
    if name.startswith(N2_TOP):
        return 30
    if name.startswith(H2_TOP):
        return 40
    if name.startswith(CV_TOP):
        return 70
    return 999


def ordered_files(root: Path) -> list[tuple[str, Path]]:
    """Return (order section, source path) in the requested dataset order."""
    top = [p for p in root.iterdir() if is_dataset_file(p)]
    closed = sorted((p for p in (root / CLOSED_DIR).iterdir() if is_dataset_file(p)), key=natural_key)
    opened = sorted((p for p in (root / OPEN_DIR).iterdir() if is_dataset_file(p)), key=natural_key)

    sections: list[tuple[str, list[Path]]] = [
        ("conventional_PEIS", sorted((p for p in top if p.name.startswith(CONVENTIONAL)), key=natural_key)),
        ("TRPEIS_open_cell", sorted((p for p in top if p.name.startswith(OPEN_TOP)), key=natural_key)),
        ("TRPEIS_N2", sorted((p for p in top if p.name.startswith(N2_TOP)), key=natural_key)),
        ("TRPEIS_H2", sorted((p for p in top if p.name.startswith(H2_TOP)), key=natural_key)),
        ("closed_cell_time_gaps", closed),
        ("open_cell_time_gaps", opened),
        ("TRPEIS_CV", sorted((p for p in top if p.name.startswith(CV_TOP)), key=natural_key)),
    ]

    return [(section, path) for section, paths in sections for path in paths]


def build_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    files = ordered_files(root)
    for number, (section, source) in enumerate(files, start=1):
        # Preserve the selected source extension.  CSV files must remain CSV
        # in the copied dataset; MPT files remain MPT files.
        proposed = f"{DATASET_PREFIX}_{number:04d}_{DATASET_SUFFIX}{source.suffix.lower()}"
        rows.append(
            {
                "dataset_number": str(number),
                "proposed_filename": proposed,
                "section": section,
                "source_relative_path": str(source.relative_to(root)),
                "source_extension": source.suffix,
                "technique": technique_from_name(source.name),
                "content_label": content_label(source),
            }
        )
    return rows


def write_preview(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "dataset_filename_preview.csv"
    readme_path = output_dir / "README_preview.txt"

    fields = list(rows[0]) if rows else [
        "dataset_number", "proposed_filename", "section", "source_relative_path",
        "source_extension", "technique", "content_label"
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write("DATASET FILENAME PREVIEW\n")
        handle.write("=========================\n\n")
        handle.write("This is a preview only. No measured files were renamed or modified.\n")
        handle.write(f"Source directory: {DATA_ROOT}\n")
        handle.write(f"Files in preview: {len(rows)}\n\n")
        handle.write("Requested order:\n")
        handle.write("1. conventional_PEIS_*\n")
        handle.write("2. TRPEIS__1,46V_OCV180s_open_cell_*\n")
        handle.write("3. TRPEIS__1,46V_OCV180s_N2_*\n")
        handle.write("4. TRPEIS__1,46V_OCV180s_H2_*\n")
        handle.write("5. 1,46 V different time gaps closed cell/*\n")
        handle.write("6. 1,46 V different time gaps open cell/*\n")
        handle.write("7. TRPEIS_CV__1,9V_OCV180s_*\n\n")
        handle.write("Proposed filename and content label\n")
        handle.write("-----------------------------------\n")
        for row in rows:
            handle.write(f"{row['proposed_filename']} {row['content_label']}\n")

    print(f"Preview written to: {csv_path}")
    print(f"README written to:  {readme_path}")
    print(f"Files included:     {len(rows)}")
    if rows:
        print(f"First file:         {rows[0]['proposed_filename']} <- {rows[0]['source_relative_path']}")
        print(f"Last file:          {rows[-1]['proposed_filename']} <- {rows[-1]['source_relative_path']}")


def copy_dataset(rows: list[dict[str, str]], root: Path, output_dir: Path) -> None:
    """Copy the selected files using their proposed names and write README."""
    output_dir.mkdir(exist_ok=True)
    for row in rows:
        source = root / row["source_relative_path"]
        destination = output_dir / row["proposed_filename"]
        shutil.copy2(source, destination)

    readme_path = output_dir / "README.txt"
    with readme_path.open("w", encoding="utf-8") as handle:
        handle.write("Electrochemical dataset\n")
        handle.write("=======================\n\n")
        handle.write(f"Files: {len(rows)}\n\n")
        for row in rows:
            handle.write(f"{row['proposed_filename']} {row['content_label']}\n")

    print(f"Dataset copied to:  {output_dir}")
    print(f"README written to:  {readme_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA_ROOT, help="Source dataset directory")
    parser.add_argument("--output", type=Path, default=PREVIEW_DIR, help="Preview output directory")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files into the dataset directory and write the final README",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DATASET_DIR,
        help="Destination directory used with --copy",
    )
    args = parser.parse_args()
    root = args.source.resolve()
    if not root.is_dir():
        raise SystemExit(f"Source directory does not exist: {root}")
    rows = build_rows(root)
    if not rows:
        raise SystemExit("No files found in the source directory.")
    write_preview(rows, args.output.resolve())
    if args.copy:
        copy_dataset(rows, root, args.dataset_output.resolve())


if __name__ == "__main__":
    main()
