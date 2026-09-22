"""List CV .mpr files from oldest to newest by their modified time."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import re


DEFAULT_FOLDER = Path(
    r'\\ELECTROLYZER\PEM-WE_measurements\2026\432_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_Pressureincrease-0-5bar'
)
OUTPUT_FILENAME = "cv_mpr_modified_table.csv"

PRESSURE_PATTERN = re.compile(
    r"_(?P<first>[+-]?\d+(?:[.,]\d+)?)alebo"
    r"(?P<second>[+-]?\d+(?:[.,]\d+)?)_bar",
    re.IGNORECASE,
)


def extract_numbers(filename: str) -> tuple[str, str]:
    """Return the two numbers around 'alebo', or NaN when unavailable."""
    match = PRESSURE_PATTERN.search(filename)
    if match is None:
        return "NaN", "NaN"

    return (
        match.group("first").replace(",", "."),
        match.group("second").replace(",", "."),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List CV .mpr files sorted by modified time, oldest first."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=DEFAULT_FOLDER,
        help="Folder containing the .mpr files (defaults to the requested folder).",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"Folder not found or inaccessible: {args.folder}")

    files = sorted(
        (
            path
            for path in args.folder.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".mps"
            and "bar" in path.name.lower()
        ),
        key=lambda path: path.stat().st_mtime,
    )

    output_path = args.folder / OUTPUT_FILENAME
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        writer.writerow(
            ["Modified date", "Filename", "Number before alebo", "Number after alebo"]
        )

        for path in files:
            modified = datetime.fromtimestamp(path.stat().st_mtime)
            first, second = extract_numbers(path.name)
            writer.writerow(
                [
                    modified.strftime("%Y-%m-%d %H:%M:%S"),
                    path.name,
                    first,
                    second,
                ]
            )

    print(f"Exported {len(files)} file(s) to: {output_path}")


if __name__ == "__main__":
    main()
