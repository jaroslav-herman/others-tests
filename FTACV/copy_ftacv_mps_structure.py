"""Copy only .mps files while preserving the parameter-range directory structure."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SOURCE_ROOT = (
    Path(
        r"\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode"
        r"_etching_series_GDE\FTacV"
    )
    / "parameter range 2"
)
DESTINATION_ROOT = (
    Path(
        r"\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode"
        r"_etching_series_GDE\FTacV"
    )
    / "parameter range 3"
)


def copy_mps_structure(source_root: Path, destination_root: Path) -> int:
    """Copy .mps files and preserve their relative paths under source_root."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source folder is not accessible: {source_root}")

    copied = 0
    for source_file in source_root.rglob("*"):
        if not source_file.is_file() or source_file.suffix.lower() != ".mps":
            continue

        relative_path = source_file.relative_to(source_root)
        destination_file = destination_root / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        copied += 1
        print(f"Copied: {relative_path}")

    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy only .mps files from one parameter-range folder structure."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--destination", type=Path, default=DESTINATION_ROOT)
    args = parser.parse_args()

    copied = copy_mps_structure(args.source, args.destination)
    print(f"Copied {copied} .mps file(s) to: {args.destination}")


if __name__ == "__main__":
    main()
