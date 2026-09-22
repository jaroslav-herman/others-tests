"""Export selected columns from Bio-Logic FTacV MPR files as MPT files.

The output files are written next to their source MPR files and contain the
three requested columns: ``Ewe/V``, ``I/mA`` and ``time/s``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import wepy.basics as we


# The first path follows the folder naming used by the measurement share.  The
# second is the spelling supplied in the request, retained for portability.
INPUT_DIR_CANDIDATES = (
    Path(r"\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode_etching_series_GDE\FTacV\parameter range\50 Hz, 300 mV amp"),
)


def choose_input_dir() -> Path:
    for candidate in INPUT_DIR_CANDIDATES:
        try:
            if candidate.is_dir():
                return candidate
        except OSError:
            # A disconnected or inaccessible candidate should not prevent the
            # other supported spelling from being checked.
            continue
    raise FileNotFoundError(
        "None of the candidate input folders is accessible:\n"
        + "\n".join(f"  {path}" for path in INPUT_DIR_CANDIDATES)
    )


def find_column(data: pd.DataFrame, *names: str) -> str:
    """Find a column using exact names, then a case-insensitive match."""
    for name in names:
        if name in data.columns:
            return name
    folded = {str(column).casefold(): column for column in data.columns}
    for name in names:
        if name.casefold() in folded:
            return folded[name.casefold()]
    raise KeyError(f"Could not find any of {names}; available columns: {list(data.columns)}")


def export_file(source: Path) -> Path:
    data = we.read_file_safe(
        source,
        error_on_unknown_column=False,
        on_error="raise",
    )
    if data is None or data.empty:
        raise ValueError("file contains no data")

    ewe = find_column(data, "Ewe/V")
    current = find_column(data, "I/mA", "<I>/mA")
    time = find_column(data, "time/s", "Time/s")

    output = data[[ewe, current, time]].copy()
    output.columns = ["Ewe/V", "I/mA", "time/s"]

    destination = source.with_suffix(".mpt")
    # This is a plain tab-separated Bio-Logic-compatible text export with one
    # header row, suitable for loading with we.read_file(..., skiprows=0).
    output.to_csv(destination, sep="\t", index=False, lineterminator="\n")
    return destination


def main() -> None:
    input_dir = choose_input_dir()
    discovered = we.load_files(
        str(input_dir),
        extension=".mpr",
        natural_sort=True,
    )
    if isinstance(discovered, str):
        raise FileNotFoundError(discovered)
    sources = [Path(source) for source in discovered]
    if not sources:
        raise FileNotFoundError(f"No .mpr files found in {input_dir}")

    exported = 0
    for source in sources:
        try:
            destination = export_file(source)
        except Exception as error:
            print(f"SKIPPED {source.name}: {error}")
            continue
        exported += 1
        print(f"EXPORTED {destination}")

    if not exported:
        raise RuntimeError("No .mpt files were exported")
    print(f"Exported {exported} of {len(sources)} .mpr file(s) in {input_dir}")


if __name__ == "__main__":
    main()
