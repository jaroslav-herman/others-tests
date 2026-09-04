"""Portable paths for the shared PEM-WE measurement folders."""

from __future__ import annotations

import os
from pathlib import Path


def year_folder_name(year: int | str) -> str:
    """Return the directory name used for a measurement year."""
    return "2024_new_naming" if str(year) == "2024" else str(year)


def find_measurements_root(year: int | str) -> Path:
    """Find the local or network measurement directory for *year*."""
    override = os.environ.get("ELECTROCHEMICAL_DATA_ROOT")
    if override:
        candidates = [Path(override)]
    else:
        year_name = year_folder_name(year)
        candidates = [
            Path(r"C:\PEM-WE_measurements") / year_name,
            Path(r"\\ELECTROLYZER\PEM-WE_measurements") / year_name,
        ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    checked = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find PEM-WE measurements for {year}. Checked:\n{checked}"
    )


def candidate_measurement_roots(
    year: int | str, sample_type: str | None = None
) -> list[Path]:
    """Return search roots selected by the sample ``Type`` from the sheet.

    AEM samples are stored below ``AEM-WE``. Known non-AEM types are stored
    directly below the year directory. If the type is blank or unavailable,
    both locations are searched, in that order.
    """
    year_root = find_measurements_root(year)
    normalized_type = (sample_type or "").strip().upper()
    aem_root = year_root / "AEM-WE"
    if normalized_type == "AEM":
        return [aem_root]
    if normalized_type:
        return [year_root]
    return [year_root, aem_root]
