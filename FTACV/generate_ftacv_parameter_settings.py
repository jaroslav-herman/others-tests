"""Generate EC-Lab FTACV settings files for frequency/amplitude sweeps.

The input is an existing, working EC-Lab ``.mps`` file.  Its first AC
Voltammetry technique is used as the template, so device-specific settings
such as electrode limits, bandwidth, current range, and cycle count are kept.
Each generated file contains one technique for every scan rate, while each
frequency/amplitude pair gets its own folder and settings file.
"""

from __future__ import annotations

import argparse
import re
from itertools import product
from pathlib import Path


DEFAULT_SCAN_RATES = (30, 50, 70, 100, 150)  # mV/s
DEFAULT_FREQUENCIES = (5,10,20,30,40, 50)  # Hz
DEFAULT_AMPLITUDES = ( 100, 150, 200, 250, 300)  # mV
INITIAL_POTENTIAL_V = 0.400
FINAL_POTENTIAL_V = 0.400
DEFAULT_TEMPLATE = Path(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\FTACV\100 mV amp, different rates, 50 Hz\FTACV_different_rates.mps"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\FTACV\parameter range 2"
)


def read_text(path: Path) -> str:
    """Read an EC-Lab settings file using a sensible encoding fallback."""
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeDecodeError("unknown", raw, 0, len(raw), "Could not decode .mps file")


def first_technique_template(text: str) -> tuple[str, str]:
    """Return the file header and the first technique block."""
    starts = list(re.finditer(r"(?m)^Technique\s*:\s*\d+\s*$", text))
    if not starts:
        raise ValueError("The template does not contain an EC-Lab Technique block")
    first = starts[0]
    end = starts[1].start() if len(starts) > 1 else len(text)
    header = text[: first.start()]
    technique = text[first.start() : end]
    return header, technique


def replace_setting_line(block: str, label: str, value: str) -> str:
    """Replace the value on a labelled EC-Lab line, preserving its layout."""
    pattern = re.compile(rf"(?m)^(\s*{re.escape(label)}\s+)[^\r\n]*(\r?\n|$)")
    block, count = pattern.subn(lambda m: f"{m.group(1)}{value}{m.group(2)}", block, count=1)
    if count != 1:
        raise ValueError(f"Could not find '{label}' in the first technique")
    return block


def make_settings(template: str, frequency: int, amplitude: int, filename: Path) -> str:
    """Create a file containing all scan-rate techniques for one pair."""
    header, technique_template = first_technique_template(template)

    header = re.sub(
        r"(?m)^(Number of linked techniques\s*:\s*)\d+",
        rf"\g<1>{len(DEFAULT_SCAN_RATES)}",
        header,
        count=1,
    )
    header = re.sub(
        r"(?m)^Filename\s*:\s*[^\r\n]*",
        lambda _match: f"Filename : {filename}",
        header,
        count=1,
    )
    techniques = []
    for technique_number, scan_rate in enumerate(DEFAULT_SCAN_RATES, start=1):
        technique = re.sub(
            r"(?m)^Technique\s*:\s*\d+",
            f"Technique : {technique_number}",
            technique_template,
            count=1,
        )
        technique = replace_setting_line(technique, "Ei (V)", f"{INITIAL_POTENTIAL_V:.3f}")
        technique = replace_setting_line(technique, "dE/dt", f"{scan_rate:.3f}")
        technique = replace_setting_line(technique, "E2 (V)", f"{FINAL_POTENTIAL_V:.3f}")
        technique = replace_setting_line(technique, "fs", f"{frequency:.3f}")
        technique = replace_setting_line(technique, "A (mV)", f"{amplitude:.3f}")
        techniques.append(technique)
    return header + "".join(techniques)


def folder_name(frequency: int, amplitude: int) -> str:
    return f"{frequency:g} Hz, {amplitude:g} mV amp"


def generate(template_path: Path, output_root: Path) -> int:
    template = read_text(template_path)
    output_root.mkdir(parents=True, exist_ok=True)
    count = 0

    for frequency, amplitude in product(DEFAULT_FREQUENCIES, DEFAULT_AMPLITUDES):
        folder = output_root / folder_name(frequency, amplitude)
        filename = folder / "FTACV.mps"
        folder.mkdir(parents=True, exist_ok=True)
        filename.write_text(
            make_settings(template, frequency, amplitude, filename.resolve()),
            encoding="utf-8",
            newline="",
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "template",
        type=Path,
        nargs="?",
        default=DEFAULT_TEMPLATE,
        help="Working FTACV .mps template",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_ROOT,
        help="Folder in which parameter folders are created",
    )
    args = parser.parse_args()

    if not args.template.is_file():
        parser.error(f"Template file does not exist: {args.template}")
    count = generate(args.template, args.output_root)
    print(f"Created {count} settings files in {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
