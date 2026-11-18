"""
Parse a SpecsLab Prodigy XY export (fixed-energy XPS series),
split it into regions by '# Region:' header (e.g. 'O1s 533.2'),
and save each region into its own file named after its binding energy.

Each output file contains two data columns:
    counts/s    ErrorBar
(one row per scan, 100 scans per region).

Output files: O1s_529.0.dat, O1s_529.3.dat, ... in ./regions/
"""

import os
import re

FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_5mVs_03.xy"  # path to your file
OUT_DIR = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS/CV_5mbar_Fixed_energy_5mVs_03/"                        # output folder

def parse_file(filename):
    regions = {}
    current = None
    acquisition_time = None
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            # Acquisition metadata: export the clock time, but not the date
            m = re.match(
                r"#\s*Acquisition Date:\s*\d{4}-\d{2}-\d{2}\s+"
                r"(\d{2}:\d{2}:\d{2})(?:\s+(UTC))?\s*$",
                line,
            )
            if m:
                acquisition_time = f"{m.group(1)}" + (
                    f" {m.group(2)}" if m.group(2) else ""
                )
                continue

            # Region header -> start a new region
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                current = m.group(1)
                regions[current] = {"index": [], "counts": [], "error": []}
                continue

            # Numeric data row: index  counts/s  ErrorBar
            m = re.match(r"^([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if m and current is not None:
                idx, cps, err = (float(x) for x in m.groups())
                regions[current]["index"].append(idx)
                regions[current]["counts"].append(cps)
                regions[current]["error"].append(err)
    return regions, acquisition_time


def save_regions(regions, acquisition_time, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, d in regions.items():
        # Extract 'O1s' and the energy from the region name
        m = re.match(r"(\S+)\s+([-\d.]+)\s*$", name)
        element, energy = (m.group(1), m.group(2)) if m else (name, "unknown")
        energy_label = energy.replace(".", "p")   # '533.2' -> '533p2'
        fname = f"{element}_{energy_label}.dat"
        path = os.path.join(out_dir, fname)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Region: {name}\n")
            f.write(f"# Binding Energy: {energy} eV\n")
            if acquisition_time:
                f.write(f"# Acquisition Time: {acquisition_time}\n")
            f.write(f"# N scans: {len(d['counts'])}\n")
            f.write("# counts/s\tErrorBar\n")
            for cps, err in zip(d["counts"], d["error"]):
                f.write(f"{cps}\t{err}\n")
        print(f"wrote {path}  ({len(d['counts'])} rows)")


if __name__ == "__main__":
    regions, acquisition_time = parse_file(FILE)
    print(f"Found {len(regions)} regions\n")
    save_regions(regions, acquisition_time, OUT_DIR)