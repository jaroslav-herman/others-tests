"""
Plot the averaged XPS spectrum (mean counts/s over all scans) versus binding
energy from a SpecsLab Prodigy fixed-energy .xy series.

The .xy file contains one '# Region: <name> <binding energy eV>' block per
binding energy; inside each block every '# Cycle: ..., Scan: N' section holds
data rows 'index  counts/s  ErrorBar'. All counts/s datapoints of a region
are averaged (error bars show the standard error of the mean) and the result
is plotted against the binding energy.

Usage: python CVXPS-average-spectrum.py [xy_file] [out_png]
Output: <xy_basename>_averaged_spectrum.png in ./CVXPS/ by default.
"""

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

XY_FILE = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_02.xy")
OUT_FILE = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "CVXPS",
    os.path.splitext(os.path.basename(XY_FILE))[0] + "_averaged_spectrum.png"))


def parse_xy(filename):
    """Return dict binding_energy_eV -> list of counts/s over all scans."""
    regions = {}
    current_be = None
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                em = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", m.group(1))
                current_be = float(em.group(2)) if em else None
                if current_be is not None:
                    regions.setdefault(current_be, [])
                continue
            m = re.match(r"^(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line)
            if m and current_be is not None:
                regions[current_be].append(float(m.group(2)))
    return regions


def main():
    regions = parse_xy(XY_FILE)
    if not regions:
        raise ValueError(f"no XPS datapoints found in {XY_FILE}")

    be = np.array(sorted(regions))
    mean = np.array([np.mean(regions[b]) for b in be])
    sem = np.array([np.std(regions[b], ddof=1) / np.sqrt(len(regions[b]))
                    if len(regions[b]) > 1 else 0.0 for b in be])
    n = np.array([len(regions[b]) for b in be])
    total = int(n.sum())

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for b in be:
        ax.scatter(np.full(len(regions[b]), b), regions[b],
                   s=6, color="0.75", zorder=1)
    ax.errorbar(be, mean, yerr=sem, fmt="o-", color="tab:blue",
                ecolor="tab:blue", elinewidth=1, capsize=3, ms=5, zorder=2,
                label=f"mean of all scans ± SEM (n = {n.min()}–{n.max()} per point)")
    ax.set_xlabel("Binding energy / eV")
    ax.set_ylabel("Intensity / counts s$^{-1}$")
    ax.set_title(f"Averaged XPS spectrum: {os.path.basename(XY_FILE)}\n"
                 f"{total} datapoints over {len(be)} binding energies")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"saved plot to {OUT_FILE}")
    print(f"{total} datapoints over {len(be)} binding energies "
          f"({be.min():g}–{be.max():g} eV)")
    for b, m_, s_, nn in zip(be, mean, sem, n):
        print(f"  {b:6.2f} eV : mean {m_:10.2f} counts/s  SEM {s_:8.2f}  (n={nn})")


if __name__ == "__main__":
    main()
