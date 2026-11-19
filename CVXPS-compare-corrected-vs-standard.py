"""
Compare the corrected averaged channel spectrum with the standard-mode reference.

Loads:
  - averaged corrected spectrum CSV (from CVXPS-xy-channel-average.py)
  - standard .xy file (FixedAnalyzerTransmission mode, channel-averaged)

Plots both on the same graph with literature BE lines.

Usage: python CVXPS-compare-corrected-vs-standard.py [avg_csv] [standard_xy] [out_png]
"""

import csv
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AVG_CSV = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_averaged_corrected_spectrum.csv")
STD_XY = (sys.argv[2] if len(sys.argv) > 2 else
          r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_standard.xy")
OUT_PNG = (sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(AVG_CSV),
    "Au_ref_corrected_vs_standard.png"))

BE_LIT_PEAK1 = 84.0   # Au 4f 7/2
BE_LIT_PEAK2 = 87.67  # Au 4f 5/2


def load_averaged(csv_file):
    energies, mean, std = [], [], []
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        i_e = header.index("binding_energy_eV")
        i_m = header.index("counts_s_mean")
        i_s = header.index("counts_s_std")
        for row in r:
            energies.append(float(row[i_e]))
            mean.append(float(row[i_m]))
            std.append(float(row[i_s]))
    return np.array(energies), np.array(mean), np.array(std)


def parse_xy_standard(filename):
    """Parse standard .xy, average over all channels/scans."""
    energies = []
    seen_e = set()
    sums = {}
    in_scan = False
    col_energy_first = False
    region_name = ""

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# ColumnLabels:") and "energy" in line.lower():
                col_energy_first = True
                continue
            m = re.match(
                r"#\s*Cycle:\s*\d+\s*,\s*Curve:\s*\d+\s*,\s*Scan:\s*(\d+)"
                r"(?:\s*,\s*Channel:\s*(\d+))?\s*$", line)
            if m:
                in_scan = m.group(2) is not None
                continue
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                region_name = m.group(1).strip()
                in_scan = False
                continue
            m = re.match(r"^(\d+(?:\.\d+)?)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line)
            if m and in_scan and col_energy_first:
                e = float(m.group(1))
                counts = float(m.group(2))
                if e not in seen_e:
                    seen_e.add(e); energies.append(e)
                if e in sums:
                    sums[e][0] += counts; sums[e][1] += 1
                else:
                    sums[e] = [counts, 1]

    energies.sort()
    mean_counts = np.array([sums[e][0] / sums[e][1] for e in energies])
    return np.array(energies), mean_counts, region_name


def main():
    e_avg, avg, std = load_averaged(AVG_CSV)
    e_std, std_counts, region = parse_xy_standard(STD_XY)
    print(f"Averaged corrected: {len(e_avg)} points, {e_avg[0]:.1f}..{e_avg[-1]:.1f} eV, "
          f"max {avg.max():.2f} counts/s")
    print(f"Standard reference: {len(e_std)} points, {e_std[0]:.1f}..{e_std[-1]:.1f} eV, "
          f"max {std_counts.max():.2f} counts/s, region={region}")

    fig, ax = plt.subplots(figsize=(12, 7))

    # standard reference (full range, thick gray)
    ax.plot(e_std, std_counts, "-", lw=2.5, color="gray", alpha=0.8,
            label="standard mode (Au_ref_standard.xy, channel-averaged)")

    # corrected averaged channel spectrum with ±1 std
    ax.plot(e_avg, avg, "-", lw=2.5, color="blue",
            label="corrected averaged channels (Au_ref_Fixed_energy, 131 ch)")
    ax.fill_between(e_avg, avg - std, avg + std, alpha=0.2, color="blue",
                    label="±1 std across channels")

    # literature lines
    ax.axvline(BE_LIT_PEAK1, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Au 4f 7/2 = {BE_LIT_PEAK1} eV")
    ax.axvline(BE_LIT_PEAK2, color="orange", ls="--", lw=1.5, alpha=0.7,
               label=f"Au 4f 5/2 = {BE_LIT_PEAK2} eV")

    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("counts/s (time-averaged)")
    ax.set_title(f"{region}: corrected channel-averaged spectrum vs standard-mode reference")
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")

    # Also save a zoomed version around the Au 4f peaks
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(e_std, std_counts, "-", lw=2.5, color="gray", alpha=0.8,
            label="standard mode reference")
    ax.plot(e_avg, avg, "-", lw=2.5, color="blue",
            label="corrected averaged channels")
    ax.fill_between(e_avg, avg - std, avg + std, alpha=0.2, color="blue")
    ax.axvline(BE_LIT_PEAK1, color="red", ls="--", lw=1.5, alpha=0.7)
    ax.axvline(BE_LIT_PEAK2, color="orange", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlim(89, 82)  # zoom to Au 4f region (inverted)
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("counts/s")
    ax.set_title(f"{region} Au 4f region: corrected vs standard")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f_zoom = OUT_PNG.replace(".png", "_zoom.png")
    fig.savefig(f_zoom, dpi=150)
    plt.close(fig)
    print(f"Saved {f_zoom}")


if __name__ == "__main__":
    main()
