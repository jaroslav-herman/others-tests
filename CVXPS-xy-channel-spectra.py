"""
Plot time-averaged XPS spectra for all individual channels from a raw .xy file,
with an optional standard-mode reference spectrum as background.

Parses a SpecsLab Prodigy .xy export with channel-split scans, averages
counts/s over all scans (time-averaged) per (channel, binding energy), and
plots all channel spectra in one graph with a vertical y-shift per channel
and a viridis color gradient.

If a standard .xy file (FixedAnalyzerTransmission mode) is given, its spectrum
averaged over all channels is plotted as a thick background line ("true" spectrum).

Usage: python CVXPS-xy-channel-spectra.py [xy_file] [out_png] [standard_xy]
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

XY_FILE = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy.xy")
OUT_PNG = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(XY_FILE),
    os.path.splitext(os.path.basename(XY_FILE))[0] + "_channel_spectra.png"))
STANDARD_XY = (sys.argv[3] if len(sys.argv) > 3 else
               r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_standard.xy")

Y_SHIFT = 0.2  # counts/s vertical offset between consecutive channels


def parse_xy_channels(filename):
    """Parse FixedEnergies or FixedAnalyzerTransmission .xy with channel-split scans.
    
    For FixedEnergies: Region line has binding energy, data rows are 'index counts/s ErrorBar'
    For FixedAnalyzerTransmission: Region line has no energy, data rows are 'energy counts/s ErrorBar'
    
    Return (channels, energies, matrix[ch_idx, e_idx] = mean counts/s, region_name)."""
    channels, energies = [], []
    seen_ch, seen_e = set(), set()
    sums = {}
    binding_energy = None
    scan_channel = None
    in_scan = False
    region_name = ""
    col_energy_first = False

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            
            # Check for column labels to detect format
            if line.startswith("# ColumnLabels:") and "energy" in line.lower():
                col_energy_first = True
                continue
            
            m = re.match(
                r"#\s*Cycle:\s*\d+\s*,\s*Curve:\s*\d+\s*,\s*Scan:\s*(\d+)"
                r"(?:\s*,\s*Channel:\s*(\d+))?\s*$", line)
            if m:
                scan_channel = int(m.group(2)) if m.group(2) else None
                in_scan = scan_channel is not None
                continue
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                current_region = m.group(1)
                em = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", current_region)
                binding_energy = float(em.group(2)) if em else None
                if em:
                    region_name = em.group(1)
                else:
                    region_name = current_region
                in_scan = False
                continue
            
            # For FixedEnergies: data is "index counts/s ErrorBar" with binding_energy from Region
            m = re.match(r"^(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line)
            if m and in_scan and binding_energy is not None and not col_energy_first:
                counts = float(m.group(2))
                ch, e = scan_channel, binding_energy
                if ch not in seen_ch:
                    seen_ch.add(ch); channels.append(ch)
                if e not in seen_e:
                    seen_e.add(e); energies.append(e)
                key = (ch, e)
                if key in sums:
                    sums[key][0] += counts; sums[key][1] += 1
                else:
                    sums[key] = [counts, 1]
            
            # For FixedAnalyzerTransmission: data is "energy counts/s ErrorBar"
            m = re.match(r"^([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line)
            if m and in_scan and scan_channel is not None and col_energy_first:
                energy = float(m.group(1))
                counts = float(m.group(2))
                ch = scan_channel
                if ch not in seen_ch:
                    seen_ch.add(ch); channels.append(ch)
                if energy not in seen_e:
                    seen_e.add(energy); energies.append(energy)
                key = (ch, energy)
                if key in sums:
                    sums[key][0] += counts; sums[key][1] += 1
                else:
                    sums[key] = [counts, 1]

    channels.sort(); energies.sort()
    matrix = np.full((len(channels), len(energies)), np.nan)
    for ci, ch in enumerate(channels):
        for ei, e in enumerate(energies):
            s = sums.get((ch, e))
            if s:
                matrix[ci, ei] = s[0] / s[1]
    return channels, np.array(energies), matrix, region_name


def parse_xy_standard(filename):
    """Parse a FixedAnalyzerTransmission .xy: data rows are 'energy counts/s ErrorBar',
    region name has no energy suffix. Average over all channels and scans
    (typical single-channel spectrum = 'true' shape reference).
    Return (energies, mean_counts, region_name)."""
    energies = []
    seen_e = set()
    sums = {}
    scan_channel = None
    in_scan = False
    region_name = ""
    col_energy_first = False

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
                scan_channel = int(m.group(2)) if m.group(2) else None
                in_scan = scan_channel is not None
                continue
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                region_name = m.group(1).strip()
                in_scan = False
                continue
            # data row: 'energy counts/s ErrorBar' (energy-first format)
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
    channels, energies, matrix, region = parse_xy_channels(XY_FILE)
    print(f"Loaded {matrix.size} cells: {len(channels)} channels x {len(energies)} energies")
    print(f"Region: {region}, channels {channels[0]}..{channels[-1]}, "
          f"BE {energies[0]:g}..{energies[-1]:g} eV")

    std_energies, std_counts, std_region = None, None, None
    if STANDARD_XY and os.path.exists(STANDARD_XY):
        std_energies, std_counts, std_region = parse_xy_standard(STANDARD_XY)
        print(f"Standard: {std_region}, {len(std_energies)} energies "
              f"{std_energies[0]:g}..{std_energies[-1]:g} eV, "
              f"counts {std_counts.min():.2f}..{std_counts.max():.2f}")

    norm = Normalize(vmin=channels[0], vmax=channels[-1])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(12, 10))

    # background: standard "true" spectrum as thick dark line at y=0
    if std_energies is not None:
        ax.plot(std_energies, std_counts, "-", lw=3.0, color="black", alpha=0.7,
                label=f"standard mode ({std_region})", zorder=10)

    for ci, ch in enumerate(channels):
        y = matrix[ci] + ci * Y_SHIFT
        color = cmap(norm(ch))
        ax.plot(energies, y, "-", lw=1.0, color=color, alpha=0.85, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Channel")

    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel(f"counts/s (time-averaged) + {Y_SHIFT:g} x channel index")
    title = f"{region} time-averaged spectra per channel (y-shift {Y_SHIFT:g}/ch)"
    if std_energies is not None:
        title += f"\nblack = standard mode {std_region} (assumed true)"
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    if std_energies is not None:
        ax.legend(loc="upper right")

    for ci in range(0, len(channels), max(1, len(channels) // 10)):
        ax.annotate(f"ch{channels[ci]}", xy=(energies[-1], matrix[ci, -1] + ci * Y_SHIFT),
                    xytext=(5, 0), textcoords="offset points", fontsize=7,
                    color=cmap(norm(channels[ci])), va="center")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {OUT_PNG}")


if __name__ == "__main__":
    main()
