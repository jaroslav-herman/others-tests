"""
Plot voltage-averaged XPS spectra for all individual channels in one graph.

For each channel, counts/s are averaged over all voltages (all rows) per
binding energy, producing one spectrum per channel. All channel spectra are
plotted together with:
  - a vertical y-shift per channel (stacked) so they don't overlap,
  - a color gradient across channels (viridis) to distinguish them.

Usage: python CVXPS-channel-spectra.py [correlated_csv] [out_png]
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

CORRELATED_CSV = (sys.argv[1] if len(sys.argv) > 1 else
                  r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\12_CV_ch0-145_XPS_correlated.csv")
OUT_PNG = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(CORRELATED_CSV),
    os.path.splitext(os.path.basename(CORRELATED_CSV))[0] + "_channel_spectra.png"))

Y_SHIFT = 50.0  # counts/s vertical offset between consecutive channels


def load_and_average(path):
    """Return (channels, energies, matrix[ch_idx, e_idx] = mean counts/s)."""
    # first pass: collect unique channels and energies in order of appearance
    channels = []
    energies = []
    seen_ch = set()
    seen_e = set()
    sums = {}   # (ch, e) -> [sum, count]
    region_name = ""

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_e = header.index("binding_energy_eV")
        i_c = header.index("counts_s")
        i_ch = header.index("channel")
        i_r = header.index("region")

        for row in reader:
            ch = int(row[i_ch])
            e = float(row[i_e])
            c = float(row[i_c])
            r = row[i_r]
            if not region_name:
                region_name = " ".join(r.split()[:-1])  # drop trailing energy
            if ch not in seen_ch:
                seen_ch.add(ch)
                channels.append(ch)
            if e not in seen_e:
                seen_e.add(e)
                energies.append(e)
            key = (ch, e)
            if key in sums:
                sums[key][0] += c
                sums[key][1] += 1
            else:
                sums[key] = [c, 1]

    channels.sort()
    energies.sort()
    matrix = np.full((len(channels), len(energies)), np.nan)
    for ci, ch in enumerate(channels):
        for ei, e in enumerate(energies):
            s = sums.get((ch, e))
            if s:
                matrix[ci, ei] = s[0] / s[1]
    return channels, np.array(energies), matrix, region_name


def main():
    channels, energies, matrix, region = load_and_average(CORRELATED_CSV)
    print(f"Loaded {matrix.size} cells: {len(channels)} channels x {len(energies)} energies")
    print(f"Region: {region}")
    print(f"Channels: {channels[0]}..{channels[-1]}")
    print(f"Binding energies: {energies[0]:g}..{energies[-1]:g} eV")

    norm = Normalize(vmin=channels[0], vmax=channels[-1])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(12, 10))
    for ci, ch in enumerate(channels):
        y = matrix[ci] + ci * Y_SHIFT
        color = cmap(norm(ch))
        ax.plot(energies, y, "-", lw=1.2, color=color, alpha=0.9)

    # colorbar for channel number
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Channel")

    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel(f"counts/s (averaged over voltage) + {Y_SHIFT:g} x channel index")
    ax.set_title(f"{region} voltage-averaged spectra per channel "
                 f"(y-shifted by {Y_SHIFT:g} counts/s per channel)")
    ax.invert_xaxis()  # XPS convention
    ax.grid(alpha=0.3)

    # annotate a few channel numbers on the right
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
