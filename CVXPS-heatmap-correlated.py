"""
Heatmap of the correlated CV+XPS data (11_CV_XPS_correlated.csv, produced by
CVXPS-correlate.py).

x-axis : binding energy (eV)
y-axis : electrode potential Ewe (V) -- the CV swept from the lower to the
         upper potential and back, so the y-axis covers the full excursion
fill   : counts/s, averaged over all points falling into the same
         (voltage bin, binding energy) cell

Voltage bins on the y-axis are chosen automatically: the target number of
bins follows the square-root rule (clamped to [MIN_BINS, MAX_BINS]), and the
bin width is rounded to a "nice" value (1/2/2.5/5 x 10^k V).

Output: 11_CV_XPS_heatmap.png next to the CSV.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CORRELATED_CSV = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\11_CV_XPS_correlated.csv"
PLOT_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\11_CV_XPS_heatmap.png"

MIN_BINS = 20
MAX_BINS = 120
NICE_STEPS = (1.0, 2.0, 2.5, 5.0)  # x 10^k


def nice_bin_width(raw_width):
    """Round a raw bin width to the closest nice value (1/2/2.5/5 x 10^k)."""
    exponent = np.floor(np.log10(raw_width))
    best, best_dist = None, np.inf
    for k in (exponent - 1, exponent, exponent + 1):
        for step in NICE_STEPS:
            w = step * 10.0**k
            dist = abs(np.log10(w / raw_width))
            if dist < best_dist:
                best, best_dist = w, dist
    return best


def choose_voltage_bins(voltage):
    """Pick bin width via square-root rule + nice rounding; return edges."""
    n = voltage.size
    target = int(np.clip(np.sqrt(n), MIN_BINS, MAX_BINS))
    raw = (voltage.max() - voltage.min()) / target
    width = nice_bin_width(max(raw, 1e-12))
    start = np.floor(voltage.min() / width) * width
    edges = np.arange(start, voltage.max() + width, width)
    return edges, width


def load_correlated(path):
    energy, volt, counts = [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_e, i_v, i_c = (header.index(c) for c in
                         ("binding_energy_eV", "voltage_Ewe_V", "counts_s"))
        for row in reader:
            energy.append(float(row[i_e]))
            volt.append(float(row[i_v]))
            counts.append(float(row[i_c]))
    return (np.asarray(energy), np.asarray(volt), np.asarray(counts))


def bin_average(energy, volt, counts, v_edges):
    """Mean counts/s per (voltage bin, energy) cell. Returns (energies,
    energy_edges, matrix) with matrix shape (n_v_bins, n_energies), NaN where
    a cell has no data."""
    energies = np.unique(energy)
    if energies.size == 1:
        e_edges = np.array([energies[0] - 0.5, energies[0] + 0.5])
    else:
        mid = (energies[:-1] + energies[1:]) / 2
        e_edges = np.concatenate(([energies[0] - (mid[0] - energies[0])], mid,
                                  [energies[-1] + (energies[-1] - mid[-1])]))
    e_idx = np.searchsorted(energies, energy)
    v_idx = np.clip(np.searchsorted(v_edges, volt, side="right") - 1,
                    0, len(v_edges) - 2)
    n_v, n_e = len(v_edges) - 1, len(energies)
    total = np.full((n_v, n_e), np.nan)
    sums = np.zeros((n_v, n_e))
    nums = np.zeros((n_v, n_e))
    np.add.at(sums, (v_idx, e_idx), counts)
    np.add.at(nums, (v_idx, e_idx), 1.0)
    filled = nums > 0
    total[filled] = sums[filled] / nums[filled]
    return energies, e_edges, total, nums


def main():
    energy, volt, counts = load_correlated(CORRELATED_CSV)
    print(f"Loaded {counts.size} correlated points from {CORRELATED_CSV}")
    print(f"Voltage range: {volt.min():.3f} .. {volt.max():.3f} V "
          f"(CV swept lower -> upper -> lower)")

    v_edges, width = choose_voltage_bins(volt)
    print(f"Y bins: width {width:g} V -> {len(v_edges) - 1} bins")

    energies, e_edges, matrix, nums = bin_average(energy, volt, counts, v_edges)
    print(f"X: {energies.size} binding energies ({energies.min():g}..{energies.max():g} eV)")
    print(f"Cells filled: {int((nums > 0).sum())}/{matrix.size}, "
          f"max points per cell: {int(nums.max())} (averaged)")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(e_edges, v_edges, np.ma.masked_invalid(matrix),
                         shading="flat", cmap=cmap)
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("Potential Ewe (V)")
    ax.set_title("Ir 4f counts/s vs binding energy and potential "
                 "(CV lower->upper->lower, cell-averaged)")
    ax.invert_xaxis()  # XPS convention: binding energy decreases to the right
    fig.colorbar(mesh, ax=ax, label="counts / s")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Saved heatmap to {PLOT_FILE}")


if __name__ == "__main__":
    main()
