"""
Heatmap of correlated CV+XPS data (e.g. 12_CV_XPS_correlated.csv, produced by
CVXPS-correlate.py).

x-axis : binding energy (eV)
y-axis : electrode potential Ewe (V), folded so that the up-sweep and the
         down-sweep are separated: rising potential occupies y = 0.6 -> 1.8,
         falling potential continues above it as y = 1.8 -> 3.0 (labelled
         1.8 -> 0.6 again). Tick labels always show the actual potential.
fill   : counts/s, averaged over all points in the same (y bin, energy) cell

Y bins are checked, not guessed: all "nice" widths (1/2/2.5/5 x 10^k) whose
bin count lies in [MIN_BINS, MAX_BINS] are evaluated (fill ratio, points per
cell), and the finest width that still fills >= FILL_TARGET of the cells is
chosen. In addition, heatmaps with the fixed widths FIXED_BIN_WIDTHS are
always produced, and two more with the auto (optimal) bins where each row is
normalized either to the value of the highest binding energy column or to the
lowest one (if that cell is empty, the neighboring energy cells of the row
are averaged instead).

plus a spectrum plot where counts/s are averaged over all voltage bins
(<csv_basename>_spectrum_voltage_avg.png).

Usage: python CVXPS-heatmap-correlated.py [correlated_csv] [out_png]
Output: <csv_basename>_heatmap.png (auto bins),
        <csv_basename>_heatmap_<width>V.png (each fixed width),
        <csv_basename>_heatmap_norm_maxBE.png / _norm_minBE.png
        (auto bins, row-normalized to highest / lowest binding energy),
        <csv_basename>_spectrum_voltage_avg.png (voltage-averaged spectrum),
        all next to the CSV.
"""

import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Optional CLI overrides: python CVXPS-heatmap-correlated.py [csv] [png]
CORRELATED_CSV = (sys.argv[1] if len(sys.argv) > 1 else
                  r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV_XPS_correlated.csv")
PLOT_FILE = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(CORRELATED_CSV),
    os.path.splitext(os.path.basename(CORRELATED_CSV))[0] + "_heatmap.png"))

V_LO = 0.6   # CV lower vertex (E2)
V_HI = 1.8   # CV upper vertex (E1)
MIN_BINS = 20
MAX_BINS = 120
FILL_TARGET = 0.95
FIXED_BIN_WIDTHS = (0.025, 0.05)  # V, heatmaps always made with these resolutions
NBR_COLS = 3  # neighboring energy columns averaged when the reference cell is empty
NICE_STEPS = (1.0, 2.0, 2.5, 5.0)  # x 10^k
DV_THRESHOLD = 1e-3  # V per sample below which the sweep direction is "flat"


def load_correlated(path):
    t, volt, energy, counts = [], [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_t, i_v, i_e, i_c = (header.index(c) for c in
                              ("cv_time_s", "voltage_Ewe_V",
                               "binding_energy_eV", "counts_s"))
        for row in reader:
            t.append(float(row[i_t]))
            volt.append(float(row[i_v]))
            energy.append(float(row[i_e]))
            counts.append(float(row[i_c]))
    return (np.asarray(t), np.asarray(volt), np.asarray(energy),
            np.asarray(counts))


def folded_y(t, volt):
    """Fold potential into a monotonic-in-sweep y axis: up-sweep -> y = V,
    down-sweep -> y = 2*V_HI - V (so y runs 0.6..1.8 up, then 1.8..3.0 down).
    Returns y and the per-row direction (+1 up, -1 down)."""
    ut, first = np.unique(t, return_index=True)
    uv = volt[first]
    grad = np.gradient(uv, ut)
    direction_u = np.where(grad > DV_THRESHOLD, 1,
                           np.where(grad < -DV_THRESHOLD, -1, 1))
    direction = direction_u[np.searchsorted(ut, t)]
    y = np.where(direction > 0, volt, 2 * V_HI - volt)
    return y, direction


def bin_cells(y, energy, counts, edges):
    e_idx = np.searchsorted(np.unique(energy), energy)
    y_idx = np.clip(np.searchsorted(edges, y, side="right") - 1,
                    0, len(edges) - 2)
    n_y, n_e = len(edges) - 1, len(np.unique(energy))
    sums = np.zeros((n_y, n_e))
    nums = np.zeros((n_y, n_e))
    np.add.at(sums, (y_idx, e_idx), counts)
    np.add.at(nums, (y_idx, e_idx), 1.0)
    filled = nums > 0
    mean = np.full((n_y, n_e), np.nan)
    mean[filled] = sums[filled] / nums[filled]
    return mean, nums


def choose_y_bins(y, energy, counts):
    """Evaluate nice bin widths; pick the finest with fill >= FILL_TARGET."""
    lo, hi = y.min(), y.max()
    candidates = sorted({s * 10.0**k for k in range(-3, 1) for s in NICE_STEPS})
    table = []
    for w in candidates:
        edges = np.arange(np.floor(lo / w) * w, hi + w, w)
        n_bins = len(edges) - 1
        if not (MIN_BINS <= n_bins <= MAX_BINS):
            continue
        _, nums = bin_cells(y, energy, counts, edges)
        filled = nums > 0
        fill = filled.sum() / nums.size
        avg = nums[filled].mean() if filled.any() else 0.0
        table.append((w, n_bins, fill, avg, int(nums.max())))
    print(" y-bin check (width V | bins | fill | pts/filled cell | max):")
    for w, n_bins, fill, avg, mx in table:
        print(f"   {w:g} | {n_bins} | {fill:.0%} | {avg:.1f} | {mx}")
    ok = [r for r in table if r[2] >= FILL_TARGET]
    chosen = min(ok, key=lambda r: r[0]) if ok else max(table, key=lambda r: r[2])
    w = chosen[0]
    edges = np.arange(np.floor(lo / w) * w, hi + w, w)
    print(f"chosen: width {w:g} V -> {len(edges) - 1} bins "
          f"(fill {chosen[2]:.0%}, {chosen[3]:.1f} pts/filled cell)")
    return edges


def energy_edges(energies):
    if energies.size == 1:
        return np.array([energies[0] - 0.5, energies[0] + 0.5])
    mid = (energies[:-1] + energies[1:]) / 2
    return np.concatenate(([energies[0] - (mid[0] - energies[0])], mid,
                           [energies[-1] + (energies[-1] - mid[-1])]))


def save_fig(fig, out_file):
    for attempt in range(3):  # OneDrive occasionally locks the file briefly
        try:
            fig.savefig(out_file, dpi=150)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)
    plt.close(fig)
    print(f"Saved plot to {out_file}")


def row_reference(matrix, side):
    """Per-row reference = value of the highest ('max', last column) or lowest
    ('min', first column) binding energy column; if that cell is empty,
    average the neighboring energy columns; if those are all empty too,
    fall back to the row mean."""
    col = -1 if side == "max" else 0
    nbr = (slice(None, NBR_COLS) if side == "min" else slice(-NBR_COLS, None))
    ref = matrix[:, col].astype(float).copy()
    n_fallback = 0
    for i in range(matrix.shape[0]):
        if not np.isfinite(ref[i]):
            nbrs = matrix[i, nbr]
            fin = nbrs[np.isfinite(nbrs)]
            if fin.size:
                ref[i] = fin.mean()
            else:
                row = matrix[i][np.isfinite(matrix[i])]
                ref[i] = row.mean() if row.size else np.nan
            n_fallback += 1
    return ref, n_fallback


def plot_heatmap(y, energy, counts, y_edges, out_file, note,
                 normalize=None):
    energies = np.unique(energy)
    matrix, nums = bin_cells(y, energy, counts, y_edges)
    filled = nums > 0
    print(f"[{note}] bins={len(y_edges) - 1}, width={np.diff(y_edges).mean():g} V, "
          f"cells filled {int(filled.sum())}/{matrix.size} ({filled.sum() / matrix.size:.0%}), "
          f"{nums[filled].mean():.1f} pts/filled cell (averaged)")
    cbar_label = "counts / s"
    if normalize in ("min", "max"):
        ref, n_fallback = row_reference(matrix, normalize)
        ref_e = energies[0] if normalize == "min" else energies[-1]
        side = "lowest" if normalize == "min" else "highest"
        print(f"[{note}] row reference = value at {ref_e:g} eV "
              f"({side} binding energy); {n_fallback} rows used neighboring-column average")
        matrix = np.divide(matrix, ref[:, None],
                           out=np.full_like(matrix, np.nan),
                           where=np.isfinite(ref[:, None]) & (ref[:, None] != 0))
        cbar_label = f"counts/s normalized to {ref_e:g} eV"

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("lightgray")
    fig, ax = plt.subplots(figsize=(9, 7))
    mesh = ax.pcolormesh(energy_edges(energies), y_edges,
                         np.ma.masked_invalid(matrix), shading="flat", cmap=cmap)
    ax.axhline(V_HI, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(V_LO, 2 * V_HI - V_LO)
    ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2 * V_HI - v:.1f}"
                        for v in ticks])
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("Potential Ewe (V)  (up-sweep, then down-sweep)")
    ax.set_title(f"Ir 4f counts/s vs binding energy and potential ({note})")
    ax.invert_xaxis()  # XPS convention: binding energy decreases to the right
    ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="white")
    ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="white")
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    fig.tight_layout()
    save_fig(fig, out_file)


def plot_spectrum(energies, matrix, out_file):
    """counts/s vs binding energy, averaged over all voltage bins (rows)."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
    n = np.sum(np.isfinite(matrix), axis=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(energies, mean, "-o", ms=4, color="black", label="mean over voltage bins")
    ax.fill_between(energies, mean - std, mean + std, alpha=0.3,
                    label="±1 std across voltage bins")
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel("counts / s")
    ax.set_title("Ir 4f spectrum averaged over potential "
                 f"({int(n.min())}-{int(n.max())} voltage bins per energy)")
    ax.invert_xaxis()  # XPS convention: binding energy decreases to the right
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_file)


def main():
    t, volt, energy, counts = load_correlated(CORRELATED_CSV)
    y, direction = folded_y(t, volt)
    print(f"Loaded {counts.size} points from {CORRELATED_CSV}")
    print(f"Potential {volt.min():.3f}..{volt.max():.3f} V; "
          f"up-sweep points: {int((direction > 0).sum())}, "
          f"down-sweep: {int((direction < 0).sum())}")
    print(f"Folded y axis: {y.min():.3f} .. {y.max():.3f} "
          f"(labels {V_LO}->{V_HI} up, {V_HI}->{V_LO} down)")
    print(f"X: {np.unique(energy).size} binding energies "
          f"({np.unique(energy).min():g}..{np.unique(energy).max():g} eV)")

    # 1) automatic bin choice
    y_edges = choose_y_bins(y, energy, counts)
    plot_heatmap(y, energy, counts, y_edges, PLOT_FILE,
                 f"auto {np.diff(y_edges).mean():g} V bins, cell-averaged")

    # 2) fixed-resolution heatmaps
    for w in FIXED_BIN_WIDTHS:
        fixed_edges = np.arange(np.floor(y.min() / w) * w, y.max() + w, w)
        fixed_png = PLOT_FILE.replace(".png", f"_{w:g}V.png")
        plot_heatmap(y, energy, counts, fixed_edges, fixed_png,
                     f"fixed {w:g} V bins, cell-averaged")

    # 3) optimal (auto) bins, rows normalized to highest / lowest binding energy
    for side, tag in (("max", "maxBE"), ("min", "minBE")):
        norm_png = PLOT_FILE.replace(".png", f"_norm_{tag}.png")
        plot_heatmap(y, energy, counts, y_edges, norm_png,
                     f"auto {np.diff(y_edges).mean():g} V bins, row-normalized to {tag}",
                     normalize=side)

    # 4) spectrum averaged over voltage bins (auto binning matrix, mean over rows)
    matrix, _ = bin_cells(y, energy, counts, y_edges)
    plot_spectrum(np.unique(energy), matrix,
                  PLOT_FILE.replace(".png", "_spectrum_voltage_avg.png"))


if __name__ == "__main__":
    main()
