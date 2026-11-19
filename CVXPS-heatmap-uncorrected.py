"""
Generate uncorrected heatmap from correlated CV-XPS data.
Does NOT apply any energy correction - uses raw binding energies.

Usage: python CVXPS-heatmap-uncorrected.py [correlated_csv] [out_prefix] [v_lo] [v_hi]
"""

import csv
import os
import sys
import time
from bisect import bisect_left

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IN_CSV = (sys.argv[1] if len(sys.argv) > 1 else
          r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\18_CV_ch0-145_XPS_correlated.csv")
OUT_PREFIX = (sys.argv[2] if len(sys.argv) > 2 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145_Ru3d\18_CV_Ru3d_uncorrected")

V_LO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
V_HI = float(sys.argv[4]) if len(sys.argv) > 4 else 2.2

V_BIN = 0.02
E_GRID_STEP = 0.1
DV_THRESHOLD = 1e-3
Y_SHIFT = 0.5

def save_fig(fig, out_file):
    """Save with retry (OneDrive occasionally locks the file briefly)."""
    for attempt in range(5):
        try:
            fig.savefig(out_file, dpi=150)
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    # --- load CV sweep direction ---
    cv_t, cv_v = [], []
    seen = set()
    with open(IN_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        iv = header.index("voltage_Ewe_V")
        it = header.index("cv_time_s")
        for row in reader:
            tv = (float(row[it]), float(row[iv]))
            if tv not in seen:
                seen.add(tv); cv_t.append(tv[0]); cv_v.append(tv[1])
    cv_t = np.array(cv_t); cv_v = np.array(cv_v)
    o = np.argsort(cv_t); cv_t, cv_v = cv_t[o], cv_v[o]
    grad = np.gradient(cv_v, cv_t)
    direction = np.where(grad > DV_THRESHOLD, 1, np.where(grad < -DV_THRESHOLD, -1, 1))

    # --- accumulate sums (RAW energies, no correction) ---
    sums = {}
    ch_amp = {}
    raw_e_set = set()
    region_name = ""
    
    with open(IN_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        iv, ie, ic, ich, it, ir = (header.index(c) for c in
            ["voltage_Ewe_V", "binding_energy_eV", "counts_s", "channel", "cv_time_s", "region"])
        for row in reader:
            ch = int(row[ich]); e = float(row[ie]); v = float(row[iv])
            t = float(row[it]); c = float(row[ic])
            if not region_name:
                region_name = " ".join(row[ir].split()[:-1])
            raw_e_set.add(e)
            i = bisect_left(cv_t, t); i = min(max(i, 0), len(direction) - 1)
            d = direction[i]
            # folded y
            y = v if d > 0 else 2 * V_HI - v
            yb = round(round(y / V_BIN) * V_BIN, 2)
            key = (ch, yb, e)
            if key in sums:
                sums[key][0] += c; sums[key][1] += 1
            else:
                sums[key] = [c, 1]
            if ch not in ch_amp or c > ch_amp[ch]:
                ch_amp[ch] = c

    # --- cut low-signal channels ---
    channels_all = sorted(set(k[0] for k in sums.keys()))
    amp_arr = np.array([ch_amp[ch] for ch in channels_all])
    threshold = max(amp_arr.mean() - 2 * amp_arr.std(), np.percentile(amp_arr, 10))
    kept = [ch for ch in channels_all if ch_amp[ch] >= threshold]
    print(f"Kept {len(kept)}, cut {len(channels_all) - len(kept)} channels, threshold={threshold:.2f}")

    # --- common energy grid (from raw data) ---
    raw_e = np.array(sorted(raw_e_set))
    e_min, e_max = raw_e.min(), raw_e.max()
    e_grid = np.arange(np.floor(e_min / E_GRID_STEP) * E_GRID_STEP,
                       e_max + E_GRID_STEP, E_GRID_STEP)
    y_vals = np.array(sorted(set(round(k[1], 2) for k in sums.keys() if k[0] in kept)))
    n_e, n_y = len(e_grid), len(y_vals)
    print(f"Uncorrected grid: {e_grid[0]:.1f}..{e_grid[-1]:.1f} eV ({n_e} pts)")
    print(f"Folded y bins: {n_y} ({y_vals[0]:.2f}..{y_vals[-1]:.2f})")

    # --- build UNCORRECTED heatmap ---
    heatmap = np.zeros((n_y, n_e))
    heatmap_n = np.zeros((n_y, n_e))
    
    for ch in kept:
        # collect per-y-bin spectra on RAW energy axis
        per_y = {}
        for (c, yb, e), (s, n) in sums.items():
            if c == ch:
                per_y.setdefault(yb, {})
                per_y[yb][e] = per_y[yb].get(e, [0.0, 0])
                per_y[yb][e][0] += s; per_y[yb][e][1] += n
        
        for yi, yb in enumerate(y_vals):
            if yb in per_y:
                ed = per_y[yb]
                es = np.array(sorted(ed.keys()))
                cs = np.array([ed[e][0]/ed[e][1] for e in es])
                # NO CORRECTION - use raw energies
                counts_interp = np.interp(e_grid, es, cs, left=np.nan, right=np.nan)
                valid = ~np.isnan(counts_interp)
                heatmap[yi, valid] += counts_interp[valid]
                heatmap_n[yi, valid] += 1
    
    heatmap = np.where(heatmap_n > 0, heatmap / np.maximum(heatmap_n, 1), np.nan)
    print(f"Heatmap shape: {n_y} y-bins x {n_e} energy pts")

    # --- plot UNCORRECTED heatmap ---
    e_edges = np.concatenate([e_grid - E_GRID_STEP/2, [e_grid[-1] + E_GRID_STEP/2]])
    y_edges = np.concatenate([y_vals - V_BIN/2, [y_vals[-1] + V_BIN/2]])
    
    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(heatmap),
                         shading="flat", cmap="magma")
    ax.axhline(V_HI, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(V_LO, 2 * V_HI - V_LO)
    ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
    ax.set_xlabel("Binding energy (eV, RAW - no correction)")
    ax.set_ylabel("Potential Ewe (V) (up-sweep, then down-sweep)")
    ax.set_title(f"{region_name} UNCORRECTED heatmap ({len(kept)} channels)")
    ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="white")
    ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="white")
    ax.invert_xaxis()
    fig.colorbar(mesh, ax=ax, label="counts/s")
    fig.tight_layout()
    f1 = OUT_PREFIX + "_heatmap_uncorrected.png"
    save_fig(fig, f1)

    # --- plot row-normalized UNCORRECTED heatmap ---
    norm_e_mask = e_grid >= (e_grid[-1] - 1.0)  # highest 1 eV
    with np.errstate(invalid="ignore"):
        row_ref = np.nanmean(heatmap[:, norm_e_mask], axis=1)
    heatmap_norm = np.divide(heatmap, row_ref[:, None],
                             out=np.full_like(heatmap, np.nan),
                             where=np.isfinite(row_ref[:, None]) & (row_ref[:, None] != 0))
    
    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(heatmap_norm),
                         shading="flat", cmap="magma")
    ax.axhline(V_HI, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(V_LO, 2 * V_HI - V_LO)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
    ax.set_xlabel("Binding energy (eV, RAW - no correction)")
    ax.set_ylabel("Potential Ewe (V) (up-sweep, then down-sweep)")
    ax.set_title(f"{region_name} UNCORRECTED heatmap (row-normalized to highest 1 eV)")
    ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="white")
    ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="white")
    ax.invert_xaxis()
    fig.colorbar(mesh, ax=ax, label=f"counts/s normalized to {e_grid[-1]-1:.1f}-{e_grid[-1]:.1f} eV")
    fig.tight_layout()
    f2 = OUT_PREFIX + "_heatmap_uncorrected_norm.png"
    save_fig(fig, f2)

    # --- plot differential UNCORRECTED heatmap ---
    with np.errstate(invalid="ignore"):
        voltage_avg_spectrum = np.nanmean(heatmap, axis=0)
    heatmap_diff = heatmap - voltage_avg_spectrum[None, :]
    m = np.nanmax(np.abs(heatmap_diff))
    symlog_norm = matplotlib.colors.SymLogNorm(linthresh=10.0, vmin=-m, vmax=m)
    
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap_rb = plt.get_cmap("RdBu_r").copy()
    cmap_rb.set_bad("lightgray")
    mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(heatmap_diff),
                         shading="flat", cmap=cmap_rb, norm=symlog_norm)
    ax.axhline(V_HI, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(V_LO, 2 * V_HI - V_LO)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
    ax.set_xlabel("Binding energy (eV, RAW - no correction)")
    ax.set_ylabel("Potential Ewe (V) (up-sweep, then down-sweep)")
    ax.set_title(f"{region_name} UNCORRECTED differential heatmap (sym-log)")
    ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="white")
    ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="white")
    ax.invert_xaxis()
    fig.colorbar(mesh, ax=ax, label="counts/s - voltage-avg (sym-log)")
    fig.tight_layout()
    f3 = OUT_PREFIX + "_heatmap_uncorrected_diff.png"
    save_fig(fig, f3)

    print("Done!")


if __name__ == "__main__":
    main()
