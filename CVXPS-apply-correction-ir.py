"""
Apply Au-derived energy correction to Ir 4f CV-XPS, with folded voltage axis
and low-signal channel cutting.

1. Load per-channel energy shifts from Au reference corrections CSV.
2. Load Ir 4f correlated CSV; determine up/down sweep direction from voltage
   vs time gradient; fold voltage axis (up 0.6->1.8, down continues 1.8->3.0).
3. Cut channels with low signal (amplitude < threshold = max(mean-2std, p10)).
4. Interpolate corrected spectra onto common grid limited to original BE range.
5. Outputs: folded heatmap, per-channel corrected spectra, overall average.

Usage: python CVXPS-apply-correction-ir.py [ir_csv] [au_corr_csv] [out_prefix]
"""

import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from bisect import bisect_left

IR_CSV = (sys.argv[1] if len(sys.argv) > 1 else
          r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\12_CV_ch0-145_XPS_correlated.csv")
AU_CORR_CSV = (sys.argv[2] if len(sys.argv) > 2 else
               r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv")
OUT_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\12_CV_Ir4f_corrected")

Y_SHIFT = 0.5
V_LO, V_HI = 0.6, 1.8   # CV vertices
V_BIN = 0.02            # voltage bin width
E_GRID_STEP = 0.1
DV_THRESHOLD = 1e-3


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


def load_corrections(csv_file):
    corr = {}
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            corr[int(row[header.index("channel")])] = float(row[header.index("correction_eV")])
    return corr


def interpolate(x, xs, ys):
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    i = bisect_left(xs, x)
    x0, x1 = xs[i-1], xs[i]
    y0, y1 = ys[i-1], ys[i]
    if x1 == x0: return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def main():
    corrections = load_corrections(AU_CORR_CSV)
    print(f"Loaded {len(corrections)} channel corrections")

    # --- pass 1: collect structure + determine sweep direction ---
    channels = []
    seen_ch = set()
    raw_energies = set()
    region_name = ""
    cv_t, cv_v = [], []  # for sweep direction

    with open(IR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        iv, ie, ic, ich, ir, it = (header.index(c) for c in
            ["voltage_Ewe_V", "binding_energy_eV", "counts_s", "channel", "region", "cv_time_s"])
        seen_tv = set()
        for row in reader:
            ch = int(row[ich])
            if ch not in seen_ch:
                seen_ch.add(ch); channels.append(ch)
            raw_energies.add(float(row[ie]))
            if not region_name:
                region_name = " ".join(row[ir].split()[:-1])
            tv = (float(row[it]), float(row[iv]))
            if tv not in seen_tv:
                seen_tv.add(tv)
                cv_t.append(tv[0]); cv_v.append(tv[1])

    channels.sort()
    raw_energies = np.array(sorted(raw_energies))
    n_ch = len(channels)

    # sweep direction from voltage gradient
    cv_t = np.array(cv_t); cv_v = np.array(cv_v)
    order = np.argsort(cv_t)
    cv_t, cv_v = cv_t[order], cv_v[order]
    grad = np.gradient(cv_v, cv_t)
    direction = np.where(grad > DV_THRESHOLD, 1, np.where(grad < -DV_THRESHOLD, -1, 1))

    # folded y: up -> V, down -> 2*V_HI - V
    def fold_y(v, t):
        i = bisect_left(cv_t, t)
        i = min(max(i, 0), len(direction)-1)
        d = direction[i]
        return v if d > 0 else 2 * V_HI - v

    # --- pass 2: accumulate ---
    # sums[(ch, y_bin, raw_e)] = [sum, n]
    sums = {}
    ch_amp = {}  # channel -> max counts (signal proxy)
    with open(IR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        iv, ie, ic, ich, it = (header.index(c) for c in
            ["voltage_Ewe_V", "binding_energy_eV", "counts_s", "channel", "cv_time_s"])
        for row in reader:
            ch = int(row[ich])
            e = float(row[ie])
            v = float(row[iv])
            t = float(row[it])
            c = float(row[ic])
            y = fold_y(v, t)
            y_bin = round(round(y / V_BIN) * V_BIN, 2)  # round to avoid float mismatch
            key = (ch, y_bin, e)
            if key in sums:
                sums[key][0] += c; sums[key][1] += 1
            else:
                sums[key] = [c, 1]
            if ch not in ch_amp or c > ch_amp[ch]:
                ch_amp[ch] = c

    # --- cut low-signal channels ---
    amp_arr = np.array([ch_amp.get(ch, 0) for ch in channels])
    mean_a, std_a = np.mean(amp_arr), np.std(amp_arr)
    threshold = max(mean_a - 2 * std_a, np.percentile(amp_arr, 10))
    kept = [ch for ch in channels if ch_amp.get(ch, 0) >= threshold]
    cut = [ch for ch in channels if ch not in kept]
    print(f"Channel amplitude: mean={mean_a:.2f}, std={std_a:.2f}, threshold={threshold:.2f}")
    print(f"Kept {len(kept)}, cut {len(cut)}: {cut}")

    # --- common corrected energy grid limited to ORIGINAL BE range ---
    e_min, e_max = raw_energies.min(), raw_energies.max()
    e_grid = np.arange(np.floor(e_min / E_GRID_STEP) * E_GRID_STEP,
                       e_max + E_GRID_STEP, E_GRID_STEP)
    n_e = len(e_grid)
    # y_vals only from KEPT channels (avoid empty rows from cut channels)
    y_vals = np.array(sorted(set(round(k[1], 2) for k in sums.keys() if k[0] in kept)))
    n_y = len(y_vals)
    print(f"Corrected grid: {e_grid[0]:.1f}..{e_grid[-1]:.1f} eV ({n_e} pts, original range)")
    print(f"Folded y bins: {n_y} ({y_vals[0]:.2f}..{y_vals[-1]:.2f})")

    # --- build matrices ---
    ch_spectra = np.full((n_ch, n_e), np.nan)   # voltage-averaged per channel
    heatmap = np.zeros((n_y, n_e))
    heatmap_n = np.zeros((n_y, n_e))

    for ci, ch in enumerate(channels):
        if ch not in kept:
            continue
        shift = corrections.get(ch, 0.0)
        # collect per-y-bin spectra on raw energy axis
        per_y = {}
        for (c, yb, e), (s, n) in sums.items():
            if c == ch:
                per_y.setdefault(yb, {})
                per_y[yb][e] = per_y[yb].get(e, [0.0, 0])
                per_y[yb][e][0] += s; per_y[yb][e][1] += n
        if not per_y:
            continue
        # voltage-averaged raw spectrum
        all_e = sorted(set(e for yb in per_y for e in per_y[yb]))
        raw_e = np.array(all_e)
        v_avg = np.zeros(len(raw_e))
        v_cnt = np.zeros(len(raw_e))
        for yb, ed in per_y.items():
            es = np.array(sorted(ed.keys()))
            cs = np.array([ed[e][0]/ed[e][1] for e in es])
            for e_val in raw_e:
                mask = es == e_val
                if mask.any():
                    idx = np.where(raw_e == e_val)[0][0]
                    v_avg[idx] += cs[mask].mean()
                    v_cnt[idx] += 1
        v_avg = np.where(v_cnt > 0, v_avg / np.maximum(v_cnt, 1), np.nan)
        # correct and interpolate
        e_corr = raw_e + shift
        o = np.argsort(e_corr)
        ch_spectra[ci] = np.interp(e_grid, e_corr[o], v_avg[o], left=np.nan, right=np.nan)
        # heatmap per y bin
        for yi, yb in enumerate(y_vals):
            if yb in per_y:
                ed = per_y[yb]
                es = np.array(sorted(ed.keys()))
                cs = np.array([ed[e][0]/ed[e][1] for e in es])
                e_c = es + shift
                o = np.argsort(e_c)
                h = np.interp(e_grid, e_c[o], cs[o], left=np.nan, right=np.nan)
                valid = ~np.isnan(h)
                heatmap[yi, valid] += h[valid]
                heatmap_n[yi, valid] += 1

    heatmap = np.where(heatmap_n > 0, heatmap / np.maximum(heatmap_n, 1), np.nan)

    # --- plot 1: amplitude vs channel with cut ---
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green" if ch in kept else "red" for ch in channels]
    ax.scatter(channels, [ch_amp.get(ch, 0) for ch in channels], c=colors, s=20, alpha=0.7)
    ax.axhline(threshold, color="black", ls="--", lw=1.5, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Channel"); ax.set_ylabel("Max counts/s")
    ax.set_title(f"Channel signal (green=kept {len(kept)}, red=cut {len(cut)})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    f0 = OUT_PREFIX + "_channel_amplitudes.png"
    save_fig(fig, f0)

    # --- plot 2: folded heatmap ---
    fig, ax = plt.subplots(figsize=(11, 8))
    e_edges = np.concatenate([e_grid - E_GRID_STEP/2, [e_grid[-1] + E_GRID_STEP/2]])
    y_edges = np.concatenate([y_vals - V_BIN/2, [y_vals[-1] + V_BIN/2]])
    mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(heatmap),
                         shading="flat", cmap="magma")
    ax.axhline(V_HI, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(V_LO, 2 * V_HI - V_LO)
    ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
    ax.set_xlabel("Binding energy (eV, corrected, original range)")
    ax.set_ylabel("Potential Ewe (V) (up-sweep, then down-sweep)")
    ax.set_title(f"{region_name} corrected heatmap (folded voltage, {len(kept)} channels)")
    ax.invert_xaxis()
    ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="white")
    ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="white")
    fig.colorbar(mesh, ax=ax, label="counts/s")
    fig.tight_layout()
    f1 = OUT_PREFIX + "_heatmap_corrected_folded.png"
    save_fig(fig, f1)

    # --- helper: folded-y axis decorations ---
    def decorate_folded(ax):
        ax.axhline(V_HI, color="white", lw=0.8, ls="--", alpha=0.7)
        ax.set_ylim(V_LO, 2 * V_HI - V_LO)
        ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
        ax.set_ylabel("Potential Ewe (V) (up-sweep, then down-sweep)")
        ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="white")
        ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="white")

    e_edges = np.concatenate([e_grid - E_GRID_STEP/2, [e_grid[-1] + E_GRID_STEP/2]])
    y_edges = np.concatenate([y_vals - V_BIN/2, [y_vals[-1] + V_BIN/2]])

    # --- plot 2b: row-normalized heatmap (normalize to highest 1 eV) ---
    norm_e_mask = e_grid >= (e_grid[-1] - 1.0)  # highest 1 eV
    with np.errstate(invalid="ignore"):
        row_ref = np.nanmean(heatmap[:, norm_e_mask], axis=1)
    heatmap_norm = np.divide(heatmap, row_ref[:, None],
                             out=np.full_like(heatmap, np.nan),
                             where=np.isfinite(row_ref[:, None]) & (row_ref[:, None] != 0))

    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(heatmap_norm),
                         shading="flat", cmap="magma")
    decorate_folded(ax)
    ax.set_xlabel("Binding energy (eV, corrected, original range)")
    ax.set_title(f"{region_name} heatmap row-normalized to highest 1 eV "
                 f"({e_grid[-1]-1:.1f}-{e_grid[-1]:.1f} eV)")
    ax.invert_xaxis()
    fig.colorbar(mesh, ax=ax, label=f"counts/s normalized to mean of {e_grid[-1]-1:.1f}-{e_grid[-1]:.1f} eV")
    fig.tight_layout()
    f1b = OUT_PREFIX + "_heatmap_corrected_folded_norm_maxBE.png"
    save_fig(fig, f1b)

    # --- plot 2c: differential heatmap (subtract voltage-averaged spectrum per row) ---
    with np.errstate(invalid="ignore"):
        voltage_avg_spectrum = np.nanmean(heatmap, axis=0)
    heatmap_diff = heatmap - voltage_avg_spectrum[None, :]
    m = np.nanmax(np.abs(heatmap_diff))
    symlog_norm = mcolors.SymLogNorm(linthresh=10.0, vmin=-m, vmax=m)

    fig, ax = plt.subplots(figsize=(11, 8))
    cmap_rb = plt.get_cmap("RdBu_r").copy()
    cmap_rb.set_bad("lightgray")
    mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(heatmap_diff),
                         shading="flat", cmap=cmap_rb, norm=symlog_norm)
    decorate_folded(ax)
    ax.set_xlabel("Binding energy (eV, corrected, original range)")
    ax.set_title(f"{region_name} differential heatmap (voltage-averaged spectrum subtracted per row, sym-log)")
    ax.invert_xaxis()
    fig.colorbar(mesh, ax=ax, label="counts/s - voltage-avg spectrum (sym-log)")
    fig.tight_layout()
    f1c = OUT_PREFIX + "_heatmap_corrected_folded_diff.png"
    save_fig(fig, f1c)

    # --- plot 2d: voltage-smoothed heatmaps at different averaging levels ---
    def smooth_y(matrix, window):
        """Moving average along y (voltage) axis with given odd window size."""
        if window <= 1:
            return matrix.copy()
        pad = window // 2
        padded = np.pad(matrix, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        smoothed = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)
        return smoothed

    for window in (3, 5, 9, 15):
        h_smooth = smooth_y(heatmap, window)
        fig, ax = plt.subplots(figsize=(11, 8))
        mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(h_smooth),
                             shading="flat", cmap="magma")
        decorate_folded(ax)
        ax.set_xlabel("Binding energy (eV, corrected, original range)")
        ax.set_title(f"{region_name} corrected heatmap, voltage-averaged over "
                     f"{window} bins ({window*V_BIN:.2f} V)")
        ax.invert_xaxis()
        fig.colorbar(mesh, ax=ax, label="counts/s (smoothed)")
        fig.tight_layout()
        f1d = OUT_PREFIX + f"_heatmap_corrected_folded_smooth{window}.png"
        save_fig(fig, f1d)



    # --- plot 3: per-channel corrected spectra (kept only) ---
    norm = Normalize(vmin=min(kept), vmax=max(kept))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(13, 10))
    for ci, ch in enumerate(channels):
        if ch not in kept:
            continue
        y = ch_spectra[ci] + ci * Y_SHIFT
        ax.plot(e_grid, y, "-", lw=1.0, color=cmap(norm(ch)), alpha=0.85)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02); cbar.set_label("Channel")
    ax.set_xlabel("Binding energy (eV, corrected, original range)")
    ax.set_ylabel(f"counts/s (voltage-averaged) + {Y_SHIFT:g} x channel idx")
    ax.set_title(f"{region_name} corrected spectra per kept channel ({len(kept)} ch)")
    ax.invert_xaxis(); ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = OUT_PREFIX + "_channel_spectra_corrected.png"
    save_fig(fig, f2)

    # --- plot 4: overall average ---
    kept_idx = [channels.index(ch) for ch in kept]
    overall = np.nanmean(ch_spectra[kept_idx], axis=0)
    overall_std = np.nanstd(ch_spectra[kept_idx], axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(e_grid, overall, "-", lw=2.5, color="darkblue",
            label=f"corrected avg ({len(kept)} channels x all voltages)")
    ax.fill_between(e_grid, overall - overall_std, overall + overall_std,
                    alpha=0.2, color="blue", label="±1 std")
    ax.set_xlabel("Binding energy (eV, corrected, original range)")
    ax.set_ylabel("counts/s")
    ax.set_title(f"{region_name} corrected averaged spectrum")
    ax.invert_xaxis(); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = OUT_PREFIX + "_overall_corrected.png"
    save_fig(fig, f3)

    # --- CSV ---
    csv_out = OUT_PREFIX + "_overall_corrected.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["binding_energy_eV", "counts_s_mean", "counts_s_std", "n_channels"])
        for i, e in enumerate(e_grid):
            w.writerow([f"{e:.4f}", f"{overall[i]:.6f}", f"{overall_std[i]:.6f}", len(kept)])
    print(f"Saved {csv_out}")


if __name__ == "__main__":
    main()
