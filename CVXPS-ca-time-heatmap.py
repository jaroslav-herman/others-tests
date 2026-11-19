"""
Time-resolved heatmap of O 1s XPS during repeated CA(2.2V, 30s)/OCV(30s) cycles.

Loads the correlated CSV produced by CVXPS-ca-time-correlate.py, applies the
Au-derived per-channel energy correction, folds the absolute time into the
0..CYCLE_PERIOD_S cycle interval (cycle_time_s), and plots heatmaps:

  x-axis : binding energy (eV, corrected per channel, interpolated on common grid)
  y-axis : time within cycle (0..60 s), subsequent cycles averaged
  fill   : counts/s averaged over all scans falling in the same (time bin, energy)

Outputs (next to the CSV, prefix = CSV basename):
  <prefix>_time_heatmap.png                  - main heatmap (magma)
  <prefix>_time_heatmap_norm.png             - row-normalized to max in each row
  <prefix>_time_heatmap_diff.png             - differential (time-avg spectrum subtracted, sym-log)
  <prefix>_time_heatmap_smooth{N}.png        - time-smoothed at N bins (3,5,9,15)
  <prefix>_time_spectra_corrected.png        - per-channel corrected spectra (waterfall)
  <prefix>_time_overall_corrected.png        - overall averaged corrected spectrum
  <prefix>_channel_amplitudes.png            - channel signal with cut threshold
  <prefix>_time_heatmap_voltage.png          - heatmap colored by interpolated voltage
  <prefix>_ec_vs_time.png                    - EC voltage/current vs time with XPS scan markers

Usage:
  python CVXPS-ca-time-heatmap.py <correlated_csv> <au_corr_csv> [out_prefix] [cycle_period_s] [time_bin_s]

Defaults:
  csv  = CA_OCV_..._O1s_CA_EC_correlated.csv
  corr = Au_ref_Fixed_energy_energy_corrections.csv
  cycle_period_s = 60
  time_bin_s = 1.0
"""

import csv
import os
import sys
import time as time_mod

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from bisect import bisect_left

CORR_CSV = (sys.argv[1] if len(sys.argv) > 1 else
            r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CA_OCV_2,2V_30s_30s_5mbar_Fixed_energy O1s_CA_EC_correlated.csv")
AU_CORR_CSV = (sys.argv[2] if len(sys.argv) > 2 else
               r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv")
OUT_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else
              os.path.splitext(CORR_CSV)[0])
CYCLE_PERIOD_S = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
TIME_BIN_S = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

E_GRID_STEP = 0.1       # eV step for corrected common grid
Y_SHIFT = 0.5           # waterfall shift for per-channel spectra


def save_fig(fig, out_file):
    """Save with retry (OneDrive occasionally locks the file briefly)."""
    for attempt in range(5):
        try:
            fig.savefig(out_file, dpi=150)
            break
        except OSError:
            if attempt == 4:
                raise
            time_mod.sleep(2)
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


def main():
    corrections = load_corrections(AU_CORR_CSV)
    print(f"Loaded {len(corrections)} channel corrections")

    # --- pass 1: collect structure ---
    channels = []
    seen_ch = set()
    raw_energies = set()
    region_name = ""
    scan_info = {}  # scan -> (ec_time_s, cycle_time_s, voltage, current)

    with open(CORR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_ct, i_e, i_c, i_ch, i_r, i_s = (header.index(c) for c in
            ["cycle_time_s", "binding_energy_eV", "counts_s", "channel", "region", "scan"])
        i_et, i_v, i_cur = (header.index(c) for c in
            ["ec_time_s", "voltage_Ewe_V", "current_mA"])
        for row in reader:
            ch = int(row[i_ch])
            if ch not in seen_ch:
                seen_ch.add(ch)
                channels.append(ch)
            e = float(row[i_e])
            raw_energies.add(e)
            if not region_name:
                region_name = " ".join(row[i_r].split()[:-1])
            s = int(row[i_s])
            if s not in scan_info:
                cur_str = row[i_cur].strip()
                scan_info[s] = (
                    float(row[i_et]),
                    float(row[i_ct]),
                    float(row[i_v]),
                    float(cur_str) if cur_str else float("nan"),
                )

    channels.sort()
    raw_energies = np.array(sorted(raw_energies))
    n_ch = len(channels)
    print(f"Channels: {n_ch} ({channels[0]}..{channels[-1]})")
    print(f"Raw energies: {raw_energies[0]:.1f}..{raw_energies[-1]:.1f} eV ({len(raw_energies)} pts)")
    print(f"Region: {region_name}")
    print(f"Scans: {len(scan_info)}")

    # --- pass 2: accumulate per (channel, time_bin, raw_energy) ---
    # sums[(ch, t_bin, raw_e)] = [sum_counts, n]
    sums = {}
    ch_amp = {}  # channel -> max counts
    with open(CORR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_ct, i_e, i_c, i_ch = (header.index(c) for c in
            ["cycle_time_s", "binding_energy_eV", "counts_s", "channel"])
        for row in reader:
            ch = int(row[i_ch])
            e = float(row[i_e])
            ct = float(row[i_ct])
            c = float(row[i_c])
            t_bin = round(round(ct / TIME_BIN_S) * TIME_BIN_S, 2)
            key = (ch, t_bin, e)
            if key in sums:
                sums[key][0] += c
                sums[key][1] += 1
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

    # time bins from KEPT channels only
    t_vals = np.array(sorted(set(round(k[1], 2) for k in sums.keys() if k[0] in kept)))
    n_t = len(t_vals)
    print(f"Corrected grid: {e_grid[0]:.1f}..{e_grid[-1]:.1f} eV ({n_e} pts)")
    print(f"Time bins: {n_t} ({t_vals[0]:.2f}..{t_vals[-1]:.2f} s, bin={TIME_BIN_S}s)")

    # --- build matrices ---
    ch_spectra = np.full((n_ch, n_e), np.nan)  # time-averaged per channel
    heatmap = np.zeros((n_t, n_e))
    heatmap_n = np.zeros((n_t, n_e))
    # voltage per time bin (averaged over scans in that bin)
    volt_per_t = {}

    for ci, ch in enumerate(channels):
        if ch not in kept:
            continue
        shift = corrections.get(ch, 0.0)
        # collect per-time-bin spectra on raw energy axis
        per_t = {}
        for (c, tb, e), (s, n) in sums.items():
            if c == ch:
                per_t.setdefault(tb, {})
                per_t[tb][e] = per_t[tb].get(e, [0.0, 0])
                per_t[tb][e][0] += s
                per_t[tb][e][1] += n
        if not per_t:
            continue
        # time-averaged raw spectrum for this channel
        all_e = sorted(set(e for tb in per_t for e in per_t[tb]))
        raw_e = np.array(all_e)
        t_avg = np.zeros(len(raw_e))
        t_cnt = np.zeros(len(raw_e))
        for tb, ed in per_t.items():
            es = np.array(sorted(ed.keys()))
            cs = np.array([ed[e][0] / ed[e][1] for e in es])
            for e_val in raw_e:
                mask = es == e_val
                if mask.any():
                    idx = np.where(raw_e == e_val)[0][0]
                    t_avg[idx] += cs[mask].mean()
                    t_cnt[idx] += 1
        t_avg = np.where(t_cnt > 0, t_avg / np.maximum(t_cnt, 1), np.nan)
        # correct and interpolate onto common grid
        e_corr = raw_e + shift
        o = np.argsort(e_corr)
        ch_spectra[ci] = np.interp(e_grid, e_corr[o], t_avg[o], left=np.nan, right=np.nan)
        # heatmap per time bin
        for ti, tb in enumerate(t_vals):
            if tb in per_t:
                ed = per_t[tb]
                es = np.array(sorted(ed.keys()))
                cs = np.array([ed[e][0] / ed[e][1] for e in es])
                e_c = es + shift
                o = np.argsort(e_c)
                h = np.interp(e_grid, e_c[o], cs[o], left=np.nan, right=np.nan)
                valid = ~np.isnan(h)
                heatmap[ti, valid] += h[valid]
                heatmap_n[ti, valid] += 1

    heatmap = np.where(heatmap_n > 0, heatmap / np.maximum(heatmap_n, 1), np.nan)

    # voltage per time bin
    for s, (et, ct, v, cur) in scan_info.items():
        tb = round(round(ct / TIME_BIN_S) * TIME_BIN_S, 2)
        if tb in set(t_vals):
            if tb not in volt_per_t:
                volt_per_t[tb] = []
            volt_per_t[tb].append(v)
    volt_arr = np.array([np.mean(volt_per_t.get(tb, [np.nan])) if tb in volt_per_t else np.nan
                         for tb in t_vals])

    e_edges = np.concatenate([e_grid - E_GRID_STEP / 2, [e_grid[-1] + E_GRID_STEP / 2]])
    t_edges = np.concatenate([t_vals - TIME_BIN_S / 2, [t_vals[-1] + TIME_BIN_S / 2]])

    def add_heatmap_grid(ax, e_grid, t_vals, cycle_period, color="white", alpha=0.35):
        """Draw visible grid lines on a heatmap: vertical at each 1 eV, horizontal at each 5 s."""
        e_lo, e_hi = e_grid[-1], e_grid[0]  # x is inverted
        # vertical grid lines at integer eV
        e_ticks = np.arange(int(np.floor(e_lo)), int(np.ceil(e_hi)) + 1, 1)
        for ev in e_ticks:
            ax.axvline(ev, color=color, lw=0.5, alpha=alpha, zorder=5)
        # horizontal grid lines: major at 5 s, minor at 1 s
        for tv in range(0, int(cycle_period) + 1, 5):
            ax.axhline(tv, color=color, lw=0.6, alpha=alpha, zorder=5)
        for tv in range(0, int(cycle_period) + 1, 1):
            if tv % 5 != 0:
                ax.axhline(tv, color=color, lw=0.25, alpha=alpha * 0.5, zorder=5)
        # ticks with binding energy labels
        ax.set_xticks(e_ticks)
        ax.set_xticklabels([f"{ev:.0f}" for ev in e_ticks], fontsize=8)
        ax.set_yticks(np.arange(0, cycle_period + 1, 5))
        ax.tick_params(axis="x", labelsize=8)

    # ========== PLOT 1: channel amplitudes ==========
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["green" if ch in kept else "red" for ch in channels]
    ax.scatter(channels, [ch_amp.get(ch, 0) for ch in channels], c=colors, s=20, alpha=0.7)
    ax.axhline(threshold, color="black", ls="--", lw=1.5, label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Max counts/s")
    ax.set_title(f"Channel signal (green=kept {len(kept)}, red=cut {len(cut)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_channel_amplitudes.png")

    # ========== PLOT 2: main time heatmap ==========
    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, t_edges, np.ma.masked_invalid(heatmap),
                         shading="flat", cmap="magma")
    # Mark CA/OCV boundary at 30s
    ax.axhline(30.0, color="white", lw=1.0, ls="--", alpha=0.8)
    ax.annotate("CA 2.2V", xy=(0.02, 0.25), xycoords="axes fraction", color="white", fontsize=11, fontweight="bold")
    ax.annotate("OCV", xy=(0.02, 0.75), xycoords="axes fraction", color="white", fontsize=11, fontweight="bold")
    ax.set_xlim(e_grid[-1], e_grid[0])  # invert x (high BE left)
    ax.set_ylim(0, CYCLE_PERIOD_S)
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"Time within cycle (s, 0-{CYCLE_PERIOD_S:.0f} s)")
    ax.set_title(f"{region_name} time-resolved heatmap (corrected, {len(kept)} channels, "
                 f"{len(scan_info)} scans averaged into 0-{CYCLE_PERIOD_S:.0f} s)")
    add_heatmap_grid(ax, e_grid, t_vals, CYCLE_PERIOD_S)
    fig.colorbar(mesh, ax=ax, label="counts/s")
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_heatmap.png")

    # ========== PLOT 3: row-normalized heatmap ==========
    with np.errstate(invalid="ignore"):
        row_max = np.nanmax(heatmap, axis=1)
    heatmap_norm = np.divide(heatmap, row_max[:, None],
                             out=np.full_like(heatmap, np.nan),
                             where=np.isfinite(row_max[:, None]) & (row_max[:, None] != 0))
    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, t_edges, np.ma.masked_invalid(heatmap_norm),
                         shading="flat", cmap="magma", vmin=0, vmax=1)
    ax.axhline(30.0, color="white", lw=1.0, ls="--", alpha=0.8)
    ax.annotate("CA 2.2V", xy=(0.02, 0.25), xycoords="axes fraction", color="white", fontsize=11, fontweight="bold")
    ax.annotate("OCV", xy=(0.02, 0.75), xycoords="axes fraction", color="white", fontsize=11, fontweight="bold")
    ax.set_xlim(e_grid[-1], e_grid[0])
    ax.set_ylim(0, CYCLE_PERIOD_S)
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"Time within cycle (s)")
    ax.set_title(f"{region_name} time heatmap row-normalized to row max")
    add_heatmap_grid(ax, e_grid, t_vals, CYCLE_PERIOD_S)
    fig.colorbar(mesh, ax=ax, label="counts/s / row max")
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_heatmap_norm.png")

    # ========== PLOT 4: differential heatmap (subtract time-averaged spectrum) ==========
    with np.errstate(invalid="ignore"):
        time_avg_spectrum = np.nanmean(heatmap, axis=0)
    heatmap_diff = heatmap - time_avg_spectrum[None, :]
    m = np.nanmax(np.abs(heatmap_diff))
    if m > 0:
        symlog_norm = mcolors.SymLogNorm(linthresh=max(m / 100, 1.0), vmin=-m, vmax=m)
    else:
        symlog_norm = None
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap_rb = plt.get_cmap("RdBu_r").copy()
    cmap_rb.set_bad("lightgray")
    mesh = ax.pcolormesh(e_edges, t_edges, np.ma.masked_invalid(heatmap_diff),
                         shading="flat", cmap=cmap_rb, norm=symlog_norm)
    ax.axhline(30.0, color="white", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(e_grid[-1], e_grid[0])
    ax.set_ylim(0, CYCLE_PERIOD_S)
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"Time within cycle (s)")
    ax.set_title(f"{region_name} differential time heatmap (time-avg spectrum subtracted, sym-log)")
    add_heatmap_grid(ax, e_grid, t_vals, CYCLE_PERIOD_S, color="gray", alpha=0.4)
    fig.colorbar(mesh, ax=ax, label="counts/s - time-avg (sym-log)")
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_heatmap_diff.png")

    # ========== PLOT 5: time-smoothed heatmaps ==========
    def smooth_t(matrix, window):
        if window <= 1:
            return matrix.copy()
        pad = window // 2
        padded = np.pad(matrix, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)

    for window in (3, 5, 9, 15):
        h_smooth = smooth_t(heatmap, window)
        fig, ax = plt.subplots(figsize=(11, 8))
        mesh = ax.pcolormesh(e_edges, t_edges, np.ma.masked_invalid(h_smooth),
                             shading="flat", cmap="magma")
        ax.axhline(30.0, color="white", lw=1.0, ls="--", alpha=0.8)
        ax.annotate("CA 2.2V", xy=(0.02, 0.25), xycoords="axes fraction", color="white", fontsize=11, fontweight="bold")
        ax.annotate("OCV", xy=(0.02, 0.75), xycoords="axes fraction", color="white", fontsize=11, fontweight="bold")
        ax.set_xlim(e_grid[-1], e_grid[0])
        ax.set_ylim(0, CYCLE_PERIOD_S)
        ax.set_xlabel("Binding energy (eV, corrected)")
        ax.set_ylabel(f"Time within cycle (s)")
        ax.set_title(f"{region_name} time heatmap, smoothed over {window} bins ({window * TIME_BIN_S:.1f} s)")
        add_heatmap_grid(ax, e_grid, t_vals, CYCLE_PERIOD_S)
        fig.colorbar(mesh, ax=ax, label="counts/s (smoothed)")
        fig.tight_layout()
        save_fig(fig, OUT_PREFIX + f"_time_heatmap_smooth{window}.png")

    # ========== PLOT 6: per-channel corrected spectra (waterfall) ==========
    norm = Normalize(vmin=min(kept), vmax=max(kept))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(13, 10))
    for ci, ch in enumerate(channels):
        if ch not in kept:
            continue
        y = ch_spectra[ci] + ci * Y_SHIFT
        ax.plot(e_grid, y, "-", lw=1.0, color=cmap(norm(ch)), alpha=0.85)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Channel")
    ax.set_xlim(e_grid[-1], e_grid[0])
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"counts/s (time-averaged) + {Y_SHIFT:g} x channel idx")
    ax.set_title(f"{region_name} corrected spectra per kept channel ({len(kept)} ch)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_spectra_corrected.png")

    # ========== PLOT 7: overall averaged corrected spectrum ==========
    kept_idx = [channels.index(ch) for ch in kept]
    overall = np.nanmean(ch_spectra[kept_idx], axis=0)
    overall_std = np.nanstd(ch_spectra[kept_idx], axis=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(e_grid, overall, "-", lw=2.5, color="darkblue",
            label=f"corrected avg ({len(kept)} channels x all times)")
    ax.fill_between(e_grid, overall - overall_std, overall + overall_std,
                    alpha=0.2, color="blue", label="±1 std")
    ax.set_xlim(e_grid[-1], e_grid[0])
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel("counts/s")
    ax.set_title(f"{region_name} corrected averaged spectrum (all cycles)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_overall_corrected.png")

    # ========== PLOT 8: heatmap with voltage overlay (secondary axis info) ==========
    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, t_edges, np.ma.masked_invalid(heatmap),
                         shading="flat", cmap="magma")
    ax.axhline(30.0, color="white", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(e_grid[-1], e_grid[0])
    ax.set_ylim(0, CYCLE_PERIOD_S)
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"Time within cycle (s)")
    # Overlay voltage as white line on secondary axis
    ax2 = ax.twinx()
    ax2.plot(volt_arr, t_vals, "c-", lw=2, alpha=0.8, label="Ewe (V)")
    ax2.set_ylabel("Ewe (V)", color="cyan")
    ax2.tick_params(axis="y", labelcolor="cyan")
    ax2.set_ylim(0, CYCLE_PERIOD_S)
    ax2.legend(loc="upper right")
    ax.set_title(f"{region_name} time heatmap with interpolated Ewe overlay")
    add_heatmap_grid(ax, e_grid, t_vals, CYCLE_PERIOD_S)
    fig.colorbar(mesh, ax=ax, label="counts/s", pad=0.12)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_heatmap_voltage.png")

    # ========== PLOT 9: EC voltage/current vs absolute time with XPS scan markers ==========
    fig, ax1 = plt.subplots(figsize=(14, 6))
    scan_times = sorted(set((scan_info[s][0], scan_info[s][2]) for s in scan_info))
    st = [x[0] for x in scan_times]
    sv = [x[1] for x in scan_times]
    ax1.plot(st, sv, "-", lw=1.5, color="blue", label="Ewe (V)")
    ax1.set_xlabel("EC time (s)")
    ax1.set_ylabel("Ewe (V)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    # scan markers
    all_scan_t = sorted(scan_info[s][0] for s in scan_info)
    ax1.scatter(all_scan_t, [scan_info[s][2] for s in sorted(scan_info)],
                c="red", s=15, zorder=5, label="XPS scan")
    ax2 = ax1.twinx()
    cur_data = [(scan_info[s][0], scan_info[s][3]) for s in sorted(scan_info)
                if scan_info[s][3] == scan_info[s][3]]
    if cur_data:
        ax2.plot([x[0] for x in cur_data], [x[1] for x in cur_data], "-", lw=1.0,
                 color="green", alpha=0.7, label="I (mA)")
        ax2.set_ylabel("Current (mA)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
    ax1.set_title("EC voltage/current vs time with XPS scan times (cycle period "
                  f"{CYCLE_PERIOD_S:.0f} s)")
    # cycle boundaries
    for ct in range(0, int(max(st)) + int(CYCLE_PERIOD_S), int(CYCLE_PERIOD_S)):
        ax1.axvline(ct, color="gray", ls=":", alpha=0.5)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_ec_vs_time.png")

    # ========== CSV: overall corrected spectrum ==========
    csv_out = OUT_PREFIX + "_time_overall_corrected.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["binding_energy_eV", "counts_s_mean", "counts_s_std", "n_channels"])
        for i, e in enumerate(e_grid):
            w.writerow([f"{e:.4f}", f"{overall[i]:.6f}", f"{overall_std[i]:.6f}", len(kept)])
    print(f"Saved {csv_out}")

    # ========== CSV: heatmap data ==========
    csv_hm = OUT_PREFIX + "_time_heatmap.csv"
    with open(csv_hm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"{e:.2f}" for e in e_grid])
        for ti, tb in enumerate(t_vals):
            w.writerow([f"{tb:.2f}"] + [f"{heatmap[ti, j]:.6f}" if not np.isnan(heatmap[ti, j]) else ""
                                         for j in range(n_e)])
    print(f"Saved {csv_hm}")


if __name__ == "__main__":
    main()
