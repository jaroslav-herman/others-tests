"""
Extract time cuts from the corrected O 1s time-resolved heatmap: for several
binding-energy windows (0.3 eV wide), plot counts/s vs cycle time (0-60 s).

Uses the same Au energy correction and channel cutting as CVXPS-ca-time-heatmap.py.

Usage: python CVXPS-ca-time-cuts.py [correlated_csv] [au_corr_csv] [out_prefix] [cycle_period_s] [time_bin_s]

Defaults:
  csv  = CA_OCV_time_resolved/CA_OCV_O1s_CA_EC_correlated.csv
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
import matplotlib.pyplot as plt
import numpy as np

CORR_CSV = (sys.argv[1] if len(sys.argv) > 1 else
            r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CA_OCV_time_resolved\CA_OCV_O1s_CA_EC_correlated.csv")
AU_CORR_CSV = (sys.argv[2] if len(sys.argv) > 2 else
               r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv")
OUT_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CA_OCV_time_resolved\CA_OCV_O1s")
CYCLE_PERIOD_S = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
TIME_BIN_S = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

E_GRID_STEP = 0.1
E_WINDOW = 0.3  # eV, total width of each energy window

# Energy centers to cut (eV) - spanning the O 1s range 528-537.5 eV
E_CENTERS = [529.0, 529.5, 530.0, 530.5, 531.0, 531.5, 532.0, 532.5,
             533.0, 533.5, 534.0, 534.5, 535.0, 535.5, 536.0, 536.5, 537.0]


def save_fig(fig, out_file):
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

    # --- accumulate sums: sums[(ch, t_bin, raw_e)] = [sum_counts, n] ---
    sums = {}
    ch_amp = {}
    raw_e_set = set()
    region_name = ""

    with open(CORR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_ct, i_e, i_c, i_ch, i_r = (header.index(c) for c in
            ["cycle_time_s", "binding_energy_eV", "counts_s", "channel", "region"])
        for row in reader:
            ch = int(row[i_ch])
            e = float(row[i_e])
            ct = float(row[i_ct])
            c = float(row[i_c])
            if not region_name:
                region_name = " ".join(row[i_r].split()[:-1])
            raw_e_set.add(e)
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
    channels_all = sorted(set(k[0] for k in sums.keys()))
    amp_arr = np.array([ch_amp[ch] for ch in channels_all])
    threshold = max(amp_arr.mean() - 2 * amp_arr.std(), np.percentile(amp_arr, 10))
    kept = [ch for ch in channels_all if ch_amp[ch] >= threshold]
    print(f"Kept {len(kept)}, cut {len(channels_all) - len(kept)} channels (threshold={threshold:.2f})")

    # --- common corrected energy grid ---
    raw_e = np.array(sorted(raw_e_set))
    e_min, e_max = raw_e.min(), raw_e.max()
    e_grid = np.arange(np.floor(e_min / E_GRID_STEP) * E_GRID_STEP,
                       e_max + E_GRID_STEP, E_GRID_STEP)
    t_vals = np.array(sorted(set(round(k[1], 2) for k in sums.keys() if k[0] in kept)))
    n_e, n_t = len(e_grid), len(t_vals)

    # --- build corrected heatmap ---
    heatmap = np.zeros((n_t, n_e))
    hn = np.zeros((n_t, n_e))
    for ch in kept:
        shift = corrections.get(ch, 0.0)
        per_t = {}
        for (c, tb, e), (s, n) in sums.items():
            if c == ch:
                per_t.setdefault(tb, {})
                per_t[tb][e] = s / n
        for ti, tb in enumerate(t_vals):
            if tb in per_t:
                ed = per_t[tb]
                es = np.array(sorted(ed.keys()))
                cs = np.array([ed[e] for e in es])
                e_c = es + shift
                o = np.argsort(e_c)
                h = np.interp(e_grid, e_c[o], cs[o], left=np.nan, right=np.nan)
                valid = ~np.isnan(h)
                heatmap[ti, valid] += h[valid]
                hn[ti, valid] += 1
    heatmap = np.where(hn > 0, heatmap / np.maximum(hn, 1), np.nan)
    print(f"Heatmap: {n_t} time-bins x {n_e} energy pts, BE {e_grid[0]:.1f}-{e_grid[-1]:.1f} eV")

    # --- extract time cuts for each energy window ---
    cuts = {}  # e_center -> counts per time bin
    for ec in E_CENTERS:
        mask = (e_grid >= ec - E_WINDOW / 2) & (e_grid <= ec + E_WINDOW / 2)
        if not mask.any():
            continue
        with np.errstate(invalid="ignore"):
            cut = np.nanmean(heatmap[:, mask], axis=1)
        cuts[ec] = cut
        print(f"  E={ec:.1f} eV (window {ec-E_WINDOW/2:.2f}-{ec+E_WINDOW/2:.2f}): "
              f"mean={np.nanmean(cut):.2f}, max={np.nanmax(cut):.2f} counts/s")

    # --- plot 1: all cuts on one graph ---
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(cuts.keys()), vmax=max(cuts.keys()))
    for ec in sorted(cuts.keys()):
        cut = cuts[ec]
        ax.plot(t_vals, cut, "-", lw=1.5, color=cmap(norm(ec)), alpha=0.9,
                label=f"{ec:.1f} eV")
    ax.axvline(30.0, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.annotate("CA 2.2V", xy=(0.25, 0.02), xycoords="axes fraction", color="gray",
                fontsize=10, fontweight="bold")
    ax.annotate("OCV", xy=(0.75, 0.02), xycoords="axes fraction", color="gray",
                fontsize=10, fontweight="bold")
    ax.set_xlim(0, CYCLE_PERIOD_S)
    ax.set_xlabel(f"Time within cycle (s, 0-{CYCLE_PERIOD_S:.0f} s)")
    ax.set_ylabel(f"counts/s (averaged over {E_WINDOW} eV)")
    ax.set_title(f"{region_name} time cuts at different binding energies "
                 f"({len(kept)} channels, corrected)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Binding energy (eV)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_cuts_all.png")

    # --- plot 2: cuts in separate subplots (grid) ---
    n_cuts = len(cuts)
    n_cols = 3
    n_rows = (n_cuts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for i, ec in enumerate(sorted(cuts.keys())):
        ax = axes[i]
        cut = cuts[ec]
        ax.plot(t_vals, cut, "-", lw=1.5, color="darkblue")
        ax.axvline(30.0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlim(0, CYCLE_PERIOD_S)
        ax.set_xticks(np.arange(0, CYCLE_PERIOD_S + 1, 10))
        ax.set_title(f"{ec:.1f} eV ({ec-E_WINDOW/2:.2f}-{ec+E_WINDOW/2:.2f})", fontsize=9)
        ax.set_ylabel("counts/s", fontsize=8)
        ax.grid(alpha=0.3)
    for i in range(n_cuts, len(axes)):
        axes[i].set_visible(False)
    axes[0].set_xlabel(f"Time within cycle (s)")
    fig.suptitle(f"{region_name} time cuts per {E_WINDOW} eV energy window", y=1.02)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_cuts_grid.png")

    # --- plot 3: heatmap with cut lines marked ---
    e_edges = np.concatenate([e_grid - E_GRID_STEP / 2, [e_grid[-1] + E_GRID_STEP / 2]])
    t_edges = np.concatenate([t_vals - TIME_BIN_S / 2, [t_vals[-1] + TIME_BIN_S / 2]])
    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, t_edges, np.ma.masked_invalid(heatmap),
                         shading="flat", cmap="magma")
    for ec in sorted(cuts.keys()):
        ax.axvline(ec, color="cyan", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(30.0, color="white", lw=1.0, ls="--", alpha=0.8)
    ax.annotate("CA 2.2V", xy=(0.02, 0.25), xycoords="axes fraction", color="white",
                fontsize=11, fontweight="bold")
    ax.annotate("OCV", xy=(0.02, 0.75), xycoords="axes fraction", color="white",
                fontsize=11, fontweight="bold")
    ax.set_xlim(e_grid[-1], e_grid[0])
    ax.set_ylim(0, CYCLE_PERIOD_S)
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"Time within cycle (s)")
    ax.set_title(f"{region_name} time heatmap with time-cut energies marked (cyan)")
    fig.colorbar(mesh, ax=ax, label="counts/s")
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_heatmap_with_cuts.png")

    # --- plot 4: normalized cuts (each cut divided by its own max) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    for ec in sorted(cuts.keys()):
        cut = cuts[ec]
        mx = np.nanmax(cut)
        if mx > 0:
            cut_n = cut / mx
        else:
            cut_n = cut
        ax.plot(t_vals, cut_n, "-", lw=1.5, color=cmap(norm(ec)), alpha=0.9,
                label=f"{ec:.1f} eV")
    ax.axvline(30.0, color="gray", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlim(0, CYCLE_PERIOD_S)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel(f"Time within cycle (s)")
    ax.set_ylabel("counts/s (normalized to cut max)")
    ax.set_title(f"{region_name} normalized time cuts ({E_WINDOW} eV windows)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_time_cuts_normalized.png")

    # --- save cuts to CSV ---
    csv_out = OUT_PREFIX + "_time_cuts.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"E{ec:.1f}_counts_s" for ec in sorted(cuts.keys())])
        for ti, tb in enumerate(t_vals):
            row = [f"{tb:.2f}"]
            for ec in sorted(cuts.keys()):
                v = cuts[ec][ti]
                row.append(f"{v:.6f}" if not np.isnan(v) else "")
            w.writerow(row)
    print(f"Saved {csv_out}")


if __name__ == "__main__":
    main()
