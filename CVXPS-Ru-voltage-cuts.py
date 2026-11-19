"""
Extract voltage cuts from the corrected Ru 3d heatmap: for several binding
energy windows (0.5 eV wide), plot counts/s vs folded potential (up/down sweep).

Uses the same Au energy correction and channel cutting as CVXPS-apply-correction-ir.py.
Modified for Ru 3d with 0.5 eV energy window.

Usage: python CVXPS-Ru-voltage-cuts.py [ir_csv] [au_corr_csv] [out_prefix]
"""

import csv
import os
import sys
import time
from bisect import bisect_left

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

IR_CSV = (sys.argv[1] if len(sys.argv) > 1 else
          r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\12_CV_ch0-145_XPS_correlated.csv")
AU_CORR_CSV = (sys.argv[2] if len(sys.argv) > 2 else
               r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv")
OUT_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\12_CV_Ru3d_voltage_cuts")

V_BIN = 0.02
V_LO, V_HI = 0.6, 1.8
DV_THRESHOLD = 1e-3
E_GRID_STEP = 0.1
E_WINDOW = 0.5  # eV, total width of each energy window - INCREASED TO 0.5 eV

# energy centers to cut (eV) - Ru 3d region (Ru 3d5/2 ~280, Ru 3d3/2 ~284)
E_CENTERS = [277.5, 278.0, 278.5, 279.0, 279.5, 280.0, 280.5, 281.0, 281.5, 282.0,
             282.5, 283.0, 283.5, 284.0, 284.5, 285.0, 285.5, 286.0, 286.5, 287.0]


def save_fig(fig, out_file):
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


def main():
    corrections = load_corrections(AU_CORR_CSV)

    # --- build CV sweep direction ---
    cv_t, cv_v = [], []
    seen = set()
    with open(IR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        iv, it = header.index("voltage_Ewe_V"), header.index("cv_time_s")
        for row in reader:
            tv = (float(row[it]), float(row[iv]))
            if tv not in seen:
                seen.add(tv); cv_t.append(tv[0]); cv_v.append(tv[1])
    cv_t = np.array(cv_t); cv_v = np.array(cv_v)
    o = np.argsort(cv_t); cv_t, cv_v = cv_t[o], cv_v[o]
    grad = np.gradient(cv_v, cv_t)
    direction = np.where(grad > DV_THRESHOLD, 1, np.where(grad < -DV_THRESHOLD, -1, 1))

    # --- accumulate sums (same as apply-correction script) ---
    sums = {}
    ch_amp = {}
    raw_e_set = set()
    region_name = ""
    with open(IR_CSV, "r", encoding="utf-8", newline="") as f:
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
    print(f"Kept {len(kept)}, cut {len(channels_all) - len(kept)} channels")

    # --- common corrected energy grid ---
    raw_e = np.array(sorted(raw_e_set))
    e_min, e_max = raw_e.min(), raw_e.max()
    e_grid = np.arange(np.floor(e_min / E_GRID_STEP) * E_GRID_STEP,
                       e_max + E_GRID_STEP, E_GRID_STEP)
    y_vals = np.array(sorted(set(round(k[1], 2) for k in sums.keys() if k[0] in kept)))
    n_e, n_y = len(e_grid), len(y_vals)

    # --- build corrected heatmap (same as before) ---
    heatmap = np.zeros((n_y, n_e))
    hn = np.zeros((n_y, n_e))
    for ch in kept:
        shift = corrections.get(ch, 0.0)
        per_y = {}
        for (c, yb, e), (s, n) in sums.items():
            if c == ch:
                per_y.setdefault(yb, {})
                per_y[yb][e] = s / n
        for yi, yb in enumerate(y_vals):
            if yb in per_y:
                ed = per_y[yb]
                es = np.array(sorted(ed.keys()))
                cs = np.array([ed[e] for e in es])
                e_c = es + shift
                o = np.argsort(e_c)
                h = np.interp(e_grid, e_c[o], cs[o], left=np.nan, right=np.nan)
                valid = ~np.isnan(h)
                heatmap[yi, valid] += h[valid]
                hn[yi, valid] += 1
    heatmap = np.where(hn > 0, heatmap / np.maximum(hn, 1), np.nan)
    print(f"Heatmap: {n_y} y-bins x {n_e} energy pts")

    # --- extract voltage cuts for each energy window ---
    # For each energy center, average heatmap over E_WINDOW wide slice
    cuts = {}  # e_center -> (y_vals, mean_counts)
    for ec in E_CENTERS:
        mask = (e_grid >= ec - E_WINDOW / 2) & (e_grid <= ec + E_WINDOW / 2)
        if not mask.any():
            continue
        with np.errstate(invalid="ignore"):
            cut = np.nanmean(heatmap[:, mask], axis=1)
        cuts[ec] = cut
        print(f"  E={ec:.1f} eV (window {ec-E_WINDOW/2:.2f}-{ec+E_WINDOW/2:.2f}): "
              f"mean={np.nanmean(cut):.2f}, max={np.nanmax(cut):.2f} counts/s")

    # --- plot 1: all cuts on one graph (folded y-axis) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(cuts.keys()), vmax=max(cuts.keys()))
    for ec in sorted(cuts.keys()):
        cut = cuts[ec]
        color = cmap(norm(ec))
        ax.plot(y_vals, cut, "-", lw=1.5, color=color, alpha=0.9,
                label=f"{ec:.1f} eV")
    ax.axhline(V_HI, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlim(V_LO, 2 * V_HI - V_LO)
    ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
    ax.set_xlabel("Potential Ewe (V) (up-sweep, then down-sweep)")
    ax.set_ylabel(f"counts/s (averaged over {E_WINDOW} eV)")
    ax.set_title(f"{region_name} voltage cuts at different binding energies "
                 f"({len(kept)} channels, corrected)")
    ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="gray")
    ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="gray")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02); cbar.set_label("Binding energy (eV)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = OUT_PREFIX + "_voltage_cuts_all.png"
    save_fig(fig, f1)

    # --- plot 2: cuts in separate subplots (grid) ---
    n_cuts = len(cuts)
    n_cols = 3
    n_rows = (n_cuts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for i, ec in enumerate(sorted(cuts.keys())):
        ax = axes[i]
        cut = cuts[ec]
        ax.plot(y_vals, cut, "-", lw=1.5, color="darkblue")
        ax.axhline(V_HI, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlim(V_LO, 2 * V_HI - V_LO)
        ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks],
                           fontsize=7)
        ax.set_title(f"{ec:.1f} eV ({ec-E_WINDOW/2:.2f}-{ec+E_WINDOW/2:.2f})", fontsize=9)
        ax.set_ylabel("counts/s", fontsize=8)
        ax.grid(alpha=0.3)
    for i in range(n_cuts, len(axes)):
        axes[i].set_visible(False)
    axes[0].set_xlabel("Potential (V, folded)")
    fig.suptitle(f"{region_name} voltage cuts per {E_WINDOW} eV energy window", y=1.02)
    fig.tight_layout()
    f2 = OUT_PREFIX + "_voltage_cuts_grid.png"
    save_fig(fig, f2)

    # --- plot 3: heatmap with cut lines marked ---
    e_edges = np.concatenate([e_grid - E_GRID_STEP/2, [e_grid[-1] + E_GRID_STEP/2]])
    y_edges = np.concatenate([y_vals - V_BIN/2, [y_vals[-1] + V_BIN/2]])
    fig, ax = plt.subplots(figsize=(11, 8))
    mesh = ax.pcolormesh(e_edges, y_edges, np.ma.masked_invalid(heatmap),
                         shading="flat", cmap="magma")
    for ec in sorted(cuts.keys()):
        ax.axvline(ec, color="cyan", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(V_HI, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(V_LO, 2 * V_HI - V_LO)
    ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel("Potential Ewe (V) (up, then down)")
    ax.set_title(f"{region_name} heatmap with voltage-cut energies marked (cyan)")
    ax.invert_xaxis()
    fig.colorbar(mesh, ax=ax, label="counts/s")
    fig.tight_layout()
    f3 = OUT_PREFIX + "_heatmap_with_cuts.png"
    save_fig(fig, f3)

    # --- save cuts to CSV ---
    csv_out = OUT_PREFIX + "_voltage_cuts.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["folded_y_V"] + [f"E{ec:.1f}_counts_s" for ec in sorted(cuts.keys())]
        w.writerow(header)
        for yi, yb in enumerate(y_vals):
            row = [f"{yb:.2f}"]
            for ec in sorted(cuts.keys()):
                row.append(f"{cuts[ec][yi]:.6f}")
            w.writerow(row)
    print(f"Saved {csv_out}")


if __name__ == "__main__":
    main()
