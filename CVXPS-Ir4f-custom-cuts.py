"""
Generate custom voltage cuts for Ir 4f with specific energy ranges:
- 60.5-61 eV
- 61.5-62 eV
- 62.1-63 eV

Usage: python CVXPS-Ir4f-custom-cuts.py [csv] [au_corr_csv] [out_prefix]
"""

import csv
import sys
import time
from bisect import bisect_left

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IR_CSV = (sys.argv[1] if len(sys.argv) > 1 else
          r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145_Ir4f\17_CV_ch0-145_XPS_correlated.csv")
AU_CORR_CSV = (sys.argv[2] if len(sys.argv) > 2 else
               r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv")
OUT_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145_Ir4f\17_CV_Ir4f_custom_cuts")

V_BIN = 0.02
V_LO, V_HI = 0.6, 2.2
DV_THRESHOLD = 1e-3
E_GRID_STEP = 0.1

# Custom energy ranges for Ir 4f
CUSTOM_RANGES = [
    (60.5, 61.0, "60.5-61"),
    (61.5, 62.0, "61.5-62"),
    (62.1, 63.0, "62.1-63"),
]


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

    # --- accumulate sums ---
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

    # --- build corrected heatmap ---
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

    # --- extract custom voltage cuts ---
    cuts = {}  # label -> (y_vals, mean_counts)
    for e_lo, e_hi, label in CUSTOM_RANGES:
        mask = (e_grid >= e_lo) & (e_grid <= e_hi)
        if not mask.any():
            continue
        with np.errstate(invalid="ignore"):
            cut = np.nanmean(heatmap[:, mask], axis=1)
        cuts[label] = cut
        print(f"  Range {label} eV: mean={np.nanmean(cut):.2f}, max={np.nanmax(cut):.2f} counts/s")

    # --- plot custom cuts on one graph ---
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i/len(cuts)) for i in range(len(cuts))]
    for label, color in zip(sorted(cuts.keys()), colors):
        cut = cuts[label]
        ax.plot(y_vals, cut, "-", lw=2, color=color, alpha=0.9, label=label)
    ax.axhline(V_HI, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlim(V_LO, 2 * V_HI - V_LO)
    ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks])
    ax.set_xlabel("Potential Ewe (V) (up-sweep, then down-sweep)")
    ax.set_ylabel("counts/s (averaged over custom energy range)")
    ax.set_title(f"{region_name} custom voltage cuts: 60.5-61, 61.5-62, 62.1-63 eV ({len(kept)} channels, corrected)")
    ax.annotate("up", xy=(0.02, 0.28), xycoords="axes fraction", color="gray")
    ax.annotate("down", xy=(0.02, 0.72), xycoords="axes fraction", color="gray")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = OUT_PREFIX + "_custom_cuts_all.png"
    save_fig(fig, f1)

    # --- plot custom cuts in separate subplots ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for i, (label, color) in enumerate(sorted(zip(cuts.keys(), colors))):
        ax = axes[i]
        cut = cuts[label]
        ax.plot(y_vals, cut, "-", lw=2, color=color)
        ax.axhline(V_HI, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlim(V_LO, 2 * V_HI - V_LO)
        ticks = np.arange(V_LO, 2 * V_HI - V_LO + 1e-9, 0.4)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:.1f}" if v <= V_HI + 1e-9 else f"{2*V_HI - v:.1f}" for v in ticks], fontsize=8)
        ax.set_title(f"{label} eV", fontsize=10, color=color)
        ax.set_ylabel("counts/s", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{region_name} custom voltage cuts per energy range", y=1.02)
    fig.tight_layout()
    f2 = OUT_PREFIX + "_custom_cuts_grid.png"
    save_fig(fig, f2)

    # --- save cuts to CSV ---
    csv_out = OUT_PREFIX + "_custom_cuts.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        header = ["folded_y_V"] + [label for label in sorted(cuts.keys())]
        w.writerow(header)
        for yi, yb in enumerate(y_vals):
            row = [f"{yb:.2f}"]
            for label in sorted(cuts.keys()):
                row.append(f"{cuts[label][yi]:.6f}")
            w.writerow(row)
    print(f"Saved {csv_out}")


if __name__ == "__main__":
    main()
