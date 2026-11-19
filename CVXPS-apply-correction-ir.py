"""
Apply the Au-derived per-channel energy correction to Ir 4f CV-XPS data.

Loads the correction shifts from the Au reference corrections CSV, applies them
to each channel's energy axis in the Ir 4f correlated CSV (12_CV.mpt + channel
XPS), interpolates all corrected channel spectra onto a common energy grid, and
produces:
  1. heatmap of corrected BE vs voltage (averaged over channels per voltage bin)
  2. voltage-averaged corrected spectra per channel (y-shifted, color gradient)
  3. overall corrected spectrum averaged over channels and voltages

Usage: python CVXPS-apply-correction-ir.py [ir_correlated_csv] [au_corr_csv] [out_prefix]
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

IR_CSV = (sys.argv[1] if len(sys.argv) > 1 else
          r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\12_CV_ch0-145_XPS_correlated.csv")
AU_CORR_CSV = (sys.argv[2] if len(sys.argv) > 2 else
               r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv")
OUT_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-145\12_CV_Ir4f_corrected")

Y_SHIFT = 0.5   # for per-channel plot
V_BIN = 0.05    # voltage bin width for heatmap
E_GRID_STEP = 0.1  # common corrected energy grid step (eV)


def load_corrections(csv_file):
    corr = {}
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        i_ch = header.index("channel")
        i_corr = header.index("correction_eV")
        for row in r:
            corr[int(row[i_ch])] = float(row[i_corr])
    return corr


def main():
    corrections = load_corrections(AU_CORR_CSV)
    print(f"Loaded {len(corrections)} channel corrections from Au reference")

    # --- pass 1: collect raw energies, channels, voltages ---
    channels = []
    seen_ch = set()
    raw_energies = set()
    voltages = set()
    region_name = ""

    with open(IR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_v = header.index("voltage_Ewe_V")
        i_e = header.index("binding_energy_eV")
        i_ch = header.index("channel")
        i_r = header.index("region")
        for row in reader:
            ch = int(row[i_ch])
            if ch not in seen_ch:
                seen_ch.add(ch); channels.append(ch)
            raw_energies.add(float(row[i_e]))
            voltages.add(round(float(row[i_v]) / V_BIN) * V_BIN)
            if not region_name:
                region_name = " ".join(row[i_r].split()[:-1])

    channels.sort()
    raw_energies = np.array(sorted(raw_energies))
    voltages = np.array(sorted(voltages))
    n_ch = len(channels)
    n_v = len(voltages)

    # common corrected energy grid
    e_min = raw_energies.min() + min(corrections.values())
    e_max = raw_energies.max() + max(corrections.values())
    e_grid = np.arange(np.floor(e_min / E_GRID_STEP) * E_GRID_STEP,
                       e_max + E_GRID_STEP, E_GRID_STEP)
    n_e = len(e_grid)
    print(f"Ir data: {n_ch} channels, raw BE {raw_energies[0]:g}..{raw_energies[-1]:g} eV, "
          f"{n_v} voltage bins, corrected grid {e_grid[0]:.1f}..{e_grid[-1]:.1f} eV ({n_e} pts)")
    print(f"Region: {region_name}")

    # --- pass 2: accumulate per (channel, voltage_bin, raw_energy) ---
    # sums[(ch, v_bin, raw_e)] = [sum_counts, n]
    sums = {}
    with open(IR_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(row for row in f if not row.startswith("#"))
        header = next(reader)
        i_v = header.index("voltage_Ewe_V")
        i_e = header.index("binding_energy_eV")
        i_c = header.index("counts_s")
        i_ch = header.index("channel")
        for row in reader:
            ch = int(row[i_ch])
            e = float(row[i_e])
            v = round(float(row[i_v]) / V_BIN) * V_BIN
            c = float(row[i_c])
            key = (ch, v, e)
            if key in sums:
                sums[key][0] += c; sums[key][1] += 1
            else:
                sums[key] = [c, 1]
    print(f"Accumulated {len(sums)} (ch, V, E) cells")

    # --- build per-channel voltage-averaged corrected spectra on common grid ---
    ch_spectra = np.full((n_ch, n_e), np.nan)  # voltage-averaged, corrected, on grid
    heatmap = np.zeros((n_v, n_e))
    heatmap_n = np.zeros((n_v, n_e))

    for ci, ch in enumerate(channels):
        shift = corrections.get(ch, 0.0)
        # raw energies for this channel
        ch_raw_e = []
        ch_vals_per_v = {}  # v_bin -> (raw_e array, counts array)
        for (c, v, e), (s, n) in sums.items():
            if c == ch:
                ch_raw_e.append(e)
                ch_vals_per_v.setdefault(v, ([], []))
                ch_vals_per_v[v][0].append(e)
                ch_vals_per_v[v][1].append(s / n)
        if not ch_raw_e:
            continue
        ch_raw_e = np.array(sorted(set(ch_raw_e)))
        # voltage-averaged spectrum on raw energy axis
        v_avg_raw = np.zeros(len(ch_raw_e))
        v_cnt = np.zeros(len(ch_raw_e))
        for v, (es, cs) in ch_vals_per_v.items():
            es = np.array(es); cs = np.array(cs)
            for ei, e in enumerate(ch_raw_e):
                mask = es == e
                if mask.any():
                    v_avg_raw[ei] += cs[mask].mean()
                    v_cnt[ei] += 1
        v_avg_raw = np.where(v_cnt > 0, v_avg_raw / np.maximum(v_cnt, 1), np.nan)
        # corrected energies and interpolate onto common grid
        e_corr = ch_raw_e + shift
        order = np.argsort(e_corr)
        ch_spectra[ci] = np.interp(e_grid, e_corr[order], v_avg_raw[order],
                                   left=np.nan, right=np.nan)

        # heatmap: for each voltage bin, corrected spectrum on grid
        for vi, v in enumerate(voltages):
            if v in ch_vals_per_v:
                es, cs = ch_vals_per_v[v]
                es = np.array(es); cs = np.array(cs)
                # average duplicates
                u_e, inv = np.unique(es, return_inverse=True)
                cs_avg = np.zeros(len(u_e))
                np.add.at(cs_avg, inv, cs)
                cnt = np.bincount(inv)
                cs_avg /= cnt
                e_corr_v = u_e + shift
                order = np.argsort(e_corr_v)
                h = np.interp(e_grid, e_corr_v[order], cs_avg[order],
                              left=np.nan, right=np.nan)
                valid = ~np.isnan(h)
                heatmap[vi, valid] += h[valid]
                heatmap_n[vi, valid] += 1

    heatmap = np.where(heatmap_n > 0, heatmap / np.maximum(heatmap_n, 1), np.nan)

    # --- plot 1: heatmap corrected BE vs voltage ---
    fig, ax = plt.subplots(figsize=(11, 8))
    e_edges = np.concatenate([e_grid - E_GRID_STEP/2, [e_grid[-1] + E_GRID_STEP/2]])
    v_edges = np.concatenate([voltages - V_BIN/2, [voltages[-1] + V_BIN/2]])
    mesh = ax.pcolormesh(e_edges, v_edges, np.ma.masked_invalid(heatmap),
                         shading="flat", cmap="magma")
    ax.set_xlabel("Binding energy (eV, corrected with Au reference)")
    ax.set_ylabel("Potential Ewe (V)")
    ax.set_title(f"{region_name} heatmap: corrected BE vs voltage (avg over channels)")
    ax.invert_xaxis()
    fig.colorbar(mesh, ax=ax, label="counts/s")
    fig.tight_layout()
    f1 = OUT_PREFIX + "_heatmap_corrected.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f"Saved {f1}")

    # --- plot 2: voltage-averaged corrected spectra per channel ---
    norm = Normalize(vmin=channels[0], vmax=channels[-1])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(13, 10))
    for ci, ch in enumerate(channels):
        y = ch_spectra[ci] + ci * Y_SHIFT
        color = cmap(norm(ch))
        ax.plot(e_grid, y, "-", lw=1.0, color=color, alpha=0.85)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Channel")
    ax.set_xlabel("Binding energy (eV, corrected with Au reference)")
    ax.set_ylabel(f"counts/s (voltage-averaged, corrected) + {Y_SHIFT:g} x channel index")
    ax.set_title(f"{region_name} voltage-averaged corrected spectra per channel")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = OUT_PREFIX + "_channel_spectra_corrected.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"Saved {f2}")

    # --- plot 3: overall average ---
    overall = np.nanmean(ch_spectra, axis=0)
    overall_std = np.nanstd(ch_spectra, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(e_grid, overall, "-", lw=2.5, color="darkblue",
            label="corrected spectrum (avg over channels + voltages)")
    ax.fill_between(e_grid, overall - overall_std, overall + overall_std,
                    alpha=0.2, color="blue", label="±1 std across channels")
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel("counts/s")
    ax.set_title(f"{region_name} corrected averaged spectrum (Au energy calibration applied)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = OUT_PREFIX + "_overall_corrected.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    print(f"Saved {f3}")

    # --- save CSV ---
    csv_out = OUT_PREFIX + "_overall_corrected.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["binding_energy_eV", "counts_s_mean", "counts_s_std"])
        for i, e in enumerate(e_grid):
            w.writerow([f"{e:.4f}", f"{overall[i]:.6f}", f"{overall_std[i]:.6f}"])
    print(f"Saved {csv_out}")


if __name__ == "__main__":
    main()
