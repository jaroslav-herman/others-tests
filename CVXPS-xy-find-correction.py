"""
Find the energy correction function for a FixedEnergies .xy file with channel-split
scans (e.g. Ir 4f at OCV). Pipeline:

1. Parse per-channel time-averaged spectra from the .xy file.
2. Fit each channel with two Gaussians (batch fit from middle channel outward).
3. Linear-fit peak centers vs channel (peak2 all channels, peak1 ch > 30).
4. Compute per-channel energy shift to align peaks to literature binding energies.
5. Outputs: fit overlay, params-vs-channel plots, correction plot, corrections CSV.

Usage: python CVXPS-xy-find-correction.py [xy_file] [out_prefix] [be_lit_peak1] [be_lit_peak2]
"""

import csv
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

XY_FILE = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\OCV_5mbar_Fixed_energy_Ir4f.xy")
OUT_PREFIX = (sys.argv[2] if len(sys.argv) > 2 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\OCV_5mbar_Fixed_energy_Ir4f")
BE_LIT_PEAK1 = float(sys.argv[3]) if len(sys.argv) > 3 else 60.8   # Ir 4f 7/2
BE_LIT_PEAK2 = float(sys.argv[4]) if len(sys.argv) > 4 else 63.8   # Ir 4f 5/2

Y_SHIFT = 0.5
PEAK1_MIN_CHANNEL = 30


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


def parse_xy_channels(filename):
    channels, energies = [], []
    seen_ch, seen_e = set(), set()
    sums = {}
    binding_energy = None
    scan_channel = None
    in_scan = False
    region_name = ""

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(
                r"#\s*Cycle:\s*\d+\s*,\s*Curve:\s*\d+\s*,\s*Scan:\s*(\d+)"
                r"(?:\s*,\s*Channel:\s*(\d+))?\s*$", line)
            if m:
                scan_channel = int(m.group(2)) if m.group(2) else None
                in_scan = scan_channel is not None
                continue
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                em = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", m.group(1))
                binding_energy = float(em.group(2)) if em else None
                if em:
                    region_name = em.group(1)
                in_scan = False
                continue
            m = re.match(r"^(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line)
            if m and in_scan and binding_energy is not None:
                counts = float(m.group(2))
                ch, e = scan_channel, binding_energy
                if ch not in seen_ch:
                    seen_ch.add(ch); channels.append(ch)
                if e not in seen_e:
                    seen_e.add(e); energies.append(e)
                key = (ch, e)
                if key in sums:
                    sums[key][0] += counts; sums[key][1] += 1
                else:
                    sums[key] = [counts, 1]

    channels.sort(); energies.sort()
    matrix = np.full((len(channels), len(energies)), np.nan)
    for ci, ch in enumerate(channels):
        for ei, e in enumerate(energies):
            s = sums.get((ch, e))
            if s:
                matrix[ci, ei] = s[0] / s[1]
    return channels, np.array(energies), matrix, region_name


def two_gaussians(x, a1, c1, w1, a2, c2, w2, offset):
    return (a1 * np.exp(-0.5 * ((x - c1) / w1) ** 2) +
            a2 * np.exp(-0.5 * ((x - c2) / w2) ** 2) + offset)


def fit_channel(energies, counts, p0=None):
    mask = ~np.isnan(counts)
    x, y = energies[mask], counts[mask]
    if len(x) < 6:
        return None
    if p0 is None:
        i_max = np.argmax(y)
        c1 = x[i_max]
        a1 = y[i_max] - np.min(y)
        p0 = [a1, c1, 0.5, a1 * 0.7, c1 + 3.0, 0.5, np.min(y)]
    lb = [0, x.min(), 0.05, 0, x.min(), 0.05, -np.inf]
    ub = [np.inf, x.max(), 5, np.inf, x.max(), 5, np.inf]
    try:
        popt, _ = curve_fit(two_gaussians, x, y, p0=p0, bounds=(lb, ub), maxfev=10000)
        return popt
    except Exception:
        return None


def linear_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def main():
    channels, energies, matrix, region = parse_xy_channels(XY_FILE)
    n_ch = len(channels)
    print(f"Loaded {n_ch} channels x {len(energies)} energies, region={region}")
    print(f"Lit BE: peak1={BE_LIT_PEAK1}, peak2={BE_LIT_PEAK2}")

    # --- batch 2-Gaussian fit from middle outward ---
    mid = n_ch // 2
    fits = [None] * n_ch
    fits[mid] = fit_channel(energies, matrix[mid])
    p_prev = fits[mid]
    for ci in range(mid + 1, n_ch):
        p = fit_channel(energies, matrix[ci], p0=p_prev)
        fits[ci] = p
        if p is not None:
            p_prev = p
    p_prev = fits[mid]
    for ci in range(mid - 1, -1, -1):
        p = fit_channel(energies, matrix[ci], p0=p_prev)
        fits[ci] = p
        if p is not None:
            p_prev = p
    n_ok = sum(1 for f in fits if f is not None)
    print(f"Fitted {n_ok}/{n_ch} channels")

    # --- plot: spectra with fits ---
    norm = Normalize(vmin=channels[0], vmax=channels[-1])
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(14, 10))
    for ci, ch in enumerate(channels):
        y = matrix[ci] + ci * Y_SHIFT
        color = cmap(norm(ch))
        ax.plot(energies, y, "-", lw=0.8, color=color, alpha=0.5)
        if fits[ci] is not None:
            ax.plot(energies, two_gaussians(energies, *fits[ci]) + ci * Y_SHIFT,
                    "-", lw=1.5, color=color, alpha=0.9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Channel")
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel(f"counts/s + {Y_SHIFT:g} x channel idx")
    ax.set_title(f"{region} per-channel spectra with 2-Gaussian fits ({n_ok}/{n_ch})")
    ax.invert_xaxis(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_channel_fits.png")

    # --- linear fits of peak centers ---
    ch_arr = np.array(channels)
    params = np.full((n_ch, 7), np.nan)
    for ci in range(n_ch):
        if fits[ci] is not None:
            params[ci] = fits[ci]
    c1 = params[:, 1]; c2 = params[:, 4]
    ok = ~np.isnan(c1)

    mask1 = ok & (ch_arr > PEAK1_MIN_CHANNEL)
    a1, b1 = linear_fit(ch_arr[mask1], c1[mask1])
    a2, b2 = linear_fit(ch_arr[ok], c2[ok])
    print(f"Peak1 (ch>{PEAK1_MIN_CHANNEL}): center = {a1:.4f}*ch + {b1:.2f}")
    print(f"Peak2 (all ok):         center = {a2:.4f}*ch + {b2:.2f}")

    # --- per-channel correction ---
    corrections = {}
    for ch in channels:
        pred_c1 = a1 * ch + b1
        pred_c2 = a2 * ch + b2
        if ch > PEAK1_MIN_CHANNEL:
            shift = 0.5 * ((BE_LIT_PEAK1 - pred_c1) + (BE_LIT_PEAK2 - pred_c2))
        else:
            shift = BE_LIT_PEAK2 - pred_c2
        corrections[ch] = shift

    # --- correction plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(ch_arr[mask1], c1[mask1], "o", ms=4, alpha=0.6)
    xf = np.linspace(ch_arr[mask1].min(), ch_arr[mask1].max(), 100)
    ax.plot(xf, a1 * xf + b1, "-", lw=2, color="blue", label=f"{a1:.4f}*ch+{b1:.2f}")
    ax.axhline(BE_LIT_PEAK1, color="red", ls="--", label=f"lit {BE_LIT_PEAK1}")
    ax.set_xlabel("Channel"); ax.set_ylabel("Peak 1 center (eV)")
    ax.set_title(f"Peak 1 (ch>{PEAK1_MIN_CHANNEL})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ch_arr[ok], c2[ok], "o", ms=4, alpha=0.6)
    xf = np.linspace(ch_arr[ok].min(), ch_arr[ok].max(), 100)
    ax.plot(xf, a2 * xf + b2, "-", lw=2, color="blue", label=f"{a2:.4f}*ch+{b2:.2f}")
    ax.axhline(BE_LIT_PEAK2, color="red", ls="--", label=f"lit {BE_LIT_PEAK2}")
    ax.set_xlabel("Channel"); ax.set_ylabel("Peak 2 center (eV)")
    ax.set_title("Peak 2 (all)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    shifts = np.array([corrections[ch] for ch in channels])
    ax.plot(channels, shifts, "o-", ms=3, lw=1, color="green")
    ax.set_xlabel("Channel"); ax.set_ylabel("Energy correction (eV)")
    ax.set_title("Per-channel correction"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    so = []
    for ch in channels:
        if ch > PEAK1_MIN_CHANNEL:
            pc1 = a1 * ch + b1; pc2 = a2 * ch + b2
            so.append((ch, (pc2 + corrections[ch]) - (pc1 + corrections[ch])))
    if so:
        ax.plot([s[0] for s in so], [s[1] for s in so], "o-", ms=3, lw=1, color="purple")
    ax.axhline(BE_LIT_PEAK2 - BE_LIT_PEAK1, color="red", ls="--",
               label=f"lit {BE_LIT_PEAK2-BE_LIT_PEAK1:.2f}")
    ax.set_xlabel("Channel"); ax.set_ylabel("Spin-orbit splitting (eV)")
    ax.set_title("Corrected splitting"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"{region} energy correction from {os.path.basename(XY_FILE)}")
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_correction.png")

    # --- save corrections CSV ---
    csv_out = OUT_PREFIX + "_energy_corrections.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "correction_eV", "pred_center1", "pred_center2",
                    "corrected_center1", "corrected_center2"])
        for ch in channels:
            pc1 = a1 * ch + b1; pc2 = a2 * ch + b2; corr = corrections[ch]
            w.writerow([ch, f"{corr:.4f}", f"{pc1:.4f}", f"{pc2:.4f}",
                        f"{pc1+corr:.4f}", f"{pc2+corr:.4f}"])
    print(f"Saved {csv_out}")

    # --- save fit params CSV ---
    fit_csv = OUT_PREFIX + "_fit_params.csv"
    with open(fit_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "amp1", "center1", "width1", "amp2", "center2", "width2", "offset", "success"])
        for ci, ch in enumerate(channels):
            row = [ch] + (list(params[ci]) if fits[ci] is not None else [np.nan]*7)
            row.append(fits[ci] is not None)
            w.writerow(row)
    print(f"Saved {fit_csv}")


if __name__ == "__main__":
    main()
