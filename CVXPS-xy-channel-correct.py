"""
Correct per-channel XPS spectra using linear fits of fitted peak positions.

1. Load 2-Gaussian fit parameters (CSV from CVXPS-xy-channel-gaussfit.py).
2. Fit peak center vs channel with a linear function:
   - peak2 (4f 5/2): all channels
   - peak1 (4f 7/2): only channels > 30
3. Compute per-channel energy shift so that each peak lands at its literature
   binding energy (Au 4f 7/2 = 84.0 eV, Au 4f 5/2 = 87.67 eV).
4. Shift each channel's spectrum by that amount and plot corrected spectra.

Usage: python CVXPS-xy-channel-correct.py [xy_file] [fit_csv] [out_prefix]
"""

import csv
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

XY_FILE = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy.xy")
FIT_CSV = (sys.argv[2] if len(sys.argv) > 2 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_fit_params.csv")
OUT_PREFIX = (sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(XY_FILE),
    os.path.splitext(os.path.basename(XY_FILE))[0]))

Y_SHIFT = 0.2

# Literature binding energies for Au 4f
BE_LIT_PEAK1 = 84.0    # Au 4f 7/2
BE_LIT_PEAK2 = 87.67   # Au 4f 5/2

# Channel threshold for peak1 fit
PEAK1_MIN_CHANNEL = 30


def parse_xy_channels(filename):
    """Parse FixedEnergies .xy. Return (channels, energies, matrix, region_name)."""
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


def load_fit_params(csv_file):
    """Return (channels, params dict per channel)."""
    channels = []
    params = {}
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for row in r:
            if row[idx["success"]] == "True":
                ch = int(row[idx["channel"]])
                channels.append(ch)
                params[ch] = {
                    "amp1": float(row[idx["amp1"]]),
                    "center1": float(row[idx["center1"]]),
                    "width1": float(row[idx["width1"]]),
                    "amp2": float(row[idx["amp2"]]),
                    "center2": float(row[idx["center2"]]),
                    "width2": float(row[idx["width2"]]),
                    "offset": float(row[idx["offset"]]),
                }
    return channels, params


def linear_fit(x, y):
    """Fit y = a*x + b. Return (a, b)."""
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a, b


def main():
    channels, energies, matrix, region = parse_xy_channels(XY_FILE)
    fit_channels, fit_params = load_fit_params(FIT_CSV)
    n_ch = len(channels)
    print(f"Loaded {n_ch} channels, {len(fit_channels)} fit params")

    # --- linear fits of peak centers vs channel ---
    ch_arr = np.array(fit_channels)
    c1 = np.array([fit_params[ch]["center1"] for ch in fit_channels])
    c2 = np.array([fit_params[ch]["center2"] for ch in fit_channels])

    # peak2: all channels
    a2, b2 = linear_fit(ch_arr, c2)
    # peak1: only channels > 30
    mask1 = ch_arr > PEAK1_MIN_CHANNEL
    a1, b1 = linear_fit(ch_arr[mask1], c1[mask1])

    print(f"Peak1 (ch>{PEAK1_MIN_CHANNEL}): center = {a1:.4f}*ch + {b1:.2f}")
    print(f"Peak2 (all ch):          center = {a2:.4f}*ch + {b2:.2f}")

    # --- compute per-channel energy correction ---
    # For each channel, predicted centers from linear fit; shift so they match literature
    corrections = {}
    for ci, ch in enumerate(channels):
        pred_c1 = a1 * ch + b1
        pred_c2 = a2 * ch + b2
        # Use peak2 correction for all channels; for ch<=30 use peak2 only
        # Average the two shifts when both are available
        if ch > PEAK1_MIN_CHANNEL and ch in fit_params:
            shift1 = BE_LIT_PEAK1 - pred_c1
            shift2 = BE_LIT_PEAK2 - pred_c2
            shift = 0.5 * (shift1 + shift2)  # average both
        else:
            shift = BE_LIT_PEAK2 - pred_c2
        corrections[ch] = shift

    # --- plot: correction and linear fits ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # linear fit peak1
    ax = axes[0, 0]
    ax.plot(ch_arr[mask1], c1[mask1], "o", ms=4, alpha=0.6, label="fitted center1")
    xf = np.linspace(ch_arr[mask1].min(), ch_arr[mask1].max(), 100)
    ax.plot(xf, a1 * xf + b1, "-", lw=2, color="blue", label=f"linear fit: {a1:.4f}*ch+{b1:.2f}")
    ax.axhline(BE_LIT_PEAK1, color="red", ls="--", label=f"lit {BE_LIT_PEAK1} eV")
    ax.set_xlabel("Channel"); ax.set_ylabel("Peak 1 center (eV)")
    ax.set_title(f"Peak 1 (4f 7/2) center vs channel (ch>{PEAK1_MIN_CHANNEL})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # linear fit peak2
    ax = axes[0, 1]
    ax.plot(ch_arr, c2, "o", ms=4, alpha=0.6, label="fitted center2")
    xf = np.linspace(ch_arr.min(), ch_arr.max(), 100)
    ax.plot(xf, a2 * xf + b2, "-", lw=2, color="blue", label=f"linear fit: {a2:.4f}*ch+{b2:.2f}")
    ax.axhline(BE_LIT_PEAK2, color="red", ls="--", label=f"lit {BE_LIT_PEAK2} eV")
    ax.set_xlabel("Channel"); ax.set_ylabel("Peak 2 center (eV)")
    ax.set_title("Peak 2 (4f 5/2) center vs channel (all)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # correction per channel
    ax = axes[1, 0]
    ch_all = np.array(channels)
    shifts = np.array([corrections[ch] for ch in ch_all])
    ax.plot(ch_all, shifts, "o-", ms=3, lw=1, color="green")
    ax.set_xlabel("Channel"); ax.set_ylabel("Energy correction (eV)")
    ax.set_title("Per-channel energy correction (shift to literature BE)")
    ax.grid(alpha=0.3)

    # spin-orbit splitting after correction
    ax = axes[1, 1]
    so = []
    for ch in ch_all:
        if ch in fit_params:
            pred_c1 = a1 * ch + b1 if ch > PEAK1_MIN_CHANNEL else None
            pred_c2 = a2 * ch + b2
            c1_corr = (pred_c1 + corrections[ch]) if pred_c1 else None
            c2_corr = pred_c2 + corrections[ch]
            if c1_corr:
                so.append((ch, c2_corr - c1_corr))
    if so:
        ax.plot([s[0] for s in so], [s[1] for s in so], "o-", ms=3, lw=1, color="purple")
    ax.axhline(3.67, color="red", ls="--", label="lit 3.67 eV")
    ax.set_xlabel("Channel"); ax.set_ylabel("Spin-orbit splitting (eV)")
    ax.set_title("Corrected spin-orbit splitting vs channel")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    f_corr = OUT_PREFIX + "_correction.png"
    fig.savefig(f_corr, dpi=150)
    plt.close(fig)
    print(f"Saved {f_corr}")

    # --- plot corrected spectra ---
    norm = Normalize(vmin=channels[0], vmax=channels[-1])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(14, 10))
    for ci, ch in enumerate(channels):
        shift = corrections[ch]
        e_corr = energies + shift  # shift energy axis
        y = matrix[ci] + ci * Y_SHIFT
        color = cmap(norm(ch))
        ax.plot(e_corr, y, "-", lw=1.0, color=color, alpha=0.85)

    # literature lines
    ax.axvline(BE_LIT_PEAK1, color="red", ls="--", lw=1.5, alpha=0.7, label=f"Au 4f 7/2 = {BE_LIT_PEAK1} eV")
    ax.axvline(BE_LIT_PEAK2, color="orange", ls="--", lw=1.5, alpha=0.7, label=f"Au 4f 5/2 = {BE_LIT_PEAK2} eV")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Channel")
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"counts/s (time-averaged) + {Y_SHIFT:g} x channel index")
    ax.set_title(f"{region} per-channel spectra with energy correction applied "
                 f"(peaks aligned to literature)")
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f_spec = OUT_PREFIX + "_channel_spectra_corrected.png"
    fig.savefig(f_spec, dpi=150)
    plt.close(fig)
    print(f"Saved {f_spec}")

    # --- save corrections to CSV ---
    corr_csv = OUT_PREFIX + "_energy_corrections.csv"
    with open(corr_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "correction_eV", "pred_center1", "pred_center2",
                    "corrected_center1", "corrected_center2"])
        for ch in channels:
            pred_c1 = a1 * ch + b1
            pred_c2 = a2 * ch + b2
            corr = corrections[ch]
            w.writerow([ch, f"{corr:.4f}", f"{pred_c1:.4f}", f"{pred_c2:.4f}",
                        f"{pred_c1 + corr:.4f}", f"{pred_c2 + corr:.4f}"])
    print(f"Saved {corr_csv}")


if __name__ == "__main__":
    main()
