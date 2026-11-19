"""
Find the per-channel energy correction for a FixedEnergies .xy file using
cross-correlation (robust, no peak fitting needed).

1. Parse per-channel time-averaged spectra from the .xy file.
2. Compute the average spectrum across all channels as reference.
3. For each channel, cross-correlate its spectrum with the reference on a fine
   energy grid (via interpolation) to find the energy shift that best aligns them.
4. Linear-fit shift vs channel; compute per-channel correction to align the
   reference centroid to the literature binding energy.
5. Outputs: shift-vs-channel plot, corrected spectra plot, corrections CSV.

Usage: python CVXPS-xy-find-correction-xcorr.py [xy_file] [out_prefix] [be_lit]
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

XY_FILE = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\OCV_5mbar_Fixed_energy_Ir4f.xy")
OUT_PREFIX = (sys.argv[2] if len(sys.argv) > 2 else
              r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\OCV_5mbar_Fixed_energy_Ir4f")
BE_LIT = float(sys.argv[3]) if len(sys.argv) > 3 else 61.5  # centroid target eV

Y_SHIFT = 0.5
FINE_STEP = 0.01  # eV, fine grid for cross-correlation
MAX_SHIFT = 5.0   # eV, max allowed shift in xcorr


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


def xcorr_shift(ref_e, ref_y, ch_e, ch_y, e_range, max_shift=MAX_SHIFT, step=FINE_STEP):
    """Find energy shift that best aligns ch spectrum to ref via cross-correlation,
    restricted to e_range = (e_lo, e_hi). Returns shift in eV."""
    e_lo_r, e_hi_r = e_range
    # restrict to peak region and subtract linear background
    mask_r = (ref_e >= e_lo_r) & (ref_e <= e_hi_r)
    mask_c = (ch_e >= e_lo_r) & (ch_e <= e_hi_r)
    re, ry = ref_e[mask_r], ref_y[mask_r]
    ce, cy = ch_e[mask_c], ch_y[mask_c]
    # subtract min as background
    ry = ry - np.nanmin(ry)
    cy = cy - np.nanmin(cy)

    # fine grid covering range plus max_shift margin
    lo = min(re.min(), ce.min()) - max_shift
    hi = max(re.max(), ce.max()) + max_shift
    fine = np.arange(lo, hi + step, step)
    ref_f = np.interp(fine, re, ry, left=0, right=0)
    ch_f = np.interp(fine, ce, cy, left=0, right=0)

    ref_f = ref_f - ref_f.mean()
    ch_f = ch_f - ch_f.mean()
    denom = np.sqrt(np.sum(ref_f**2) * np.sum(ch_f**2))
    if denom == 0:
        return 0.0

    shifts = np.arange(-max_shift, max_shift + step, step)
    best_shift, best_corr = 0.0, -np.inf
    for s in shifts:
        shifted = np.interp(fine + s, fine, ch_f, left=0, right=0)
        corr = np.sum(ref_f * shifted) / denom
        if corr > best_corr:
            best_corr = corr
            best_shift = s
    return best_shift


def main():
    channels, energies, matrix, region = parse_xy_channels(XY_FILE)
    n_ch = len(channels)
    print(f"Loaded {n_ch} channels x {len(energies)} energies, region={region}")
    print(f"Target centroid BE = {BE_LIT} eV")

    # reference = average spectrum across all channels (on raw energy axis)
    with np.errstate(invalid="ignore"):
        ref_spec = np.nanmean(matrix, axis=0)
    # centroid of the peak region only (59-67 eV for Ir 4f)
    peak_mask = (energies >= 59) & (energies <= 67)
    ref_peak = ref_spec.copy()
    ref_peak[~peak_mask] = np.nan
    with np.errstate(invalid="ignore"):
        ref_centroid = np.nansum(energies[peak_mask] * ref_peak[peak_mask]) / np.nansum(ref_peak[peak_mask])
    print(f"Reference peak centroid (59-67 eV): {ref_centroid:.3f} eV")

    # --- per-channel shift via cross-correlation on peak region ---
    shifts = np.zeros(n_ch)
    E_RANGE = (59.0, 67.0)  # Ir 4f peak region
    for ci, ch in enumerate(channels):
        if ci % 20 == 0:
            print(f"  xcorr ch{ch} ({ci+1}/{n_ch})...")
        shifts[ci] = xcorr_shift(energies, ref_spec, energies, matrix[ci], E_RANGE)
    print(f"Shifts: {shifts.min():.3f}..{shifts.max():.3f} eV")

    # --- linear fit shift vs channel ---
    ch_arr = np.array(channels)
    A = np.vstack([ch_arr, np.ones(n_ch)]).T
    a, b = np.linalg.lstsq(A, shifts, rcond=None)[0]
    pred = a * ch_arr + b
    resid = shifts - pred
    print(f"Linear fit: shift = {a:.5f}*ch + {b:.3f}, resid std = {resid.std():.4f} eV")

    # --- correction: shift each channel so its centroid matches literature ---
    # channel centroid after correction = ref_centroid + shift_ch + correction
    # we want this = BE_LIT, so correction = BE_LIT - ref_centroid - shift_ch
    corrections = BE_LIT - ref_centroid - pred  # use linear prediction for smoothness

    # --- plot 1: shift vs channel ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(ch_arr, shifts, "o", ms=4, alpha=0.6, label="xcorr shift")
    ax.plot(ch_arr, pred, "-", lw=2, color="blue", label=f"linear: {a:.5f}*ch+{b:.3f}")
    ax.set_xlabel("Channel"); ax.set_ylabel("Energy shift (eV)")
    ax.set_title("Cross-correlation shift vs channel"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ch_arr, corrections, "o-", ms=3, lw=1, color="green")
    ax.set_xlabel("Channel"); ax.set_ylabel("Correction (eV)")
    ax.set_title(f"Per-channel correction (target centroid {BE_LIT} eV)"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(resid, bins=30, color="gray", alpha=0.7)
    ax.set_xlabel("Residual (eV)"); ax.set_ylabel("Count")
    ax.set_title(f"Linear fit residuals (std={resid.std():.4f} eV)"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    # corrected spectra overlay (all channels on common corrected axis)
    norm0 = Normalize(0, n_ch - 1)
    cmap0 = plt.get_cmap("viridis")
    e_corr_grid = np.arange(energies.min() - 5, energies.max() + 5, 0.05)
    for ci, ch in enumerate(channels):
        e_c = energies + corrections[ci]
        o = np.argsort(e_c)
        y_interp = np.interp(e_corr_grid, e_c[o], matrix[ci][o], left=np.nan, right=np.nan)
        ax.plot(e_corr_grid, y_interp, "-", lw=0.8, color=cmap0(norm0(ci)), alpha=0.5)
    ax.axvline(BE_LIT, color="red", ls="--", label=f"lit {BE_LIT}")
    ax.set_xlabel("Corrected BE (eV)"); ax.set_ylabel("counts/s")
    ax.set_title("Corrected spectra overlay"); ax.legend(); ax.invert_xaxis(); ax.grid(alpha=0.3)

    fig.suptitle(f"{region} energy correction from {os.path.basename(XY_FILE)} (xcorr)")
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_xcorr_correction.png")

    # --- plot 2: corrected spectra stacked ---
    fig, ax = plt.subplots(figsize=(13, 10))
    norm = Normalize(vmin=channels[0], vmax=channels[-1])
    cmap = plt.get_cmap("viridis")
    for ci, ch in enumerate(channels):
        e_c = energies + corrections[ci]
        o = np.argsort(e_c)
        y = matrix[ci][o] + ci * Y_SHIFT
        ax.plot(e_c[o], y, "-", lw=1.0, color=cmap(norm(ch)), alpha=0.85)
    ax.axvline(BE_LIT, color="red", ls="--", lw=1.5, alpha=0.7, label=f"lit {BE_LIT} eV")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Channel")
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"counts/s + {Y_SHIFT:g} x channel idx")
    ax.set_title(f"{region} corrected per-channel spectra (xcorr, {n_ch} ch)")
    ax.invert_xaxis(); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, OUT_PREFIX + "_xcorr_corrected_spectra.png")

    # --- save corrections CSV ---
    csv_out = OUT_PREFIX + "_energy_corrections_xcorr.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "correction_eV", "xcorr_shift", "linear_pred", "residual"])
        for ci, ch in enumerate(channels):
            w.writerow([ch, f"{corrections[ci]:.4f}", f"{shifts[ci]:.4f}",
                        f"{pred[ci]:.4f}", f"{resid[ci]:.4f}"])
    print(f"Saved {csv_out}")

    # summary
    print(f"\nCorrection function: correction(ch) = {BE_LIT} - {ref_centroid:.3f} - ({a:.5f}*ch + {b:.3f})")
    print(f"  = {-a:.5f}*ch + {BE_LIT - ref_centroid - b:.3f}")
    print(f"Range: ch0={corrections[0]:+.3f} eV, ch145={corrections[-1]:+.3f} eV")


if __name__ == "__main__":
    main()
