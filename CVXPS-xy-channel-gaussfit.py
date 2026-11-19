"""
Fit each channel's time-averaged XPS spectrum with two Gaussians (Au 4f 7/2 + 5/2).

Strategy: start from the middle channel where the two peaks are best separated,
fit it to get initial parameters, then propagate the fit outward to both ends
using the previous channel's parameters as starting guesses (batch fit).

Outputs:
  1. <base>_channel_fits.png       - all channel spectra with fitted curves overlaid
  2. <base>_fit_params_vs_channel.png - both peaks' parameters in one plot
  3. <base>_fit_param_<name>.png   - separate plots for each parameter vs channel

Usage: python CVXPS-xy-channel-gaussfit.py [xy_file] [out_prefix]
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

XY_FILE = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy.xy")
OUT_PREFIX = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(XY_FILE),
    os.path.splitext(os.path.basename(XY_FILE))[0]))

Y_SHIFT = 0.2  # same as channel spectra plot


def parse_xy_channels(filename):
    """Parse FixedEnergies .xy. Return (channels, energies, matrix, region_name)."""
    channels, energies = [], []
    seen_ch, seen_e = set(), set()
    sums = {}
    binding_energy = None
    in_scan = False
    region_name = ""

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(
                r"#\s*Cycle:\s*\d+\s*,\s*Curve:\s*\d+\s*,\s*Scan:\s*(\d+)"
                r"(?:\s*,\s*Channel:\s*(\d+))?\s*$", line)
            if m:
                in_scan = m.group(2) is not None
                if in_scan and int(m.group(2)) not in seen_ch:
                    seen_ch.add(int(m.group(2)))
                    channels.append(int(m.group(2)))
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
                ch = int(re.search(r"Channel:\s*(\d+)", line).group(1)) if "Channel:" in line else None
                # we need the channel from the last scan header; re-parse
                # (simpler: track scan_channel)
                pass

    # simpler second pass with explicit channel tracking
    channels, energies = [], []
    seen_ch, seen_e = set(), set()
    sums = {}
    binding_energy = None
    scan_channel = None
    in_scan = False

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
    """Sum of two Gaussians plus constant offset."""
    g1 = a1 * np.exp(-0.5 * ((x - c1) / w1) ** 2)
    g2 = a2 * np.exp(-0.5 * ((x - c2) / w2) ** 2)
    return g1 + g2 + offset


def fit_channel(energies, counts, p0=None):
    """Fit one channel spectrum with two Gaussians. Returns params array or None."""
    # mask out NaN
    mask = ~np.isnan(counts)
    x = energies[mask]
    y = counts[mask]
    if len(x) < 6:
        return None

    if p0 is None:
        # initial guess: peak1 at max position, peak2 ~3.5 eV higher (Au 4f spin-orbit)
        i_max = np.argmax(y)
        c1 = x[i_max]
        a1 = y[i_max] - np.min(y)
        w1 = 0.5
        c2 = c1 + 3.5
        a2 = a1 * 0.7
        w2 = 0.5
        offset = np.min(y)
        p0 = [a1, c1, w1, a2, c2, w2, offset]

    # bounds: amplitudes >= 0, centers within range, widths positive
    lb = [0, x.min(), 0.05, 0, x.min(), 0.05, -np.inf]
    ub = [np.inf, x.max(), 5, np.inf, x.max(), 5, np.inf]
    try:
        popt, _ = curve_fit(two_gaussians, x, y, p0=p0, bounds=(lb, ub), maxfev=10000)
        return popt
    except Exception as e:
        print(f"  fit failed: {e}")
        return None


def main():
    channels, energies, matrix, region = parse_xy_channels(XY_FILE)
    n_ch = len(channels)
    print(f"Loaded {matrix.size} cells: {n_ch} channels x {len(energies)} energies")
    print(f"Region: {region}, channels {channels[0]}..{channels[-1]}")

    # --- batch fit: start from middle, propagate to both ends ---
    mid = n_ch // 2
    print(f"Starting batch fit from middle channel {channels[mid]}...")

    fits = [None] * n_ch  # params per channel

    # fit middle first
    p_mid = fit_channel(energies, matrix[mid])
    if p_mid is None:
        # fallback initial guess
        p_mid = None
    fits[mid] = p_mid

    # propagate to higher channels (mid+1 .. end)
    p_prev = p_mid
    for ci in range(mid + 1, n_ch):
        p = fit_channel(energies, matrix[ci], p0=p_prev)
        fits[ci] = p
        if p is not None:
            p_prev = p
        if ci % 20 == 0:
            print(f"  fitted ch{channels[ci]} ({ci+1}/{n_ch})")

    # propagate to lower channels (mid-1 .. 0)
    p_prev = p_mid
    for ci in range(mid - 1, -1, -1):
        p = fit_channel(energies, matrix[ci], p0=p_prev)
        fits[ci] = p
        if p is not None:
            p_prev = p
        if ci % 20 == 0:
            print(f"  fitted ch{channels[ci]} ({ci+1}/{n_ch})")

    n_ok = sum(1 for f in fits if f is not None)
    print(f"Fitted {n_ok}/{n_ch} channels successfully")

    # --- plot 1: all spectra with fitted curves ---
    norm = Normalize(vmin=channels[0], vmax=channels[-1])
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(14, 10))
    for ci, ch in enumerate(channels):
        y = matrix[ci] + ci * Y_SHIFT
        color = cmap(norm(ch))
        ax.plot(energies, y, "-", lw=0.8, color=color, alpha=0.6)
        if fits[ci] is not None:
            yfit = two_gaussians(energies, *fits[ci]) + ci * Y_SHIFT
            ax.plot(energies, yfit, "-", lw=1.5, color=color, alpha=0.95)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Channel")
    ax.set_xlabel("Binding energy (eV)")
    ax.set_ylabel(f"counts/s + {Y_SHIFT:g} x channel index")
    ax.set_title(f"{region} per-channel spectra with 2-Gaussian fits "
                 f"({n_ok}/{n_ch} fitted)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = OUT_PREFIX + "_channel_fits.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f"Saved {f1}")

    # --- extract parameters ---
    # params: a1, c1, w1, a2, c2, w2, offset
    param_names = ["amp1", "center1", "width1", "amp2", "center2", "width2", "offset"]
    params = np.full((n_ch, len(param_names)), np.nan)
    for ci in range(n_ch):
        if fits[ci] is not None:
            params[ci] = fits[ci]

    ch_arr = np.array(channels)

    # --- plot 2: both peaks' parameters in one plot (two subplots) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    colors_p = ["#1f77b4", "#d62728"]  # peak1 blue, peak2 red
    for pi, pname in enumerate(param_names[:6]):  # first 6 params (both peaks)
        ax = axes[0] if pi < 3 else axes[1]
        peak_label = "peak1 (4f 7/2)" if pi < 3 else "peak2 (4f 5/2)"
        color = colors_p[0] if pi < 3 else colors_p[1]
        ax.plot(ch_arr, params[:, pi], "o-", ms=3, lw=1, color=color, alpha=0.8,
                label=f"{pname} {peak_label}")
    axes[0].set_ylabel("peak 1 parameters")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{region} 2-Gaussian fit parameters vs channel")
    axes[1].set_ylabel("peak 2 parameters")
    axes[1].set_xlabel("Channel")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    f2 = OUT_PREFIX + "_fit_params_vs_channel.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"Saved {f2}")

    # --- plot 3: separate plots for each parameter ---
    for pi, pname in enumerate(param_names):
        fig, ax = plt.subplots(figsize=(10, 6))
        for peak in range(2):
            idx = peak * 3 + pi if pi < 3 else None
            if pi < 3:
                ax.plot(ch_arr, params[:, peak * 3 + pi], "o-", ms=3, lw=1,
                        label=f"peak{peak+1} {pname}")
            else:
                ax.plot(ch_arr, params[:, pi], "o-", ms=3, lw=1, color="gray",
                        label=pname)
        ax.set_xlabel("Channel")
        ax.set_ylabel(pname)
        ax.set_title(f"{region} fit parameter '{pname}' vs channel")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f3 = OUT_PREFIX + f"_fit_param_{pname}.png"
        fig.savefig(f3, dpi=150)
        plt.close(fig)
        print(f"Saved {f3}")

    # --- save fit results to CSV ---
    import csv
    csv_file = OUT_PREFIX + "_fit_params.csv"
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel"] + param_names + ["success"])
        for ci, ch in enumerate(channels):
            row = [ch] + (list(params[ci]) if fits[ci] is not None else [np.nan]*len(param_names))
            row.append(fits[ci] is not None)
            w.writerow(row)
    print(f"Saved {csv_file}")


if __name__ == "__main__":
    main()
