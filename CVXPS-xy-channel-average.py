"""
Average corrected channel spectra after cutting low-amplitude channels.

1. Load 2-Gaussian fit params to identify channels with significantly low
   peak amplitude (amp1 below threshold = mean - 2*std, or a percentile).
2. Load per-channel corrected spectra (energy-shifted so peaks align to
   literature BE).
3. Average only the kept channels.
4. Plot: (a) amplitude vs channel with cut threshold, (b) averaged corrected
   spectrum with individual kept channels faintly in background.

Usage: python CVXPS-xy-channel-average.py [xy_file] [fit_csv] [corr_csv] [out_prefix]
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
CORR_CSV = (sys.argv[3] if len(sys.argv) > 3 else
            r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv")
OUT_PREFIX = (sys.argv[4] if len(sys.argv) > 4 else os.path.join(
    os.path.dirname(XY_FILE),
    os.path.splitext(os.path.basename(XY_FILE))[0]))

Y_SHIFT = 0.2
BE_LIT_PEAK1 = 84.0
BE_LIT_PEAK2 = 87.67


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
    channels = []
    amps = {}
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for row in r:
            if row[idx["success"]] == "True":
                ch = int(row[idx["channel"]])
                channels.append(ch)
                # use max of both peak amplitudes as channel intensity proxy
                amps[ch] = max(float(row[idx["amp1"]]), float(row[idx["amp2"]]))
    return channels, amps


def load_corrections(csv_file):
    corrections = {}
    with open(csv_file, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for row in r:
            ch = int(row[idx["channel"]])
            corrections[ch] = float(row[idx["correction_eV"]])
    return corrections


def main():
    channels, energies, matrix, region = parse_xy_channels(XY_FILE)
    fit_channels, amps = load_fit_params(FIT_CSV)
    corrections = load_corrections(CORR_CSV)
    n_ch = len(channels)
    print(f"Loaded {n_ch} channels, {len(fit_channels)} fit params, {len(corrections)} corrections")

    # --- determine amplitude threshold ---
    amp_arr = np.array([amps[ch] for ch in fit_channels if ch in amps])
    mean_amp = np.mean(amp_arr)
    std_amp = np.std(amp_arr)
    threshold = mean_amp - 2 * std_amp
    # also use percentile as fallback if threshold is too low
    p10 = np.percentile(amp_arr, 10)
    threshold = max(threshold, p10)
    print(f"Amplitude stats: mean={mean_amp:.2f}, std={std_amp:.2f}, "
          f"mean-2std={mean_amp - 2*std_amp:.2f}, p10={p10:.2f}")
    print(f"Threshold: {threshold:.2f}")

    kept = [ch for ch in channels if ch in amps and amps[ch] >= threshold]
    cut = [ch for ch in channels if ch not in kept]
    print(f"Kept {len(kept)} channels, cut {len(cut)} channels: {cut}")

    # --- plot 1: amplitude vs channel with threshold ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ch_all = np.array(channels)
    amp_vals = np.array([amps.get(ch, 0) for ch in ch_all])
    colors = ["green" if amps.get(ch, 0) >= threshold else "red" for ch in ch_all]
    ax.scatter(ch_all, amp_vals, c=colors, s=20, alpha=0.7)
    ax.axhline(threshold, color="black", ls="--", lw=1.5,
               label=f"cut threshold = {threshold:.2f}")
    ax.axhline(mean_amp, color="blue", ls=":", lw=1, label=f"mean = {mean_amp:.2f}")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Peak amplitude (max of 2 Gaussians)")
    ax.set_title(f"{region} channel amplitude (green=kept, red=cut, {len(kept)}/{n_ch} kept)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = OUT_PREFIX + "_channel_amplitudes.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f"Saved {f1}")

    # --- plot 2: corrected spectra + average ---
    norm = Normalize(vmin=min(kept), vmax=max(kept))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(12, 8))

    # faint individual corrected channels
    corrected_spectra = []
    for ci, ch in enumerate(channels):
        if ch not in kept:
            continue
        shift = corrections.get(ch, 0.0)
        e_corr = energies + shift
        y = matrix[ci] + ci * Y_SHIFT
        color = cmap(norm(ch))
        ax.plot(e_corr, y, "-", lw=0.6, color=color, alpha=0.25)
        corrected_spectra.append((e_corr, matrix[ci]))

    # average across kept channels (interpolate onto common energy grid)
    if corrected_spectra:
        # use the corrected energy grid of a middle channel as reference
        ref_e = corrected_spectra[len(corrected_spectra) // 2][0]
        spectra_interp = []
        for e_corr, y in corrected_spectra:
            yi = np.interp(ref_e, e_corr, y)
            spectra_interp.append(yi)
        avg_spectrum = np.mean(spectra_interp, axis=0)
        std_spectrum = np.std(spectra_interp, axis=0)

        # plot average as thick black line (no y-shift)
        ax.plot(ref_e, avg_spectrum, "-", lw=3, color="black",
                label=f"average of {len(kept)} corrected channels", zorder=10)
        ax.fill_between(ref_e, avg_spectrum - std_spectrum, avg_spectrum + std_spectrum,
                        alpha=0.2, color="black", label="±1 std")

    ax.axvline(BE_LIT_PEAK1, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Au 4f 7/2 = {BE_LIT_PEAK1} eV")
    ax.axvline(BE_LIT_PEAK2, color="orange", ls="--", lw=1.5, alpha=0.7,
               label=f"Au 4f 5/2 = {BE_LIT_PEAK2} eV")

    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel(f"counts/s (time-averaged, y-shifted for individuals)")
    ax.set_title(f"{region} corrected channel spectra + average "
                 f"({len(kept)} channels kept, {len(cut)} cut)")
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = OUT_PREFIX + "_averaged_corrected_spectrum.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"Saved {f2}")

    # --- plot 3: just the average spectrum clean ---
    fig, ax = plt.subplots(figsize=(10, 6))
    if corrected_spectra:
        ax.plot(ref_e, avg_spectrum, "-", lw=2.5, color="black", label="averaged spectrum")
        ax.fill_between(ref_e, avg_spectrum - std_spectrum, avg_spectrum + std_spectrum,
                        alpha=0.2, color="gray", label="±1 std across channels")
    ax.axvline(BE_LIT_PEAK1, color="red", ls="--", lw=1.5, alpha=0.7,
               label=f"Au 4f 7/2 = {BE_LIT_PEAK1} eV")
    ax.axvline(BE_LIT_PEAK2, color="orange", ls="--", lw=1.5, alpha=0.7,
               label=f"Au 4f 5/2 = {BE_LIT_PEAK2} eV")
    ax.set_xlabel("Binding energy (eV, corrected)")
    ax.set_ylabel("counts/s (time-averaged)")
    ax.set_title(f"{region} averaged corrected spectrum ({len(kept)} channels)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = OUT_PREFIX + "_averaged_corrected_spectrum_clean.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    print(f"Saved {f3}")

    # --- save average to CSV ---
    avg_csv = OUT_PREFIX + "_averaged_corrected_spectrum.csv"
    with open(avg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["binding_energy_eV", "counts_s_mean", "counts_s_std", "n_channels"])
        for i, e in enumerate(ref_e):
            w.writerow([f"{e:.4f}", f"{avg_spectrum[i]:.6f}", f"{std_spectrum[i]:.6f}", len(kept)])
    print(f"Saved {avg_csv}")

    # --- save kept/cut channel list ---
    list_csv = OUT_PREFIX + "_kept_channels.csv"
    with open(list_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "amplitude", "status"])
        for ch in channels:
            a = amps.get(ch, 0)
            w.writerow([ch, f"{a:.4f}", "kept" if ch in kept else "cut"])
    print(f"Saved {list_csv}")


if __name__ == "__main__":
    main()
