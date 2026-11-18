"""
Plot an XPS spectrum from a SpecsLab Prodigy fixed-energy .xy series.

The .xy file contains one '# Region: <name> <binding energy eV>' block per
binding energy; inside each block every '# Cycle: ..., Scan: N' section holds
data rows 'index  counts/s  ErrorBar' with its own '# Acquisition Date' (UTC).
All regions are measured in interleaved cycles, so each cycle contributes one
datapoint per binding energy.

Two modes (optional third CLI argument N):
  - default : average ALL counts/s datapoints of each binding energy
              (error bars show the standard error of the mean).
  - N given : use only the first N datapoints in time (by acquisition date).

Usage: python CVXPS-average-spectrum.py [xy_file] [out_png] [N_first]
Output: <xy_basename>_averaged_spectrum.png or <xy_basename>_first<N>_spectrum.png
        in ./CVXPS/ by default.
"""

import os
import re
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

XY_FILE = (sys.argv[1] if len(sys.argv) > 1 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_02.xy")
N_FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else None
_base = os.path.splitext(os.path.basename(XY_FILE))[0]
OUT_FILE = (sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "CVXPS",
    f"{_base}_first{N_FIRST}_spectrum.png" if N_FIRST else f"{_base}_averaged_spectrum.png"))

UTC = timezone.utc


def parse_time(date_str, time_str):
    s = f"{date_str} {time_str}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {s!r}")


def parse_xy(filename):
    """Return list of datapoints (epoch_utc, binding_energy_eV, counts_s)."""
    points = []
    current_be = None
    scan_ts = None
    in_scan = False
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                em = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", m.group(1))
                current_be = float(em.group(2)) if em else None
                in_scan = False
                continue
            if re.match(r"#\s*Cycle:\s*\d+\s*,\s*Curve:\s*\d+\s*,\s*Scan:\s*\d+\s*$", line):
                in_scan = True
                scan_ts = None
                continue
            m = re.match(r"#\s*Acquisition Date:\s*(\d{4}-\d{2}-\d{2})\s+"
                         r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)(?:\s+UTC)?\s*$", line)
            if m and in_scan:
                scan_ts = parse_time(m.group(1), m.group(2))
                continue
            m = re.match(r"^(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line)
            if m and current_be is not None and in_scan:
                points.append((scan_ts, current_be, float(m.group(2))))
    return points


def fmt_utc(epoch):
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def main():
    points = parse_xy(XY_FILE)
    if not points:
        raise ValueError(f"no XPS datapoints found in {XY_FILE}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    if N_FIRST is not None:
        # first N datapoints in time (by acquisition date)
        points.sort(key=lambda p: (p[0] is None, p[0]))
        sel = points[:N_FIRST]
        be = np.array([p[1] for p in sel])
        counts = np.array([p[2] for p in sel])
        order = np.argsort(be)
        ax.plot(be[order], counts[order], "o-", color="tab:blue", ms=6, zorder=2)
        for i, (ts, b, c) in enumerate(sel):
            ax.annotate(str(i + 1), (b, c), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")
        t0 = fmt_utc(sel[0][0]) if sel[0][0] else "?"
        t1 = datetime.fromtimestamp(sel[-1][0], tz=UTC).strftime("%H:%M:%S") if sel[-1][0] else "?"
        ax.set_title(f"First {len(sel)} XPS datapoints in time: {os.path.basename(XY_FILE)}\n"
                     f"{t0} – {t1} UTC (numbers = time order)")
        print(f"first {len(sel)} datapoints in time:")
        for i, (ts, b, c) in enumerate(sel):
            print(f"  {i + 1:2d}. {fmt_utc(ts) if ts else '?'} UTC  "
                  f"{b:6.2f} eV : {c:10.2f} counts/s")
    else:
        # average of all datapoints per binding energy
        regions = {}
        for _, b, c in points:
            regions.setdefault(b, []).append(c)
        be = np.array(sorted(regions))
        mean = np.array([np.mean(regions[b]) for b in be])
        sem = np.array([np.std(regions[b], ddof=1) / np.sqrt(len(regions[b]))
                        if len(regions[b]) > 1 else 0.0 for b in be])
        n = np.array([len(regions[b]) for b in be])
        total = int(n.sum())
        for b in be:
            ax.scatter(np.full(len(regions[b]), b), regions[b],
                       s=6, color="0.75", zorder=1)
        ax.errorbar(be, mean, yerr=sem, fmt="o-", color="tab:blue",
                    ecolor="tab:blue", elinewidth=1, capsize=3, ms=5, zorder=2,
                    label=f"mean of all scans ± SEM (n = {n.min()}–{n.max()} per point)")
        ax.legend()
        ax.set_title(f"Averaged XPS spectrum: {os.path.basename(XY_FILE)}\n"
                     f"{total} datapoints over {len(be)} binding energies")
        print(f"{total} datapoints over {len(be)} binding energies "
              f"({be.min():g}–{be.max():g} eV)")
        for b, m_, s_, nn in zip(be, mean, sem, n):
            print(f"  {b:6.2f} eV : mean {m_:10.2f} counts/s  SEM {s_:8.2f}  (n={nn})")

    ax.set_xlabel("Binding energy / eV")
    ax.set_ylabel("Intensity / counts s$^{-1}$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"saved plot to {OUT_FILE}")


if __name__ == "__main__":
    main()
