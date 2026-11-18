"""
Correlate an EC-Lab CV export (.mpt) with a SpecsLab Prodigy fixed-energy
XPS series (.xy) with respect to absolute time, and export one combined file.

Time bases:
  - .mpt: "Acquisition started on : 11/18/2026 15:58:37.946" is LOCAL time;
    each data row has an elapsed time/s. Absolute time = start + time/s.
  - .xy : each scan has "# Acquisition Date: 2026-11-18 14:58:53 UTC" (UTC),
    which is UTC_OFFSET_HOURS lower than the .mpt local time.

To align both on one absolute (UTC) axis, the .mpt local start is shifted
by -UTC_OFFSET_HOURS to UTC; the .xy timestamps are already UTC.

For every XPS scan the CV voltage (Ewe/V) and current (<I>/mA) are linearly
interpolated at the scan's absolute time, and one row per XPS scan is written
to the output CSV:

    abs_time_utc, cv_time_s, voltage_Ewe_V, current_mA,
    binding_energy_eV, counts_s, error_bar, region, scan, channel

Channel-split .xy files ("# Cycle: ..., Scan: N, Channel: C") are filtered to
CH_MIN <= channel <= CH_MAX before correlating.

Output: <mpt_basename>_XPS_correlated.csv next to the .mpt file
(or the path given as the optional third CLI argument).
"""

import csv
import os
import re
import sys
from bisect import bisect_left
from datetime import datetime, timedelta, timezone

# Optional CLI overrides: python CVXPS-correlate-channel.py [mpt] [xy] [out_csv] [ch_min] [ch_max]
MPT_FILE = (sys.argv[1] if len(sys.argv) > 1 else
            r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\11_CV.mpt")
XY_FILE = (sys.argv[2] if len(sys.argv) > 2 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_50mVs_new_01.xy")
OUT_FILE = (sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(MPT_FILE),
    os.path.splitext(os.path.basename(MPT_FILE))[0] + "_XPS_correlated.csv"))

# .mpt timestamps are local time; .xy timestamps are UTC.
# Local = UTC + UTC_OFFSET_HOURS (here CET, 1 hour ahead of UTC).
UTC_OFFSET_HOURS = 1.0

# Channel-split .xy files carry "# Cycle: ..., Scan: N, Channel: C"; keep only
# scans whose channel lies in [CH_MIN, CH_MAX].
CH_MIN = int(sys.argv[4]) if len(sys.argv) > 4 else 50
CH_MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 80

UTC = timezone.utc


def parse_time(date_str, time_str):
    """Parse 'YYYY-MM-DD' + 'HH:MM:SS[.ffffff]' or 'MM/DD/YYYY' variants."""
    s = f"{date_str} {time_str}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {s!r}")


# --------------------------------------------------------------------------
# .mpt parsing
# --------------------------------------------------------------------------

def parse_mpt(filename):
    """Return (start_utc_epoch, rows) where rows = list of dicts with
    time/s, control/V, Ewe/V, <I>/mA, cycle number, abs_time_utc_epoch."""
    start_local = None
    header_seen = False
    rows = []
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = re.match(
                r"Acquisition started on\s*:\s*(\d{1,2}/\d{1,2}/\d{4})\s+"
                r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*$",
                line,
            )
            if m:
                start_local = parse_time(m.group(1), m.group(2))
                continue
            if line.startswith("mode\t"):
                header_seen = True
                continue
            if not header_seen or not line:
                continue
            fields = line.split("\t")
            if len(fields) < 10:
                continue
            try:
                time_s = float(fields[5])
                control_v = float(fields[6])
                ewe_v = float(fields[7])
                current_ma = float(fields[8])
                cycle = float(fields[9])
            except ValueError:
                continue
            rows.append({
                "time_s": time_s,
                "control_V": control_v,
                "Ewe_V": ewe_v,
                "I_mA": current_ma,
                "cycle": cycle,
            })
    if start_local is None:
        raise ValueError("no 'Acquisition started on' line found in .mpt")
    # local start -> UTC, then add elapsed time/s
    start_utc = (start_local - timedelta(hours=UTC_OFFSET_HOURS)).replace(tzinfo=UTC)
    start_epoch = start_utc.timestamp()
    for r in rows:
        r["abs_epoch"] = start_epoch + r["time_s"]
    rows.sort(key=lambda r: r["abs_epoch"])
    return start_local, start_epoch, rows


# --------------------------------------------------------------------------
# .xy parsing
# --------------------------------------------------------------------------

def parse_xy(filename):
    """Return (scans, stats). Each scan dict: abs_epoch (UTC), region,
    binding_energy_eV, counts_s, error_bar, scan, channel. Only scans with
    CH_MIN <= channel <= CH_MAX are kept."""
    scans = []
    stats = {"total": 0, "kept": 0, "no_channel": 0}
    current = None
    binding_energy = None
    scan_no = None
    scan_channel = None
    scan_timestamp = None
    in_scan = False
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            # Scan header: "# Cycle: 0, Curve: 0, Scan: 3[, Channel: 145]"
            m = re.match(
                r"#\s*Cycle:\s*\d+\s*,\s*Curve:\s*\d+\s*,\s*Scan:\s*(\d+)"
                r"(?:\s*,\s*Channel:\s*(\d+))?\s*$", line
            )
            if m:
                stats["total"] += 1
                scan_no = int(m.group(1))
                scan_channel = int(m.group(2)) if m.group(2) else None
                if scan_channel is None:
                    stats["no_channel"] += 1
                    in_scan = False
                elif CH_MIN <= scan_channel <= CH_MAX:
                    stats["kept"] += 1
                    in_scan = True
                else:
                    in_scan = False
                scan_timestamp = None
                continue

            # In-scan acquisition timestamp (UTC)
            m = re.match(
                r"#\s*Acquisition Date:\s*(\d{4}-\d{2}-\d{2})\s+"
                r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)(?:\s+(UTC))?\s*$",
                line,
            )
            if m and in_scan:
                scan_timestamp = parse_time(m.group(1), m.group(2)).replace(tzinfo=UTC)
                continue

            # Region header: "# Region: Ir 4f 60"
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                current = m.group(1)
                em = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", current)
                binding_energy = float(em.group(2)) if em else None
                in_scan = False
                scan_timestamp = None
                continue

            # Data row: index  counts/s  ErrorBar
            m = re.match(
                r"^(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line
            )
            if m and current is not None and in_scan:
                counts, err = float(m.group(2)), float(m.group(3))
                scans.append({
                    "abs_epoch": scan_timestamp.timestamp() if scan_timestamp else None,
                    "region": current,
                    "binding_energy_eV": binding_energy,
                    "counts_s": counts,
                    "error_bar": err,
                    "scan": scan_no,
                    "channel": scan_channel,
                })
    return scans, stats


# --------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------

def interpolate(x, xs, ys):
    """Linear interpolation of ys at x; clamps to endpoints outside range."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def correlate(cv_rows, xps_scans):
    cv_t = [r["abs_epoch"] for r in cv_rows]
    cv_ewe = [r["Ewe_V"] for r in cv_rows]
    cv_i = [r["I_mA"] for r in cv_rows]
    out = []
    max_dt = 0.0
    n_outside = 0
    for s in xps_scans:
        t = s["abs_epoch"]
        if t < cv_t[0] or t > cv_t[-1]:
            n_outside += 1  # no CV data at this time: drop, do not clamp
            continue
        ewe = interpolate(t, cv_t, cv_ewe)
        cur = interpolate(t, cv_t, cv_i)
        i = bisect_left(cv_t, t)
        i = min(max(i, 1), len(cv_t) - 1)
        dt = min(abs(t - cv_t[i - 1]), abs(t - cv_t[i]))
        max_dt = max(max_dt, dt)
        out.append({
            "abs_epoch": t,
            "cv_time_s": t - cv_t[0] + cv_rows[0]["time_s"],
            "voltage_Ewe_V": ewe,
            "current_mA": cur,
            "binding_energy_eV": s["binding_energy_eV"],
            "counts_s": s["counts_s"],
            "error_bar": s["error_bar"],
            "region": s["region"],
            "scan": s["scan"],
            "channel": s["channel"],
        })
    out.sort(key=lambda r: r["abs_epoch"])
    return out, max_dt, n_outside


def fmt_utc(epoch):
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export(out_rows, out_file, start_local, start_epoch, cv_rows):
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        f.write(f"# Correlated CV (.mpt) + XPS (.xy) data on absolute UTC time\n")
        f.write(f"# MPT acquisition started (local): {start_local} (local = UTC+{UTC_OFFSET_HOURS:g} h)\n")
        f.write(f"# MPT acquisition started (UTC):   {fmt_utc(start_epoch)} UTC\n")
        f.write(f"# CV rows: {len(cv_rows)}   XPS scans: {len(out_rows)}\n")
        f.write("# voltage_Ewe_V and current_mA are linearly interpolated from the CV at each XPS scan time\n")
        w = csv.writer(f)
        w.writerow([
            "abs_time_utc", "cv_time_s", "voltage_Ewe_V", "current_mA",
            "binding_energy_eV", "counts_s", "error_bar", "region", "scan",
            "channel",
        ])
        for r in out_rows:
            w.writerow([
                fmt_utc(r["abs_epoch"]), f"{r['cv_time_s']:.6f}",
                f"{r['voltage_Ewe_V']:.6f}", f"{r['current_mA']:.6f}",
                f"{r['binding_energy_eV']:g}", f"{r['counts_s']:.5f}",
                f"{r['error_bar']:.6f}", r["region"], r["scan"], r["channel"],
            ])
    print(f"wrote {out_file}  ({len(out_rows)} rows)")


def main():
    start_local, start_epoch, cv_rows = parse_mpt(MPT_FILE)
    xps_scans, ch_stats = parse_xy(XY_FILE)
    print(f"Channel filter: {ch_stats['kept']} of {ch_stats['total']} scan blocks "
          f"kept (channel {CH_MIN}-{CH_MAX}); "
          f"{ch_stats['total'] - ch_stats['kept'] - ch_stats['no_channel']} outside range, "
          f"{ch_stats['no_channel']} without channel")

    # --- compare the time bases of both files ---
    cv_end = cv_rows[-1]["abs_epoch"]
    xy_times = [s["abs_epoch"] for s in xps_scans if s["abs_epoch"] is not None]
    print("--- time comparison ---")
    print(f"MPT acquisition started (local) : {start_local}  (UTC+{UTC_OFFSET_HOURS:g} h)")
    print(f"MPT acquisition started (UTC)   : {fmt_utc(start_epoch)} UTC")
    print(f"MPT last data point (UTC)       : {fmt_utc(cv_end)} UTC "
          f"(time/s = {cv_rows[-1]['time_s']:.3f} s, {len(cv_rows)} rows)")
    print(f"XY first scan (UTC)             : {fmt_utc(min(xy_times))} UTC")
    print(f"XY last scan  (UTC)             : {fmt_utc(max(xy_times))} UTC "
          f"({len(xps_scans)} scans)")
    print(f"Offset applied: MPT local - {UTC_OFFSET_HOURS:g} h -> UTC; XY already UTC")

    out_rows, max_dt, n_outside = correlate(cv_rows, xps_scans)
    print(f"XPS scans matched to CV: {len(out_rows)} "
          f"({n_outside} dropped, outside CV time range), max |dt| to nearest CV point: {max_dt:.3f} s")

    export(out_rows, OUT_FILE, start_local, start_epoch, cv_rows)


if __name__ == "__main__":
    main()
