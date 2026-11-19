"""
Correlate EC-Lab CA+OCV exports (.mpt) with a SpecsLab Prodigy fixed-energy
XPS series (.xy) on absolute UTC time, for time-resolved (not voltage-resolved)
XPS during repeated CA/OCV cycles.

Two .mpt files share the same acquisition start time:
  - CA file  : chronoamperometry at 2.2 V (has Ewe/V and <I>/mA, only during CA segments)
  - OCV file : open-circuit voltage (has <Ewe/V>, only during OCV segments)
They are merged into one time series: CA rows provide current, OCV rows provide
the actual potential during OCV; the merged series is interpolated at each XPS scan.

Time bases:
  - .mpt: "Acquisition started on : 11/19/2026 15:57:24.849" is LOCAL time (CET, UTC+1);
    each data row has elapsed time/s. Absolute time = start + time/s.
  - .xy : each scan has "# Acquisition Date: 2026-11-19 15:02:21 UTC" (UTC).

Computer clock offset: EC computer is TIME_OFFSET_SECONDS ahead of XPS computer.
Positive value => subtract from EC UTC time to align with XPS UTC.

For every XPS scan (per channel, per energy region) the merged EC voltage and
current are linearly interpolated at the scan's absolute time, and one row per
(scan, channel, energy) is written to the output CSV:

    abs_time_utc, ec_time_s, cycle_time_s, voltage_Ewe_V, current_mA,
    binding_energy_eV, counts_s, error_bar, region, scan, channel

cycle_time_s = ec_time_s % CYCLE_PERIOD_S  (position within the 0-60 s cycle)

Usage:
  python CVXPS-ca-time-correlate.py <ca_mpt> <ocv_mpt> <xy> <out_csv> [ch_min] [ch_max] [time_offset_s] [cycle_period_s]

Defaults:
  CA  = 23_CA_OCV_2,2V_30s_30s_01_CA.mpt
  OCV = 23_CA_OCV_2,2V_30s_30s_02_OCV.mpt
  XY  = CA_OCV_2,2V_30s_30s_5mbar_Fixed_energy O1s.xy
  ch  = 0..145
  time_offset_s = 64  (EC ahead of XPS, subtracted)
  cycle_period_s = 60
"""

import csv
import os
import re
import sys
from bisect import bisect_left
from datetime import datetime, timedelta, timezone

# --- CLI args ---
CA_MPT = (sys.argv[1] if len(sys.argv) > 1 else
          r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\23_CA_OCV_2,2V_30s_30s_01_CA.mpt")
OCV_MPT = (sys.argv[2] if len(sys.argv) > 2 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\23_CA_OCV_2,2V_30s_30s_02_OCV.mpt")
XY_FILE = (sys.argv[3] if len(sys.argv) > 3 else
           r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CA_OCV_2,2V_30s_30s_5mbar_Fixed_energy O1s.xy")
OUT_FILE = (sys.argv[4] if len(sys.argv) > 4 else os.path.join(
    os.path.dirname(XY_FILE),
    os.path.splitext(os.path.basename(XY_FILE))[0] + "_CA_EC_correlated.csv"))
CH_MIN = int(sys.argv[5]) if len(sys.argv) > 5 else 0
CH_MAX = int(sys.argv[6]) if len(sys.argv) > 6 else 145
# EC computer is TIME_OFFSET_SECONDS ahead of XPS; subtract to align
TIME_OFFSET_SECONDS = float(sys.argv[7]) if len(sys.argv) > 7 else 64.0
CYCLE_PERIOD_S = float(sys.argv[8]) if len(sys.argv) > 8 else 60.0

UTC_OFFSET_HOURS = 1.0  # CET
UTC = timezone.utc


def parse_time(date_str, time_str):
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

def parse_mpt_start(filename):
    """Return the local start datetime from the 'Acquisition started on' line."""
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(
                r"Acquisition started on\s*:\s*(\d{1,2}/\d{1,2}/\d{4})\s+"
                r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*$",
                line.strip(),
            )
            if m:
                return parse_time(m.group(1), m.group(2))
    raise ValueError(f"no 'Acquisition started on' line in {filename}")


def parse_ca_mpt(filename):
    """Parse CA .mpt: return list of (time_s, Ewe_V, I_mA)."""
    rows = []
    header_seen = False
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("mode\t"):
                header_seen = True
                continue
            if not header_seen or not line:
                continue
            fields = line.split("\t")
            if len(fields) < 11:
                continue
            try:
                t = float(fields[7])    # time/s
                ewe = float(fields[9])  # Ewe/V
                cur = float(fields[10]) # <I>/mA
            except ValueError:
                continue
            rows.append((t, ewe, cur))
    return rows


def parse_ocv_mpt(filename):
    """Parse OCV .mpt: return list of (time_s, Ewe_V)."""
    rows = []
    header_seen = False
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("mode\t"):
                header_seen = True
                continue
            if not header_seen or not line:
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            try:
                t = float(fields[2])   # time/s
                ewe = float(fields[3]) # <Ewe/V>
            except ValueError:
                continue
            rows.append((t, ewe))
    return rows


def merge_ec(ca_rows, ocv_rows):
    """Merge CA and OCV rows into one sorted time series.
    CA rows have current; OCV rows have only voltage (current = NaN).
    During OCV periods, the CA file has no data, so OCV fills those gaps.
    Returns list of dicts: time_s, Ewe_V, I_mA (None if from OCV).
    """
    merged = []
    for t, ewe, cur in ca_rows:
        merged.append({"time_s": t, "Ewe_V": ewe, "I_mA": cur})
    for t, ewe in ocv_rows:
        merged.append({"time_s": t, "Ewe_V": ewe, "I_mA": None})
    merged.sort(key=lambda r: r["time_s"])
    return merged


# --------------------------------------------------------------------------
# .xy parsing
# --------------------------------------------------------------------------

def parse_xy(filename):
    """Return (scans, stats). Each scan dict: abs_epoch (UTC), region,
    binding_energy_eV, counts_s, error_bar, scan, channel.
    The .xy has regions at top level, each containing scan-channel blocks
    with a single data point (index 0, counts/s, ErrorBar).
    Only scans with CH_MIN <= channel <= CH_MAX are kept.
    """
    scans = []
    stats = {"total": 0, "kept": 0, "no_channel": 0}
    current_region = None
    binding_energy = None
    scan_no = None
    scan_channel = None
    scan_timestamp = None
    in_scan = False

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            # Region header (top level): "# Region: O1s 528"
            m = re.match(r"#\s*Region:\s*(.+?)\s*$", line)
            if m:
                current_region = m.group(1)
                em = re.match(r"^(.+?)\s+(-?\d+(?:\.\d+)?)\s*$", current_region)
                binding_energy = float(em.group(2)) if em else None
                # Reset scan state when region changes
                in_scan = False
                scan_timestamp = None
                continue

            # Scan header: "# Cycle: 0, Curve: 0, Scan: 3, Channel: 145"
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

            # Data row: index  counts/s  ErrorBar
            m = re.match(
                r"^(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$", line
            )
            if m and current_region is not None and in_scan:
                counts, err = float(m.group(2)), float(m.group(3))
                scans.append({
                    "abs_epoch": scan_timestamp.timestamp() if scan_timestamp else None,
                    "region": current_region,
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
    if not xs:
        return float("nan")
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


def correlate(ec_rows, start_epoch, xps_scans):
    """Interpolate EC voltage/current at each XPS scan time."""
    ec_t = [r["time_s"] for r in ec_rows]
    ec_abs = [start_epoch + r["time_s"] for r in ec_rows]
    ec_v = [r["Ewe_V"] for r in ec_rows]
    # For current: forward-fill None values from CA, or use NaN
    ec_i_raw = [r["I_mA"] for r in ec_rows]
    # Build a version with NaN for OCV rows (no current measurement)
    ec_i = [v if v is not None else float("nan") for v in ec_i_raw]

    out = []
    n_outside = 0
    max_dt = 0.0
    for s in xps_scans:
        t = s["abs_epoch"]
        if t is None:
            continue
        if t < ec_abs[0] or t > ec_abs[-1]:
            n_outside += 1
            continue
        ewe = interpolate(t, ec_abs, ec_v)
        cur = interpolate(t, ec_abs, ec_i)
        i = bisect_left(ec_abs, t)
        i = min(max(i, 1), len(ec_abs) - 1)
        dt = min(abs(t - ec_abs[i - 1]), abs(t - ec_abs[i]))
        max_dt = max(max_dt, dt)
        ec_time_s = t - start_epoch
        cycle_time_s = ec_time_s % CYCLE_PERIOD_S
        out.append({
            "abs_epoch": t,
            "ec_time_s": ec_time_s,
            "cycle_time_s": cycle_time_s,
            "voltage_Ewe_V": ewe,
            "current_mA": cur,
            "binding_energy_eV": s["binding_energy_eV"],
            "counts_s": s["counts_s"],
            "error_bar": s["error_bar"],
            "region": s["region"],
            "scan": s["scan"],
            "channel": s["channel"],
        })
    out.sort(key=lambda r: (r["abs_epoch"], r["channel"], r["binding_energy_eV"] or 0))
    return out, max_dt, n_outside


def fmt_utc(epoch):
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def export(out_rows, out_file, start_local, start_epoch, ec_rows, ca_rows, ocv_rows):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        f.write("# Correlated CA+OCV EC (.mpt) + XPS (.xy) data on absolute UTC time\n")
        f.write(f"# CA mpt:  {CA_MPT}\n")
        f.write(f"# OCV mpt: {OCV_MPT}\n")
        f.write(f"# XY:      {XY_FILE}\n")
        f.write(f"# EC acquisition started (local): {start_local} (local = UTC+{UTC_OFFSET_HOURS:g} h)\n")
        f.write(f"# EC acquisition started (UTC, before offset): {fmt_utc(start_epoch)} UTC\n")
        f.write(f"# Computer time offset: EC is {TIME_OFFSET_SECONDS:g} s ahead of XPS (subtracted)\n")
        f.write(f"# EC acquisition started (aligned UTC): {fmt_utc(start_epoch - TIME_OFFSET_SECONDS)} UTC\n")
        f.write(f"# Cycle period: {CYCLE_PERIOD_S:g} s (cycle_time_s = ec_time_s % {CYCLE_PERIOD_S:g})\n")
        f.write(f"# CA rows: {len(ca_rows)}   OCV rows: {len(ocv_rows)}   merged EC rows: {len(ec_rows)}\n")
        f.write(f"# Correlated XPS rows: {len(out_rows)}\n")
        f.write("# voltage_Ewe_V and current_mA interpolated from merged EC at each XPS scan time\n")
        w = csv.writer(f)
        w.writerow([
            "abs_time_utc", "ec_time_s", "cycle_time_s", "voltage_Ewe_V", "current_mA",
            "binding_energy_eV", "counts_s", "error_bar", "region", "scan", "channel",
        ])
        for r in out_rows:
            cur_str = f"{r['current_mA']:.6f}" if r['current_mA'] == r['current_mA'] else ""
            w.writerow([
                fmt_utc(r["abs_epoch"]), f"{r['ec_time_s']:.6f}", f"{r['cycle_time_s']:.6f}",
                f"{r['voltage_Ewe_V']:.6f}", cur_str,
                f"{r['binding_energy_eV']:g}" if r['binding_energy_eV'] is not None else "",
                f"{r['counts_s']:.5f}", f"{r['error_bar']:.6f}",
                r["region"], r["scan"], r["channel"],
            ])
    print(f"wrote {out_file}  ({len(out_rows)} rows)")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    start_ca = parse_mpt_start(CA_MPT)
    start_ocv = parse_mpt_start(OCV_MPT)
    if start_ca != start_ocv:
        print(f"WARNING: CA start ({start_ca}) != OCV start ({start_ocv}); using CA start")
    start_local = start_ca

    # EC local -> UTC, then subtract computer offset (EC ahead of XPS)
    start_utc_aligned = (start_local - timedelta(hours=UTC_OFFSET_HOURS)
                         - timedelta(seconds=TIME_OFFSET_SECONDS)).replace(tzinfo=UTC)
    start_epoch = start_utc_aligned.timestamp()

    ca_rows = parse_ca_mpt(CA_MPT)
    ocv_rows = parse_ocv_mpt(OCV_MPT)
    ec_rows = merge_ec(ca_rows, ocv_rows)
    print(f"CA rows: {len(ca_rows)}  OCV rows: {len(ocv_rows)}  merged: {len(ec_rows)}")
    print(f"EC time range: {ec_rows[0]['time_s']:.3f} .. {ec_rows[-1]['time_s']:.3f} s")

    xps_scans, ch_stats = parse_xy(XY_FILE)
    print(f"Channel filter: {ch_stats['kept']} of {ch_stats['total']} scan blocks kept "
          f"(channel {CH_MIN}-{CH_MAX})")

    xy_times = [s["abs_epoch"] for s in xps_scans if s["abs_epoch"] is not None]
    print("--- time comparison ---")
    print(f"EC acquisition started (local)      : {start_local} (UTC+{UTC_OFFSET_HOURS:g} h)")
    print(f"EC acquisition started (UTC raw)    : {fmt_utc(start_local.timestamp() - UTC_OFFSET_HOURS*3600)} UTC")
    print(f"EC computer ahead of XPS by         : {TIME_OFFSET_SECONDS:g} s")
    print(f"EC acquisition started (aligned UTC): {fmt_utc(start_epoch)} UTC")
    print(f"EC last data point (aligned UTC)    : {fmt_utc(start_epoch + ec_rows[-1]['time_s'])} UTC "
          f"(time/s = {ec_rows[-1]['time_s']:.3f} s)")
    print(f"XY first scan (UTC)                 : {fmt_utc(min(xy_times))} UTC")
    print(f"XY last scan  (UTC)                 : {fmt_utc(max(xy_times))} UTC")

    out_rows, max_dt, n_outside = correlate(ec_rows, start_epoch, xps_scans)
    print(f"XPS rows matched to EC: {len(out_rows)} "
          f"({n_outside} dropped, outside EC time range), max |dt|: {max_dt:.3f} s")

    # Show cycle time distribution
    cycle_times = [r["cycle_time_s"] for r in out_rows]
    if cycle_times:
        print(f"cycle_time_s range: {min(cycle_times):.2f} .. {max(cycle_times):.2f} s")

    export(out_rows, OUT_FILE, start_local, start_utc_aligned.timestamp(), ec_rows, ca_rows, ocv_rows)


if __name__ == "__main__":
    main()
