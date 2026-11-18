#!/usr/bin/env python3
"""
Generate correlation CSVs and heatmaps for ALL channels EXCEPT those in range 48-96.

This means processing:
- Channel range 0-47 (all channels below 48)
- Channel range 97-145 (all channels above 96)

Usage: python plot_channels_excluding_48_96.py

Outputs are saved in:
  CVXPS/channel_ranges/ch0-47/
  CVXPS/channel_ranges/ch97-145/
"""

import os
import subprocess
import sys

# Base paths - modify these if needed
MPT_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV.mpt"
XY_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy"
CHANNEL_DIR = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges"

# Channel ranges to process (all channels EXCEPT 48-96)
CHANNEL_RANGES = [
    (0, 47, "ch0-47", "All channels below 48"),
    (97, 145, "ch97-145", "All channels above 96"),
]

def run_command(cmd):
    """Run a command and print output."""
    print(f"  Running: {' '.join(cmd[1:] if len(cmd) > 1 else cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.splitlines():
            print(f"    {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            print(f"    [STDERR] {line}")
    if result.returncode != 0:
        print(f"  ERROR: Command failed with return code {result.returncode}")
    return result.returncode

def main():
    python_exe = sys.executable
    
    print("="*70)
    print("Generating plots for ALL channels EXCEPT 48-96")
    print("="*70)
    
    for ch_min, ch_max, tag, description in CHANNEL_RANGES:
        print(f"\n{'-'*70}")
        print(f"{tag}: {description} (channels {ch_min}-{ch_max})")
        print('-'*70)
        
        # Create output directory
        out_dir = os.path.join(CHANNEL_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        
        # CSV output path
        csv_path = os.path.join(out_dir, "12_CV_XPS_correlated.csv")
        
        # Step 1: Correlate CV and XPS data
        print("\n  Step 1: Correlating CV and XPS data...")
        cmd = [
            python_exe,
            "CVXPS-correlate-channel.py",
            MPT_FILE,
            XY_FILE,
            csv_path,
            str(ch_min),
            str(ch_max)
        ]
        if run_command(cmd) != 0:
            print(f"  Skipping heatmap generation for {tag} due to correlation error")
            continue
        
        # Step 2: Generate heatmaps
        print("\n  Step 2: Generating heatmaps...")
        cmd = [
            python_exe,
            "CVXPS-heatmap-channel.py",
            csv_path,
            "",  # No specific PNG path - uses default
            tag
        ]
        run_command(cmd)
        
        print(f"\n  Completed: {tag}")
        print(f"    CSV: {csv_path}")
        print(f"    Heatmaps: {out_dir}")
    
    print(f"\n{'='*70}")
    print("All done!")
    print(f"Results saved in: {CHANNEL_DIR}")
    print(f"  - {CHANNEL_DIR}\ch0-47")
    print(f"  - {CHANNEL_DIR}\ch97-145")
    print('='*70)

if __name__ == "__main__":
    main()
