#!/usr/bin/env python3
"""
Generate correlation CSVs and heatmaps for channels just outside the 48-96 range.

This script creates outputs for:
- ch47-47 (single channel just below 48)
- ch97-97 (single channel just above 96)

Usage: python plot_channels_just_outside_48_96.py

Outputs are saved in CVXPS/channel_ranges/ch47-47/ and CVXPS/channel_ranges/ch97-97/
"""

import os
import subprocess
import sys

# Base paths - modify these if needed
MPT_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV.mpt"
XY_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy"
CHANNEL_DIR = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges"

# Channel ranges to process (just outside 48-96)
CHANNEL_RANGES = [
    (47, 47, "ch47-47"),
    (97, 97, "ch97-97"),
]

def run_command(cmd):
    """Run a command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
    return result.returncode

def main():
    python_exe = sys.executable
    
    for ch_min, ch_max, tag in CHANNEL_RANGES:
        print(f"\n{'='*60}")
        print(f"Processing channel range: {tag} (channels {ch_min}-{ch_max})")
        print('='*60)
        
        # Create output directory
        out_dir = os.path.join(CHANNEL_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        
        # CSV output path
        csv_path = os.path.join(out_dir, "12_CV_XPS_correlated.csv")
        
        # Step 1: Correlate CV and XPS data
        print(f"\n1. Correlating CV and XPS data for {tag}...")
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
            print(f"Skipping heatmap generation for {tag} due to correlation error")
            continue
        
        # Step 2: Generate heatmaps
        print(f"\n2. Generating heatmaps for {tag}...")
        cmd = [
            python_exe,
            "CVXPS-heatmap-channel.py",
            csv_path,
            "",  # No specific PNG path - uses default
            tag
        ]
        run_command(cmd)
        
        print(f"\nCompleted: {tag}")
        print(f"  CSV: {csv_path}")
        print(f"  Heatmaps: {out_dir}")
    
    print(f"\n{'='*60}")
    print("All done!")
    print(f"Results saved in: {CHANNEL_DIR}")
    print('='*60)

if __name__ == "__main__":
    main()
