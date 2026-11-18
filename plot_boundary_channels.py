#!/usr/bin/env python3
"""
Generate correlation CSVs and heatmaps for channels at the boundaries of 48-96 range.

This script creates outputs for channels just outside and at the edges of 48-96:
- ch47-47 (just below 48)
- ch48-48 (lower boundary of range)
- ch96-96 (upper boundary of range)
- ch97-97 (just above 96)
- ch47-48 (transition at lower boundary)
- ch96-97 (transition at upper boundary)

Usage: python plot_boundary_channels.py

Outputs are saved in CVXPS/channel_ranges/<tag>/
"""

import os
import subprocess
import sys

# Base paths - modify these if needed
MPT_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV.mpt"
XY_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy"
CHANNEL_DIR = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges"

# Channel ranges to process
CHANNEL_RANGES = [
    # Single channels at boundaries
    (47, 47, "ch47-47", "Single channel just below 48"),
    (48, 48, "ch48-48", "Single channel at lower boundary"),
    (96, 96, "ch96-96", "Single channel at upper boundary"),
    (97, 97, "ch97-97", "Single channel just above 96"),
    # Transition ranges
    (47, 48, "ch47-48", "Transition at lower boundary"),
    (96, 97, "ch96-97", "Transition at upper boundary"),
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
    print("Generating plots for channels at boundaries of 48-96 range")
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
    print('='*70)

if __name__ == "__main__":
    main()
