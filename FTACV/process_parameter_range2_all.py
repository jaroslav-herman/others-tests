"""FTACV processing script for parameter range 2 folder structure.

This script processes all .mpr files in all subfolders of the parameter range 2 directory.

For each FTACV .mpr file, it calculates:
- Harmonics 1-9 using Fourier transform
- Filtered DC signal
- Harmonic envelopes
- Power spectrum
- Generates 10-panel plots

Output files are saved beside each .mpr file with suffixes:
- _harmonic_envelopes.csv
- _filtered_dc_current.csv
- _power_spectrum.csv
- _summary.csv
- _ten_panel.png

Usage:
    Run this file directly from VS Code or command line.
    The script will automatically find and process all folders.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import sleep

# The main processing script
PROCESSING_SCRIPT = Path(__file__).parent / "ftacv_fourier_harmonic_filter.py"
PYTHON_EXE = Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"

# Default input folder
DEFAULT_INPUT = Path(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\FTACV\480_VIII_cathode_etching_series_210min_more_C\parameter range 2"
)


def process_folder(folder: Path) -> bool:
    """Process a single folder with the FTACV script."""
    print(f"Processing: {folder.name}")
    
    cmd = [
        str(PYTHON_EXE),
        str(PROCESSING_SCRIPT),
        "--input", str(folder),
        "--all-files",
        "--recalculate-all"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes per folder
        )
        
        if result.returncode == 0:
            print(f"[OK] Completed: {folder.name}")
            # Show processing summary
            for line in result.stdout.strip().split('\n'):
                if 'Found' in line or 'elapsed' in line:
                    print(f"    {line}")
            return True
        else:
            print(f"[FAIL] Error in {folder.name}")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[FAIL] Timeout: {folder.name}")
        return False
    except Exception as e:
        print(f"[FAIL] Exception in {folder.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input folder containing subfolders with .mpr files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show folders that would be processed without actually processing"
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input folder does not exist: {args.input}")
        sys.exit(1)
    
    # Find all subfolders
    folders = sorted([f for f in args.input.iterdir() if f.is_dir()])
    print(f"Found {len(folders)} subfolders in {args.input.name}")
    
    if args.dry_run:
        print("Dry run - folders that would be processed:")
        for folder in folders:
            print(f"  - {folder.name}")
        return
    
    print("Starting processing...")
    print("=" * 60)
    
    success_count = 0
    failure_count = 0
    
    for folder in folders:
        if process_folder(folder):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 60)
        sleep(0.5)  # Small delay between folders
    
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Successful: {success_count}/{len(folders)}")
    print(f"Failed: {failure_count}/{len(folders)}")
    
    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
