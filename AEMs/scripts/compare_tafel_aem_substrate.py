#!/usr/bin/env python3
"""
Compare Tafel slope Time evolution for AEM Substrate Series samples.
Loads Tafel summary CSV files and plots Time evolution of Tafel slope for all samples.

Usage:
    python compare_tafel_aem_substrate.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Disable interactive backend for faster rendering
plt.ioff()

# ============================================================================
# CONFIGURATION
# ============================================================================

# AEM Substrate Series samples (474, 499, 505 only - NOT 498)
SAMPLES = ['474', '499', '505']

# Sample display names from Google Sheets
SAMPLE_NAMES = {
    '474': 'VI_Ni_on_Ni_PTL',
    '499': 'VI_Ni_on_Ni_etched',
    '505': 'VI_Ni_on_Ni_plain',
}

# Rainbow colors for samples (purple to red)
SAMPLE_COLORS = {
    '474': 'purple',
    '499': 'blue',
    '505': 'red'
}

# Plot configuration
PLOT_DPI = 200
PLOT_FORMAT = 'png'

# ============================================================================
# Load Tafel Data
# ============================================================================

def load_tafel_data(sample_id):
    """Load Tafel summary data for a single sample."""
    csv_path = Path(f"sample_{sample_id}/tafel_700ma_{sample_id}/{sample_id}_tafel_summary.csv")
    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df['sample_id'] = sample_id
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    base_dir = Path('/workspace/github__jaroslav-herman__others-tests/AEMs')
    os.chdir(base_dir)
    
    # Load all sample Tafel data
    sample_dfs = {}
    for sample_id in SAMPLES:
        df = load_tafel_data(sample_id)
        if df is not None:
            sample_dfs[sample_id] = df
    
    # Create output directory
    output_dir = base_dir / "AEM Substrate Series"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for sample_id in SAMPLES:
        if sample_id not in sample_dfs:
            continue
        
        df = sample_dfs[sample_id]
        
        # Get sample display name
        sample_name = SAMPLE_NAMES.get(sample_id, sample_id)
        label = f"{sample_id} ({sample_name})"
        color = SAMPLE_COLORS.get(sample_id, 'black')
        
        # Plot Tafel slope vs Time
        ax.plot(df['time'], df['tafel_slope'],
                'o-', color=color, markersize=8, linewidth=2,
                label=label)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
    ax.set_title('Tafel Slope Time Evolution - AEM Substrate Series', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    output_path = output_dir / f"tafel_slope_time_evolution_comparison.{PLOT_FORMAT}"
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    print("\nComparison graph generated successfully!")


if __name__ == '__main__':
    main()
