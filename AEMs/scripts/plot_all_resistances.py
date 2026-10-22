#!/usr/bin/env python3
"""
Plot all resistances (R0, R1, R2, R3) vs Current and vs Cell Voltage for a single sample.
- Split by Time with rainbow colors (purple to red)
- Save graphs directly to sample folder

Usage:
    python plot_all_resistances.py
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

# Samples to analyze
SAMPLES = ['474', '498', '499', '505']

# Resistance types to plot
RESISTANCE_TYPES = ['R0', 'R1', 'R2', 'R3']
RESISTANCE_LABELS = {
    'R0': 'R0 (Ohm)',
    'R1': 'R1 (Ohm)',
    'R2': 'R2 (Ohm)',
    'R3': 'R3 (Ohm)'
}
RESISTANCE_COLORS = {
    'R0': 'blue',
    'R1': 'orange',
    'R2': 'green',
    'R3': 'red'
}

# Plot configuration
PLOT_DPI = 200
PLOT_FORMAT = 'png'

# ============================================================================
# Load Data
# ============================================================================

def load_sample_data(sample_id):
    """Load and preprocess data for a single sample."""
    csv_path = Path(f"sample_{sample_id}/{sample_id}_fit_export.csv")
    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Select relevant columns
    cols = ['Time', 'Ecell_V', 'I_mA', 'R0', 'R1', 'R2', 'R3']
    for col in cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in sample {sample_id}")
            return None
    
    df = df[cols].copy()
    
    # Remove rows with NaN or infinite values in resistances
    for r in RESISTANCE_TYPES:
        df = df[np.isfinite(df[r])]
        df = df[df[r] > 0]
    
    return df


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_resistances_vs_variable(df, sample_id, x_var, x_label, output_dir):
    """Plot all resistances vs a variable (current or voltage) split by Time with rainbow colors."""
    times = sorted(df['Time'].unique())
    n_times = len(times)
    
    # Generate rainbow colors from purple to red
    colors = plt.cm.rainbow(np.linspace(1, 0, n_times))
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for i, time in enumerate(times):
        time_data = df[df['Time'] == time].copy()
        time_data = time_data.sort_values(x_var)
        
        for r_type in RESISTANCE_TYPES:
            ax.plot(time_data[x_var], time_data[r_type],
                    'o-', color=colors[i], markersize=4, alpha=0.7, linewidth=1,
                    label=f'Time {time}' if i == 0 else "")
    
    # Add legend for resistance types
    legend_handles = []
    for r_type in RESISTANCE_TYPES:
        legend_handles.append(plt.Line2D([0], [0],
            marker='o', color=RESISTANCE_COLORS[r_type], linestyle='-',
            markersize=4, label=RESISTANCE_LABELS[r_type]))
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Resistance (Ohm)', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=10, loc='upper right')
    
    output_path = output_dir / f"{sample_id}_resistances_vs_{x_var}.{PLOT_FORMAT}"
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


def main():
    base_dir = Path('/workspace/github__jaroslav-herman__others-tests/AEMs')
    os.chdir(base_dir)
    
    for sample_id in SAMPLES:
        print(f"\nProcessing sample {sample_id}...")
        
        # Load data
        df = load_sample_data(sample_id)
        if df is None or df.empty:
            print(f"  Skipping sample {sample_id} - no data")
            continue
        
        # Create sample output directory if it doesn't exist
        sample_dir = base_dir / f"sample_{sample_id}"
        sample_dir.mkdir(exist_ok=True)
        
        # Plot resistances vs Current
        plot_resistances_vs_variable(df, sample_id, 'I_mA', 'Current (mA)', sample_dir)
        
        # Plot resistances vs Cell Voltage
        plot_resistances_vs_variable(df, sample_id, 'Ecell_V', 'Cell Voltage (V)', sample_dir)
    
    print("\nAll resistance plots generated successfully!")


if __name__ == '__main__':
    main()
