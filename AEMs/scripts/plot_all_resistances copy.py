#!/usr/bin/env python3
"""
Plot all resistances and time constants vs Current and Cell Voltage for a single sample.
- Split by Time, using grayscale for R0 and rainbow colors for the other parameters
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

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})

# Disable interactive backend for faster rendering
plt.ioff()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Samples to analyze
SAMPLES = [ '499']

# Parameters to plot
RESISTANCE_TYPES = ['R0', 'R1', 'R3', 'R2', 'R4']
RESISTANCE_SCALE = 4.84
RESISTANCE_LABELS = {
    'R0': 'R0 (Ohm)',
    'R1': 'R1 (Ohm)',
    'R3': 'R3 (Ohm)',
    'R2': 'R2 (Ohm)',
    'R4': 'R4 (Ohm)'
}
TAU_TYPES = ['tau1', 'tau3', 'tau2', 'tau4']
TAU_LABELS = {tau: f'{tau} (s)' for tau in TAU_TYPES}

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
    
    required_cols = ['Time', 'Ecell_V', 'I_mA', 'R0', 'R1', 'R3', 'R2',
                     'tau1', 'tau2', 'tau3']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in sample {sample_id}")
            return None
    
    optional_cols = ['R4', 'tau4']
    df = df[required_cols + [col for col in optional_cols if col in df.columns]].copy()
    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan
    df[RESISTANCE_TYPES] *= RESISTANCE_SCALE
    
    return df


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_parameters_vs_variable(df, sample_id, x_var, x_label, output_dir,
                               parameter_types, labels, output_name, y_label):
    """Plot parameter values against a variable, with a separate color sequence per parameter."""
    times = sorted(df['Time'].unique())
    n_times = len(times)
    rainbow_types = [parameter for parameter in parameter_types if parameter != 'R0']
    colors = plt.cm.rainbow(np.linspace(1, 0, len(rainbow_types) * n_times))
    r0_colors = plt.cm.Greys(np.linspace(0.75, 0.25, n_times))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for i, time in enumerate(times):
        time_data = df[df['Time'] == time].copy()
        time_data = time_data.sort_values(x_var)
        
        for parameter in parameter_types:
            if parameter == 'R0':
                color = r0_colors[i]
            else:
                color_index = rainbow_types.index(parameter) * n_times + i
                color = colors[color_index]
            plot_data = time_data[
                np.isfinite(time_data[x_var])
                & np.isfinite(time_data[parameter])
                & (time_data[parameter] > 0)
            ]
            ax.plot(plot_data[x_var], plot_data[parameter],
                    'o-', color=color, markersize=4, alpha=0.7, linewidth=1,
                    label=f'Time {time}' if i == 0 else "")
    
    # Show one representative shade for each parameter in the legend.
    legend_handles = []
    for parameter in parameter_types:
        if parameter == 'R0':
            color = r0_colors[n_times // 2]
        else:
            color_index = rainbow_types.index(parameter) * n_times + n_times // 2
            color = colors[color_index]
        legend_handles.append(plt.Line2D([0], [0],
            marker='o', color=color, linestyle='-',
            markersize=4, label=labels[parameter]))
    
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    ax.tick_params(axis='both', which='both', labelsize=16)
    ax.set_yscale('log')

    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    #ax.legend(handles=legend_handles, fontsize=10, loc='upper right')
    
    output_path = output_dir / f"{sample_id}_{output_name}_vs_{x_var}.{PLOT_FORMAT}"
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")
    return output_path


def main():
    base_dir = Path('/workspace/github__jaroslav-herman__others-tests/AEMs')
    base_dir = Path(r'C:\Users\Herman\Python\WE\others-tests\AEMs')
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
        
        # Plot resistances and time constants against current and cell voltage.
        for x_var, x_label in [('I_mA', 'Current (mA)'), ('Ecell_V', 'Cell Voltage (V)')]:
            plot_parameters_vs_variable(
                df, sample_id, x_var, x_label, sample_dir,
                RESISTANCE_TYPES, RESISTANCE_LABELS, 'resistances', r'Resistance $\mathrm{(\Omega~cm^{2})}$'
            )
            plot_parameters_vs_variable(
                df, sample_id, x_var, x_label, sample_dir,
                TAU_TYPES, TAU_LABELS, 'taus', 'Time constant (s)'
            )
    
    print("\nAll resistance and time-constant plots generated successfully!")


if __name__ == '__main__':
    main()
