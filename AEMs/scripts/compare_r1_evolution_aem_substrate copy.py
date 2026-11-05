#!/usr/bin/env python3
"""
Compare R1 Time evolution at specific currents for AEM Substrate Series samples.
Creates one graph per current showing R1 vs Time for all samples in the series.

Usage:
    python compare_r1_evolution_aem_substrate.py
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

# AEM Substrate Series samples (474, 499, 505 only - NOT 498)
SAMPLES = ['474', '499', '505', '516']

# Sample display names from Google Sheets
SAMPLE_NAMES = {
    '474': 'Ni on Ni PTL',
    '499': 'Ni on Ni etched membrane',
    '505': 'Ni on Ni plain membrane',
    '516': 'Ni PTL, no sputtering',
}

# Target currents for interpolation (mA)
TARGET_CURRENTS = [150]

# Plot configuration
PLOT_DPI = 200
PLOT_FORMAT = 'png'

# Rainbow colors for samples (purple to red)
SAMPLE_COLORS = {
    '474': 'purple',
    '499': 'blue',
    '505': 'red',
    '516': 'green'
}

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
    cols = ['Time', 'Ecell_V', 'I_mA', 'R1']
    for col in cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in sample {sample_id}")
            return None
    
    df = df[cols].copy()
    
    # Remove rows with NaN or infinite values
    df = df[np.isfinite(df['R1'])]
    df = df[df['R1'] > 0]
    
    return df


# ============================================================================
# Interpolation
# ============================================================================

def interpolate_r1_at_current(df, target_current):
    """Interpolate R1 at a specific current for each Time point."""
    results = []
    
    times = sorted(df['Time'].unique())
    
    for t in times:
        time_data = df[df['Time'] == t].copy()
        time_data = time_data.sort_values('I_mA')
        
        # Remove duplicates by taking mean
        time_data = time_data.groupby('I_mA').agg({'R1': 'mean'}).reset_index()
        
        # Interpolate at target current
        if target_current < time_data['I_mA'].min() or target_current > time_data['I_mA'].max():
            interp_val = np.nan
        else:
            interp_val = np.interp(target_current, time_data['I_mA'], time_data['R1'])
        
        results.append({
            'Time': t,
            'I_mA': target_current,
            'R1': interp_val
        })
    
    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

def main():
    base_dir = Path('/workspace/github__jaroslav-herman__others-tests/AEMs')
    base_dir = Path(r'C:\Users\Herman\Python\WE\others-tests\AEMs')
    os.chdir(base_dir)
    
    # Load all sample data
    sample_dfs = {}
    for sample_id in SAMPLES:
        df = load_sample_data(sample_id)
        if df is not None:
            sample_dfs[sample_id] = df
    
    # Create output directory
    output_dir = base_dir / "AEM Substrate Series"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # For each current, create a comparison graph
    for current in TARGET_CURRENTS:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for sample_id in SAMPLES:
            if sample_id not in sample_dfs:
                continue
            
            df = sample_dfs[sample_id]
            interp_df = interpolate_r1_at_current(df, current)
            
            # Get sample display name
            sample_name = SAMPLE_NAMES.get(sample_id, sample_id)
            label = f"{sample_name}"
            color = SAMPLE_COLORS.get(sample_id, 'black')
            
            # Plot R1 vs Time
            valid_data = interp_df.dropna()
            if len(valid_data) > 0:
                ax.plot(valid_data['Time'], valid_data['R1']*4.84,
                        'o-', color=color, markersize=6, linewidth=2,
                        label=label)
        
        measurement_times = sorted({
            time for df in sample_dfs.values() for time in df['Time'].unique()
        })
        day_ticks = measurement_times[::3]
        ax.set_xticks(day_ticks, labels=range(1, len(day_ticks) + 1))
        ax.set_xlabel('Day', fontsize=18)
        ax.set_ylabel(r'Anode Charge Transfer Resistance at 30 $\mathrm{mA~cm^{-2}}$ $\mathrm{(\Omega~cm^{2})}$', fontsize=18)
       # ax.set_title(f'R1 Time Evolution at {current} mA - AEM Substrate Series', fontsize=16)
        ax.tick_params(axis='both', which='both', labelsize=16)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.set_xlim(0.5,16.5)
        ax.legend(fontsize=15)
        
        output_path = output_dir / f"r1_evolution_at_{current}mA.png"
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    print("\nAll comparison graphs generated successfully!")


if __name__ == '__main__':
    main()
