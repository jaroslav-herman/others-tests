#!/usr/bin/env python3
"""
Interpolate R1 with respect to I_mA and plot Time evolution of R1 at specific currents.

Usage:
    python plot_r1_at_current.py
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

# Target currents for interpolation (mA)
TARGET_CURRENTS = [50, 100, 200, 500, 1000]

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
# Interpolation and Plotting
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


def plot_r1_evolution_at_currents(df, sample_id, output_dir):
    """Plot Time evolution of R1 at target currents."""
    
    # Generate rainbow colors from purple to red
    n_currents = len(TARGET_CURRENTS)
    colors = plt.cm.rainbow(np.linspace(1, 0, n_currents))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, current in enumerate(TARGET_CURRENTS):
        interp_df = interpolate_r1_at_current(df, current)
        
        # Drop NaN values
        valid_data = interp_df.dropna()
        
        if len(valid_data) > 0:
            ax.plot(valid_data['Time'], valid_data['R1'],
                    'o-', color=colors[i], markersize=6, linewidth=2,
                    label=f'{current} mA')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('R1 (Ohm)', fontsize=12)
    ax.set_title(f'Sample {sample_id}: R1 Time Evolution at Specific Currents', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    output_path = output_dir / f"{sample_id}_R1_evolution_at_currents.{PLOT_FORMAT}"
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
        
        # Plot R1 evolution at target currents
        plot_r1_evolution_at_currents(df, sample_id, sample_dir)
    
    print("\nAll R1 evolution plots generated successfully!")


if __name__ == '__main__':
    main()
