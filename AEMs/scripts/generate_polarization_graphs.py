#!/usr/bin/env python3
"""
Generate polarization curves and current evolution graphs for AEM samples.
Uses EIS fitting data from CSV files.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Sample IDs
SAMPLES = ['474', '498', '499', '505']

# Target voltages for interpolation
TARGET_VOLTAGES = [1.4, 1.45, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9]

# Rainbow color scheme from purple to red
# Colors will be generated dynamically based on number of items, from purple (0) to red (1)
RAINBOW_COLORS = None  # Will be set dynamically per plot

def load_sample_data(sample_id):
    """Load sample data from CSV file."""
    csv_path = Path(f"sample_{sample_id}/{sample_id}_fit_export.csv")
    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    # Use the first Ecell_V column (there are duplicates)
    df = df.rename(columns={'Ecell_V.1': 'Ecell_V_dup'})
    
    # Select relevant columns
    df = df[['Ecell_V', 'I_mA', 'Time']].copy()
    df = df.dropna()
    
    return df

def generate_polarization_curve(df, sample_id, output_dir):
    """Generate I/mA vs Ecell/V plot split by Time with rainbow colors from purple to red."""
    times = sorted(df['Time'].unique())
    
    # Generate rainbow colors from purple to red based on number of time points
    n_colors = len(times)
    colors = plt.cm.rainbow(np.linspace(1, 0, n_colors))  # 1=purple, 0=red
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, t in enumerate(times):
        time_data = df[df['Time'] == t]
        time_data = time_data.sort_values('Ecell_V')
        
        ax.plot(time_data['Ecell_V'], time_data['I_mA'], 
                marker='o', markersize=4, linestyle='-', 
                color=colors[i], 
                label=f'Time {t}')
    
    ax.set_xlabel('Ecell / V', fontsize=14)
    ax.set_ylabel('I / mA', fontsize=14)
    ax.set_title(f'Sample {sample_id}: Polarization Curve (I vs Ecell)', fontsize=16)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / f"{sample_id}_polarization_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path

def interpolate_current_at_voltages(df, target_voltages):
    """Interpolate current values at specific voltages for each Time point."""
    results = []
    
    times = sorted(df['Time'].unique())
    
    for t in times:
        time_data = df[df['Time'] == t].copy()
        time_data = time_data.sort_values('Ecell_V')
        
        # Remove duplicates by taking mean
        time_data = time_data.groupby('Ecell_V').agg({'I_mA': 'mean'}).reset_index()
        
        # Interpolate at target voltages
        for v in target_voltages:
            if v < time_data['Ecell_V'].min() or v > time_data['Ecell_V'].max():
                # Extrapolate if needed
                interp_val = np.nan
            else:
                interp_val = np.interp(v, time_data['Ecell_V'], time_data['I_mA'])
            
            results.append({
                'Time': t,
                'Ecell_V': v,
                'I_mA': interp_val
            })
    
    return pd.DataFrame(results)

def generate_current_evolution_graph(df, sample_id, output_dir):
    """Generate current evolution at specific voltages as function of Time with rainbow colors from purple to red."""
    # Interpolate currents at target voltages
    interp_df = interpolate_current_at_voltages(df, TARGET_VOLTAGES)
    
    if interp_df.empty:
        print(f"Warning: No interpolation data for sample {sample_id}")
        return None
    
    # Pivot for plotting
    pivot_df = interp_df.pivot(index='Time', columns='Ecell_V', values='I_mA')
    
    # Generate rainbow colors from purple to red based on number of voltages
    n_voltages = len(TARGET_VOLTAGES)
    colors = plt.cm.rainbow(np.linspace(1, 0, n_voltages))  # 1=purple, 0=red
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, v in enumerate(TARGET_VOLTAGES):
        if v in pivot_df.columns:
            voltage_data = pivot_df[v].dropna()
            ax.plot(voltage_data.index, voltage_data.values, 
                    marker='o', markersize=5, linestyle='-',
                    color=colors[i],
                    label=f'{v} V')
    
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Current / mA', fontsize=14)
    ax.set_yscale('log')
    ax.set_title(f'Sample {sample_id}: Current Evolution at Specific Voltages', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    output_path = output_dir / f"{sample_id}_current_evolution_at_voltages.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    return output_path

def main():
    base_dir = Path('/workspace/github__jaroslav-herman__others-tests/AEMs')
    os.chdir(base_dir)
    
    for sample_id in SAMPLES:
        print(f"\nProcessing sample {sample_id}...")
        
        # Load data
        df = load_sample_data(sample_id)
        if df is None or df.empty:
            print(f"Skipping sample {sample_id} - no data")
            continue
        
        # Create sample output directory if it doesn't exist
        sample_dir = base_dir / f"sample_{sample_id}"
        sample_dir.mkdir(exist_ok=True)
        
        # Generate polarization curve
        generate_polarization_curve(df, sample_id, sample_dir)
        
        # Generate current evolution graph
        generate_current_evolution_graph(df, sample_id, sample_dir)
    
    print("\nAll graphs generated successfully!")

if __name__ == '__main__':
    main()
