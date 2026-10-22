#!/usr/bin/env python3
"""
Compare current time evolution for AEM Substrate Series samples.
Creates one graph per voltage showing current vs Time for all samples.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# AEM Substrate Series samples (474, 499, 505 only - NOT 498)
SAMPLES = ['474', '499', '505']

# Target voltages for comparison (same as in generate_polarization_graphs.py)
TARGET_VOLTAGES = [1.4, 1.45, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9]

# Sample display names from Google Sheets
# Legend format: sample_number (sample_name)
SAMPLE_NAMES = {
    '474': 'VI_Ni_on_Ni_PTL',
    '499': 'VI_Ni_on_Ni_etched',
    '505': 'VI_Ni_on_Ni_plain',
}

def load_sample_data(sample_id):
    """Load sample data from CSV file."""
    csv_path = Path(f"sample_{sample_id}/{sample_id}_fit_export.csv")
    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df = df[['Ecell_V', 'I_mA', 'Time']].copy()
    df = df.dropna()
    
    return df

def interpolate_current_at_voltage(df, voltage):
    """Interpolate current at a specific voltage for each Time point."""
    results = []
    
    times = sorted(df['Time'].unique())
    
    for t in times:
        time_data = df[df['Time'] == t].copy()
        time_data = time_data.sort_values('Ecell_V')
        
        # Remove duplicates by taking mean
        time_data = time_data.groupby('Ecell_V').agg({'I_mA': 'mean'}).reset_index()
        
        # Interpolate at target voltage
        if voltage < time_data['Ecell_V'].min() or voltage > time_data['Ecell_V'].max():
            interp_val = np.nan
        else:
            interp_val = np.interp(voltage, time_data['Ecell_V'], time_data['I_mA'])
        
        results.append({
            'Time': t,
            'Ecell_V': voltage,
            'I_mA': interp_val
        })
    
    return pd.DataFrame(results)

def main():
    base_dir = Path('/workspace/github__jaroslav-herman__others-tests/AEMs')
    os.chdir(base_dir)
    
    # Load all sample data
    sample_dfs = {}
    for sample_id in SAMPLES:
        df = load_sample_data(sample_id)
        if df is not None:
            sample_dfs[sample_id] = df
    
    # Create output directory
    output_dir = base_dir / "AEM Substrate Series"
    output_dir.mkdir(exist_ok=True)
    
    # For each voltage, create a comparison graph
    for voltage in TARGET_VOLTAGES:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for sample_id in SAMPLES:
            if sample_id not in sample_dfs:
                continue
            
            df = sample_dfs[sample_id]
            interp_df = interpolate_current_at_voltage(df, voltage)
            
            # Get sample display name
            sample_name = SAMPLE_NAMES.get(sample_id, sample_id)
            label = f"{sample_id} ({sample_name})"
            
            # Plot current vs Time
            ax.plot(interp_df['Time'], interp_df['I_mA'], 
                    marker='o', markersize=6, linestyle='-',
                    label=label)
        
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Current / mA', fontsize=14)
        ax.set_title(f'Current Time Evolution at {voltage} V - AEM Substrate Series', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / f"current_evolution_at_{voltage}V.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {output_path}")
    
    print("\nAll comparison graphs generated successfully!")

if __name__ == '__main__':
    main()
