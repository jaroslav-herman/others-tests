#!/usr/bin/env python3
"""
Compare total resistance (R0+R1+R2+R3, +R4 where fitted) Time evolution
at specific currents for AEM Substrate Series samples.
Creates one graph per current showing R_total vs Time for all samples in the series.

Usage:
    python compare_rtotal_evolution_aem_substrate.py
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

# AEM Substrate Series samples (474, 499, 505, 516 - NOT 498)
SAMPLES = ['474', '499', '505', '516']

# Sample display names from Google Sheets
SAMPLE_NAMES = {
    '474': 'VI_Ni_on_Ni_PTL',
    '499': 'VI_Ni_on_Ni_etched',
    '505': 'VI_Ni_on_Ni_plain',
    '516': 'Ni PTL no sputtering',
}

# Target currents for interpolation (mA)
TARGET_CURRENTS = [100, 200, 500, 1000, 2000, 3000]

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
    """Load and preprocess data for a single sample, computing total resistance."""
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

    # Add R4 when present in the export (treated as 0 when not fitted)
    if 'R4' in pd.read_csv(csv_path, nrows=0).columns:
        r4 = pd.read_csv(csv_path, usecols=['R4'])['R4']
        df['R4'] = pd.to_numeric(r4, errors='coerce').fillna(0.0)
    else:
        df['R4'] = 0.0

    # Total resistance: R0+R1+R2+R3 (+R4 when fitted)
    df['R_total'] = df[['R0', 'R1', 'R2', 'R3', 'R4']].sum(axis=1)

    # Remove rows with NaN or infinite values
    df = df[np.isfinite(df['R_total'])]
    df = df[df['R_total'] > 0]

    return df[['Time', 'Ecell_V', 'I_mA', 'R_total']]


# ============================================================================
# Interpolation
# ============================================================================

def interpolate_rtotal_at_current(df, target_current):
    """Interpolate R_total at a specific current for each Time point."""
    results = []

    times = sorted(df['Time'].unique())

    for t in times:
        time_data = df[df['Time'] == t].copy()
        time_data = time_data.sort_values('I_mA')

        # Remove duplicates by taking mean
        time_data = time_data.groupby('I_mA').agg({'R_total': 'mean'}).reset_index()

        # Interpolate at target current
        if target_current < time_data['I_mA'].min() or target_current > time_data['I_mA'].max():
            interp_val = np.nan
        else:
            interp_val = np.interp(target_current, time_data['I_mA'], time_data['R_total'])

        results.append({
            'Time': t,
            'I_mA': target_current,
            'R_total': interp_val
        })

    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

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
    output_dir.mkdir(exist_ok=True, parents=True)

    # For each current, create a comparison graph
    for current in TARGET_CURRENTS:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for sample_id in SAMPLES:
            if sample_id not in sample_dfs:
                continue

            df = sample_dfs[sample_id]
            interp_df = interpolate_rtotal_at_current(df, current)

            # Get sample display name
            sample_name = SAMPLE_NAMES.get(sample_id, sample_id)
            label = f"{sample_id} ({sample_name})"
            color = SAMPLE_COLORS.get(sample_id, 'black')

            # Plot R_total vs Time
            valid_data = interp_df.dropna()
            if len(valid_data) > 0:
                ax.plot(valid_data['Time'], valid_data['R_total'],
                        'o-', color=color, markersize=6, linewidth=2,
                        label=label)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('R_total = R0+R1+R2+R3(+R4) (Ohm)', fontsize=12)
        ax.set_title(f'Total Resistance Time Evolution at {current} mA - AEM Substrate Series', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)

        output_path = output_dir / f"rtotal_evolution_at_{current}mA.{PLOT_FORMAT}"
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

    print("\nAll comparison graphs generated successfully!")


if __name__ == '__main__':
    main()
