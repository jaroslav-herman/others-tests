#!/usr/bin/env python3
"""
Detailed Tafel Analysis for Sample 499 - One plot per Time point
Shows all measured points and highlights the fit range
Allows visual inspection of whether current range is well chosen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Load and prepare data
# ============================================================================

df = pd.read_csv('499_fit_export.csv')
df['sample_id'] = '499'

# Clean up duplicate column names
if 'Ecell_V.1' in df.columns:
    df = df.drop(columns=['Ecell_V.1'])

# ============================================================================
# Parameters
# ============================================================================

MIN_CURRENT = 5    # mA - minimum current for analysis
MAX_CURRENT = 300  # mA - maximum current for analysis
FIT_RANGE_START = 0.2  # Fraction of data to start fit from
FIT_RANGE_END = 0.8    # Fraction of data to end fit at

# ============================================================================
# Analysis and Plotting Functions
# ============================================================================

def plot_single_time_point(group: pd.DataFrame, time: int, output_dir: Path = None) -> dict:
    """
    Create a detailed plot for a single time point showing:
    - All measured points
    - Fit range highlighted
    - Fit line
    - Annotations
    """
    # Sort by current
    group = group.sort_values('I_mA')
    
    # Get all data (don't filter yet)
    i_short = group['I_mA'].iloc[0]
    I_all = group['I_mA'].values - i_short
    R1_inv_all = (1 / group['R1'].values)
    
    # Filter valid points
    valid_mask = (R1_inv_all > 0) & (I_all > 0)
    I_all = I_all[valid_mask]
    R1_inv_all = R1_inv_all[valid_mask]
    
    if len(I_all) < 5:
        return None
    
    # Filter to analysis range
    in_range = (I_all >= MIN_CURRENT) & (I_all <= MAX_CURRENT)
    I_range = I_all[in_range]
    R1_inv_range = R1_inv_all[in_range]
    
    # Find fit indices within the range
    start_idx = int(len(I_range) * FIT_RANGE_START)
    end_idx = int(len(I_range) * FIT_RANGE_END)
    
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = len(I_range)
    
    I_fit = I_range[start_idx:end_idx]
    R1_inv_fit = R1_inv_range[start_idx:end_idx]
    
    # Perform linear fit
    k, q = np.polyfit(I_fit, R1_inv_fit, 1)
    tafel_slope = 2.3 / k if k != 0 else np.nan
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all points
    ax.plot(I_all, R1_inv_all, 'o', color='lightgray', alpha=0.5, 
            markersize=6, label='All points')
    
    # Plot points in analysis range
    ax.plot(I_range, R1_inv_range, 'o', color='blue', alpha=0.7, 
            markersize=6, label=f'In range ({MIN_CURRENT}-{MAX_CURRENT} mA)')
    
    # Highlight fit range
    ax.plot(I_fit, R1_inv_fit, 'o', color='red', alpha=1, 
            markersize=8, label='Fit points')
    
    # Plot fit line (extended to full range)
    I_line = np.linspace(I_range.min() * 0.9, I_range.max() * 1.1, 100)
    R1_inv_line = k * I_line + q
    ax.plot(I_line, R1_inv_line, '-', color='darkred', linewidth=2, 
            label=f'Fit: k={k:.4f}, q={q:.4f}')
    
    # Add annotations
    ax.set_title(f'Sample 499 - Time {time}\nTafel Slope = {tafel_slope:.2f} mV/dec', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax.set_ylabel('1/R1 (1/Ohm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add vertical lines to show range boundaries
    ax.axvline(x=MIN_CURRENT, color='green', linestyle='--', alpha=0.5, 
               label=f'Min current: {MIN_CURRENT} mA')
    ax.axvline(x=MAX_CURRENT, color='purple', linestyle='--', alpha=0.5, 
               label=f'Max current: {MAX_CURRENT} mA')
    
    # Mark fit range boundaries
    if len(I_range) > 0:
        ax.axvline(x=I_range[start_idx], color='orange', linestyle=':', alpha=0.7,
                   label=f'Fit start: {I_range[start_idx]:.1f} mA')
        ax.axvline(x=I_range[end_idx-1], color='orange', linestyle=':', alpha=0.7,
                   label=f'Fit end: {I_range[end_idx-1]:.1f} mA')
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'499_tafel_time_{time:02d}.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'time': time,
        'tafel_slope': tafel_slope,
        'k': k,
        'q': q,
        'n_all': len(I_all),
        'n_range': len(I_range),
        'n_fit': len(I_fit),
        'i_short': i_short,
        'current_range': f'{I_range.min():.1f}-{I_range.max():.1f} mA'
    }


# ============================================================================
# Main Analysis
# ============================================================================

# Group by Time
time_groups = {int(t): group for t, group in df.groupby('Time')}

# Create output directory
output_dir = Path('tafel_plots_499')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Detailed Tafel Analysis: Sample 499")
print("Creating one plot per Time point with all data and fit range")
print("=" * 70)

# Store all results
all_results = []

# Process each time point
for time in sorted(time_groups.keys()):
    group = time_groups[time]
    print(f"\nProcessing Time {time}...")
    
    result = plot_single_time_point(group, time, output_dir)
    
    if result:
        all_results.append(result)
        print(f"  → Tafel slope: {result['tafel_slope']:.2f} mV/dec")
        print(f"  → Current range: {result['current_range']}")
        print(f"  → Points: {result['n_all']} total, {result['n_range']} in range, {result['n_fit']} for fit")
    else:
        print(f"  → Insufficient data")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Save summary
summary_file = output_dir / '499_tafel_detailed_summary.csv'
results_df.to_csv(summary_file, index=False)
print(f"\nDetailed summary saved to: {summary_file}")

# Count files created
plot_files = list(output_dir.glob('499_tafel_time_*.png'))
print(f"Created {len(plot_files)} individual time point plots")
print(f"All plots saved to: {output_dir}")

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
