#!/usr/bin/env python3
"""
Tafel Analysis for Sample 499 - OER from R1
Plots 1/R1 vs current for different Time points
Corrects for short circuit and fits optimal linear range
Calculates Tafel slope = 2.3/k
Plots time evolution of Tafel slope
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Load and prepare data
# ============================================================================

df = pd.read_csv('sample_499/499_fit_export.csv')
df['sample_id'] = '499'

# Clean up duplicate column names
if 'Ecell_V.1' in df.columns:
    df = df.drop(columns=['Ecell_V.1'])

# ============================================================================
# Parameters
# ============================================================================

MAX_CURRENT = 300  # mA - filter to reasonable current range
MIN_CURRENT = 5    # mA - minimum current for fitting
FIT_RANGE_START = 0.2  # Fraction of data to start fit from
FIT_RANGE_END = 0.8    # Fraction of data to end fit at

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_time_point(group: pd.DataFrame, time: int, color: str, ax: plt.Axes) -> dict:
    """
    Analyze a single time point:
    - Correct current for short circuit (subtract current at lowest voltage)
    - Plot 1/R1 vs corrected current
    - Fit linear range and calculate Tafel slope
    """
    # Sort by current
    group = group.sort_values('I_mA')
    
    # Filter to reasonable current range
    group = group[(group['I_mA'] >= MIN_CURRENT) & (group['I_mA'] <= MAX_CURRENT)]
    
    if len(group) < 10:
        print(f"Time {time}: Not enough data points ({len(group)})")
        return None
    
    # Correct for short circuit: subtract current at lowest voltage
    # Find current at lowest voltage (approximately short circuit current)
    i_short = group['I_mA'].iloc[0]
    
    # Corrected current
    I_corr = group['I_mA'].values - i_short
    R1_inv = (1 / group['R1'].values)
    
    # Filter valid points (R1 > 0, I_corr > 0)
    valid_mask = (R1_inv > 0) & (I_corr > 0)
    I_corr = I_corr[valid_mask]
    R1_inv = R1_inv[valid_mask]
    
    if len(I_corr) < 5:
        print(f"Time {time}: Not enough valid points after filtering ({len(I_corr)})")
        return None
    
    # Plot all points
    ax.plot(I_corr, R1_inv, 'o', color=color, alpha=0.5, markersize=4, label=f'Time {time}')
    
    # Find optimal fit range - use middle portion of data
    start_idx = int(len(I_corr) * FIT_RANGE_START)
    end_idx = int(len(I_corr) * FIT_RANGE_END)
    
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = len(I_corr)
    
    I_fit = I_corr[start_idx:end_idx]
    R1_inv_fit = R1_inv[start_idx:end_idx]
    
    # Perform linear fit: 1/R1 = k * I + q
    k, q = np.polyfit(I_fit, R1_inv_fit, 1)
    
    # Calculate Tafel slope: 2.3 / k
    tafel_slope = 2.3 / k if k != 0 else np.nan
    
    # Plot fit line
    I_line = np.linspace(I_corr.min(), I_corr.max(), 100)
    R1_inv_line = k * I_line + q
    ax.plot(I_line, R1_inv_line, '-', color=color, linewidth=1, alpha=0.8)
    
    # Annotate Tafel slope on plot
    ax.annotate(f'Tafel: {tafel_slope:.2f} mV/dec',
                xy=(I_corr[-1], R1_inv[-1]),
                xytext=(I_corr[-1] * 0.7, R1_inv[-1] * 1.1),
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.7),
                fontsize=8, color=color)
    
    return {
        'time': time,
        'I_short': i_short,
        'k': k,
        'q': q,
        'tafel_slope': tafel_slope,
        'n_points': len(I_corr),
        'fit_points': len(I_fit)
    }


# ============================================================================
# Main Analysis
# ============================================================================

# Group by Time
time_groups = {int(t): group for t, group in df.groupby('Time')}

# Get colors for each time point
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(time_groups)))

# ============================================================================
# Plot 1: 1/R1 vs Corrected Current for all Time points
# ============================================================================

print("=" * 60)
print("Tafel Analysis: Sample 499")
print("Plotting 1/R1 vs Corrected Current for each Time point")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Store results
results = []

for (time, group), color in zip(sorted(time_groups.items()), colors):
    result = analyze_time_point(group, time, color, ax)
    if result:
        results.append(result)
        print(f"Time {time}: Tafel slope = {result['tafel_slope']:.2f} mV/dec "
              f"(k={result['k']:.4e}, n_points={result['n_points']}, fit_points={result['fit_points']})")

# Format plot
ax.set_xlabel('Corrected Current (mA)', fontsize=12)
ax.set_ylabel('1/R1 (1/Ohm)', fontsize=12)
ax.set_title('Sample 499: 1/R1 vs Corrected Current (Tafel Analysis)', fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(fontsize=8, ncol=3)

plt.tight_layout()
plt.savefig('499_tafel_1_over_R1_vs_current.png', dpi=300, bbox_inches='tight')
print(f"\nSaved plot: 499_tafel_1_over_R1_vs_current.png")
plt.show()

# ============================================================================
# Plot 2: Time Evolution of Tafel Slope
# ============================================================================

print("\n" + "=" * 60)
print("Plotting Time Evolution of Tafel Slope")
print("=" * 60)

# Convert results to DataFrame for easier manipulation
results_df = pd.DataFrame(results)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot Tafel slope vs Time
ax.plot(results_df['time'], results_df['tafel_slope'], 
        'o-', color='darkred', linewidth=2, markersize=8, label='Tafel Slope')

# Add horizontal line for reference (typical OER Tafel slope ~40-60 mV/dec)
ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='40 mV/dec')
ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='60 mV/dec')

# Annotate each point with its value
for i, row in results_df.iterrows():
    ax.annotate(f"{row['tafel_slope']:.1f}",
                xy=(row['time'], row['tafel_slope']),
                xytext=(row['time'], row['tafel_slope'] + 1),
                ha='center', va='bottom', fontsize=9)

# Format plot
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
ax.set_title('Sample 499: Time Evolution of Tafel Slope from R1', fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('499_tafel_slope_time_evolution.png', dpi=300, bbox_inches='tight')
print(f"Saved plot: 499_tafel_slope_time_evolution.png")
plt.show()

# ============================================================================
# Plot 3: Detailed view with fit information
# ============================================================================

print("\n" + "=" * 60)
print("Creating detailed analysis table")
print("=" * 60)

# Create a summary table
summary = results_df[['time', 'tafel_slope', 'k', 'q', 'n_points', 'fit_points']].copy()
summary.columns = ['Time', 'Tafel Slope (mV/dec)', 'k (1/Ohm/mA)', 'q (1/Ohm)', 'Total Points', 'Fit Points']

# Round values for display
summary['Tafel Slope (mV/dec)'] = summary['Tafel Slope (mV/dec)'].round(2)
summary['k (1/Ohm/mA)'] = summary['k (1/Ohm/mA)'].round(4)
summary['q (1/Ohm)'] = summary['q (1/Ohm)'].round(4)

print("\nTafel Analysis Summary for Sample 499:")
print(summary.to_string(index=False))

# Save summary to CSV
summary.to_csv('499_tafel_analysis_summary.csv', index=False)
print(f"\nSaved summary: 499_tafel_analysis_summary.csv")

# ============================================================================
# Additional Plot: Log-scale view
# ============================================================================

print("\n" + "=" * 60)
print("Plotting log-scale view of 1/R1 vs Current")
print("=" * 60)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for (time, group), color in zip(sorted(time_groups.items()), colors):
    result = analyze_time_point(group, time, color, ax)
    if result:
        # Re-do the analysis to get the data for log plot
        group = group.sort_values('I_mA')
        group = group[(group['I_mA'] >= MIN_CURRENT) & (group['I_mA'] <= MAX_CURRENT)]
        i_short = group['I_mA'].iloc[0]
        I_corr = group['I_mA'].values - i_short
        R1_inv = (1 / group['R1'].values)
        valid_mask = (R1_inv > 0) & (I_corr > 0)
        I_corr = I_corr[valid_mask]
        R1_inv = R1_inv[valid_mask]
        
        if len(I_corr) >= 5:
            ax.plot(I_corr, R1_inv, 'o', color=color, alpha=0.5, markersize=4)

# Format log plot
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Corrected Current (mA) - Log Scale', fontsize=12)
ax.set_ylabel('1/R1 (1/Ohm) - Log Scale', fontsize=12)
ax.set_title('Sample 499: 1/R1 vs Corrected Current (Log-Log Scale)', fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('499_tafel_1_over_R1_vs_current_loglog.png', dpi=300, bbox_inches='tight')
print(f"Saved plot: 499_tafel_1_over_R1_vs_current_loglog.png")
plt.show()

print("\n" + "=" * 60)
print("Tafel Analysis Complete for Sample 499!")
print("=" * 60)
