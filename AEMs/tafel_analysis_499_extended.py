#!/usr/bin/env python3
"""
Extended Tafel Analysis for Sample 499
- Current range extended to 5-500 mA
- Force line intersection with y-axis to 0 (q=0)
- One plot per Time point showing all data and fit
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

CURRENT_MIN = 5    # mA
CURRENT_MAX = 500  # mA - EXTENDED from 300
FORCE_ZERO_INTERCEPT = True  # Force q=0
FIT_START_PCT = 0.2
FIT_END_PCT = 0.8

# ============================================================================
# Analysis Functions
# ============================================================================

def plot_time_point_extended(group: pd.DataFrame, time: int, 
                             output_dir: Path = None) -> dict:
    """
    Plot a single time point with extended range and forced zero intercept.
    """
    # Sort by current
    group = group.sort_values('I_mA')
    
    # Correct for short circuit
    i_short = group['I_mA'].iloc[0]
    I_corr = group['I_mA'].values - i_short
    R1_inv = 1 / group['R1'].values
    Ecell = group['Ecell_V'].values
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ===== Top plot: 1/R1 vs Corrected Current =====
    
    # All points
    ax1.plot(I_corr, R1_inv, 'o', color='lightgray', alpha=0.5, 
             markersize=6, label='All points')
    
    # Points below current_min
    below_min = I_corr < CURRENT_MIN
    if np.any(below_min):
        ax1.plot(I_corr[below_min], R1_inv[below_min], 'o', 
                 color='blue', alpha=0.7, markersize=6, 
                 label=f'I < {CURRENT_MIN} mA')
    
    # Points above current_max
    above_max = I_corr > CURRENT_MAX
    if np.any(above_max):
        ax1.plot(I_corr[above_max], R1_inv[above_max], 'o', 
                 color='purple', alpha=0.7, markersize=6, 
                 label=f'I > {CURRENT_MAX} mA')
    
    # Points in range
    in_range = (I_corr >= CURRENT_MIN) & (I_corr <= CURRENT_MAX)
    I_in = I_corr[in_range]
    R1_inv_in = R1_inv[in_range]
    
    if np.any(in_range):
        ax1.plot(I_in, R1_inv_in, 'o', color='green', alpha=0.9, 
                 markersize=6, label=f'In range [{CURRENT_MIN}-{CURRENT_MAX} mA]')
    
    # Fit line with forced zero intercept (q=0)
    if len(I_in) >= 5:
        start_idx = int(len(I_in) * FIT_START_PCT)
        end_idx = int(len(I_in) * FIT_END_PCT)
        I_fit = I_in[start_idx:end_idx]
        R1_inv_fit = R1_inv_in[start_idx:end_idx]
        
        if FORCE_ZERO_INTERCEPT:
            # Force q=0, fit only k
            k = np.sum(I_fit * R1_inv_fit) / np.sum(I_fit ** 2)
            q = 0
            tafel = 2.3 / k if k != 0 else np.nan
            
            # Plot fit line (through origin)
            I_line = np.linspace(0, I_in.max() * 1.2, 100)
            R1_inv_line = k * I_line
            ax1.plot(I_line, R1_inv_line, '-', color='red', linewidth=2, 
                     label=f'Fit (q=0): Tafel={tafel:.1f} mV/dec')
        else:
            k, q = np.polyfit(I_fit, R1_inv_fit, 1)
            tafel = 2.3 / k if k != 0 else np.nan
            I_line = np.linspace(I_in.min() * 0.8, I_in.max() * 1.2, 100)
            R1_inv_line = k * I_line + q
            ax1.plot(I_line, R1_inv_line, '-', color='red', linewidth=2, 
                     label=f'Fit: Tafel={tafel:.1f} mV/dec')
        
        # Mark fit range
        ax1.axvline(x=I_in[start_idx], color='orange', linestyle='--', alpha=0.7)
        ax1.axvline(x=I_in[end_idx-1], color='orange', linestyle='--', alpha=0.7)
        
        # Annotate
        ax1.annotate(f'Tafel = {tafel:.1f} mV/dec',
                     xy=(I_in[-1], R1_inv_in[-1]),
                     xytext=(I_in[-1] * 0.6, R1_inv_in[-1] * 1.2),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=12, color='red', fontweight='bold')
    
    ax1.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax1.set_ylabel('1/R1 (1/Ω)', fontsize=12)
    ax1.set_title(f'Sample 499 - Time {time}: 1/R1 vs Corrected Current (5-500 mA, q=0)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.legend(fontsize=9, loc='upper left')
    
    # ===== Bottom plot: IV curve =====
    
    ax2.plot(I_corr, Ecell, 'o', color='lightgray', alpha=0.5, markersize=6)
    if np.any(below_min):
        ax2.plot(I_corr[below_min], Ecell[below_min], 'o', 
                 color='blue', alpha=0.7, markersize=6)
    if np.any(above_max):
        ax2.plot(I_corr[above_max], Ecell[above_max], 'o', 
                 color='purple', alpha=0.7, markersize=6)
    if np.any(in_range):
        ax2.plot(I_in, Ecell[in_range], 'o', color='green', alpha=0.9, 
                 markersize=6, label=f'Fit range')
    
    ax2.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax2.set_ylabel('Cell Voltage Ecell (V)', fontsize=12)
    ax2.set_title(f'IV Curve - Time {time}', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend(fontsize=9, loc='lower right')
    
    # Vertical lines for range
    for ax in [ax1, ax2]:
        ax.axvline(x=CURRENT_MIN, color='blue', linestyle=':', alpha=0.5)
        ax.axvline(x=CURRENT_MAX, color='purple', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'499_tafel_ext_time_{time:02d}.png', 
                   dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'time': time,
        'tafel': tafel if len(I_in) >= 5 else None,
        'k': k if len(I_in) >= 5 else None,
        'q': q,
        'n_total': len(I_corr),
        'n_in_range': np.sum(in_range),
        'current_range': f'{I_corr.min():.1f}-{I_corr.max():.1f} mA'
    }


# ============================================================================
# Main Analysis
# ============================================================================

# Group by Time
time_groups = {int(t): group for t, group in df.groupby('Time')}

# Create output directory
output_dir = Path('tafel_extended_499')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Extended Tafel Analysis: Sample 499")
print(f"Current range: {CURRENT_MIN}-{CURRENT_MAX} mA")
print(f"Forced zero intercept: {FORCE_ZERO_INTERCEPT}")
print("=" * 80)

# Process each time point
all_results = []

for time in sorted(time_groups.keys()):
    group = time_groups[time]
    print(f"\nProcessing Time {time}...")
    
    result = plot_time_point_extended(group, time, output_dir)
    
    if result and result['tafel']:
        all_results.append(result)
        print(f"  → Tafel slope: {result['tafel']:.1f} mV/dec")
        print(f"  → k: {result['k']:.4e}")
        print(f"  → Points in range: {result['n_in_range']}")
        print(f"  → Current range: {result['current_range']}")
    else:
        print(f"  → Insufficient data")

# ============================================================================
# Comparison Plot
# ============================================================================

print("\n" + "=" * 80)
print("Creating time evolution plot")
print("=" * 80)

results_df = pd.DataFrame(all_results)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot Tafel slope vs Time
ax.plot(results_df['time'], results_df['tafel'], 
        'o-', color='darkred', linewidth=2, markersize=8, label='Tafel Slope (q=0)')

# Add reference lines
ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='40 mV/dec')
ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='60 mV/dec')
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80 mV/dec')

# Annotate each point
for i, row in results_df.iterrows():
    ax.annotate(f"{row['tafel']:.1f}",
                xy=(row['time'], row['tafel']),
                xytext=(row['time'], row['tafel'] + 1),
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
ax.set_title(f'Sample 499: Time Evolution of Tafel Slope (5-500 mA, q=0)', fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '499_tafel_time_evolution_extended.png', dpi=300, bbox_inches='tight')
print(f"Saved: 499_tafel_time_evolution_extended.png")
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

summary = results_df[['time', 'tafel', 'k', 'n_in_range', 'current_range']].copy()
summary.columns = ['Time', 'Tafel Slope (mV/dec)', 'k', 'Points in Range', 'Current Range']
summary['Tafel Slope (mV/dec)'] = summary['Tafel Slope (mV/dec)'].round(1)
summary['k'] = summary['k'].round(4)

print(summary.to_string(index=False))

# Calculate statistics
print(f"\nStatistics:")
print(f"  Mean Tafel slope: {results_df['tafel'].mean():.1f} mV/dec")
print(f"  Std Tafel slope: {results_df['tafel'].std():.1f} mV/dec")
print(f"  Min Tafel slope: {results_df['tafel'].min():.1f} mV/dec")
print(f"  Max Tafel slope: {results_df['tafel'].max():.1f} mV/dec")

# Save summary
summary_file = output_dir / '499_tafel_extended_summary.csv'
summary.to_csv(summary_file, index=False)
print(f"\nSummary saved to: {summary_file}")

# Count files
plot_files = list(output_dir.glob('499_tafel_ext_time_*.png'))
print(f"Created {len(plot_files)} individual plots")
print(f"All files in: {output_dir}")

print("\n" + "=" * 80)
print("Extended Analysis Complete!")
print("Key changes: Current range 5-500 mA, forced zero intercept (q=0)")
print("=" * 80)
