#!/usr/bin/env python3
"""
Optimized Tafel Analysis for Sample 499
Finds the best current range for each Time point
Shows all points and lets user see where the linear region is
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
# Parameters - MORE FLEXIBLE
# ============================================================================

MIN_CURRENT = 5    # mA - minimum current
MAX_CURRENT = 1000  # mA - maximum current (increased from 300)
MIN_POINTS_FOR_FIT = 8  # Minimum points needed for a good fit

# ============================================================================
# Analysis Functions
# ============================================================================

def find_optimal_fit_range(I: np.ndarray, R1_inv: np.ndarray) -> tuple:
    """
    Find the optimal range for linear fit by checking R^2 for different ranges.
    Returns the best start and end indices.
    """
    n = len(I)
    if n < MIN_POINTS_FOR_FIT:
        return 0, n
    
    best_r2 = -1
    best_start = 0
    best_end = n
    
    # Try different start positions (from 0 to 50% of data)
    for start_pct in np.linspace(0, 0.5, 11):
        start_idx = int(start_pct * n)
        
        # Try different end positions (from start+MIN_POINTS to 100%)
        for end_pct in np.linspace((start_idx + MIN_POINTS_FOR_FIT) / n, 1.0, 11):
            end_idx = int(end_pct * n)
            
            if end_idx <= start_idx + MIN_POINTS_FOR_FIT:
                continue
            
            I_fit = I[start_idx:end_idx]
            R1_inv_fit = R1_inv[start_idx:end_idx]
            
            # Perform fit
            k, q = np.polyfit(I_fit, R1_inv_fit, 1)
            
            # Calculate R^2
            R1_inv_pred = k * I_fit + q
            ss_res = np.sum((R1_inv_fit - R1_inv_pred) ** 2)
            ss_tot = np.sum((R1_inv_fit - np.mean(R1_inv_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            if r2 > best_r2:
                best_r2 = r2
                best_start = start_idx
                best_end = end_idx
    
    return best_start, best_end


def plot_time_point_with_optimization(group: pd.DataFrame, time: int, 
                                      output_dir: Path = None, 
                                      show_all: bool = True) -> dict:
    """
    Plot a single time point with:
    - All measured points
    - Optimal fit range highlighted
    - Fit line
    - Multiple possible fit ranges for comparison
    """
    # Sort by current
    group = group.sort_values('I_mA')
    
    # Correct for short circuit
    i_short = group['I_mA'].iloc[0]
    I_all = group['I_mA'].values - i_short
    R1_inv_all = (1 / group['R1'].values)
    
    # Filter valid points
    valid_mask = (R1_inv_all > 0) & (I_all > 0) & (I_all <= MAX_CURRENT) & (I_all >= MIN_CURRENT)
    I_valid = I_all[valid_mask]
    R1_inv_valid = R1_inv_all[valid_mask]
    
    if len(I_valid) < MIN_POINTS_FOR_FIT:
        print(f"Time {time}: Only {len(I_valid)} valid points, skipping")
        return None
    
    # Find optimal fit range
    opt_start, opt_end = find_optimal_fit_range(I_valid, R1_inv_valid)
    I_opt = I_valid[opt_start:opt_end]
    R1_inv_opt = R1_inv_valid[opt_start:opt_end]
    
    # Perform fit on optimal range
    k_opt, q_opt = np.polyfit(I_opt, R1_inv_opt, 1)
    tafel_opt = 2.3 / k_opt if k_opt != 0 else np.nan
    
    # Also try fit on full valid range
    k_full, q_full = np.polyfit(I_valid, R1_inv_valid, 1)
    tafel_full = 2.3 / k_full if k_full != 0 else np.nan
    
    # Also try fit on first half of valid range
    mid = len(I_valid) // 2
    if mid >= MIN_POINTS_FOR_FIT:
        k_half1, q_half1 = np.polyfit(I_valid[:mid], R1_inv_valid[:mid], 1)
        tafel_half1 = 2.3 / k_half1 if k_half1 != 0 else np.nan
    else:
        tafel_half1 = np.nan
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Plot all valid points
    ax.plot(I_valid, R1_inv_valid, 'o', color='blue', alpha=0.6, 
            markersize=6, label=f'All valid points (n={len(I_valid)})')
    
    # Highlight optimal fit range
    ax.plot(I_opt, R1_inv_opt, 'o', color='red', alpha=1, 
            markersize=8, fillstyle='none', linewidth=2, 
            label=f'Optimal fit range (n={len(I_opt)})')
    
    # Plot optimal fit line
    I_line = np.linspace(I_valid.min() * 0.9, I_valid.max() * 1.1, 100)
    R1_inv_opt_line = k_opt * I_line + q_opt
    ax.plot(I_line, R1_inv_opt_line, '-', color='darkred', linewidth=2, 
            label=f'Optimal fit: Tafel={tafel_opt:.1f} mV/dec')
    
    # Plot full range fit line
    R1_inv_full_line = k_full * I_line + q_full
    ax.plot(I_line, R1_inv_full_line, '--', color='green', linewidth=1.5, 
            label=f'Full range fit: Tafel={tafel_full:.1f} mV/dec')
    
    # Add reference Tafel slopes
    for tafel_ref in [40, 60, 80, 100]:
        k_ref = 2.3 / tafel_ref
        R1_inv_ref = k_ref * I_line + q_opt  # Use same intercept
        ax.plot(I_line, R1_inv_ref, ':', color='gray', linewidth=1, 
                label=f'{tafel_ref} mV/dec' if tafel_ref == 40 else "")
    
    # Add annotations
    ax.set_title(f'Sample 499 - Time {time}\n' + 
                 f'Optimal: {tafel_opt:.1f} mV/dec | Full: {tafel_full:.1f} mV/dec | Half: {tafel_half1:.1f} mV/dec',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax.set_ylabel('1/R1 (1/Ohm)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    
    # Add vertical lines for range
    ax.axvline(x=I_valid[opt_start], color='red', linestyle=':', alpha=0.7)
    ax.axvline(x=I_valid[opt_end-1], color='red', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'499_tafel_opt_time_{time:02d}.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'time': time,
        'tafel_optimal': tafel_opt,
        'tafel_full': tafel_full,
        'tafel_half1': tafel_half1,
        'k_opt': k_opt,
        'q_opt': q_opt,
        'n_valid': len(I_valid),
        'n_optimal': len(I_opt),
        'i_short': i_short,
        'current_range': f'{I_valid.min():.1f}-{I_valid.max():.1f} mA',
        'fit_range': f'{I_opt.min():.1f}-{I_opt.max():.1f} mA'
    }


# ============================================================================
# Main Analysis
# ============================================================================

# Group by Time
time_groups = {int(t): group for t, group in df.groupby('Time')}

# Create output directory
output_dir = Path('tafel_plots_499_optimized')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Optimized Tafel Analysis: Sample 499")
print(f"Using current range: {MIN_CURRENT}-{MAX_CURRENT} mA")
print(f"Minimum points for fit: {MIN_POINTS_FOR_FIT}")
print("=" * 80)

# Store all results
all_results = []

# Process each time point
for time in sorted(time_groups.keys()):
    group = time_groups[time]
    print(f"\nProcessing Time {time}...")
    
    result = plot_time_point_with_optimization(group, time, output_dir)
    
    if result:
        all_results.append(result)
        print(f"  → Optimal Tafel: {result['tafel_optimal']:.1f} mV/dec")
        print(f"  → Full range Tafel: {result['tafel_full']:.1f} mV/dec")
        print(f"  → Current range: {result['current_range']}")
        print(f"  → Fit range: {result['fit_range']}")
        print(f"  → Points: {result['n_valid']} valid, {result['n_optimal']} optimal")
    else:
        print(f"  → Insufficient data")

# ============================================================================
# Comparison Plot
# ============================================================================

print("\n" + "=" * 80)
print("Creating comparison plot of all three fitting methods")
print("=" * 80)

results_df = pd.DataFrame(all_results)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot all three methods
ax.plot(results_df['time'], results_df['tafel_optimal'], 
        'o-', color='red', linewidth=2, markersize=8, label='Optimal range fit')
ax.plot(results_df['time'], results_df['tafel_full'], 
        's-', color='green', linewidth=2, markersize=8, label='Full range fit')
ax.plot(results_df['time'], results_df['tafel_half1'], 
        '^-', color='blue', linewidth=2, markersize=8, label='First half fit')

# Add reference lines
ax.axhline(y=40, color='gray', linestyle='--', alpha=0.5, label='40 mV/dec')
ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='60 mV/dec')
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80 mV/dec')

# Annotate
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
ax.set_title('Sample 499: Tafel Slope Comparison - Three Fitting Methods', fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '499_tafel_comparison_all_methods.png', dpi=300, bbox_inches='tight')
print(f"Saved comparison plot: 499_tafel_comparison_all_methods.png")
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

# Create detailed summary
summary_cols = ['time', 'tafel_optimal', 'tafel_full', 'tafel_half1', 
               'n_valid', 'n_optimal', 'current_range', 'fit_range']
summary = results_df[summary_cols].copy()
summary.columns = ['Time', 'Tafel Optimal', 'Tafel Full', 'Tafel Half1',
                 'Valid Pts', 'Optimal Pts', 'Current Range', 'Fit Range']

# Round values
for col in ['Tafel Optimal', 'Tafel Full', 'Tafel Half1']:
    summary[col] = summary[col].round(1)

print(summary.to_string(index=False))

# Save summary
summary_file = output_dir / '499_tafel_optimized_summary.csv'
summary.to_csv(summary_file, index=False)
print(f"\nOptimized summary saved to: {summary_file}")

# Count files created
plot_files = list(output_dir.glob('499_tafel_opt_time_*.png'))
print(f"Created {len(plot_files)} individual optimized time point plots")
print(f"All plots saved to: {output_dir}")

print("\n" + "=" * 80)
print("Optimized Analysis Complete!")
print("=" * 80)
