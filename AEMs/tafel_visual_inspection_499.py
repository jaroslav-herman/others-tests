#!/usr/bin/env python3
"""
Visual Inspection of Tafel Analysis for Sample 499
Creates one plot per Time point showing ALL measured points
Highlights which points are in/out of current range
Helps determine optimal fit range
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Load and prepare data
# ============================================================================

df = pd.read_csv('499_fit_export.csv')

# Clean up duplicate column names
if 'Ecell_V.1' in df.columns:
    df = df.drop(columns=['Ecell_V.1'])

# ============================================================================
# Parameters - User can adjust these
# ============================================================================

CURRENT_MIN = 5    # mA - minimum current for analysis
CURRENT_MAX = 300  # mA - maximum current for analysis (can be extended)
FIT_START_PCT = 0.2  # 20% of data from start
FIT_END_PCT = 0.8    # 80% of data from start

# ============================================================================
# Plotting Function
# ============================================================================

def plot_time_point_inspection(group: pd.DataFrame, time: int, 
                               current_min: float = CURRENT_MIN,
                               current_max: float = CURRENT_MAX,
                               output_dir: Path = None) -> dict:
    """
    Create a detailed inspection plot for a single time point.
    Shows ALL points and clearly marks which are in/out of range.
    """
    # Sort by current
    group = group.sort_values('I_mA')
    
    # Correct for short circuit (subtract current at lowest voltage)
    i_short = group['I_mA'].iloc[0]
    I_corr = group['I_mA'].values - i_short
    R1_inv = 1 / group['R1'].values
    Ecell = group['Ecell_V'].values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ===== Top plot: 1/R1 vs Corrected Current =====
    
    # All points in gray
    ax1.plot(I_corr, R1_inv, 'o', color='lightgray', alpha=0.5, 
             markersize=6, label='All points')
    
    # Points below current_min
    below_min = I_corr < current_min
    if np.any(below_min):
        ax1.plot(I_corr[below_min], R1_inv[below_min], 'o', 
                 color='blue', alpha=0.7, markersize=6, 
                 label=f'I < {current_min} mA (excluded)')
    
    # Points above current_max
    above_max = I_corr > current_max
    if np.any(above_max):
        ax1.plot(I_corr[above_max], R1_inv[above_max], 'o', 
                 color='purple', alpha=0.7, markersize=6, 
                 label=f'I > {current_max} mA (excluded)')
    
    # Points in range
    in_range = (I_corr >= current_min) & (I_corr <= current_max)
    I_in = I_corr[in_range]
    R1_inv_in = R1_inv[in_range]
    
    if np.any(in_range):
        ax1.plot(I_in, R1_inv_in, 'o', color='green', alpha=0.9, 
                 markersize=6, label=f'In range [{current_min}-{current_max} mA]')
    
    # Fit line on in-range points
    if len(I_in) >= 5:
        # Use middle portion for fit
        start_idx = int(len(I_in) * FIT_START_PCT)
        end_idx = int(len(I_in) * FIT_END_PCT)
        I_fit = I_in[start_idx:end_idx]
        R1_inv_fit = R1_inv_in[start_idx:end_idx]
        
        k, q = np.polyfit(I_fit, R1_inv_fit, 1)
        tafel = 2.3 / k if k != 0 else np.nan
        
        # Plot fit line
        I_line = np.linspace(I_in.min() * 0.8, I_in.max() * 1.2, 100)
        R1_inv_line = k * I_line + q
        ax1.plot(I_line, R1_inv_line, '-', color='red', linewidth=2, 
                 label=f'Fit on {start_idx*100:.0f}-{end_idx*100:.0f}% of range')
        
        # Mark fit range boundaries
        ax1.axvline(x=I_in[start_idx], color='orange', linestyle='--', alpha=0.7)
        ax1.axvline(x=I_in[end_idx-1], color='orange', linestyle='--', alpha=0.7)
        
        # Annotate Tafel slope
        ax1.annotate(f'Tafel = {tafel:.1f} mV/dec',
                     xy=(I_in[-1], R1_inv_in[-1]),
                     xytext=(I_in[-1] * 0.6, R1_inv_in[-1] * 1.2),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=12, color='red', fontweight='bold')
    
    ax1.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax1.set_ylabel('1/R1 (1/Ω)', fontsize=12)
    ax1.set_title(f'Sample 499 - Time {time}: 1/R1 vs Corrected Current', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.legend(fontsize=9, loc='upper left')
    
    # ===== Bottom plot: Ecell vs Current (IV curve) =====
    
    # All points
    ax2.plot(I_corr, Ecell, 'o', color='lightgray', alpha=0.5, markersize=6)
    
    # Points below current_min
    if np.any(below_min):
        ax2.plot(I_corr[below_min], Ecell[below_min], 'o', 
                 color='blue', alpha=0.7, markersize=6)
    
    # Points above current_max
    if np.any(above_max):
        ax2.plot(I_corr[above_max], Ecell[above_max], 'o', 
                 color='purple', alpha=0.7, markersize=6)
    
    # Points in range
    if np.any(in_range):
        ax2.plot(I_in, Ecell[in_range], 'o', color='green', alpha=0.9, 
                 markersize=6, label=f'Fit range')
    
    ax2.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax2.set_ylabel('Cell Voltage Ecell (V)', fontsize=12)
    ax2.set_title(f'IV Curve - Time {time}', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend(fontsize=9, loc='lower right')
    
    # Add vertical lines for range boundaries
    ax1.axvline(x=current_min, color='blue', linestyle=':', alpha=0.5)
    ax1.axvline(x=current_max, color='purple', linestyle=':', alpha=0.5)
    ax2.axvline(x=current_min, color='blue', linestyle=':', alpha=0.5)
    ax2.axvline(x=current_max, color='purple', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'499_inspect_time_{time:02d}.png', 
                   dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Return analysis results
    return {
        'time': time,
        'i_short': i_short,
        'n_total': len(I_corr),
        'n_below_min': np.sum(below_min),
        'n_above_max': np.sum(above_max),
        'n_in_range': np.sum(in_range),
        'current_range': f'{I_corr.min():.1f}-{I_corr.max():.1f} mA',
        'tafel': tafel if len(I_in) >= 5 else None
    }


# ============================================================================
# Main Analysis
# ============================================================================

# Group by Time
time_groups = {int(t): group for t, group in df.groupby('Time')}

# Create output directory
output_dir = Path('tafel_inspection_499')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Visual Inspection: Sample 499 - Tafel Analysis")
print(f"Current range for analysis: {CURRENT_MIN}-{CURRENT_MAX} mA")
print(f"Fit uses {FIT_START_PCT*100:.0f}-{FIT_END_PCT*100:.0f}% of in-range points")
print("=" * 80)

# Process each time point
all_results = []

for time in sorted(time_groups.keys()):
    group = time_groups[time]
    print(f"\nProcessing Time {time}...")
    
    result = plot_time_point_inspection(group, time, CURRENT_MIN, CURRENT_MAX, output_dir)
    
    if result:
        all_results.append(result)
        print(f"  → Total points: {result['n_total']}")
        print(f"  → Below {CURRENT_MIN} mA: {result['n_below_min']}")
        print(f"  → Above {CURRENT_MAX} mA: {result['n_above_max']}")
        print(f"  → In range: {result['n_in_range']}")
        print(f"  → Current range: {result['current_range']}")
        if result['tafel']:
            print(f"  → Tafel slope: {result['tafel']:.1f} mV/dec")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Save summary
summary_file = output_dir / '499_inspection_summary.csv'
results_df.to_csv(summary_file, index=False)
print(f"\nSummary saved to: {summary_file}")

# Count files
plot_files = list(output_dir.glob('499_inspect_time_*.png'))
print(f"Created {len(plot_files)} inspection plots")
print(f"All files in: {output_dir}")

print("\n" + "=" * 80)
print("Inspection Complete!")
print("Each plot shows:")
print("  - Top: 1/R1 vs Corrected Current with points color-coded by range")
print("  - Bottom: IV curve (Ecell vs Current)")
print("  - Gray: All points")
print("  - Blue: Below current_min (excluded)")
print("  - Purple: Above current_max (excluded)")
print("  - Green: In analysis range")
print("  - Red line: Linear fit on middle portion of in-range points")
print("=" * 80)
