#!/usr/bin/env python3
"""
Tafel Analysis for Sample 499 with 700 mA upper limit
Extended range to capture more data points
Force q=0 (zero intercept)
Focus on outlier impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ============================================================================
# Load and prepare data
# ============================================================================

df = pd.read_csv('499_fit_export.csv')

# Clean up duplicate column names
if 'Ecell_V.1' in df.columns:
    df = df.drop(columns=['Ecell_V.1'])

# ============================================================================
# Parameters
# ============================================================================

CURRENT_MIN = 5    # mA
CURRENT_MAX = 700  # mA - EXTENDED from 500
FORCE_ZERO_INTERCEPT = True
FIT_START_PCT = 0.2
FIT_END_PCT = 0.8
OUTLIER_Z_SCORE = 2.5

# ============================================================================
# Outlier Detection
# ============================================================================

def detect_outliers_residuals(I: np.ndarray, R1_inv: np.ndarray, k: float, q: float = 0) -> np.ndarray:
    """Detect outliers based on residuals from fit line."""
    R1_inv_pred = k * I + q
    residuals = R1_inv - R1_inv_pred
    std_residuals = np.std(residuals)
    if std_residuals == 0 or np.isnan(std_residuals):
        return np.zeros(len(residuals), dtype=bool)
    z_residuals = np.abs(residuals / std_residuals)
    return z_residuals > OUTLIER_Z_SCORE


# ============================================================================
# Analysis Function
# ============================================================================

def analyze_time_point(group: pd.DataFrame, time: int, output_dir: Path = None) -> dict:
    """
    Analyze a single time point with outlier detection.
    Compare Tafel slopes with and without outliers.
    """
    # Sort by current
    group = group.sort_values('I_mA')
    
    # Correct for short circuit
    i_short = group['I_mA'].iloc[0]
    I_corr = group['I_mA'].values - i_short
    R1_inv = 1 / group['R1'].values
    
    # Filter to range
    in_range = (I_corr >= CURRENT_MIN) & (I_corr <= CURRENT_MAX)
    I_in = I_corr[in_range]
    R1_inv_in = R1_inv[in_range]
    
    if len(I_in) < 5:
        return None
    
    # ===== Fit WITH all points =====
    start_idx = int(len(I_in) * FIT_START_PCT)
    end_idx = int(len(I_in) * FIT_END_PCT)
    I_fit = I_in[start_idx:end_idx]
    R1_inv_fit = R1_inv_in[start_idx:end_idx]
    
    k_with = np.sum(I_fit * R1_inv_fit) / np.sum(I_fit ** 2)
    tafel_with = 2.3 / k_with if k_with != 0 else np.nan
    
    # Calculate residuals for ALL in-range points
    R1_inv_pred_all = k_with * I_in
    residuals = R1_inv_in - R1_inv_pred_all
    std_res = np.std(residuals)
    
    # Detect outliers
    if std_res > 0 and not np.isnan(std_res):
        outliers = np.abs(residuals / std_res) > OUTLIER_Z_SCORE
    else:
        outliers = np.zeros(len(I_in), dtype=bool)
    
    n_outliers = np.sum(outliers)
    
    # ===== Fit WITHOUT outliers =====
    I_clean = I_in[~outliers]
    R1_inv_clean = R1_inv_in[~outliers]
    
    if len(I_clean) >= 5:
        start_idx_clean = int(len(I_clean) * FIT_START_PCT)
        end_idx_clean = int(len(I_clean) * FIT_END_PCT)
        
        if end_idx_clean > start_idx_clean and start_idx_clean >= 0:
            I_fit_clean = I_clean[start_idx_clean:end_idx_clean]
            R1_inv_fit_clean = R1_inv_clean[start_idx_clean:end_idx_clean]
            
            k_without = np.sum(I_fit_clean * R1_inv_fit_clean) / \
                       np.sum(I_fit_clean ** 2)
            tafel_without = 2.3 / k_without if k_without != 0 else np.nan
            difference = tafel_without - tafel_with
        else:
            tafel_without = np.nan
            difference = np.nan
    else:
        tafel_without = np.nan
        difference = np.nan
    
    # ===== Create Plot =====
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: 1/R1 vs Current
    ax1.plot(I_in, R1_inv_in, 'o', color='green', alpha=0.6, 
             markersize=8, label=f'All in-range ({len(I_in)} pts)')
    
    if n_outliers > 0:
        ax1.plot(I_in[outliers], R1_inv_in[outliers], 'o', 
                 color='red', alpha=1, markersize=10, markeredgecolor='black',
                 markeredgewidth=2, label=f'Outliers ({n_outliers})')
    
    # Fit lines
    I_line = np.linspace(0, I_in.max() * 1.2, 100)
    
    # With outliers
    R1_inv_line_with = k_with * I_line
    ax1.plot(I_line, R1_inv_line_with, '-', color='blue', linewidth=2, 
             label=f'With outliers: {tafel_with:.1f} mV/dec')
    
    # Without outliers
    if not np.isnan(tafel_without):
        R1_inv_line_without = k_without * I_line
        ax1.plot(I_line, R1_inv_line_without, '-', color='darkgreen', linewidth=2, 
                 label=f'Without outliers: {tafel_without:.1f} mV/dec')
    
    ax1.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax1.set_ylabel('1/R1 (1/Ω)', fontsize=12)
    ax1.set_title(f'Sample 499 - Time {time}: 1/R1 vs Current (5-700 mA, q=0)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.legend(fontsize=9, loc='upper left')
    
    # Bottom: Residuals
    ax2.plot(I_in, residuals, 'o', color='blue', alpha=0.6, markersize=6, label='Residuals')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    if n_outliers > 0:
        ax2.plot(I_in[outliers], residuals[outliers], 'o', 
                 color='red', alpha=1, markersize=8, markeredgecolor='black',
                 label='Outlier residuals')
    
    # Std deviation lines
    if std_res > 0:
        ax2.axhline(y=std_res, color='green', linestyle='--', alpha=0.5, label=f'±1σ ({std_res:.4f})')
        ax2.axhline(y=-std_res, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=2*std_res, color='orange', linestyle='--', alpha=0.5, label=f'±2σ ({2*std_res:.4f})')
        ax2.axhline(y=-2*std_res, color='orange', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax2.set_ylabel('Residual (1/Ω)', fontsize=12)
    ax2.set_title(f'Residuals - Time {time}', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend(fontsize=9, loc='upper left')
    
    # Annotate difference
    if not np.isnan(difference):
        ax1.annotate(f'ΔTafel = {difference:+.1f} mV/dec',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, color='purple', fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'499_700ma_time_{time:02d}.png', 
                   dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'time': time,
        'tafel_with': tafel_with,
        'tafel_without': tafel_without,
        'difference': difference,
        'k_with': k_with,
        'k_without': k_without if not np.isnan(tafel_without) else np.nan,
        'n_in_range': len(I_in),
        'n_outliers': n_outliers,
        'n_without': len(I_clean),
        'std_residuals': float(std_res) if not np.isnan(std_res) else None,
        'outlier_I': list(I_in[outliers]) if n_outliers > 0 else [],
        'outlier_R1_inv': list(R1_inv_in[outliers]) if n_outliers > 0 else []
    }


# ============================================================================
# Main Analysis
# ============================================================================

# Group by Time
time_groups = {int(t): group for t, group in df.groupby('Time')}

# Create output directory
output_dir = Path('tafel_700ma_499')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Tafel Analysis: Sample 499 with 700 mA Upper Limit")
print(f"Current range: {CURRENT_MIN}-{CURRENT_MAX} mA")
print(f"Forced zero intercept: {FORCE_ZERO_INTERCEPT}")
print("=" * 80)

# Process each time point
all_results = []

for time in sorted(time_groups.keys()):
    group = time_groups[time]
    print(f"\nProcessing Time {time}...")
    
    result = analyze_time_point(group, time, output_dir)
    
    if result:
        all_results.append(result)
        print(f"  → With outliers: {result['tafel_with']:.1f} mV/dec")
        if not np.isnan(result['tafel_without']):
            print(f"  → Without outliers: {result['tafel_without']:.1f} mV/dec")
            print(f"  → Difference: {result['difference']:+.1f} mV/dec")
        print(f"  → Points in range: {result['n_in_range']}, Outliers: {result['n_outliers']}")
        if result['n_outliers'] > 0:
            print(f"  → Outlier I: {result['outlier_I']}")
            print(f"  → Outlier 1/R1: {result['outlier_R1_inv']}")
    else:
        print(f"  → Insufficient data")

# ============================================================================
# Summary and Comparison
# ============================================================================

print("\n" + "=" * 80)
print("Summary: Impact of Removing Outliers")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Calculate statistics
print(f"\nOverall Statistics:")
print(f"  Mean Tafel (with outliers): {results_df['tafel_with'].mean():.1f} mV/dec")
print(f"  Mean Tafel (without outliers): {results_df['tafel_without'].mean():.1f} mV/dec")
print(f"  Mean difference: {results_df['difference'].mean():+.1f} mV/dec")
print(f"  Max difference: {results_df['difference'].abs().max():.1f} mV/dec")
print(f"  Total outliers detected: {results_df['n_outliers'].sum()}")

# Detailed table
print("\nDetailed Results:")
detailed = results_df[['time', 'tafel_with', 'tafel_without', 'difference', 
                        'n_in_range', 'n_outliers']].copy()
detailed.columns = ['Time', 'With Outliers', 'Without Outliers', 'ΔTafel', 
                     'Pts In Range', 'Outliers']
detailed['With Outliers'] = detailed['With Outliers'].round(1)
detailed['Without Outliers'] = detailed['Without Outliers'].round(1)
detailed['ΔTafel'] = detailed['ΔTafel'].round(1)

print(detailed.to_string(index=False))

# ===== Plot: Tafel Slope Comparison =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# With vs Without
ax1.plot(results_df['time'], results_df['tafel_with'], 
        'o-', color='blue', linewidth=2, markersize=8, label='With outliers')
ax1.plot(results_df['time'], results_df['tafel_without'], 
        's-', color='green', linewidth=2, markersize=8, label='Without outliers')

ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
ax1.set_title('Tafel Slope: With vs Without Outliers (5-700 mA, q=0)', fontsize=14)
ax1.grid(True, which='both', linestyle='--', alpha=0.3)
ax1.legend(fontsize=10)

# Difference
ax2.bar(results_df['time'], results_df['difference'], 
        color='purple', alpha=0.7, label='ΔTafel (without - with)')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('ΔTafel (mV/dec)', fontsize=12)
ax2.set_title('Difference in Tafel Slope When Removing Outliers', fontsize=14)
ax2.grid(True, which='both', linestyle='--', alpha=0.3)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '499_700ma_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved comparison plot: 499_700ma_comparison.png")
plt.show()

# ===== Plot: Outlier Count vs Tafel Difference =====
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Scatter: n_outliers vs difference
for _, row in results_df.iterrows():
    if row['n_outliers'] > 0:
        ax.scatter(row['n_outliers'], row['difference'], 
                   s=100, alpha=0.7, label=f'Time {row["time"]}')

# Also add times with 0 outliers
for _, row in results_df[results_df['n_outliers'] == 0].iterrows():
    ax.scatter(row['n_outliers'], row['difference'], 
               s=100, alpha=0.7, color='gray', label=f'Time {row["time"]}')

ax.set_xlabel('Number of Outliers Detected', fontsize=12)
ax.set_ylabel('ΔTafel (mV/dec)', fontsize=12)
ax.set_title('Outlier Count vs Tafel Slope Change', fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# Annotate
for _, row in results_df.iterrows():
    ax.annotate(f"{row['time']}", xy=(row['n_outliers'], row['difference']),
                xytext=(row['n_outliers'] + 0.1, row['difference'] + 0.1),
                fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / '499_700ma_outliers_vs_difference.png', dpi=300, bbox_inches='tight')
print(f"Saved: 499_700ma_outliers_vs_difference.png")
plt.show()

# ============================================================================
# Save Results
# ============================================================================

# Save main summary
summary_file = output_dir / '499_700ma_summary.csv'
results_df.to_csv(summary_file, index=False)
print(f"\nSummary saved to: {summary_file}")

# Save detailed
comparison_file = output_dir / '499_700ma_comparison.csv'
detailed.to_csv(comparison_file, index=False)
print(f"Comparison saved to: {comparison_file}")

# Count files
plot_files = list(output_dir.glob('499_700ma_time_*.png'))
print(f"\nCreated {len(plot_files)} individual plots")
print(f"All files in: {output_dir}")

print("\n" + "=" * 80)
print("Analysis Complete!")
print(f"Current range: {CURRENT_MIN}-{CURRENT_MAX} mA, q=0")
print(f"Outlier detection: Z-score > {OUTLIER_Z_SCORE} on residuals")
print("=" * 80)
