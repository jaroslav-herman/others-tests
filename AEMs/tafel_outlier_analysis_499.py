#!/usr/bin/env python3
"""
Outlier Analysis for Tafel Slope - Sample 499
Evaluates which points are outliers and should not be fitted
Uses statistical methods to identify outliers in 1/R1 vs I data
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
CURRENT_MAX = 500  # mA
FORCE_ZERO_INTERCEPT = True
FIT_START_PCT = 0.2
FIT_END_PCT = 0.8
OUTLIER_Z_SCORE = 2.5  # Z-score threshold for outlier detection

# ============================================================================
# Outlier Detection Functions
# ============================================================================

def detect_outliers_zscore(data: np.ndarray) -> np.ndarray:
    """Detect outliers using Z-score method."""
    if len(data) < 3:
        return np.zeros(len(data), dtype=bool)
    z_scores = np.abs(stats.zscore(data))
    return z_scores > OUTLIER_Z_SCORE


def detect_outliers_iqr(data: np.ndarray) -> np.ndarray:
    """Detect outliers using IQR method."""
    if len(data) < 4:
        return np.zeros(len(data), dtype=bool)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_residuals(I: np.ndarray, R1_inv: np.ndarray, 
                              k: float, q: float) -> np.ndarray:
    """Detect outliers based on residuals from fit line."""
    R1_inv_pred = k * I + q
    residuals = R1_inv - R1_inv_pred
    std_residuals = np.std(residuals)
    if std_residuals == 0:
        return np.zeros(len(residuals), dtype=bool)
    z_residuals = np.abs(residuals / std_residuals)
    return z_residuals > OUTLIER_Z_SCORE


# ============================================================================
# Analysis and Plotting Function
# ============================================================================

def analyze_time_point_outliers(group: pd.DataFrame, time: int,
                                output_dir: Path = None) -> dict:
    """
    Analyze a single time point for outliers.
    Shows all points with outlier detection.
    """
    # Sort by current
    group = group.sort_values('I_mA')
    
    # Correct for short circuit
    i_short = group['I_mA'].iloc[0]
    I_corr = group['I_mA'].values - i_short
    R1_inv = 1 / group['R1'].values
    Ecell = group['Ecell_V'].values
    
    # Filter to range
    in_range = (I_corr >= CURRENT_MIN) & (I_corr <= CURRENT_MAX)
    I_in = I_corr[in_range]
    R1_inv_in = R1_inv[in_range]
    
    if len(I_in) < 5:
        return None
    
    # Fit with zero intercept
    start_idx = int(len(I_in) * FIT_START_PCT)
    end_idx = int(len(I_in) * FIT_END_PCT)
    I_fit = I_in[start_idx:end_idx]
    R1_inv_fit = R1_inv_in[start_idx:end_idx]
    
    k = np.sum(I_fit * R1_inv_fit) / np.sum(I_fit ** 2)
    q = 0
    tafel = 2.3 / k if k != 0 else np.nan
    
    # Calculate residuals for ALL in-range points
    R1_inv_pred_all = k * I_in
    residuals = R1_inv_in - R1_inv_pred_all
    
    # Detect outliers using multiple methods
    outliers_zscore = detect_outliers_zscore(R1_inv_in)
    outliers_iqr = detect_outliers_iqr(R1_inv_in)
    outliers_residuals = detect_outliers_residuals(I_in, R1_inv_in, k, q)
    
    # Combined: a point is outlier if ANY method flags it
    outliers_combined = outliers_zscore | outliers_iqr | outliers_residuals
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ===== Top plot: 1/R1 vs Current with outlier detection =====
    
    # All in-range points
    ax1.plot(I_in, R1_inv_in, 'o', color='green', alpha=0.6, 
             markersize=8, label='In-range points')
    
    # Outliers (combined)
    if np.any(outliers_combined):
        ax1.plot(I_in[outliers_combined], R1_inv_in[outliers_combined], 'o',
                 color='red', alpha=1, markersize=10, markeredgecolor='black',
                 markeredgewidth=2, label='Outliers (combined)')
    
    # Fit line
    I_line = np.linspace(0, I_in.max() * 1.2, 100)
    R1_inv_line = k * I_line
    ax1.plot(I_line, R1_inv_line, '-', color='darkred', linewidth=2, 
             label=f'Fit (q=0): Tafel={tafel:.1f} mV/dec')
    
    # Annotate outliers
    for i, (I_val, R1_inv_val) in enumerate(zip(I_in, R1_inv_in)):
        if outliers_combined[i]:
            ax1.annotate(f'Outlier {i}',
                         xy=(I_val, R1_inv_val),
                         xytext=(I_val * 1.05, R1_inv_val * 1.05),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=9, color='red')
    
    # Residuals subplot
    ax_res = ax1.twinx()
    # Don't show residuals on main plot, use second subplot instead
    
    ax1.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax1.set_ylabel('1/R1 (1/Ω)', fontsize=12)
    ax1.set_title(f'Sample 499 - Time {time}: Outlier Detection (5-500 mA, q=0)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.legend(fontsize=9, loc='upper left')
    
    # ===== Bottom plot: Residuals =====
    
    # Plot residuals
    ax2.plot(I_in, residuals, 'o', color='blue', alpha=0.6, markersize=6, label='Residuals')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Highlight outlier residuals
    if np.any(outliers_combined):
        ax2.plot(I_in[outliers_combined], residuals[outliers_combined], 'o',
                 color='red', alpha=1, markersize=8, markeredgecolor='black',
                 label='Outlier residuals')
    
    # Add std deviation lines
    std_res = np.std(residuals)
    ax2.axhline(y=std_res, color='green', linestyle='--', alpha=0.5, label=f'±1σ ({std_res:.4f})')
    ax2.axhline(y=-std_res, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=2*std_res, color='orange', linestyle='--', alpha=0.5, label=f'±2σ')
    ax2.axhline(y=-2*std_res, color='orange', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Corrected Current (mA)', fontsize=12)
    ax2.set_ylabel('Residual (1/Ω)', fontsize=12)
    ax2.set_title(f'Residuals from Fit - Time {time}', fontsize=14)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    ax2.legend(fontsize=9, loc='upper left')
    
    plt.tight_layout()
    
    # Save
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f'499_outliers_time_{time:02d}.png', 
                   dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return {
        'time': time,
        'tafel': tafel,
        'k': k,
        'n_in_range': len(I_in),
        'n_outliers_zscore': np.sum(outliers_zscore),
        'n_outliers_iqr': np.sum(outliers_iqr),
        'n_outliers_residuals': np.sum(outliers_residuals),
        'n_outliers_combined': np.sum(outliers_combined),
        'outlier_indices': list(np.where(outliers_combined)[0]),
        'outlier_I': list(I_in[outliers_combined]),
        'outlier_R1_inv': list(R1_inv_in[outliers_combined]),
        'std_residuals': float(std_res)
    }


# ============================================================================
# Main Analysis
# ============================================================================

# Group by Time
time_groups = {int(t): group for t, group in df.groupby('Time')}

# Create output directory
output_dir = Path('tafel_outliers_499')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Outlier Analysis: Sample 499 - Tafel Slope")
print(f"Current range: {CURRENT_MIN}-{CURRENT_MAX} mA")
print(f"Forced zero intercept: {FORCE_ZERO_INTERCEPT}")
print(f"Outlier detection: Z-score > {OUTLIER_Z_SCORE}, IQR, Residuals")
print("=" * 80)

# Process each time point
all_results = []

for time in sorted(time_groups.keys()):
    group = time_groups[time]
    print(f"\nProcessing Time {time}...")
    
    result = analyze_time_point_outliers(group, time, output_dir)
    
    if result:
        all_results.append(result)
        print(f"  → Tafel slope: {result['tafel']:.1f} mV/dec")
        print(f"  → Points in range: {result['n_in_range']}")
        print(f"  → Outliers (Z-score): {result['n_outliers_zscore']}")
        print(f"  → Outliers (IQR): {result['n_outliers_iqr']}")
        print(f"  → Outliers (Residuals): {result['n_outliers_residuals']}")
        print(f"  → Outliers (Combined): {result['n_outliers_combined']}")
        if result['n_outliers_combined'] > 0:
            print(f"  → Outlier I values: {result['outlier_I']}")
            print(f"  → Outlier 1/R1 values: {result['outlier_R1_inv']}")
    else:
        print(f"  → Insufficient data")

# ============================================================================
# Summary and Statistics
# ============================================================================

print("\n" + "=" * 80)
print("Outlier Summary")
print("=" * 80)

results_df = pd.DataFrame(all_results)

# Summary statistics
print(f"\nOverall Statistics:")
print(f"  Total time points analyzed: {len(results_df)}")
print(f"  Total outliers detected: {results_df['n_outliers_combined'].sum()}")
print(f"  Mean outliers per time point: {results_df['n_outliers_combined'].mean():.1f}")
print(f"  Time points with outliers: {(results_df['n_outliers_combined'] > 0).sum()}")

# Find time points with most outliers
top_outliers = results_df.nlargest(5, 'n_outliers_combined')
print(f"\nTop 5 Time Points by Number of Outliers:")
for _, row in top_outliers.iterrows():
    print(f"  Time {row['time']}: {row['n_outliers_combined']} outliers, Tafel={row['tafel']:.1f} mV/dec")

# ============================================================================
# Comparison: With vs Without Outliers
# ============================================================================

print("\n" + "=" * 80)
print("Comparison: Tafel Slopes With vs Without Outliers")
print("=" * 80)

# Calculate Tafel slopes WITH outliers removed
results_no_outliers = []

for time in sorted(time_groups.keys()):
    group = time_groups[time]
    group = group.sort_values('I_mA')
    i_short = group['I_mA'].iloc[0]
    I_corr = group['I_mA'].values - i_short
    R1_inv = 1 / group['R1'].values
    
    in_range = (I_corr >= CURRENT_MIN) & (I_corr <= CURRENT_MAX)
    I_in = I_corr[in_range]
    R1_inv_in = R1_inv[in_range]
    
    if len(I_in) < 5:
        continue
    
    # Detect outliers
    k_temp = np.sum(I_in * R1_inv_in) / np.sum(I_in ** 2)
    residuals_temp = R1_inv_in - k_temp * I_in
    std_res_temp = np.std(residuals_temp)
    if std_res_temp > 0:
        outliers_temp = np.abs(residuals_temp / std_res_temp) > OUTLIER_Z_SCORE
    else:
        outliers_temp = np.zeros(len(I_in), dtype=bool)
    
    # Fit WITH outliers
    start_idx = int(len(I_in) * FIT_START_PCT)
    end_idx = int(len(I_in) * FIT_END_PCT)
    k_with = np.sum(I_in[start_idx:end_idx] * R1_inv_in[start_idx:end_idx]) / \
            np.sum(I_in[start_idx:end_idx] ** 2)
    tafel_with = 2.3 / k_with if k_with != 0 else np.nan
    
    # Fit WITHOUT outliers
    I_clean = I_in[~outliers_temp]
    R1_inv_clean = R1_inv_in[~outliers_temp]
    
    if len(I_clean) >= 5:
        start_idx_clean = int(len(I_clean) * FIT_START_PCT)
        end_idx_clean = int(len(I_clean) * FIT_END_PCT)
        if end_idx_clean > start_idx_clean:
            k_without = np.sum(I_clean[start_idx_clean:end_idx_clean] * 
                               R1_inv_clean[start_idx_clean:end_idx_clean]) / \
                       np.sum(I_clean[start_idx_clean:end_idx_clean] ** 2)
            tafel_without = 2.3 / k_without if k_without != 0 else np.nan
        else:
            tafel_without = np.nan
    else:
        tafel_without = np.nan
    
    results_no_outliers.append({
        'time': time,
        'tafel_with': tafel_with,
        'tafel_without': tafel_without,
        'n_outliers': np.sum(outliers_temp),
        'n_with': len(I_in),
        'n_without': len(I_clean)
    })

comparison_df = pd.DataFrame(results_no_outliers)
comparison_df['difference'] = comparison_df['tafel_without'] - comparison_df['tafel_with']

print(comparison_df.to_string(index=False))

# Plot comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot both
comparison_df_sorted = comparison_df.sort_values('time')
ax.plot(comparison_df_sorted['time'], comparison_df_sorted['tafel_with'], 
        'o-', color='blue', linewidth=2, markersize=8, label='With outliers')
ax.plot(comparison_df_sorted['time'], comparison_df_sorted['tafel_without'], 
        's-', color='green', linewidth=2, markersize=8, label='Without outliers')

# Add reference lines
ax.axhline(y=60, color='gray', linestyle='--', alpha=0.5, label='60 mV/dec')
ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='70 mV/dec')

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
ax.set_title('Sample 499: Tafel Slope With vs Without Outliers', fontsize=14)
ax.grid(True, which='both', linestyle='--', alpha=0.3)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / '499_tafel_with_vs_without_outliers.png', dpi=300, bbox_inches='tight')
print(f"\nSaved comparison plot: 499_tafel_with_vs_without_outliers.png")
plt.show()

# ============================================================================
# Save Results
# ============================================================================

# Save detailed outlier info
outlier_details = []
for result in all_results:
    if result['n_outliers_combined'] > 0:
        for i, (I_val, R1_inv_val) in enumerate(zip(result['outlier_I'], result['outlier_R1_inv'])):
            outlier_details.append({
                'time': result['time'],
                'outlier_index': i,
                'I_mA': I_val + result['i_short'] if 'i_short' in result else I_val,
                'I_corr': I_val,
                'R1_inv': R1_inv_val,
                'R1': 1/R1_inv_val if R1_inv_val != 0 else np.nan
            })

if outlier_details:
    outlier_df = pd.DataFrame(outlier_details)
    outlier_file = output_dir / '499_outlier_details.csv'
    outlier_df.to_csv(outlier_file, index=False)
    print(f"Outlier details saved to: {outlier_file}")

# Save main summary
summary_file = output_dir / '499_outlier_summary.csv'
results_df.to_csv(summary_file, index=False)
print(f"Summary saved to: {summary_file}")

# Save comparison
comparison_file = output_dir / '499_outlier_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"Comparison saved to: {comparison_file}")

# Count files
plot_files = list(output_dir.glob('499_outliers_time_*.png'))
print(f"\nCreated {len(plot_files)} outlier analysis plots")
print(f"All files in: {output_dir}")

print("\n" + "=" * 80)
print("Outlier Analysis Complete!")
print("Each plot shows:")
print("  - Top: 1/R1 vs Current with outliers marked in red")
print("  - Bottom: Residuals with ±1σ and ±2σ lines")
print("  - Outliers detected using Z-score, IQR, and residual methods")
print("=" * 80)
