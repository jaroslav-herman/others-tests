#!/usr/bin/env python3
"""
Reusable Tafel Analysis for All EIS Samples
Performs optimized Tafel slope analysis on EIS fitting data from eisyFIT

Procedure:
1. Load CSV data from eisyFIT export
2. Correct for short circuit (subtract current at lowest voltage)
3. Filter to current range 5-700 mA
4. Fit 1/R1 = k * I (forced zero intercept, q=0)
5. Calculate Tafel slope = 2.3 / k
6. Detect outliers using Z-score on residuals (>2.5)
7. Compare results with and without outliers
8. Generate plots for visual verification
9. Export results to CSV

Usage:
    python tafel_analysis_all_samples.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION - Optimized parameters from sample 499 analysis
# ============================================================================

# Current range (mA)
CURRENT_MIN = 5
CURRENT_MAX = 700

# Fit configuration
FORCE_ZERO_INTERCEPT = True
FIT_RANGE_START = 0.2  # Start fit from 20% of in-range points
FIT_RANGE_END = 0.8    # End fit at 80% of in-range points

# Outlier detection
OUTLIER_Z_SCORE = 2.5

# Plot configuration
PLOT_DPI = 300
PLOT_FORMAT = 'png'

# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_tafel_sample(csv_path, sample_id,
                        current_min=CURRENT_MIN,
                        current_max=CURRENT_MAX,
                        force_zero_intercept=FORCE_ZERO_INTERCEPT,
                        fit_start=FIT_RANGE_START,
                        fit_end=FIT_RANGE_END,
                        outlier_z_score=OUTLIER_Z_SCORE):
    """
    Perform Tafel analysis on a single EIS CSV file.
    
    Parameters:
    - csv_path: Path to CSV file from eisyFIT
    - sample_id: Sample identifier (e.g., '499')
    
    Returns:
    - results_df: DataFrame with Tafel slopes for each Time point
    - Generates plots and CSV files in output directory
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Sample {sample_id}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    df['sample_id'] = sample_id
    
    # Clean up duplicate column names
    if 'Ecell_V.1' in df.columns:
        df = df.drop(columns=['Ecell_V.1'])
    
    # Sort by Time and current within each Time group
    df = df.sort_values(['Time', 'I_mA'])
    
    # Create output directory
    output_dir = Path(f"sample_{sample_id}/tafel_{current_max}ma_{sample_id}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get unique Time points
    time_points = sorted(df['Time'].unique())
    n_times = len(time_points)
    
    print(f"Found {n_times} Time points")
    print(f"Current range: {current_min}-{current_max} mA")
    print(f"Force zero intercept: {force_zero_intercept}")
    print(f"Outlier Z-score threshold: {outlier_z_score}")
    
    # Store results
    results = []
    comparison_data = []
    
    # Get colors for plotting
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_times))
    
    # ========================================================================
    # Process each Time point
    # ========================================================================
    
    for time, group in df.groupby('Time'):
        time = int(time)
        color = colors[time_points.index(time)]
        
        # Filter to current range
        group = group[(group['I_mA'] >= current_min) & 
                      (group['I_mA'] <= current_max)]
        
        if len(group) < 5:
            print(f"  Time {time}: Insufficient data ({len(group)} points)")
            continue
        
        # Short circuit correction: subtract current at lowest voltage
        i_short = group['I_mA'].iloc[0]
        
        # Corrected current and 1/R1
        I_corr = group['I_mA'].values - i_short
        R1_inv = 1 / group['R1'].values
        
        # Filter valid points (R1 > 0, I_corr > 0)
        valid = (R1_inv > 0) & (I_corr > 0) & np.isfinite(R1_inv) & np.isfinite(I_corr)
        I_corr = I_corr[valid]
        R1_inv = R1_inv[valid]
        
        if len(I_corr) < 5:
            print(f"  Time {time}: Insufficient valid points ({len(I_corr)} points)")
            continue
        
        # Fit range (middle portion to avoid edge effects)
        start_idx = int(len(I_corr) * fit_start)
        end_idx = int(len(I_corr) * fit_end)
        
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = len(I_corr)
        
        I_fit = I_corr[start_idx:end_idx]
        R1_inv_fit = R1_inv[start_idx:end_idx]
        
        # ====================================================================
        # Linear fit: 1/R1 = k * I + q
        # ====================================================================
        
        if force_zero_intercept:
            # Force q=0, fit only k
            k = np.sum(I_fit * R1_inv_fit) / np.sum(I_fit ** 2)
            q = 0
        else:
            k, q = np.polyfit(I_fit, R1_inv_fit, 1)
        
        # Calculate Tafel slope: 2.3 / k
        tafel_with = 2.3 / k if k != 0 else np.nan
        
        # ====================================================================
        # Outlier detection
        # ====================================================================
        
        # Predict 1/R1 for all in-range points
        R1_inv_pred = k * I_corr + q
        residuals = R1_inv - R1_inv_pred
        std_res = np.std(residuals)
        
        # Calculate Z-scores
        z_scores = np.abs(residuals / std_res) if std_res > 0 else np.zeros_like(residuals)
        outliers = z_scores > outlier_z_score
        n_outliers = np.sum(outliers)
        
        # Get outlier information
        outlier_I = list(I_corr[outliers])
        outlier_R1_inv = list(R1_inv[outliers])
        
        # ====================================================================
        # Fit without outliers
        # ====================================================================
        
        if n_outliers > 0:
            I_no_out = I_corr[~outliers]
            R1_inv_no_out = R1_inv[~outliers]
            
            # Recalculate fit range for data without outliers
            start_idx_no = int(len(I_no_out) * fit_start)
            end_idx_no = int(len(I_no_out) * fit_end)
            
            if end_idx_no <= start_idx_no:
                start_idx_no = 0
                end_idx_no = len(I_no_out)
            
            I_fit_no = I_no_out[start_idx_no:end_idx_no]
            R1_inv_fit_no = R1_inv_no_out[start_idx_no:end_idx_no]
            
            if force_zero_intercept:
                k_no_out = np.sum(I_fit_no * R1_inv_fit_no) / np.sum(I_fit_no ** 2)
            else:
                k_no_out, _ = np.polyfit(I_fit_no, R1_inv_fit_no, 1)
            
            tafel_without = 2.3 / k_no_out if k_no_out != 0 else np.nan
        else:
            k_no_out = k
            tafel_without = tafel_with
        
        # ====================================================================
        # Store results
        # ====================================================================
        
        results.append({
            'time': time,
            'tafel_slope': tafel_without,  # Use outlier-removed value as primary
            'k': k_no_out,
            'q': 0 if force_zero_intercept else q,
            'n_points': len(I_corr),
            'n_fit_points': len(I_fit_no) if n_outliers > 0 else len(I_fit),
            'i_short': i_short,
            'std_residuals': std_res
        })
        
        comparison_data.append({
            'time': time,
            'tafel_with': tafel_with,
            'tafel_without': tafel_without,
            'difference': tafel_without - tafel_with,
            'k_with': k,
            'k_without': k_no_out,
            'n_in_range': len(I_corr),
            'n_outliers': n_outliers,
            'n_without': len(I_corr) - n_outliers,
            'std_residuals': std_res,
            'outlier_I': outlier_I,
            'outlier_R1_inv': outlier_R1_inv
        })
        
        print(f"  Time {time}: Tafel = {tafel_without:.2f} mV/dec "
              f"(with outliers: {tafel_with:.2f}, diff: {tafel_without - tafel_with:+.2f}, "
              f"outliers: {n_outliers}, points: {len(I_corr)})")
        
        # ====================================================================
        # Generate individual plot for this Time point
        # ====================================================================
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: 1/R1 vs Corrected Current
        ax1.plot(I_corr, R1_inv, 'o', color=color, alpha=0.6, markersize=5, label='All points')
        
        # Plot fit line (with outliers)
        I_line = np.linspace(I_corr.min(), I_corr.max(), 100)
        R1_inv_line = k * I_line + q
        ax1.plot(I_line, R1_inv_line, '-', color='red', linewidth=1.5, 
                 alpha=0.8, label=f'Fit with outliers (Tafel: {tafel_with:.2f})')
        
        # Plot fit line (without outliers)
        if n_outliers > 0:
            R1_inv_line_no = k_no_out * I_line
            ax1.plot(I_line, R1_inv_line_no, '--', color='blue', linewidth=1.5,
                     alpha=0.8, label=f'Fit w/o outliers (Tafel: {tafel_without:.2f})')
        
        # Highlight outliers
        if n_outliers > 0:
            ax1.plot(I_corr[outliers], R1_inv[outliers], 'x', color='red',
                     markersize=8, markeredgewidth=2, label='Outliers')
        
        # Highlight fit range
        ax1.axvspan(I_fit.min(), I_fit.max(), color='green', alpha=0.1, label='Fit range')
        
        ax1.set_xlabel('Corrected Current (mA)', fontsize=10)
        ax1.set_ylabel('1/R1 (1/Ohm)', fontsize=10)
        ax1.set_title(f'Sample {sample_id} - Time {time}', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot 2: Residuals
        ax2.plot(I_corr, residuals, 'o', color=color, alpha=0.6, markersize=5, label='Residuals')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.axhline(y=std_res * outlier_z_score, color='red', linestyle=':', linewidth=1, label=f'Outlier threshold (±{outlier_z_score}σ)')
        ax2.axhline(y=-std_res * outlier_z_score, color='red', linestyle=':', linewidth=1)
        
        if n_outliers > 0:
            ax2.plot(I_corr[outliers], residuals[outliers], 'x', color='red',
                     markersize=8, markeredgewidth=2, label='Outliers')
        
        ax2.set_xlabel('Corrected Current (mA)', fontsize=10)
        ax2.set_ylabel('Residuals (1/Ohm)', fontsize=10)
        ax2.set_title(f'Residuals (std={std_res:.4f})', fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.3)
        ax2.legend(fontsize=8)
        
        plt.tight_layout()
        plot_path = output_dir / f"{sample_id}_tafel_time_{time:02d}.{PLOT_FORMAT}"
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # Create summary DataFrames
    # ========================================================================
    
    results_df = pd.DataFrame(results)
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by time
    results_df = results_df.sort_values('time').reset_index(drop=True)
    comparison_df = comparison_df.sort_values('time').reset_index(drop=True)
    
    # ========================================================================
    # Save CSV files
    # ========================================================================
    
    summary_path = output_dir / f"{sample_id}_tafel_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    
    comparison_path = output_dir / f"{sample_id}_tafel_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison: {comparison_path}")
    
    # ========================================================================
    # Summary Plot: Time Evolution of Tafel Slope
    # ========================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot Tafel slope vs Time
    ax.plot(results_df['time'], results_df['tafel_slope'], 
            'o-', color='darkred', linewidth=2, markersize=8, label='Tafel Slope (w/o outliers)')
    
    # Add comparison with outliers
    if 'tafel_with' in comparison_df.columns:
        ax.plot(comparison_df['time'], comparison_df['tafel_with'], 
                'x--', color='gray', linewidth=1, markersize=6, label='Tafel Slope (with outliers)')
    
    # Add horizontal reference lines
    ax.axhline(y=40, color='gray', linestyle=':', alpha=0.5, label='40 mV/dec')
    ax.axhline(y=60, color='gray', linestyle=':', alpha=0.5, label='60 mV/dec')
    ax.axhline(y=80, color='gray', linestyle=':', alpha=0.5, label='80 mV/dec')
    
    # Removed value annotations as requested
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
    ax.set_title(f'Sample {sample_id}: Time Evolution of Tafel Slope from R1', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    plot_path = output_dir / f"{sample_id}_tafel_time_evolution.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved time evolution plot: {plot_path}")
    
    # ========================================================================
    # Comparison Plot: Outliers vs Tafel Difference
    # ========================================================================
    
    if 'difference' in comparison_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot difference vs n_outliers
        scatter = ax.scatter(comparison_df['n_outliers'], comparison_df['difference'],
                             c=comparison_df['time'], cmap='viridis', s=100, alpha=0.7)
        
        # Annotate each point
        for i, row in comparison_df.iterrows():
            ax.annotate(f"{row['time']}",
                        xy=(row['n_outliers'], row['difference']),
                        xytext=(row['n_outliers'], row['difference'] + 0.5),
                        ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Number of Outliers', fontsize=12)
        ax.set_ylabel('Tafel Slope Difference (without - with)', fontsize=12)
        ax.set_title(f'Sample {sample_id}: Outlier Impact on Tafel Slope', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        plt.colorbar(scatter, label='Time')
        
        plot_path = output_dir / f"{sample_id}_tafel_outliers_vs_difference.{PLOT_FORMAT}"
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved outlier comparison plot: {plot_path}")
    
    # ========================================================================
    # Print Summary Statistics
    # ========================================================================
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics for Sample {sample_id}")
    print(f"{'='*60}")
    
    if len(results_df) > 0:
        print(f"Time points analyzed: {len(results_df)}")
        print(f"Mean Tafel slope (w/o outliers): {results_df['tafel_slope'].mean():.2f} mV/dec")
        print(f"Std Tafel slope: {results_df['tafel_slope'].std():.2f} mV/dec")
        print(f"Min Tafel slope: {results_df['tafel_slope'].min():.2f} mV/dec")
        print(f"Max Tafel slope: {results_df['tafel_slope'].max():.2f} mV/dec")
        print(f"Total outliers detected: {comparison_df['n_outliers'].sum()}")
        print(f"Mean outliers per Time: {comparison_df['n_outliers'].mean():.2f}")
    
    if 'difference' in comparison_df.columns:
        print(f"\nOutlier Impact:")
        print(f"  Mean difference (w/o - with): {comparison_df['difference'].mean():.2f} mV/dec")
        print(f"  Max difference: {comparison_df['difference'].abs().max():.2f} mV/dec")
        print(f"  Outliers tend to: {'INCREASE' if comparison_df['difference'].mean() < 0 else 'DECREASE'} Tafel slopes")
    
    return results_df, comparison_df


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run Tafel analysis on all samples in AEMs folder."""
    
    # Define samples
    samples = [
        {'id': '474', 'file': 'sample_474/474_fit_export.csv'},
        {'id': '498', 'file': 'sample_498/498_fit_export.csv'},
        {'id': '499', 'file': 'sample_499/499_fit_export.csv'},
        {'id': '505', 'file': 'sample_505/505_fit_export.csv'},
    ]
    
    # Run analysis for each sample
    all_results = {}
    all_comparisons = {}
    
    for sample in samples:
        csv_path = Path(sample['file'])
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        try:
            results_df, comparison_df = analyze_tafel_sample(
                csv_path, 
                sample['id']
            )
            all_results[sample['id']] = results_df
            all_comparisons[sample['id']] = comparison_df
        except Exception as e:
            print(f"Error analyzing sample {sample['id']}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # Create Combined Summary
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("Combined Summary for All Samples")
    print(f"{'='*60}")
    
    combined_summary = []
    
    for sample_id, results_df in all_results.items():
        if len(results_df) > 0:
            combined_summary.append({
                'sample': sample_id,
                'mean_tafel': results_df['tafel_slope'].mean(),
                'std_tafel': results_df['tafel_slope'].std(),
                'min_tafel': results_df['tafel_slope'].min(),
                'max_tafel': results_df['tafel_slope'].max(),
                'n_time_points': len(results_df),
                'total_outliers': all_comparisons[sample_id]['n_outliers'].sum()
            })
    
    if combined_summary:
        summary_df = pd.DataFrame(combined_summary)
        print("\n" + summary_df.to_string(index=False))
        
        # Save combined summary
        summary_df.to_csv('scripts/tafel_combined_summary.csv', index=False)
        print(f"\nSaved combined summary: scripts/tafel_combined_summary.csv")
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print("\nTo apply this procedure to new samples:")
    print("1. Place CSV file from eisyFIT in AEMs/ folder")
    print("2. Add sample to the samples list in this script")
    print("3. Run: python tafel_analysis_all_samples.py")


if __name__ == '__main__':
    main()
