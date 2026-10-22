#!/usr/bin/env python3
"""
C1 Linear Fit Analysis for EIS Data
Following the same pattern as Tafel analysis:
1. Set low current limit (5 mA)
2. Do linear fit: C1 = k * I_corr + q
3. Evaluate outliers using Z-score on residuals
4. Show time evolution of intercept (q) from linear fit

Usage:
    python c1_linear_fit_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION - Following Tafel analysis pattern
# ============================================================================

# Samples to analyze
SAMPLES = ['474', '498', '499', '505']

# Current range for analysis (mA)
CURRENT_MIN = 10
CURRENT_MAX = 700

# Fit configuration
FIT_RANGE_START = 0.2  # Start fit from 20% of in-range points
FIT_RANGE_END = 0.8    # End fit at 80% of in-range points

# Outlier detection
OUTLIER_Z_SCORE = 2.5

# Plot configuration
PLOT_DPI = 300
PLOT_FORMAT = 'png'

# ============================================================================
# Analysis Function
# ============================================================================

def analyze_c1_linear_sample(csv_path, sample_id,
                           current_min=CURRENT_MIN,
                           current_max=CURRENT_MAX,
                           fit_start=FIT_RANGE_START,
                           fit_end=FIT_RANGE_END,
                           outlier_z_score=OUTLIER_Z_SCORE):
    """
    Analyze C1 with linear fit following Tafel analysis pattern.
    
    For each Time point:
    - Correct current for short circuit
    - Filter to current range
    - Fit C1 = k * I_corr + q
    - Detect outliers using Z-score on residuals
    - Track intercept (q) evolution over Time
    """
    print(f"\n{'='*60}")
    print(f"C1 Linear Fit Analysis for Sample {sample_id}")
    print(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(csv_path)
    df['sample_id'] = sample_id
    
    # Clean up duplicate column names
    if 'Ecell_V.1' in df.columns:
        df = df.drop(columns=['Ecell_V.1'])
    
    # Sort by Time and current
    df = df.sort_values(['Time', 'I_mA'])
    
    # Create output directory
    output_dir = Path(f"c1_linear_{sample_id}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get unique Time points
    time_points = sorted(df['Time'].unique())
    n_times = len(time_points)
    
    print(f"Found {n_times} Time points")
    print(f"Current range: {current_min}-{current_max} mA")
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
        
        # Sort by current
        group = group.sort_values('I_mA')
        
        # Short circuit correction
        i_short = group['I_mA'].iloc[0]
        I_corr = group['I_mA'].values - i_short
        C1 = group['C1'].values
        
        # Filter valid points
        valid = (np.isfinite(C1)) & (C1 > 0) & (I_corr > 0) & np.isfinite(I_corr)
        I_corr = I_corr[valid]
        C1 = C1[valid]
        
        if len(I_corr) < 5:
            print(f"  Time {time}: Insufficient valid points ({len(I_corr)} points)")
            continue
        
        # Fit range (middle portion)
        start_idx = int(len(I_corr) * fit_start)
        end_idx = int(len(I_corr) * fit_end)
        
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = len(I_corr)
        
        I_fit = I_corr[start_idx:end_idx]
        C1_fit = C1[start_idx:end_idx]
        
        # ====================================================================
        # Linear fit: C1 = k * I_corr + q
        # ====================================================================
        
        k, q = np.polyfit(I_fit, C1_fit, 1)
        
        # Calculate C1 for all in-range points using fit
        C1_pred = k * I_corr + q
        residuals = C1 - C1_pred
        std_res = np.std(residuals)
        
        # ====================================================================
        # Outlier detection
        # ====================================================================
        
        # Calculate Z-scores
        z_scores = np.abs(residuals / std_res) if std_res > 0 else np.zeros_like(residuals)
        outliers = z_scores > outlier_z_score
        n_outliers = np.sum(outliers)
        
        # Get outlier information
        outlier_I = list(I_corr[outliers])
        outlier_C1 = list(C1[outliers])
        
        # ====================================================================
        # Fit without outliers
        # ====================================================================
        
        if n_outliers > 0:
            I_no_out = I_corr[~outliers]
            C1_no_out = C1[~outliers]
            
            # Recalculate fit range for data without outliers
            start_idx_no = int(len(I_no_out) * fit_start)
            end_idx_no = int(len(I_no_out) * fit_end)
            
            if end_idx_no <= start_idx_no:
                start_idx_no = 0
                end_idx_no = len(I_no_out)
            
            I_fit_no = I_no_out[start_idx_no:end_idx_no]
            C1_fit_no = C1_no_out[start_idx_no:end_idx_no]
            
            k_no_out, q_no_out = np.polyfit(I_fit_no, C1_fit_no, 1)
        else:
            k_no_out = k
            q_no_out = q
        
        # ====================================================================
        # Store results
        # ====================================================================
        
        results.append({
            'time': time,
            'k': k_no_out,
            'q': q_no_out,
            'n_points': len(I_corr),
            'n_fit_points': len(I_fit_no) if n_outliers > 0 else len(I_fit),
            'i_short': i_short,
            'std_residuals': std_res
        })
        
        comparison_data.append({
            'time': time,
            'k_with': k,
            'q_with': q,
            'k_without': k_no_out,
            'q_without': q_no_out,
            'q_difference': q_no_out - q,
            'n_in_range': len(I_corr),
            'n_outliers': n_outliers,
            'n_without': len(I_corr) - n_outliers,
            'std_residuals': std_res,
            'outlier_I': outlier_I,
            'outlier_C1': outlier_C1
        })
        
        print(f"  Time {time}: q = {q_no_out:.4f} F "
              f"(with outliers: {q:.4f}, diff: {q_no_out - q:+.4f}, "
              f"outliers: {n_outliers}, points: {len(I_corr)})")
        
        # ====================================================================
        # Generate individual plot for this Time point
        # ====================================================================
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: C1 vs Corrected Current
        ax1.plot(I_corr, C1, 'o', color=color, alpha=0.6, markersize=5, label='All points')
        
        # Plot fit line (with outliers)
        I_line = np.linspace(I_corr.min(), I_corr.max(), 100)
        C1_line = k * I_line + q
        ax1.plot(I_line, C1_line, '-', color='red', linewidth=1.5, 
                 alpha=0.8, label=f'Fit with outliers (q={q:.4f})')
        
        # Plot fit line (without outliers)
        if n_outliers > 0:
            C1_line_no = k_no_out * I_line + q_no_out
            ax1.plot(I_line, C1_line_no, '--', color='blue', linewidth=1.5,
                     alpha=0.8, label=f'Fit w/o outliers (q={q_no_out:.4f})')
        
        # Highlight outliers
        if n_outliers > 0:
            ax1.plot(I_corr[outliers], C1[outliers], 'x', color='red',
                     markersize=8, markeredgewidth=2, label='Outliers')
        
        # Highlight fit range
        ax1.axvspan(I_fit.min(), I_fit.max(), color='green', alpha=0.1, label='Fit range')
        
        ax1.set_xlabel('Corrected Current (mA)', fontsize=10)
        ax1.set_ylabel('C1 (F)', fontsize=10)
        ax1.set_title(f'Sample {sample_id} - Time {time}: C1 vs I with linear fit', fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot 2: Residuals
        ax2.plot(I_corr, residuals, 'o', color=color, alpha=0.6, markersize=5, label='Residuals')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.axhline(y=std_res * outlier_z_score, color='red', linestyle=':', linewidth=1, 
                    label=f'Outlier threshold (±{outlier_z_score}σ)')
        ax2.axhline(y=-std_res * outlier_z_score, color='red', linestyle=':', linewidth=1)
        
        if n_outliers > 0:
            ax2.plot(I_corr[outliers], residuals[outliers], 'x', color='red',
                     markersize=8, markeredgewidth=2, label='Outliers')
        
        ax2.set_xlabel('Corrected Current (mA)', fontsize=10)
        ax2.set_ylabel('Residuals (F)', fontsize=10)
        ax2.set_title(f'Residuals (std={std_res:.4f})', fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.3)
        ax2.legend(fontsize=8)
        
        plt.tight_layout()
        plot_path = output_dir / f"{sample_id}_C1_fit_time_{time:02d}.{PLOT_FORMAT}"
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
    
    summary_path = output_dir / f"{sample_id}_C1_fit_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    
    comparison_path = output_dir / f"{sample_id}_C1_fit_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison: {comparison_path}")
    
    # ========================================================================
    # Summary Plot: Time Evolution of Intercept (q)
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("Plotting Time Evolution of Intercept (q)")
    print(f"{'='*60}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot q vs Time
    ax.plot(results_df['time'], results_df['q'], 
            'o-', color='darkred', linewidth=2, markersize=8, label='Intercept q (w/o outliers)')
    
    # Add comparison with outliers
    if 'q_with' in comparison_df.columns:
        ax.plot(comparison_df['time'], comparison_df['q_with'], 
                'x--', color='gray', linewidth=1, markersize=6, label='Intercept q (with outliers)')
    
    # Add horizontal line at q=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Removed value annotations as requested
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Intercept q (F)', fontsize=12)
    ax.set_title(f'Sample {sample_id}: Time Evolution of Intercept from C1 = k*I + q', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    plot_path = output_dir / f"{sample_id}_C1_intercept_time_evolution.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved intercept evolution plot: {plot_path}")
    
    # ========================================================================
    # Plot: Slope (k) Time Evolution
    # ========================================================================
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(results_df['time'], results_df['k'], 
            'o-', color='darkblue', linewidth=2, markersize=8, label='Slope k (w/o outliers)')
    
    if 'k_with' in comparison_df.columns:
        ax.plot(comparison_df['time'], comparison_df['k_with'], 
                'x--', color='gray', linewidth=1, markersize=6, label='Slope k (with outliers)')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Removed value annotations as requested
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Slope k (F/mA)', fontsize=12)
    ax.set_title(f'Sample {sample_id}: Time Evolution of Slope from C1 = k*I + q', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    plot_path = output_dir / f"{sample_id}_C1_slope_time_evolution.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved slope evolution plot: {plot_path}")
    
    # ========================================================================
    # Comparison Plot: Outliers vs q Difference
    # ========================================================================
    
    if 'q_difference' in comparison_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot difference vs n_outliers
        scatter = ax.scatter(comparison_df['n_outliers'], comparison_df['q_difference'],
                             c=comparison_df['time'], cmap='viridis', s=100, alpha=0.7)
        
        # Annotate each point
        for i, row in comparison_df.iterrows():
            ax.annotate(f"{row['time']}",
                        xy=(row['n_outliers'], row['q_difference']),
                        xytext=(row['n_outliers'], row['q_difference'] + 0.01),
                        ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Number of Outliers', fontsize=12)
        ax.set_ylabel('Intercept q Difference (without - with)', fontsize=12)
        ax.set_title(f'Sample {sample_id}: Outlier Impact on Intercept q', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        plt.colorbar(scatter, label='Time')
        
        plot_path = output_dir / f"{sample_id}_C1_outliers_vs_q_difference.{PLOT_FORMAT}"
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
        print(f"Mean intercept q (w/o outliers): {results_df['q'].mean():.4f} F")
        print(f"Std intercept q: {results_df['q'].std():.4f} F")
        print(f"Min intercept q: {results_df['q'].min():.4f} F")
        print(f"Max intercept q: {results_df['q'].max():.4f} F")
        print(f"Mean slope k (w/o outliers): {results_df['k'].mean():.2e} F/mA")
        print(f"Std slope k: {results_df['k'].std():.2e} F/mA")
        print(f"Total outliers detected: {comparison_df['n_outliers'].sum()}")
        print(f"Mean outliers per Time: {comparison_df['n_outliers'].mean():.2f}")
    
    if 'q_difference' in comparison_df.columns:
        print(f"\nOutlier Impact on Intercept:")
        print(f"  Mean q difference (w/o - with): {comparison_df['q_difference'].mean():.4f} F")
        print(f"  Max |q difference|: {np.abs(comparison_df['q_difference']).max():.4f} F")
        if comparison_df['q_difference'].mean() < 0:
            print(f"  Outliers tend to: INCREASE intercept q")
        else:
            print(f"  Outliers tend to: DECREASE intercept q")
    
    return results_df, comparison_df


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run C1 linear fit analysis on all samples."""
    
    # Run analysis for each sample
    all_results = {}
    all_comparisons = {}
    
    for sample_id in SAMPLES:
        csv_path = Path(f"sample_{sample_id}/{sample_id}_fit_export.csv")
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        try:
            results_df, comparison_df = analyze_c1_linear_sample(
                csv_path, 
                sample_id
            )
            all_results[sample_id] = results_df
            all_comparisons[sample_id] = comparison_df
        except Exception as e:
            print(f"Error analyzing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # Create Combined Summary
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("Combined C1 Linear Fit Summary Across Samples")
    print(f"{'='*60}")
    
    combined_summary = []
    
    for sample_id, results_df in all_results.items():
        if len(results_df) > 0:
            combined_summary.append({
                'sample': sample_id,
                'mean_q': results_df['q'].mean(),
                'std_q': results_df['q'].std(),
                'min_q': results_df['q'].min(),
                'max_q': results_df['q'].max(),
                'mean_k': results_df['k'].mean(),
                'std_k': results_df['k'].std(),
                'n_time_points': len(results_df),
                'total_outliers': all_comparisons[sample_id]['n_outliers'].sum()
            })
    
    if combined_summary:
        summary_df = pd.DataFrame(combined_summary)
        print("\n" + summary_df.to_string(index=False))
        
        # Save combined summary
        summary_df.to_csv('c1_linear_combined_summary.csv', index=False)
        print(f"\nSaved combined summary: c1_linear_combined_summary.csv")
    
    # ========================================================================
    # Combined comparison plot: Intercept q across samples
    # ========================================================================
    
    if all_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_results)))
        
        for (sample_id, results_df), color in zip(all_results.items(), colors):
            ax.plot(results_df['time'], results_df['q'], 
                    'o-', color=color, linewidth=2, markersize=6, label=sample_id)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Intercept q (F)', fontsize=12)
        ax.set_title('C1 Linear Fit: Intercept q Time Evolution - All Samples', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        plot_path = Path(f"c1_linear_combined_q_evolution.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved combined q evolution plot: {plot_path}")
    
    print(f"\n{'='*60}")
    print("C1 Linear Fit Analysis Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
