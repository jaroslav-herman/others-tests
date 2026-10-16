#!/usr/bin/env python3
"""
Anode Capacitance (C1) Analysis for EIS Data
Plots C1 dependence on current for each Time point in separate graphs

Usage:
    python capacitance_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Samples to analyze
SAMPLES = ['474', '498', '499']

# Current range for analysis (mA)
CURRENT_MIN = 5
CURRENT_MAX = 700

# Plot configuration
PLOT_DPI = 300
PLOT_FORMAT = 'png'

# ============================================================================
# Analysis Function
# ============================================================================

def analyze_capacitance_sample(csv_path, sample_id,
                             current_min=CURRENT_MIN,
                             current_max=CURRENT_MAX):
    """
    Analyze C1 (anode capacitance) dependence on current for each Time point.
    Generates separate graph for each Time point.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing C1 for Sample {sample_id}")
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
    output_dir = Path(f"capacitance_{sample_id}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get unique Time points
    time_points = sorted(df['Time'].unique())
    n_times = len(time_points)
    
    print(f"Found {n_times} Time points")
    
    # Get colors for plotting
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_times))
    
    # Store summary data
    summary_data = []
    
    # ========================================================================
    # Process each Time point
    # ========================================================================
    
    for time, group in df.groupby('Time'):
        time = int(time)
        color = colors[time_points.index(time)]
        
        # Filter to current range
        group = group[(group['I_mA'] >= current_min) & 
                      (group['I_mA'] <= current_max)]
        
        if len(group) < 3:
            print(f"  Time {time}: Insufficient data ({len(group)} points)")
            continue
        
        # Sort by current
        group = group.sort_values('I_mA')
        
        # Short circuit correction
        i_short = group['I_mA'].iloc[0]
        I_corr = group['I_mA'].values - i_short
        C1 = group['C1'].values
        
        # Filter valid points
        valid = (np.isfinite(C1)) & (C1 > 0) & (I_corr > 0)
        I_corr = I_corr[valid]
        C1 = C1[valid]
        
        if len(I_corr) < 3:
            print(f"  Time {time}: Insufficient valid points ({len(I_corr)} points)")
            continue
        
        # Calculate statistics
        mean_C1 = np.mean(C1)
        std_C1 = np.std(C1)
        min_C1 = np.min(C1)
        max_C1 = np.max(C1)
        
        summary_data.append({
            'time': time,
            'mean_C1': mean_C1,
            'std_C1': std_C1,
            'min_C1': min_C1,
            'max_C1': max_C1,
            'n_points': len(I_corr),
            'i_short': i_short
        })
        
        print(f"  Time {time}: C1 = {mean_C1:.4f} ± {std_C1:.4f} F "
              f"(range: {min_C1:.4f}-{max_C1:.4f}, n={len(I_corr)})")
        
        # ====================================================================
        # Generate plot for this Time point
        # ====================================================================
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot C1 vs Corrected Current
        ax.plot(I_corr, C1, 'o-', color=color, linewidth=1.5, 
                markersize=5, alpha=0.7, label=f'Time {time}')
        
        # Add horizontal line for mean
        ax.axhline(y=mean_C1, color='red', linestyle='--', linewidth=1,
                   label=f'Mean C1 = {mean_C1:.4f} F')
        
        # Add trend line (linear fit)
        if len(I_corr) >= 3:
            k, q = np.polyfit(I_corr, C1, 1)
            I_line = np.linspace(I_corr.min(), I_corr.max(), 100)
            C1_line = k * I_line + q
            ax.plot(I_line, C1_line, '--', color='gray', linewidth=1,
                    label=f'Linear fit: k={k:.2e}')
        
        ax.set_xlabel('Corrected Current (mA)', fontsize=12)
        ax.set_ylabel('C1 (F)', fontsize=12)
        ax.set_title(f'Sample {sample_id} - Time {time}: C1 vs Corrected Current', 
                      fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plot_path = output_dir / f"{sample_id}_C1_vs_I_time_{time:02d}.{PLOT_FORMAT}"
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # Save summary CSV
    # ========================================================================
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"{sample_id}_C1_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")
    
    # ========================================================================
    # Combined plot: All Time points together
    # ========================================================================
    
    if len(summary_data) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for time, group in df.groupby('Time'):
            time = int(time)
            color = colors[time_points.index(time)]
            
            group = group[(group['I_mA'] >= current_min) & 
                          (group['I_mA'] <= current_max)]
            group = group.sort_values('I_mA')
            
            if len(group) < 3:
                continue
            
            i_short = group['I_mA'].iloc[0]
            I_corr = group['I_mA'].values - i_short
            C1 = group['C1'].values
            
            valid = (np.isfinite(C1)) & (C1 > 0) & (I_corr > 0)
            I_corr = I_corr[valid]
            C1 = C1[valid]
            
            if len(I_corr) >= 3:
                ax.plot(I_corr, C1, 'o-', color=color, linewidth=1, 
                        markersize=4, alpha=0.7, label=f'Time {time}')
        
        ax.set_xlabel('Corrected Current (mA)', fontsize=12)
        ax.set_ylabel('C1 (F)', fontsize=12)
        ax.set_title(f'Sample {sample_id}: C1 vs Corrected Current (All Time Points)', 
                      fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
        
        plot_path = output_dir / f"{sample_id}_C1_vs_I_all_times.{PLOT_FORMAT}"
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved combined plot: {plot_path}")
    
    # ========================================================================
    # Time evolution plot
    # ========================================================================
    
    if len(summary_data) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(summary_df['time'], summary_df['mean_C1'], 
                'o-', color='darkblue', linewidth=2, markersize=8, label='Mean C1')
        
        # Add error bars
        ax.errorbar(summary_df['time'], summary_df['mean_C1'],
                    yerr=summary_df['std_C1'], fmt='none', 
                    color='darkblue', alpha=0.3, capsize=3)
        
        # Annotate each point
        for i, row in summary_df.iterrows():
            ax.annotate(f"{row['mean_C1']:.3f}",
                        xy=(row['time'], row['mean_C1']),
                        xytext=(row['time'], row['mean_C1'] + row['std_C1'] * 0.5),
                        ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Mean C1 (F)', fontsize=12)
        ax.set_title(f'Sample {sample_id}: Time Evolution of Mean C1', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        plot_path = output_dir / f"{sample_id}_C1_time_evolution.{PLOT_FORMAT}"
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved time evolution plot: {plot_path}")
    
    return summary_df


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run C1 analysis on all samples."""
    
    all_summaries = {}
    
    for sample_id in SAMPLES:
        csv_path = Path(f"{sample_id}_fit_export.csv")
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        try:
            summary_df = analyze_capacitance_sample(csv_path, sample_id)
            all_summaries[sample_id] = summary_df
        except Exception as e:
            print(f"Error analyzing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # Combined comparison across samples
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("Combined C1 Comparison Across Samples")
    print(f"{'='*60}")
    
    if all_summaries:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_summaries)))
        
        for (sample_id, summary_df), color in zip(all_summaries.items(), colors):
            ax.plot(summary_df['time'], summary_df['mean_C1'], 
                    'o-', color=color, linewidth=2, markersize=6, label=sample_id)
            ax.errorbar(summary_df['time'], summary_df['mean_C1'],
                        yerr=summary_df['std_C1'], fmt='none', 
                        color=color, alpha=0.3, capsize=3)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Mean C1 (F)', fontsize=12)
        ax.set_title('C1 Time Evolution: All Samples', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        plot_path = Path(f"capacitance_combined_comparison.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved combined comparison plot: {plot_path}")
    
    print(f"\n{'='*60}")
    print("Capacitance Analysis Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
