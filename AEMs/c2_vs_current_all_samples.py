#!/usr/bin/env python3
"""
Plot C2 dependence on current for all 3 samples.
Show separate graphs for each Time point, with all samples on the same plot.

Usage:
    python c2_vs_current_all_samples.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLES = ['474', '498', '499']
SAMPLE_COLORS = {
    '474': 'blue',
    '498': 'green',
    '499': 'red'
}
SAMPLE_LABELS = {
    '474': 'Sample 474',
    '498': 'Sample 498',
    '499': 'Sample 499'
}

PLOT_DPI = 300
PLOT_FORMAT = 'png'

# ============================================================================
# Load Data
# ============================================================================

def load_sample_data(sample_id):
    """Load and preprocess data for a single sample."""
    csv_path = Path(f"sample_{sample_id}/{sample_id}_fit_export.csv")
    df = pd.read_csv(csv_path)
    
    # Select relevant columns
    cols = ['Time', 'I_mA', 'C2']
    for col in cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in sample {sample_id}")
            return None
    
    df = df[cols].copy()
    df['sample_id'] = sample_id
    
    # Remove rows with NaN or infinite values
    df = df.dropna()
    df = df[np.isfinite(df['C2'])]
    df = df[df['C2'] > 0]
    
    return df

# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Plot C2 vs Current for all samples, separately for each Time."""
    
    # Load all data
    all_data = []
    for sample_id in SAMPLES:
        df = load_sample_data(sample_id)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        print("No data loaded. Exiting.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get unique Time points across all samples
    all_times = sorted(combined_df['Time'].unique())
    
    print(f"Found {len(all_times)} unique Time points")
    print(f"Time points: {all_times}")
    
    # Create output directory
    output_dir = Path("c2_vs_current_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # Create one plot per Time point
    # ========================================================================
    
    for time in all_times:
        # Filter data for this Time
        time_data = combined_df[combined_df['Time'] == time]
        
        # Get samples present at this Time
        samples_at_time = time_data['sample_id'].unique()
        
        if len(samples_at_time) == 0:
            continue
        
        print(f"\nProcessing Time {time}: {len(samples_at_time)} samples")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot each sample
        for sample_id in samples_at_time:
            sample_data = time_data[time_data['sample_id'] == sample_id]
            sample_data = sample_data.sort_values('I_mA')
            
            color = SAMPLE_COLORS.get(sample_id, 'gray')
            label = SAMPLE_LABELS.get(sample_id, f'Sample {sample_id}')
            
            ax.plot(sample_data['I_mA'], sample_data['C2'], 
                    'o-', color=color, linewidth=2, markersize=6, 
                    label=label, alpha=0.7)
        
        ax.set_xlabel('Current (mA)', fontsize=12)
        ax.set_ylabel('C2 (F)', fontsize=12)
        ax.set_title(f'C2 vs Current - Time {time}', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        
        # Save plot
        plot_path = output_dir / f"c2_vs_current_time_{time:02d}.{PLOT_FORMAT}"
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")
    
    # ========================================================================
    # Create combined plot with all Time points as subplots
    # ========================================================================
    
    print(f"\nCreating combined subplot figure...")
    
    # Calculate grid size
    n_times = len(all_times)
    n_cols = 5  # 5 columns
    n_rows = (n_times + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    
    for idx, time in enumerate(all_times):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Filter data for this Time
        time_data = combined_df[combined_df['Time'] == time]
        samples_at_time = time_data['sample_id'].unique()
        
        if len(samples_at_time) == 0:
            ax.set_title(f'Time {time}: No data')
            ax.axis('off')
            continue
        
        # Plot each sample
        for sample_id in samples_at_time:
            sample_data = time_data[time_data['sample_id'] == sample_id]
            sample_data = sample_data.sort_values('I_mA')
            
            color = SAMPLE_COLORS.get(sample_id, 'gray')
            label = SAMPLE_LABELS.get(sample_id, f'Sample {sample_id}')
            
            ax.plot(sample_data['I_mA'], sample_data['C2'], 
                    'o-', color=color, linewidth=1.5, markersize=5, 
                    label=label, alpha=0.7)
        
        ax.set_xlabel('Current (mA)', fontsize=10)
        ax.set_ylabel('C2 (F)', fontsize=10)
        ax.set_title(f'Time {time}', fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(all_times), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_plot_path = output_dir / f"c2_vs_current_all_times_combined.{PLOT_FORMAT}"
    plt.savefig(combined_plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {combined_plot_path}")
    
    print(f"\nC2 vs Current analysis complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
