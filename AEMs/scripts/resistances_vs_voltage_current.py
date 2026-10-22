#!/usr/bin/env python3
"""
Plot all resistances (R0, R1, R2, R3) as a function of:
1. Cell voltage (with logarithmic Y scale)
2. Current (with logarithmic Y scale)

Following the pattern from eis_parameters.py:
- Different colors based on Time (using rainbow colormap like original)
- Each sample processed separately with its own color array
- All samples overlaid in combined plots

Usage:
    python resistances_vs_voltage_current.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil


def get_colors(n=10, colormap="rainbow"):
    """
    Generate a list of colors sampled evenly from a matplotlib colormap.
    Same as wepy.basics.get_colors() from original script.
    """
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, n))
    return colors


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

# Resistance types
RESISTANCE_TYPES = ['R0', 'R1', 'R2', 'R3']
RESISTANCE_MARKERS = {
    'R0': 'o',
    'R1': 's',
    'R2': '^',
    'R3': 'D'
}
RESISTANCE_LABELS = {
    'R0': 'R0 (Ohm)',
    'R1': 'R1 (Ohm)',
    'R2': 'R2 (Ohm)',
    'R3': 'R3 (Ohm)'
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
    cols = ['Time', 'Ecell_V', 'I_mA', 'R0', 'R1', 'R2', 'R3']
    for col in cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in sample {sample_id}")
            return None
    
    df = df[cols].copy()
    df['sample_id'] = sample_id
    
    # Remove rows with NaN or infinite values in resistances
    for r in RESISTANCE_TYPES:
        df = df[np.isfinite(df[r])]
        df = df[df[r] > 0]
    
    return df


# ============================================================================
# Plotting Functions (using original rainbow color scheme from eis_parameters.py)
# ============================================================================

def plot_all_resistances_vs_voltage(combined_df, output_dir):
    """Plot all resistances vs voltage - all samples overlaid, original color scheme per sample."""
    print("Creating all resistances vs voltage plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create legend handles
    legend_handles = []
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        
        # Get unique Time points for this sample
        sample_times = sorted(sample_data['Time'].unique())
        len_data = len(sample_times)
        
        # Generate colors for this sample: 4 resistance types * len_data
        colors = get_colors(4 * len_data, colormap="rainbow")
        
        index = 0
        for time in sample_times:
            time_data = sample_data[sample_data['Time'] == time]
            time_data = time_data.sort_values('Ecell_V')
            
            # Plot each resistance with its assigned color (original pattern)
            # R0: colors[index]
            # R1: colors[len_data + index]
            # R2: colors[2*len_data + index]
            # R3: colors[3*len_data + index]
            
            ax.plot(time_data['Ecell_V'], time_data['R0'],
                    'o-', color=colors[index], markersize=5, alpha=0.7, linewidth=1)
            
            ax.plot(time_data['Ecell_V'], time_data['R1'],
                    'o-', color=colors[len_data + index], markersize=5, alpha=0.7, linewidth=1)
            
            ax.plot(time_data['Ecell_V'], time_data['R2'],
                    'o-', color=colors[2*len_data + index], markersize=5, alpha=0.7, linewidth=1)
            
            ax.plot(time_data['Ecell_V'], time_data['R3'],
                    'o-', color=colors[3*len_data + index], markersize=5, alpha=0.7, linewidth=1)
            
            index += 1
    
    # Add legend for resistance types
    for r_type in RESISTANCE_TYPES:
        legend_handles.append(plt.Line2D([0], [0],
            marker=RESISTANCE_MARKERS[r_type], color='black', linestyle='-',
            markersize=5, label=RESISTANCE_LABELS[r_type]))
    
    ax.set_xlabel('Cell Voltage (V)', fontsize=12)
    ax.set_ylabel('Resistance (Ohm)', fontsize=12)
    ax.set_title('All Resistances vs Cell Voltage (Log Scale)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    
    plot_path = output_dir / f"all_resistances_vs_voltage.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_all_resistances_vs_current(combined_df, output_dir):
    """Plot all resistances vs current - all samples overlaid, original color scheme per sample."""
    print("Creating all resistances vs current plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create legend handles
    legend_handles = []
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        
        # Get unique Time points for this sample
        sample_times = sorted(sample_data['Time'].unique())
        len_data = len(sample_times)
        
        # Generate colors for this sample: 4 resistance types * len_data
        colors = get_colors(4 * len_data, colormap="rainbow")
        
        index = 0
        for time in sample_times:
            time_data = sample_data[sample_data['Time'] == time]
            time_data = time_data.sort_values('I_mA')
            
            # Plot each resistance with its assigned color (original pattern)
            ax.plot(time_data['I_mA'], time_data['R0'],
                    'o-', color=colors[index], markersize=5, alpha=0.7, linewidth=1)
            
            ax.plot(time_data['I_mA'], time_data['R1'],
                    'o-', color=colors[len_data + index], markersize=5, alpha=0.7, linewidth=1)
            
            ax.plot(time_data['I_mA'], time_data['R2'],
                    'o-', color=colors[2*len_data + index], markersize=5, alpha=0.7, linewidth=1)
            
            ax.plot(time_data['I_mA'], time_data['R3'],
                    'o-', color=colors[3*len_data + index], markersize=5, alpha=0.7, linewidth=1)
            
            index += 1
    
    # Add legend for resistance types
    for r_type in RESISTANCE_TYPES:
        legend_handles.append(plt.Line2D([0], [0],
            marker=RESISTANCE_MARKERS[r_type], color='black', linestyle='-',
            markersize=5, label=RESISTANCE_LABELS[r_type]))
    
    ax.set_xlabel('Current (mA)', fontsize=12)
    ax.set_ylabel('Resistance (Ohm)', fontsize=12)
    ax.set_title('All Resistances vs Current (Log Scale)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right')
    
    plot_path = output_dir / f"all_resistances_vs_current.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_sample_resistances_vs_voltage(sample_data, sample_id, output_dir):
    """Plot resistances vs voltage for a single sample with original color scheme."""
    print(f"Creating resistances vs voltage plot for sample {sample_id}...")
    
    # Get unique Time points for this sample
    sample_times = sorted(sample_data['Time'].unique())
    len_data = len(sample_times)
    
    # Generate colors for this sample: 4 resistance types * len_data
    colors = get_colors(4 * len_data, colormap="rainbow")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    legend_handles = []
    
    index = 0
    for time in sample_times:
        time_data = sample_data[sample_data['Time'] == time]
        time_data = time_data.sort_values('Ecell_V')
        
        # Plot each resistance with its assigned color (original pattern)
        ax.plot(time_data['Ecell_V'], time_data['R0'],
                'o-', color=colors[index], markersize=5, alpha=0.7, linewidth=1)
        
        ax.plot(time_data['Ecell_V'], time_data['R1'],
                'o-', color=colors[len_data + index], markersize=5, alpha=0.7, linewidth=1)
        
        ax.plot(time_data['Ecell_V'], time_data['R2'],
                'o-', color=colors[2*len_data + index], markersize=5, alpha=0.7, linewidth=1)
        
        ax.plot(time_data['Ecell_V'], time_data['R3'],
                'o-', color=colors[3*len_data + index], markersize=5, alpha=0.7, linewidth=1)
        
        index += 1
    
    # Add legend for resistance types
    for r_type in RESISTANCE_TYPES:
        legend_handles.append(plt.Line2D([0], [0],
            marker=RESISTANCE_MARKERS[r_type], color='black', linestyle='-',
            markersize=5, label=RESISTANCE_LABELS[r_type]))
    
    ax.set_xlabel('Cell Voltage (V)', fontsize=12)
    ax.set_ylabel('Resistance (Ohm)', fontsize=12)
    ax.set_title(f'Sample {sample_id}: All Resistances vs Cell Voltage (Log Scale)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8)
    
    plot_path = output_dir / f"{sample_id}_resistances_vs_voltage.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_sample_resistances_vs_current(sample_data, sample_id, output_dir):
    """Plot resistances vs current for a single sample with original color scheme."""
    print(f"Creating resistances vs current plot for sample {sample_id}...")
    
    # Get unique Time points for this sample
    sample_times = sorted(sample_data['Time'].unique())
    len_data = len(sample_times)
    
    # Generate colors for this sample: 4 resistance types * len_data
    colors = get_colors(4 * len_data, colormap="rainbow")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    legend_handles = []
    
    index = 0
    for time in sample_times:
        time_data = sample_data[sample_data['Time'] == time]
        time_data = time_data.sort_values('I_mA')
        
        # Plot each resistance with its assigned color (original pattern)
        ax.plot(time_data['I_mA'], time_data['R0'],
                'o-', color=colors[index], markersize=5, alpha=0.7, linewidth=1)
        
        ax.plot(time_data['I_mA'], time_data['R1'],
                'o-', color=colors[len_data + index], markersize=5, alpha=0.7, linewidth=1)
        
        ax.plot(time_data['I_mA'], time_data['R2'],
                'o-', color=colors[2*len_data + index], markersize=5, alpha=0.7, linewidth=1)
        
        ax.plot(time_data['I_mA'], time_data['R3'],
                'o-', color=colors[3*len_data + index], markersize=5, alpha=0.7, linewidth=1)
        
        index += 1
    
    # Add legend for resistance types
    for r_type in RESISTANCE_TYPES:
        legend_handles.append(plt.Line2D([0], [0],
            marker=RESISTANCE_MARKERS[r_type], color='black', linestyle='-',
            markersize=5, label=RESISTANCE_LABELS[r_type]))
    
    ax.set_xlabel('Current (mA)', fontsize=12)
    ax.set_ylabel('Resistance (Ohm)', fontsize=12)
    ax.set_title(f'Sample {sample_id}: All Resistances vs Current (Log Scale)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8)
    
    plot_path = output_dir / f"{sample_id}_resistances_vs_current.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_all_samples_resistances_vs_voltage(combined_df, output_dir):
    """Plot all samples resistances vs voltage - simple scatter with sample colors."""
    print("Creating combined resistances vs voltage plot (all samples)...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    legend_handles = []
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        color = SAMPLE_COLORS[sample_id]
        
        for r_type in RESISTANCE_TYPES:
            marker = RESISTANCE_MARKERS[r_type]
            label = f'{sample_id} {r_type}'
            
            ax.plot(sample_data['Ecell_V'], sample_data[r_type],
                    marker=marker, color=color, linestyle='',
                    markersize=4, alpha=0.5, label=label)
    
    ax.set_xlabel('Cell Voltage (V)', fontsize=12)
    ax.set_ylabel('Resistance (Ohm)', fontsize=12)
    ax.set_title('All Samples: Resistances vs Cell Voltage (Log Scale)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    
    plot_path = output_dir / f"all_samples_resistances_vs_voltage.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


def plot_all_samples_resistances_vs_current(combined_df, output_dir):
    """Plot all samples resistances vs current - simple scatter with sample colors."""
    print("Creating combined resistances vs current plot (all samples)...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    legend_handles = []
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        color = SAMPLE_COLORS[sample_id]
        
        for r_type in RESISTANCE_TYPES:
            marker = RESISTANCE_MARKERS[r_type]
            label = f'{sample_id} {r_type}'
            
            ax.plot(sample_data['I_mA'], sample_data[r_type],
                    marker=marker, color=color, linestyle='',
                    markersize=4, alpha=0.5, label=label)
    
    ax.set_xlabel('Current (mA)', fontsize=12)
    ax.set_ylabel('Resistance (Ohm)', fontsize=12)
    ax.set_title('All Samples: Resistances vs Current (Log Scale)', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    
    plot_path = output_dir / f"all_samples_resistances_vs_current.{PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Plot resistances vs voltage and current for all samples."""
    
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
    
    # Create output directory
    output_dir = Path("resistances_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Also ensure sample folders exist
    for sample_id in SAMPLES:
        sample_dir = Path(f"sample_{sample_id}")
        sample_dir.mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # Plot 1: All resistances vs Cell Voltage (log Y scale)
    # Original color scheme: rainbow per sample, 4*len_data colors
    # ========================================================================
    plot_all_resistances_vs_voltage(combined_df, output_dir)
    
    # ========================================================================
    # Plot 2: All resistances vs Current (log Y scale)
    # Original color scheme: rainbow per sample, 4*len_data colors
    # ========================================================================
    plot_all_resistances_vs_current(combined_df, output_dir)
    
    # ========================================================================
    # Plot 3-4: Per-sample plots with original color scheme
    # ========================================================================
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        plot_sample_resistances_vs_voltage(sample_data, sample_id, output_dir)
        plot_sample_resistances_vs_current(sample_data, sample_id, output_dir)
        
        # Copy per-sample plots to sample folder
        for suffix in ['vs_voltage', 'vs_current']:
            src = output_dir / f"{sample_id}_resistances_{suffix}.{PLOT_FORMAT}"
            dst = Path(f"sample_{sample_id}") / f"{sample_id}_resistances_{suffix}.{PLOT_FORMAT}"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"  Copied: {src} -> {dst}")
    
    # ========================================================================
    # Plot 5-6: Combined all samples plots
    # ========================================================================
    plot_all_samples_resistances_vs_voltage(combined_df, output_dir)
    plot_all_samples_resistances_vs_current(combined_df, output_dir)
    
    print(f"\nResistances vs Voltage/Current analysis complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
