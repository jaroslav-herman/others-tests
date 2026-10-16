#!/usr/bin/env python3
"""
Plot all resistances (R0, R1, R2, R3) as a function of:
1. Cell voltage (with logarithmic Y scale)
2. Current (with logarithmic Y scale)

Following the pattern from eis_parameters.py:
- Different colors based on Time
- Different line styles/markers based on resistance type
- All samples in one graph for comparison

Usage:
    python resistances_vs_voltage_current.py
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

# Resistance types and their colors/markers
RESISTANCE_TYPES = ['R0', 'R1', 'R2', 'R3']
RESISTANCE_COLORS = {
    'R0': 'C0',  # First color in tab10
    'R1': 'C1',  # Second color in tab10
    'R2': 'C2',  # Third color in tab10
    'R3': 'C3'   # Fourth color in tab10
}
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
    
    # ========================================================================
    # Plot 1: All resistances vs Cell Voltage (log Y scale)
    # Different colors based on Time, different markers based on resistance type
    # ========================================================================
    
    print("Creating resistances vs voltage plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Get unique Time points for color mapping
    all_times = sorted(combined_df['Time'].unique())
    n_times = len(all_times)
    
    # Create a colormap for Time points
    time_colors = plt.cm.viridis(np.linspace(0, 1, n_times))
    
    # Create legend handles
    legend_handles = []
    legend_labels = []
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        
        for time in all_times:
            time_data = sample_data[sample_data['Time'] == time]
            if len(time_data) == 0:
                continue
            
            time_data = time_data.sort_values('Ecell_V')
            
            # Get color for this Time
            time_idx = all_times.index(time)
            color = time_colors[time_idx]
            
            for r_type in RESISTANCE_TYPES:
                marker = RESISTANCE_MARKERS[r_type]
                label = f'{sample_id} {r_type}'
                
                ax.plot(time_data['Ecell_V'], time_data[r_type],
                        marker=marker, color=color, linestyle='-',
                        markersize=5, alpha=0.7, linewidth=1)
            
            # Add Time legend entry (only once per sample)
            if time == all_times[0]:
                legend_handles.append(plt.Line2D([0], [0], 
                    marker='o', color='black', linestyle='-',
                    markersize=5, label=f'{sample_id} Time {time}'))
    
    # Add resistance type legend entries
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
    
    # ========================================================================
    # Plot 2: All resistances vs Current (log Y scale)
    # ========================================================================
    
    print("Creating resistances vs current plot...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    legend_handles = []
    legend_labels = []
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        
        for time in all_times:
            time_data = sample_data[sample_data['Time'] == time]
            if len(time_data) == 0:
                continue
            
            time_data = time_data.sort_values('I_mA')
            
            # Get color for this Time
            time_idx = all_times.index(time)
            color = time_colors[time_idx]
            
            for r_type in RESISTANCE_TYPES:
                marker = RESISTANCE_MARKERS[r_type]
                
                ax.plot(time_data['I_mA'], time_data[r_type],
                        marker=marker, color=color, linestyle='-',
                        markersize=5, alpha=0.7, linewidth=1)
            
            # Add Time legend entry (only once per sample)
            if time == all_times[0]:
                legend_handles.append(plt.Line2D([0], [0], 
                    marker='o', color='black', linestyle='-',
                    markersize=5, label=f'{sample_id} Time {time}'))
    
    # Add resistance type legend entries
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
    
    # ========================================================================
    # Plot 3: Resistances vs Voltage - Separate by Sample
    # Each sample gets its own plot with all Times
    # ========================================================================
    
    print("Creating per-sample resistances vs voltage plots...")
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        
        # Get unique Time points for this sample
        sample_times = sorted(sample_data['Time'].unique())
        n_sample_times = len(sample_times)
        
        # Create colormap for Times
        sample_time_colors = plt.cm.viridis(np.linspace(0, 1, n_sample_times))
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        legend_handles = []
        
        for time_idx, time in enumerate(sample_times):
            time_data = sample_data[sample_data['Time'] == time]
            time_data = time_data.sort_values('Ecell_V')
            color = sample_time_colors[time_idx]
            
            for r_type in RESISTANCE_TYPES:
                marker = RESISTANCE_MARKERS[r_type]
                ax.plot(time_data['Ecell_V'], time_data[r_type],
                        marker=marker, color=color, linestyle='-',
                        markersize=5, alpha=0.7, linewidth=1,
                        label=f'Time {time} {r_type}' if time_idx == 0 else '')
        
        # Add legend entries for resistance types
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
    
    # ========================================================================
    # Plot 4: Resistances vs Current - Separate by Sample
    # ========================================================================
    
    print("Creating per-sample resistances vs current plots...")
    
    for sample_id in SAMPLES:
        sample_data = combined_df[combined_df['sample_id'] == sample_id]
        
        # Get unique Time points for this sample
        sample_times = sorted(sample_data['Time'].unique())
        n_sample_times = len(sample_times)
        
        # Create colormap for Times
        sample_time_colors = plt.cm.viridis(np.linspace(0, 1, n_sample_times))
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        
        legend_handles = []
        
        for time_idx, time in enumerate(sample_times):
            time_data = sample_data[sample_data['Time'] == time]
            time_data = time_data.sort_values('I_mA')
            color = sample_time_colors[time_idx]
            
            for r_type in RESISTANCE_TYPES:
                marker = RESISTANCE_MARKERS[r_type]
                ax.plot(time_data['I_mA'], time_data[r_type],
                        marker=marker, color=color, linestyle='-',
                        markersize=5, alpha=0.7, linewidth=1,
                        label=f'Time {time} {r_type}' if time_idx == 0 else '')
        
        # Add legend entries for resistance types
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
    
    # ========================================================================
    # Plot 5: Combined - All samples, resistances vs Voltage
    # Different colors for each sample, different markers for each resistance
    # ========================================================================
    
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
    
    # ========================================================================
    # Plot 6: Combined - All samples, resistances vs Current
    # ========================================================================
    
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
    
    print(f"\nResistances vs Voltage/Current analysis complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
