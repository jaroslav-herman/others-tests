#!/usr/bin/env python3
"""
Regenerate R2 and R3 comparison graphs with new current limits (5-700 mA).
This script reads the updated summary CSV files and creates comparison plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Sample names from Google Sheet
sample_names = {
    '474': 'VI_Ni_on_Ni_PTL',
    '499': 'V_Ni90nm(100)_on_etched_piperion',
    '505': 'V_Ni90nm(100)_non-etched_piperion'
}

# Colors and markers for each sample
colors = {
    '474': 'blue',
    '499': 'orange',
    '505': 'green'
}

markers = {
    '474': 'o',
    '499': 's',
    '505': '^'
}

param_units = {
    'R0': 'Ohm',
    'R2': 'Ohm',
    'R3': 'Ohm'
}

param_labels = {
    'R0': 'R0 (Ohm)',
    'R2': 'R2 (Ohm)',
    'R3': 'R3 (Ohm)'
}

# ============================================================================
# Create comparison graph for a single parameter
# ============================================================================

def create_comparison_graph(param_name, output_dir):
    """Create comparison graph for a single parameter (q vs Time)."""
    
    # Handle case sensitivity for R0 (r0_linear vs R0_linear) and R2/R3 (R2_linear vs r2_linear)
    dir_prefix = 'r0' if param_name == 'R0' else param_name
    samples = {
        '474': Path(f'../sample_474/{dir_prefix}_linear_474/474_{param_name}_fit_summary.csv'),
        '499': Path(f'../sample_499/{dir_prefix}_linear_499/499_{param_name}_fit_summary.csv'),
        '505': Path(f'../sample_505/{dir_prefix}_linear_505/505_{param_name}_fit_summary.csv')
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for sample_id, csv_path in samples.items():
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue
        
        df = pd.read_csv(csv_path)
        label = f'{sample_id} ({sample_names.get(sample_id, "N/A")})'
        
        ax.plot(df['time'], df['q'], 
                marker=markers[sample_id], 
                color=colors[sample_id],
                linewidth=2, 
                markersize=8, 
                label=label)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(f'Intercept q ({param_units.get(param_name, "")})', fontsize=12)
    ax.set_title(f'Time Evolution of Intercept q from {param_labels.get(param_name, param_name)} = k*I + q', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    
    output_path = output_dir / f'{param_name.lower()}_time_evolution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

# ============================================================================
# Create combined comparison graph for all three resistances
# ============================================================================

def create_combined_comparison_graph(output_dir):
    """Create combined comparison graph with R0, R2, R3 in separate subplots."""
    
    parameters = ['R0', 'R2', 'R3']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx, param_name in enumerate(parameters):
        ax = axes[idx]
        
        # Handle case sensitivity for R0 (r0_linear vs R0_linear) and R2/R3 (R2_linear vs r2_linear)
        dir_prefix = 'r0' if param_name == 'R0' else param_name
        samples = {
            '474': Path(f'../sample_474/{dir_prefix}_linear_474/474_{param_name}_fit_summary.csv'),
            '499': Path(f'../sample_499/{dir_prefix}_linear_499/499_{param_name}_fit_summary.csv'),
            '505': Path(f'../sample_505/{dir_prefix}_linear_505/505_{param_name}_fit_summary.csv')
        }
        
        for sample_id, csv_path in samples.items():
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping...")
                continue
            
            df = pd.read_csv(csv_path)
            label = f'{sample_id} ({sample_names.get(sample_id, "N/A")})'
            
            ax.plot(df['time'], df['q'], 
                    marker=markers[sample_id], 
                    color=colors[sample_id],
                    linewidth=2, 
                    markersize=8, 
                    label=label)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel(f'Intercept q ({param_units.get(param_name, "")})', fontsize=12)
        ax.set_title(f'{param_labels.get(param_name, param_name)}: {param_name} = k*I + q', fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    
    plt.tight_layout()
    output_path = output_dir / 'all_resistances_q_time_evolution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = Path('.')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Creating R2 and R3 comparison graphs with updated current limits...")
    
    # Create individual comparison graphs for R2 and R3
    create_comparison_graph('R2', output_dir)
    create_comparison_graph('R3', output_dir)
    
    # Create combined comparison graph
    create_combined_comparison_graph(output_dir)
    
    print("\nAll comparison graphs updated!")

if __name__ == '__main__':
    main()
