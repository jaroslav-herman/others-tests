#!/usr/bin/env python3
"""
EIS Data Processing Script for AEMs
Processes EIS fitting results from eisyFIT program in CSV files.

Usage:
    python process_eis_data.py [--sample SAMPLE_ID] [--output OUTPUT_DIR]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_SAMPLES_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = DEFAULT_SAMPLES_DIR / "processed_results"

# Color palette for consistent plotting
COLORS = {
    '474': '#1f77b4',  # blue
    '498': '#ff7f0e',  # orange
    '499': '#2ca02c',  # green
}

# ============================================================================
# Data Loading
# ============================================================================

def load_sample_data(sample_id: str, data_dir: Path = None) -> pd.DataFrame:
    """Load EIS fitting data for a specific sample."""
    if data_dir is None:
        data_dir = DEFAULT_SAMPLES_DIR
    
    csv_file = data_dir / f"{sample_id}_fit_export.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"Sample data file not found: {csv_file}")
    
    # Load CSV with proper encoding handling
    df = pd.read_csv(csv_file)
    
    # Add sample_id column for reference
    df['sample_id'] = sample_id
    
    # Clean up duplicate column names (Ecell_V appears twice in some files)
    if 'Ecell_V.1' in df.columns:
        df = df.drop(columns=['Ecell_V.1'])
    
    return df


def load_all_samples(data_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """Load all sample data files from the directory."""
    if data_dir is None:
        data_dir = DEFAULT_SAMPLES_DIR
    
    samples = {}
    for csv_file in data_dir.glob("*_fit_export.csv"):
        sample_id = csv_file.stem.split("_")[0]
        samples[sample_id] = load_sample_data(sample_id, data_dir)
    
    return samples


# ============================================================================
# Data Processing Functions
# ============================================================================

def calculate_derived_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived parameters from EIS fitting results."""
    df = df.copy()
    
    # Calculate total resistance (sum of all R components)
    r_columns = [col for col in df.columns if col.startswith('R') and col != 'R0_e']
    df['R_total'] = df[[col for col in r_columns if 'e' not in col]].sum(axis=1)
    
    # Calculate total capacitance (sum of all C components)
    c_columns = [col for col in df.columns if col.startswith('C') and not col.startswith('CPE')]
    df['C_total'] = df[c_columns].sum(axis=1)
    
    # Calculate current density (assuming area is 4.84 cm2 based on existing code)
    # I_mA is in mA, convert to mA/cm2
    df['I_mA_cm2'] = df['I_mA'] / 4.84
    
    # Calculate overpotential (Ecell_V - theoretical voltage at 0 current)
    # Approximate: find Ecell at I=0 for each time point
    df['eta'] = df['Ecell_V'] - df.groupby('Time')['Ecell_V'].transform('min')
    
    return df


def filter_data_by_current(df: pd.DataFrame, max_current: float = 300) -> pd.DataFrame:
    """Filter data to only include points below a certain current threshold."""
    return df[df['I_mA'] < max_current].copy()


def group_by_time(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Group data by Time column."""
    return {int(time): group.copy() for time, group in df.groupby('Time')}


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_ohmic_resistance(samples: Dict[str, pd.DataFrame], 
                           output_dir: Path = None) -> Dict[str, Dict]:
    """Analyze R0 (ohmic resistance) vs current for each sample."""
    results = {}
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        sample_results = {
            'time_points': [],
            'R0_at_zero_current': [],
            'R0_slope': [],
            'R0_intercept': []
        }
        
        for time, group in sorted(time_groups.items()):
            # Filter to reasonable current range
            group = group[(group['I_mA'] < 300) & (group['I_mA'] > 0)]
            
            if len(group) < 5:
                continue
            
            # Fit R0 vs I_mA
            x = group['I_mA'].values
            y = group['R0'].values
            
            # Linear fit
            k, q = np.polyfit(x, y, 1)
            
            sample_results['time_points'].append(time)
            sample_results['R0_at_zero_current'].append(q)
            sample_results['R0_slope'].append(k)
            sample_results['R0_intercept'].append(q)
        
        results[sample_id] = sample_results
    
    return results


def analyze_charge_transfer_resistance(samples: Dict[str, pd.DataFrame],
                                     output_dir: Path = None) -> Dict[str, Dict]:
    """Analyze R1 (charge transfer resistance) vs current."""
    results = {}
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        sample_results = {
            'time_points': [],
            'R1_at_200mA': [],
            'R1_slope': [],
            'R1_intercept': [],
            'tafel_slope': []
        }
        
        for time, group in sorted(time_groups.items()):
            group = group[(group['I_mA'] < 300) & (group['I_mA'] > 10)]
            
            if len(group) < 10:  # Need enough points for fitting
                continue
            
            x = group['I_mA'].values - group['I_mA'].iloc[0]
            y = group['R1'].values
            
            # Fit R1 vs I - use middle 80% of data
            start_idx = int(len(x) * 0.1)
            end_idx = int(len(x) * 0.9)
            if end_idx <= start_idx:
                continue
            
            try:
                # Fit R1 vs I
                k, q = np.polyfit(x[start_idx:end_idx], y[start_idx:end_idx], 1)
                
                # Calculate Tafel slope from 1/R1 vs I
                y_inv = 1 / group['R1'].values
                k_inv, q_inv = np.polyfit(x[start_idx:end_idx], y_inv[start_idx:end_idx], 1)
                
                # Interpolate R1 at 200 mA
                if x[-1] > 200:
                    interp_r1 = np.interp(200, x, y)
                elif x[-1] > 0:
                    interp_r1 = y[-1]  # Use last available value
                else:
                    interp_r1 = np.nan
            except (ValueError, TypeError) as e:
                continue
            
            sample_results['time_points'].append(time)
            sample_results['R1_at_200mA'].append(float(interp_r1) if not np.isnan(interp_r1) else None)
            sample_results['R1_slope'].append(float(k))
            sample_results['R1_intercept'].append(float(q))
            sample_results['tafel_slope'].append(float(2.3 / k_inv) if k_inv != 0 and not np.isnan(k_inv) else None)
        
        results[sample_id] = sample_results
    
    return results


def analyze_capacitance(samples: Dict[str, pd.DataFrame],
                      output_dir: Path = None) -> Dict[str, Dict]:
    """Analyze capacitance parameters vs current."""
    results = {}
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        sample_results = {
            'time_points': [],
            'C1_at_zero_current': [],
            'C1_slope': [],
            'C1_intercept': [],
            'C_total': []
        }
        
        for time, group in sorted(time_groups.items()):
            group = group[group['I_mA'] < 300]
            
            if len(group) < 10:
                continue
            
            x = group['I_mA'].values - group['I_mA'].iloc[0]
            y = group['C1'].values
            
            # Use middle 80% of data for fitting
            start_idx = int(len(x) * 0.1)
            end_idx = int(len(x) * 0.9)
            if end_idx <= start_idx:
                continue
            
            try:
                k, q = np.polyfit(x[start_idx:end_idx], y[start_idx:end_idx], 1)
            except (ValueError, TypeError):
                continue
            
            sample_results['time_points'].append(time)
            sample_results['C1_at_zero_current'].append(float(q))
            sample_results['C1_slope'].append(float(k))
            sample_results['C1_intercept'].append(float(q))
            sample_results['C_total'].append(float(group['C1'].mean()))
        
        results[sample_id] = sample_results
    
    return results


def analyze_time_evolution(samples: Dict[str, pd.DataFrame],
                         output_dir: Path = None) -> Dict[str, Dict]:
    """Analyze how parameters evolve over time."""
    results = {}
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        sample_results = {
            'time_points': [],
            'current_at_1.65V': [],
            'R1_at_1.65V': [],
            'R_total_at_1.65V': [],
            'Ecell_at_200mA': []
        }
        
        for time, group in sorted(time_groups.items()):
            # Find current at Ecell = 1.65V
            group_sorted = group.sort_values('Ecell_V')
            idx = (group_sorted['Ecell_V'] - 1.65).abs().idxmin()
            current_at_165 = group_sorted.loc[idx, 'I_mA'] - group_sorted.iloc[0]['I_mA']
            
            # Find R1 at Ecell = 1.65V
            r1_at_165 = group_sorted.loc[idx, 'R1']
            
            # Find Ecell at I = 200 mA
            x = group['I_mA'].values
            y = group['Ecell_V'].values
            try:
                ecell_at_200 = np.interp(200, x - x[0], y)
            except:
                ecell_at_200 = np.nan
            
            sample_results['time_points'].append(time)
            sample_results['current_at_1.65V'].append(current_at_165)
            sample_results['R1_at_1.65V'].append(r1_at_165)
            sample_results['R_total_at_1.65V'].append(group_sorted.loc[idx, 'R_total'])
            sample_results['Ecell_at_200mA'].append(ecell_at_200)
        
        results[sample_id] = sample_results
    
    return results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_R_vs_current(samples: Dict[str, pd.DataFrame], 
                     output_dir: Path = None,
                     max_current: float = 300) -> None:
    """Plot R0, R1, R2, R3 vs current for each sample."""
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Sample {sample_id} - Resistance vs Current', fontsize=16)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_groups)))
        
        for (time, group), color in zip(sorted(time_groups.items()), colors):
            group = group[group['I_mA'] < max_current]
            x = group['I_mA'].values - group['I_mA'].iloc[0]
            
            # R0
            axes[0, 0].plot(x, group['R0'].values, 'o-', color=color, 
                           label=f'Time {time}', alpha=0.7)
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_ylabel('R0 (Ohm)')
            
            # R1
            axes[0, 1].plot(x, group['R1'].values, 'o-', color=color, label=f'Time {time}', alpha=0.7)
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_ylabel('R1 (Ohm)')
            
            # R2
            axes[1, 0].plot(x, group['R2'].values, 'o-', color=color, label=f'Time {time}', alpha=0.7)
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_ylabel('R2 (Ohm)')
            
            # R3
            axes[1, 1].plot(x, group['R3'].values, 'o-', color=color, label=f'Time {time}', alpha=0.7)
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_ylabel('R3 (Ohm)')
        
        for ax in axes.flat:
            ax.set_xlabel('Current (mA)')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(fontsize=6)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f'{sample_id}_R_vs_current.png', dpi=300)
        
        plt.show()


def plot_capacitance_vs_current(samples: Dict[str, pd.DataFrame],
                               output_dir: Path = None,
                               max_current: float = 300) -> None:
    """Plot C1, C2, C3, C4 vs current for each sample."""
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Sample {sample_id} - Capacitance vs Current', fontsize=16)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_groups)))
        
        for (time, group), color in zip(sorted(time_groups.items()), colors):
            group = group[group['I_mA'] < max_current]
            x = group['I_mA'].values - group['I_mA'].iloc[0]
            
            if 'C1' in group.columns:
                axes[0, 0].plot(x, group['C1'].values, 'o-', color=color, 
                               label=f'Time {time}', alpha=0.7)
                axes[0, 0].set_ylabel('C1 (F)')
            
            if 'C2' in group.columns:
                axes[0, 1].plot(x, group['C2'].values, 'o-', color=color, label=f'Time {time}', alpha=0.7)
                axes[0, 1].set_ylabel('C2 (F)')
            
            if 'C3' in group.columns:
                axes[1, 0].plot(x, group['C3'].values, 'o-', color=color, label=f'Time {time}', alpha=0.7)
                axes[1, 0].set_ylabel('C3 (F)')
            
            if 'C4' in group.columns:
                axes[1, 1].plot(x, group['C4'].values, 'o-', color=color, label=f'Time {time}', alpha=0.7)
                axes[1, 1].set_ylabel('C4 (F)')
        
        for ax in axes.flat:
            ax.set_xlabel('Current (mA)')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(fontsize=6)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f'{sample_id}_C_vs_current.png', dpi=300)
        
        plt.show()


def plot_time_evolution(samples: Dict[str, pd.DataFrame],
                       output_dir: Path = None) -> None:
    """Plot time evolution of key parameters."""
    
    # Analyze time evolution
    time_evolution = analyze_time_evolution(samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Time Evolution of Key Parameters', fontsize=16)
    
    for sample_id, color in COLORS.items():
        if sample_id not in time_evolution:
            continue
        
        data = time_evolution[sample_id]
        
        # Current at 1.65V
        axes[0, 0].plot(data['time_points'], data['current_at_1.65V'], 
                        'o-', color=color, label=f'Sample {sample_id}')
        axes[0, 0].set_ylabel('Current at 1.65V (mA)')
        
        # R1 at 1.65V
        axes[0, 1].plot(data['time_points'], data['R1_at_1.65V'], 
                        'o-', color=color, label=f'Sample {sample_id}')
        axes[0, 1].set_ylabel('R1 at 1.65V (Ohm)')
        axes[0, 1].set_yscale('log')
        
        # R_total at 1.65V
        axes[1, 0].plot(data['time_points'], data['R_total_at_1.65V'], 
                        'o-', color=color, label=f'Sample {sample_id}')
        axes[1, 0].set_ylabel('R_total at 1.65V (Ohm)')
        axes[1, 0].set_yscale('log')
        
        # Ecell at 200mA
        axes[1, 1].plot(data['time_points'], data['Ecell_at_200mA'], 
                        'o-', color=color, label=f'Sample {sample_id}')
        axes[1, 1].set_ylabel('Ecell at 200mA (V)')
    
    for ax in axes.flat:
        ax.set_xlabel('Time')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'time_evolution.png', dpi=300)
    
    plt.show()


def plot_iv_curves(samples: Dict[str, pd.DataFrame],
                   output_dir: Path = None) -> None:
    """Plot IV curves for each sample."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        color = COLORS.get(sample_id, '#888888')
        
        for time, group in sorted(time_groups.items()):
            group = group[group['Ecell_V'] < 1.65]
            ax.plot(group['Ecell_V'].values, group['I_mA'].values - group['I_mA'].iloc[0],
                    '-', color=color, alpha=0.3, label=f'{sample_id} Time {time}')
    
    ax.set_xlabel('Cell Voltage (V)')
    ax.set_ylabel('Current (mA)')
    ax.set_title('IV Curves')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=6)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / 'iv_curves.png', dpi=300)
    
    plt.show()


def plot_nyquist(samples: Dict[str, pd.DataFrame],
                output_dir: Path = None) -> None:
    """Plot Nyquist plots (R vs -Im(Z)) for each sample."""
    # Note: This would require impedance data, not just fitting parameters
    # For now, we'll plot R_total vs frequency as a proxy
    
    for sample_id, df in samples.items():
        df = calculate_derived_parameters(df)
        time_groups = group_by_time(df)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_groups)))
        
        for (time, group), color in zip(sorted(time_groups.items()), colors):
            # Sort by frequency (approximate from fmin_act_Hz)
            group = group.sort_values('fmin_act_Hz', ascending=False)
            ax.plot(group['R0'].values, group['R1'].values, 
                    'o-', color=color, label=f'Time {time}', alpha=0.7)
        
        ax.set_xlabel('R0 (Ohm)')
        ax.set_ylabel('R1 (Ohm)')
        ax.set_title(f'Sample {sample_id} - R0 vs R1')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize=6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f'{sample_id}_R0_vs_R1.png', dpi=300)
        
        plt.show()


# ============================================================================
# Main Processing Function
# ============================================================================

def process_samples(sample_ids: List[str] = None,
                   data_dir: Path = None,
                   output_dir: Path = None,
                   plot: bool = True,
                   save: bool = True) -> Dict:
    """
    Main function to process EIS data for AEM samples.
    
    Args:
        sample_ids: List of sample IDs to process. If None, process all.
        data_dir: Directory containing CSV files
        output_dir: Directory to save results
        plot: Whether to generate plots
        save: Whether to save results to JSON
    
    Returns:
        Dictionary containing all analysis results
    """
    # Setup directories
    if data_dir is None:
        data_dir = DEFAULT_SAMPLES_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Load samples
    if sample_ids:
        samples = {sid: load_sample_data(sid, data_dir) for sid in sample_ids}
    else:
        samples = load_all_samples(data_dir)
    
    if not samples:
        print("No samples found to process!")
        return {}
    
    print(f"Processing {len(samples)} samples: {list(samples.keys())}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    results = {
        'ohmic_resistance': analyze_ohmic_resistance(samples, output_dir),
        'charge_transfer_resistance': analyze_charge_transfer_resistance(samples, output_dir),
        'capacitance': analyze_capacitance(samples, output_dir),
        'time_evolution': analyze_time_evolution(samples, output_dir),
    }
    
    # Generate plots if requested
    if plot:
        print("Generating plots...")
        plot_R_vs_current(samples, output_dir)
        plot_capacitance_vs_current(samples, output_dir)
        plot_time_evolution(samples, output_dir)
        plot_iv_curves(samples, output_dir)
        plot_nyquist(samples, output_dir)
    
    # Save results if requested
    if save:
        print("Saving results...")
        results_file = output_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = json.loads(
                json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x))
            )
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to: {results_file}")
    
    return results


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Process EIS data from AEM samples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python process_eis_data.py                    # Process all samples
    python process_eis_data.py --sample 474      # Process only sample 474
    python process_eis_data.py --output ./results  # Save to custom directory
        """
    )
    parser.add_argument('--sample', '-s', nargs='*', default=None,
                        help='Sample ID(s) to process (default: all)')
    parser.add_argument('--data-dir', '-d', type=Path, default=None,
                        help='Directory containing CSV files')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output directory for results')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving results')
    
    args = parser.parse_args()
    
    # Process samples
    process_samples(
        sample_ids=args.sample,
        data_dir=args.data_dir,
        output_dir=args.output,
        plot=not args.no_plot,
        save=not args.no_save
    )


if __name__ == '__main__':
    main()
