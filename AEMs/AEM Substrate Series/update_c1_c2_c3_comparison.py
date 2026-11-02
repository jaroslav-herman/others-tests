#!/usr/bin/env python3
"""
Regenerate C1, C2 and C3 intercept-q time-evolution comparison graphs
for the AEM Substrate Series, including every sample in the series.

Outputs (saved to this folder, 'AEM Substrate Series/'):
  - c1_c2_c3_q_time_evolution_comparison.png  (3-row figure, one row per capacitance)
  - c1_q_time_evolution_comparison.png, c2_q_time_evolution_comparison.png,
    c3_q_time_evolution_comparison.png       (individual graphs per capacitance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.ioff()

# AEM Substrate Series samples (474, 499, 505, 516 - NOT 498)
SAMPLES = ['474', '499', '505', '516']

sample_names = {
    '474': 'VI_Ni_on_Ni_PTL',
    '499': 'VI_Ni_on_Ni_etched',
    '505': 'VI_Ni_on_Ni_plain',
    '516': 'Ni PTL no sputtering',
}

colors = {
    '474': 'blue',
    '499': 'orange',
    '505': 'green',
    '516': 'purple',
}

markers = {
    '474': 'o',
    '499': 's',
    '505': '^',
    '516': 'D',
}

# Capacitance folder prefixes differ in case: c1_linear, C2_linear, C3_linear
CAPACITORS = {
    'C1': 'c1',
    'C2': 'C2',
    'C3': 'C3',
}


def summary_path(sample_id, param_name):
    prefix = CAPACITORS[param_name]
    return Path(f'../sample_{sample_id}/{prefix}_linear_{sample_id}/{sample_id}_{param_name}_fit_summary.csv')


def plot_single(param_name, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for sample_id in SAMPLES:
        csv_path = summary_path(sample_id, param_name)
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
    ax.set_ylabel(f'Intercept q (F)', fontsize=12)
    ax.set_title(f'Time Evolution of Intercept q from {param_name} = k*I + q - AEM Substrate Series', fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)

    output_path = output_dir / f'{param_name.lower()}_q_time_evolution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined(output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    for idx, param_name in enumerate(['C1', 'C2', 'C3']):
        ax = axes[idx]

        for sample_id in SAMPLES:
            csv_path = summary_path(sample_id, param_name)
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
        ax.set_ylabel('Intercept q (F)', fontsize=12)
        ax.set_title(f'{param_name} = k*I + q', fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('Time', fontsize=12)

    plt.tight_layout()
    output_path = output_dir / 'c1_c2_c3_q_time_evolution_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path('.')
    output_dir.mkdir(exist_ok=True, parents=True)

    for param_name in ['C1', 'C2', 'C3']:
        plot_single(param_name, output_dir)
    plot_combined(output_dir)

    print("\nAll C1/C2/C3 comparison graphs updated!")


if __name__ == '__main__':
    main()
