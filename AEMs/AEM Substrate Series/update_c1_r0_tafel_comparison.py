#!/usr/bin/env python3
"""
Regenerate C1, R0 (ohmic) and Tafel slope time-evolution comparison graphs
for the AEM Substrate Series, including every sample in the series.

Outputs (saved to this folder, 'AEM Substrate Series/'):
  - c1_time_evolution_comparison.png
  - ohmic_resistance_time_evolution_comparison.png
  - tafel_slopes_time_evolution_comparison.png
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


def create_comparison_graph(title, ylabel, csv_path_fn, x_col, y_col, output_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for sample_id in SAMPLES:
        csv_path = csv_path_fn(sample_id)
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue

        df = pd.read_csv(csv_path)
        label = f'{sample_id} ({sample_names.get(sample_id, "N/A")})'

        ax.plot(df[x_col], df[y_col],
                marker=markers[sample_id],
                color=colors[sample_id],
                linewidth=2,
                markersize=8,
                label=label)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)

    output_path = Path('.') / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    create_comparison_graph(
        'Time Evolution of Intercept q from C1 = k*I + q - AEM Substrate Series',
        'Intercept q (F)',
        lambda s: Path(f'../sample_{s}/c1_linear_{s}/{s}_C1_fit_summary.csv'),
        'time', 'q',
        'c1_time_evolution_comparison.png',
    )
    create_comparison_graph(
        'Ohmic Resistance R0 Time Evolution (intercept q) - AEM Substrate Series',
        'Intercept q (Ohm)',
        lambda s: Path(f'../sample_{s}/r0_linear_{s}/{s}_R0_fit_summary.csv'),
        'time', 'q',
        'ohmic_resistance_time_evolution_comparison.png',
    )
    create_comparison_graph(
        'Tafel Slope Time Evolution - AEM Substrate Series',
        'Tafel Slope (mV/dec)',
        lambda s: Path(f'../sample_{s}/tafel_700ma_{s}/{s}_tafel_summary.csv'),
        'time', 'tafel_slope',
        'tafel_slopes_time_evolution_comparison.png',
    )
    print("\nAll comparison graphs updated!")


if __name__ == '__main__':
    main()
