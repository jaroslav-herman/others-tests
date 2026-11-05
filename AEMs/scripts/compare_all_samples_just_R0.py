#!/usr/bin/env python3
"""Compare ohmic resistance (R0) evolution for AEM Substrate Series samples."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})
plt.ioff()

SAMPLES = ['474', '499', '505', '516']
SAMPLE_NAMES = {
    '474': 'Ni on Ni PTL',
    '499': 'Ni on Ni etched membrane',
    '505': 'Ni on Ni plain membrane',
    '516': 'Ni PTL, no sputtering',
}
SAMPLE_COLORS = {
    '474': 'purple',
    '499': 'blue',
    '505': 'red',
    '516': 'green',
}
PLOT_DPI = 200
BASE_DIR = Path(r'C:\Users\Herman\Python\WE\others-tests\AEMs')
OUTPUT_DIR = BASE_DIR / 'AEM Substrate Series'


def load_r0_fit_summary(sample_id):
    """Load the time evolution of q from the R0 = k*I + q linear fits."""
    csv_path = (
        BASE_DIR / f"sample_{sample_id}" / f"r0_linear_{sample_id}"
        / f"{sample_id}_R0_fit_summary.csv"
    )
    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    required_columns = ['time', 'q']
    for column in required_columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in sample {sample_id}")
            return None
    return df[required_columns].copy()


def plot_r0_intercept_evolution(fit_summaries, output_dir):
    """Plot the linear-fit R0 intercept q against measurement time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for sample_id in SAMPLES:
        df = fit_summaries.get(sample_id)
        if df is None:
            continue
        plot_data = df[
            np.isfinite(df['time'])
            & np.isfinite(df['q'])
        ].sort_values('time')
        if not plot_data.empty:
            ax.plot(
                plot_data['time'],
                plot_data['q']*4.84,
                'o-',
                color=SAMPLE_COLORS[sample_id],
                markersize=6,
                linewidth=2,
                label=SAMPLE_NAMES[sample_id],
            )

    measurement_times = sorted({
        time for df in fit_summaries.values()
        if df is not None
        for time in df['time'].dropna().unique()
    })
    day_ticks = measurement_times[::3]
    ax.set_xticks(day_ticks, labels=range(1, len(day_ticks) + 1))
    ax.set_xlabel('Day', fontsize=18)
    ax.set_ylabel(
        r'Ohmic Resistance $\mathrm{(\Omega~cm^{2})}$',
        fontsize=18,
    )
    ax.tick_params(axis='both', which='both', labelsize=16)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_xlim(0.5, 16.5)
    ax.legend(loc='best',fontsize=15)

    output_path = output_dir / 'ohmic_resistance_time_evolution_comparison.png'
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    fit_summaries = {}
    for sample_id in SAMPLES:
        df = load_r0_fit_summary(sample_id)
        if df is not None:
            fit_summaries[sample_id] = df

    plot_r0_intercept_evolution(fit_summaries, OUTPUT_DIR)
    print("\nOhmic resistance linear-fit evolution graph generated successfully!")


if __name__ == '__main__':
    main()
