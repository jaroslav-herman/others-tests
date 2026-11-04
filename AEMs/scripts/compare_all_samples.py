#!/usr/bin/env python3
"""
Compare all AEM EIS samples: 474, 486, 491, 499, 505, 516
(AEM Substrate Series 474/499/505/516 + samples 486 and 491; NOT 498).

Generates all cross-sample comparison graphs into
'AEMs/comparisons_474_486_491_499_505_516/':
  - current_evolution_at_{V}V.png            (current vs Time at fixed voltages)
  - r1_evolution_at_{I}mA.png                (R1 vs Time at fixed currents)
  - rtotal_evolution_at_{I}mA.png            (R0+R1+R2+R3(+R4) vs Time at fixed currents)
  - tafel_slope(s)_time_evolution_comparison.png
  - {r0,r2,r3}_time_evolution_comparison.png + all_resistances_q_time_evolution_comparison.png
  - {c1,c2,c3}_q_time_evolution_comparison.png + c1_c2_c3_q_time_evolution_comparison.png
  - c1_time_evolution_comparison.png, ohmic_resistance_time_evolution_comparison.png
  - r0_r3_tradeoff_analysis.png/.svg + r0_r3_tradeoff_summary.csv

Legend format: sample_number (sample_name)

Usage:
    python compare_all_samples.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

plt.ioff()

# ============================================================================
# CONFIGURATION
# ============================================================================

# All AEM samples: AEM Substrate Series (474, 499, 505, 516) + 486, 491 (NOT 498)
SAMPLES = ['474', '486', '491', '499', '505', '516']

# Sample display names from Google Sheets (electrolyzer_samples)
# Legend format: sample_number (sample_name)
SAMPLE_NAMES = {
    '474': 'VI_Ni_on_Ni_PTL',
    '486': 'Ni90nm_on_etched_PIPERION',
    '491': 'Coni_t3',
    '499': 'VI_Ni_on_Ni_etched',
    '505': 'VI_Ni_on_Ni_plain',
    '516': 'Ni PTL no sputtering',
}

SAMPLE_COLORS = {
    '474': 'purple',
    '486': 'orange',
    '491': 'brown',
    '499': 'blue',
    '505': 'red',
    '516': 'green',
}

SAMPLE_MARKERS = {
    '474': 'o',
    '486': 's',
    '491': '^',
    '499': 'D',
    '505': 'v',
    '516': 'P',
}

TARGET_VOLTAGES = [1.4, 1.45, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9]
TARGET_CURRENTS_R1 = [50, 100, 200, 500, 1000]
TARGET_CURRENTS_RTOTAL = [100, 200, 500, 1000, 2000, 3000]

PLOT_DPI = 200
PLOT_FORMAT = 'png'

BASE_DIR = Path('/workspace/github__jaroslav-herman__others-tests/AEMs')
OUTPUT_DIR = BASE_DIR / 'comparisons_474_486_491_499_505_516'

# ============================================================================
# Load Data
# ============================================================================

def load_fit_export(sample_id):
    csv_path = BASE_DIR / f"sample_{sample_id}/{sample_id}_fit_export.csv"
    if not csv_path.exists():
        print(f"Warning: File not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    return df

def load_summary(sample_id, folder, filename):
    csv_path = BASE_DIR / f"sample_{sample_id}/{folder}/{filename}"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)

def label_of(sample_id):
    return f"{sample_id} ({SAMPLE_NAMES.get(sample_id, sample_id)})"

# ============================================================================
# Interpolation helpers
# ============================================================================

def interpolate_at_x(df, x_col, y_col, target_x, group_col='Time'):
    """Interpolate y at target x for each group (Time point). Returns DataFrame."""
    results = []
    for t in sorted(df[group_col].unique()):
        g = df[df[group_col] == t].copy().sort_values(x_col)
        g = g.groupby(x_col).agg({y_col: 'mean'}).reset_index()
        if target_x < g[x_col].min() or target_x > g[x_col].max():
            val = np.nan
        else:
            val = np.interp(target_x, g[x_col], g[y_col])
        results.append({group_col: t, 'target': target_x, y_col: val})
    return pd.DataFrame(results)

# ============================================================================
# Comparison plots
# ============================================================================

def plot_current_evolution(sample_dfs, output_dir):
    for voltage in TARGET_VOLTAGES:
        fig, ax = plt.subplots(figsize=(12, 8))
        for sample_id in SAMPLES:
            df = sample_dfs.get(sample_id)
            if df is None:
                continue
            d = df[['Ecell_V', 'I_mA', 'Time']].dropna()
            interp = interpolate_at_x(d, 'Ecell_V', 'I_mA', voltage)
            ax.plot(interp['Time'], interp['I_mA'], marker='o', markersize=5,
                    linestyle='-', color=SAMPLE_COLORS[sample_id], label=label_of(sample_id))
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Current / mA', fontsize=14)
        ax.set_title(f'Current Time Evolution at {voltage} V - All AEM Samples (474, 486, 491, 499, 505, 516)', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        path = output_dir / f"current_evolution_at_{voltage}V.{PLOT_FORMAT}"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path.name}")

def plot_r1_evolution(sample_dfs, output_dir):
    for current in TARGET_CURRENTS_R1:
        fig, ax = plt.subplots(figsize=(12, 8))
        for sample_id in SAMPLES:
            df = sample_dfs.get(sample_id)
            if df is None:
                continue
            d = df[['Time', 'I_mA', 'R1']].copy()
            d = d[np.isfinite(d['R1']) & (d['R1'] > 0)]
            interp = interpolate_at_x(d, 'I_mA', 'R1', current)
            valid = interp.dropna()
            if len(valid) > 0:
                ax.plot(valid['Time'], valid['R1'], 'o-', color=SAMPLE_COLORS[sample_id],
                        markersize=6, linewidth=2, label=label_of(sample_id))
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('R1 (Ohm)', fontsize=12)
        ax.set_title(f'R1 Time Evolution at {current} mA - All AEM Samples (474, 486, 491, 499, 505, 516)', fontsize=13)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=9)
        path = output_dir / f"r1_evolution_at_{current}mA.{PLOT_FORMAT}"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path.name}")

def plot_rtotal_evolution(sample_dfs, output_dir):
    for current in TARGET_CURRENTS_RTOTAL:
        fig, ax = plt.subplots(figsize=(12, 8))
        for sample_id in SAMPLES:
            df = sample_dfs.get(sample_id)
            if df is None:
                continue
            d = df[['Time', 'I_mA', 'R0', 'R1', 'R2', 'R3']].copy()
            if 'R4' in df.columns:
                d['R4'] = pd.to_numeric(df['R4'], errors='coerce').fillna(0.0)
            else:
                d['R4'] = 0.0
            d['R_total'] = d[['R0', 'R1', 'R2', 'R3', 'R4']].sum(axis=1)
            d = d[np.isfinite(d['R_total']) & (d['R_total'] > 0)][['Time', 'I_mA', 'R_total']]
            interp = interpolate_at_x(d, 'I_mA', 'R_total', current)
            valid = interp.dropna()
            if len(valid) > 0:
                ax.plot(valid['Time'], valid['R_total'], 'o-', color=SAMPLE_COLORS[sample_id],
                        markersize=6, linewidth=2, label=label_of(sample_id))
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('R_total = R0+R1+R2+R3(+R4) (Ohm)', fontsize=12)
        ax.set_title(f'Total Resistance Time Evolution at {current} mA - All AEM Samples (474, 486, 491, 499, 505, 516)', fontsize=13)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=9)
        path = output_dir / f"rtotal_evolution_at_{current}mA.{PLOT_FORMAT}"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path.name}")

def plot_tafel_comparison(output_dir):
    fig, ax = plt.subplots(figsize=(12, 8))
    for sample_id in SAMPLES:
        df = load_summary(sample_id, f'tafel_700ma_{sample_id}', f'{sample_id}_tafel_summary.csv')
        if df is None:
            print(f"Warning: tafel summary for {sample_id} not found")
            continue
        ax.plot(df['time'], df['tafel_slope'], 'o-', color=SAMPLE_COLORS[sample_id],
                markersize=7, linewidth=2, label=label_of(sample_id))
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Tafel Slope (mV/dec)', fontsize=12)
    ax.set_title('Tafel Slope Time Evolution - All AEM Samples (474, 486, 491, 499, 505, 516)', fontsize=13)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    for name in ['tafel_slope_time_evolution_comparison.png', 'tafel_slopes_time_evolution_comparison.png']:
        path = output_dir / name
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Saved: {path.name}")
    plt.close(fig)

def plot_q_evolution(param_name, folder_prefix, output_dir, filename_suffix=None):
    """Plot intercept q vs Time from a linear-fit summary for all samples."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for sample_id in SAMPLES:
        df = load_summary(sample_id, f'{folder_prefix}_linear_{sample_id}',
                          f'{sample_id}_{param_name}_fit_summary.csv')
        if df is None:
            print(f"Warning: {param_name} summary for {sample_id} not found")
            continue
        ax.plot(df['time'], df['q'], marker=SAMPLE_MARKERS[sample_id],
                color=SAMPLE_COLORS[sample_id], linewidth=2, markersize=7,
                label=label_of(sample_id))
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time', fontsize=12)
    unit = 'F' if param_name.startswith('C') else 'Ohm'
    ax.set_ylabel(f'Intercept q ({unit})', fontsize=12)
    ax.set_title(f'Time Evolution of Intercept q from {param_name} = k*I + q - All AEM Samples', fontsize=13)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(fontsize=9)
    suffix = filename_suffix if filename_suffix else f"{param_name.lower()}_time_evolution_comparison"
    path = output_dir / f"{suffix}.{PLOT_FORMAT}"
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path.name}")

def plot_q_evolution_combined(params, folder_prefixes, output_dir, filename):
    """Combined multi-row q evolution figure (one row per parameter)."""
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 5 * len(params)), sharex=True)
    if len(params) == 1:
        axes = [axes]
    for ax, param_name, prefix in zip(axes, params, folder_prefixes):
        for sample_id in SAMPLES:
            df = load_summary(sample_id, f'{prefix}_linear_{sample_id}',
                              f'{sample_id}_{param_name}_fit_summary.csv')
            if df is None:
                continue
            ax.plot(df['time'], df['q'], marker=SAMPLE_MARKERS[sample_id],
                    color=SAMPLE_COLORS[sample_id], linewidth=2, markersize=7,
                    label=label_of(sample_id))
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        unit = 'F' if param_name.startswith('C') else 'Ohm'
        ax.set_ylabel(f'q ({unit})', fontsize=11)
        ax.set_title(f'{param_name} = k*I + q', fontsize=11)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path.name}")

def plot_r0_r3_tradeoff(sample_dfs, output_dir):
    rows = []
    frames = []
    for sample_id in SAMPLES:
        df = sample_dfs.get(sample_id)
        if df is None:
            continue
        d = df[['Time', 'Ecell_V', 'I_mA', 'R0', 'R3', 'a3']].copy()
        for c in ['Time', 'Ecell_V', 'I_mA', 'R0', 'R3', 'a3']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna(subset=['R0', 'R3'])
        d = d[np.isfinite(d['R0']) & np.isfinite(d['R3'])]
        d = d[(d['R0'] > 0) & (d['R3'] > 0)]
        d['R0_plus_R3'] = d['R0'] + d['R3']
        d['sample'] = sample_id
        frames.append(d)
        rho, p = stats.spearmanr(d['R0'], d['R3'])
        rows.append({
            'sample': sample_id,
            'name': SAMPLE_NAMES[sample_id],
            'n_spectra': len(d),
            'R0_median_Ohm': d['R0'].median(),
            'R3_median_Ohm': d['R3'].median(),
            'R0_plus_R3_median_Ohm': d['R0_plus_R3'].median(),
            'R0_CV_percent': d['R0'].std() / d['R0'].median() * 100,
            'R3_CV_percent': d['R3'].std() / d['R3'].median() * 100,
            'R0_plus_R3_CV_percent': d['R0_plus_R3'].std() / d['R0_plus_R3'].median() * 100,
            'spearman_R0_vs_R3': rho,
            'spearman_p_value': p,
            'a3_median': d['a3'].median(),
        })
    alld = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / 'r0_r3_tradeoff_summary.csv', index=False)
    print(summary.to_string(index=False))
    print(f"Saved: r0_r3_tradeoff_summary.csv")

    n = len(SAMPLES)
    fig = plt.figure(figsize=(4.2 * n, 13))
    gs = fig.add_gridspec(3, n, height_ratios=[1.15, 1, 0.85], hspace=0.45,
                          wspace=0.30, top=0.90, bottom=0.06, left=0.06, right=0.97)
    fig.suptitle('R0/R3 Trade-off - All AEM Samples (474, 486, 491, 499, 505, 516)',
                 fontsize=17, fontweight='bold', y=0.965)

    axA = fig.add_subplot(gs[0, :])
    for i, sv in enumerate([0.015, 0.02, 0.03, 0.04]):
        xs = np.linspace(0.005, 0.05, 200)
        axA.plot(xs, sv - xs, color='grey', ls=':', lw=1.0, alpha=0.7,
                 label='iso-sum $R_0+R_3$' if i == 0 else None)
        axA.annotate(f'{sv:.3f}', xy=(0.0055, sv - 0.0055), fontsize=8, color='grey')
    for sample_id in SAMPLES:
        ds = alld[alld['sample'] == sample_id]
        rho = summary.loc[summary['sample'] == sample_id, 'spearman_R0_vs_R3'].iloc[0]
        label = f'{label_of(sample_id)}, $\\rho_s$={rho:+.2f}'
        axA.scatter(ds['R0'], ds['R3'], s=14, alpha=0.3,
                    color=SAMPLE_COLORS[sample_id], label=label)
        m0 = ds['R0'].median()
        m3 = ds['R3'].median()
        axA.scatter([m0], [m3], marker='X', s=200, color=SAMPLE_COLORS[sample_id],
                    edgecolors='black', linewidths=1.2, zorder=5)
        axA.annotate(f'median\n({m0:.4f}, {m3:.4f})\nsum={m0 + m3:.4f}',
                     xy=(m0, m3), xytext=(10, -34), textcoords='offset points',
                     fontsize=8, color=SAMPLE_COLORS[sample_id],
                     bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.75))
    axA.set_xscale('log')
    axA.set_yscale('log')
    axA.set_xlim(0.005, 0.06)
    axA.set_ylim(0.001, 0.2)
    axA.set_xlabel('$R_0$ ($\\Omega$)', fontsize=12)
    axA.set_ylabel('$R_3$ ($\\Omega$)', fontsize=12)
    axA.set_title('(A) $R_0$ vs $R_3$ for every fitted spectrum, with iso-sum diagonals.', fontsize=11)
    axA.legend(fontsize=8.5, loc='upper right', framealpha=0.9)

    bins = np.array([10, 20, 50, 100, 200, 400, 700, 1000, 1500, 2500, 4000], dtype=float)
    for i, sample_id in enumerate(SAMPLES):
        ax = fig.add_subplot(gs[1, i])
        ds = alld[alld['sample'] == sample_id].copy()
        ds['Ibin'] = pd.cut(ds['I_mA'], bins, labels=False)
        g = ds.groupby('Ibin')[['R0', 'R3', 'R0_plus_R3']].median()
        gcount = ds.groupby('Ibin').size()
        g = g[gcount >= 3]
        xc = (bins[:-1] + bins[1:]) / 2
        xc = xc[np.array(g.index, dtype=int)]
        ax.plot(xc, g['R0'], 'o-', color='grey', label='$R_0$')
        ax.plot(xc, g['R3'], 's--', color=SAMPLE_COLORS[sample_id], label='$R_3$')
        ax.plot(xc, g['R0_plus_R3'], '^-', color='black', label='$R_0+R_3$')
        ax.set_xscale('log')
        ax.set_xlabel('Current $I$ (mA)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Resistance ($\\Omega$)', fontsize=11)
        ax.set_title(f'(B{i + 1}) {label_of(sample_id)}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for i, sample_id in enumerate(SAMPLES):
        ax = fig.add_subplot(gs[2, i])
        ds = alld[alld['sample'] == sample_id]
        data = [ds['R0'] / ds['R0'].median(),
                ds['R3'] / ds['R3'].median(),
                ds['R0_plus_R3'] / ds['R0_plus_R3'].median()]
        bp = ax.boxplot(data, tick_labels=['$R_0$', '$R_3$', '$R_0+R_3$'],
                        patch_artist=True, showfliers=False, widths=0.55)
        for patch, color in zip(bp['boxes'], ['grey', SAMPLE_COLORS[sample_id], 'black']):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.axhline(1.0, color='red', ls=':', lw=1.5)
        ax.set_ylabel('Normalized to median', fontsize=10)
        cv = summary[summary['sample'] == sample_id].iloc[0]
        ax.set_title(f'(C{i + 1}) {sample_id}: CV $R_0$={cv["R0_CV_percent"]:.0f}%, '
                     f'$R_3$={cv["R3_CV_percent"]:.0f}%, sum={cv["R0_plus_R3_CV_percent"]:.0f}%',
                     fontsize=9.5)
        ax.grid(alpha=0.3, axis='y')

    fig.savefig(output_dir / 'r0_r3_tradeoff_analysis.png', dpi=150)
    fig.savefig(output_dir / 'r0_r3_tradeoff_analysis.svg')
    print("Saved: r0_r3_tradeoff_analysis.png/.svg")

# ============================================================================
# Main
# ============================================================================

def main():
    os.chdir(BASE_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    sample_dfs = {}
    for sample_id in SAMPLES:
        df = load_fit_export(sample_id)
        if df is not None:
            sample_dfs[sample_id] = df

    plot_current_evolution(sample_dfs, OUTPUT_DIR)
    plot_r1_evolution(sample_dfs, OUTPUT_DIR)
    plot_rtotal_evolution(sample_dfs, OUTPUT_DIR)
    plot_tafel_comparison(OUTPUT_DIR)

    # R0/R2/R3 intercept q (legacy names r2/r3 + ohmic for R0)
    plot_q_evolution('R0', 'r0', OUTPUT_DIR, 'r0_time_evolution_comparison')
    plot_q_evolution('R0', 'r0', OUTPUT_DIR, 'ohmic_resistance_time_evolution_comparison')
    plot_q_evolution('R2', 'R2', OUTPUT_DIR)
    plot_q_evolution('R3', 'R3', OUTPUT_DIR)
    plot_q_evolution_combined(['R0', 'R2', 'R3'], ['r0', 'R2', 'R3'], OUTPUT_DIR,
                              'all_resistances_q_time_evolution_comparison.png')

    # C1/C2/C3 intercept q (legacy name c1_time_evolution_comparison for C1)
    plot_q_evolution('C1', 'c1', OUTPUT_DIR, 'c1_q_time_evolution_comparison')
    plot_q_evolution('C1', 'c1', OUTPUT_DIR, 'c1_time_evolution_comparison')
    plot_q_evolution('C2', 'C2', OUTPUT_DIR, 'c2_q_time_evolution_comparison')
    plot_q_evolution('C3', 'C3', OUTPUT_DIR, 'c3_q_time_evolution_comparison')
    plot_q_evolution_combined(['C1', 'C2', 'C3'], ['c1', 'C2', 'C3'], OUTPUT_DIR,
                              'c1_c2_c3_q_time_evolution_comparison.png')

    plot_r0_r3_tradeoff(sample_dfs, OUTPUT_DIR)

    print("\nAll all-sample comparison graphs generated successfully!")


if __name__ == '__main__':
    main()
