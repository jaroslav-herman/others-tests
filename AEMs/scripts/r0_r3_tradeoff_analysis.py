"""
R0/R3 trade-off analysis for the AEM Substrate Series (474, 499, 505).

Investigates the fit degeneracy between the ohmic intercept R0 and the
high-frequency arc resistance R3 (strongly depressed HF arcs, a3 ~ 0.5-0.65):
individual components can trade off during fitting while their sum remains
approximately conserved.

Outputs (saved to 'AEM Substrate Series/'):
  - r0_r3_tradeoff_analysis.png   (3-row figure)
  - r0_r3_tradeoff_summary.csv    (per-sample statistics)

Usage:
    python r0_r3_tradeoff_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# AEM Substrate Series samples (474, 499, 505 only - NOT 498)
SAMPLES = [474, 499, 505]

# Sample display names from Google Sheets
SAMPLE_NAMES = {
    474: 'VI_Ni_on_Ni_PTL',
    499: 'VI_Ni_on_Ni_etched',
    505: 'VI_Ni_on_Ni_plain',
}

# Rainbow colors for samples (purple to red)
SAMPLE_COLORS = {
    474: 'purple',
    499: 'blue',
    505: 'red',
}


def load_sample_data(sample_id):
    """Load and preprocess data for a single sample."""
    csv_path = Path(f'sample_{sample_id}/{sample_id}_fit_export.csv')
    if not csv_path.exists():
        print(f'Warning: File not found: {csv_path}')
        return None

    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)

    cols = ['Time', 'Ecell_V', 'I_mA', 'R0', 'R3', 'a3']
    for col in cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in sample {sample_id}")
            return None

    df = df[cols].copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with NaN or non-finite values, keep positive resistances
    df = df.dropna(subset=['R0', 'R3'])
    df = df[np.isfinite(df['R0']) & np.isfinite(df['R3'])]
    df = df[(df['R0'] > 0) & (df['R3'] > 0)]
    df['R0_plus_R3'] = df['R0'] + df['R3']

    return df


def main():
    base_dir = Path(__file__).resolve().parent.parent
    os.chdir(base_dir)

    # ------------------------------------------------------------------ load
    sample_dfs = {}
    for sample_id in SAMPLES:
        df = load_sample_data(sample_id)
        if df is not None:
            df['sample'] = sample_id
            sample_dfs[sample_id] = df

    alld = pd.concat(sample_dfs.values(), ignore_index=True)

    # ------------------------------------------------------ per-sample stats
    rows = []
    for sample_id in SAMPLES:
        ds = alld[alld['sample'] == sample_id]
        rho, p = stats.spearmanr(ds['R0'], ds['R3'])
        rows.append({
            'sample': sample_id,
            'name': SAMPLE_NAMES[sample_id],
            'n_spectra': len(ds),
            'R0_median_Ohm': ds['R0'].median(),
            'R3_median_Ohm': ds['R3'].median(),
            'R0_plus_R3_median_Ohm': ds['R0_plus_R3'].median(),
            'R0_CV_percent': ds['R0'].std() / ds['R0'].median() * 100,
            'R3_CV_percent': ds['R3'].std() / ds['R3'].median() * 100,
            'R0_plus_R3_CV_percent': ds['R0_plus_R3'].std() / ds['R0_plus_R3'].median() * 100,
            'spearman_R0_vs_R3': rho,
            'spearman_p_value': p,
            'a3_median': ds['a3'].median(),
        })
    summary = pd.DataFrame(rows)

    output_dir = base_dir / 'AEM Substrate Series'
    output_dir.mkdir(exist_ok=True)
    summary.to_csv(output_dir / 'r0_r3_tradeoff_summary.csv', index=False)
    print(summary.to_string(index=False))

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 0.85], hspace=0.45,
                          wspace=0.30, top=0.90, bottom=0.06, left=0.06, right=0.97)
    fig.suptitle('R0/R3 Trade-off - AEM Substrate Series (474, 499, 505)',
                 fontsize=17, fontweight='bold', y=0.965)

    # Panel A: R3 vs R0 scatter, iso-sum lines, per-sample medians
    axA = fig.add_subplot(gs[0, :])
    sum_vals = [0.028, 0.030, 0.032, 0.034]
    for i, sv in enumerate(sum_vals):
        xs = np.linspace(0.008, 0.04, 200)
        axA.plot(xs, sv - xs, color='grey', ls=':', lw=1.0, alpha=0.7,
                 label='iso-sum $R_0+R_3$' if i == 0 else None)
        axA.annotate(f'{sv:.3f}', xy=(0.0085, sv - 0.0085), fontsize=8, color='grey')
    for sample_id in SAMPLES:
        ds = alld[alld['sample'] == sample_id]
        rho = summary.loc[summary['sample'] == sample_id, 'spearman_R0_vs_R3'].iloc[0]
        label = f'{sample_id} ({SAMPLE_NAMES[sample_id]}), ' \
                f'$\\rho_s$={rho:+.2f}'
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
    axA.set_xlim(0.008, 0.05)
    axA.set_ylim(0.001, 0.1)
    axA.set_xlabel('$R_0$ ($\\Omega$)', fontsize=12)
    axA.set_ylabel('$R_3$ ($\\Omega$)', fontsize=12)
    axA.set_title('(A) $R_0$ vs $R_3$ for every fitted spectrum, with iso-sum diagonals. '
                 'Sample medians (X) lie close to a common constant-sum trajectory: '
                 'lower $R_0$ is compensated by higher $R_3$.', fontsize=11)
    axA.legend(fontsize=9.5, loc='upper right', framealpha=0.9)

    # Panels B1-B3: R0, R3, R0+R3 vs current per sample
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
        ax.plot(xc, g['R0'], 'o-', color='grey', label='$R_0$ (ohmic intercept)')
        ax.plot(xc, g['R3'], 's--', color=SAMPLE_COLORS[sample_id], label='$R_3$ (HF arc)')
        ax.plot(xc, g['R0_plus_R3'], '^-', color='black', label='$R_0+R_3$ (sum)')
        ax.set_xscale('log')
        ax.set_xlabel('Current $I$ (mA)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Resistance ($\\Omega$)', fontsize=11)
        ax.set_title(f'(B{i + 1}) {sample_id} ({SAMPLE_NAMES[sample_id]})', fontsize=11)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3)

    # Panels C1-C3: normalized spread of R0, R3, R0+R3
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
                     fontsize=10.5)
        ax.grid(alpha=0.3, axis='y')

    fig.savefig(output_dir / 'r0_r3_tradeoff_analysis.png', dpi=150)
    fig.savefig(output_dir / 'r0_r3_tradeoff_analysis.svg')
    print(f"\nSaved: {output_dir / 'r0_r3_tradeoff_analysis.png'}")
    print(f"Saved: {output_dir / 'r0_r3_tradeoff_analysis.svg'}")
    print(f"Saved: {output_dir / 'r0_r3_tradeoff_summary.csv'}")


if __name__ == '__main__':
    main()
