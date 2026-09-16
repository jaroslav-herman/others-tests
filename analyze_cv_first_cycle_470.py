"""Compare the first CV cycle for techniques 05 and 11 at 1.6/1.8/2.0 V."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wepy.basics as we

matplotlib.rcParams["text.usetex"] = False

ROOT = Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\interruptiony\470 CVs H2")
FILES = {
    "1.6": {
        "05": ROOT / "inter_50_sccm_H2_CVs_1,6V_05_CV_C02.mpr",
        "11": ROOT / "inter_50_sccm_H2_CVs_1,6V_11_CV_C02.mpr",
    },
    "1.8": {
        "05": ROOT / "inter_50_sccm_H2_CVs_1,8V_05_CV_C02.mpr",
        "11": ROOT / "inter_50_sccm_H2_CVs_1,8V_11_CV_C02.mpr",
    },
    "2.0": {
        "05": ROOT / "inter_50_sccm_H2_CVs_2,0V_2_05_CV_C02.mpr",
        "11": ROOT / "inter_50_sccm_H2_CVs_2,0V_2_11_CV_C02.mpr",
    },
}
OUT = Path("results")


def load(path):
    df = we.read_file_safe(path, error_on_unknown_column=False, on_error="raise")
    if df is None or df.empty:
        raise RuntimeError(f"No data in {path}")
    df = df.copy()
    df["cell_voltage/V"] = df["Ewe/V"] - df["Ece/V"]
    df["current/mA"] = df["<I>/mA"]
    return df


def first_cycle(df):
    return df[df["cycle number"] == sorted(df["cycle number"].unique())[0]].copy()


def endpoint_stats(df):
    g = first_cycle(df)
    imax = g["cell_voltage/V"].idxmax()
    row = g.loc[imax]
    vmax = float(row["cell_voltage/V"])
    # Test the complete top 50 mV, using voltage-matched interpolation on both traces.
    top = g[g["cell_voltage/V"] >= vmax - 0.05]
    return {
        "points": len(g),
        "vmax": vmax,
        "I_at_vmax": float(row["current/mA"]),
        "I_top50_min": float(top["current/mA"].min()),
        "I_top50_max": float(top["current/mA"].max()),
        "I_top50_mean": float(top["current/mA"].mean()),
        "I_min": float(g["current/mA"].min()),
        "I_max": float(g["current/mA"].max()),
    }


def voltage_matched_test(a, b):
    """Compare currents at common voltages on the forward scan up to the apex."""
    a, b = first_cycle(a), first_cycle(b)
    ia, ib = int(a["cell_voltage/V"].idxmax()), int(b["cell_voltage/V"].idxmax())
    a, b = a.loc[:ia], b.loc[:ib]
    vmax = min(a["cell_voltage/V"].max(), b["cell_voltage/V"].max())
    grid = np.linspace(max(1.0, min(a["cell_voltage/V"].min(), b["cell_voltage/V"].min())), vmax, 500)
    def interp(g):
        v = g["cell_voltage/V"].to_numpy(float)
        i = g["current/mA"].to_numpy(float)
        order = np.argsort(v)
        v, i = v[order], i[order]
        v, idx = np.unique(v, return_index=True)
        return np.interp(grid, v, i)
    diff = interp(a) - interp(b)
    top = grid >= vmax - 0.05
    return {
        "grid_vmax": float(vmax),
        "top50_fraction_05_higher": float(np.mean(diff[top] > 0)),
        "top50_min_difference_mA": float(diff[top].min()),
        "top50_max_difference_mA": float(diff[top].max()),
        "top50_mean_difference_mA": float(diff[top].mean()),
    }


def main():
    OUT.mkdir(exist_ok=True)
    all_data = {}
    for voltage, paths in FILES.items():
        all_data[voltage] = {tech: load(path) for tech, path in paths.items()}
        print(f"\nVOLTAGE {voltage} V")
        for tech, df in all_data[voltage].items():
            print(f"{tech}: file={paths[tech].name}, rows={len(df)}, cycles={len(df['cycle number'].unique())}, first={endpoint_stats(df)}")
        print(f"05 minus 11, forward top 50 mV: {voltage_matched_test(all_data[voltage]['05'], all_data[voltage]['11'])}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (voltage, data) in zip(axes, all_data.items()):
        for tech, df in data.items():
            g = first_cycle(df)
            ax.plot(g["cell_voltage/V"], g["current/mA"], lw=1.2, label=tech)
        ax.set_title(f"{voltage} V nominal")
        ax.set_xlabel("Cell voltage (V)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Current (mA)")
    axes[-1].legend(title="Technique")
    fig.suptitle("Sample 470: first CV cycle after preceding technique")
    fig.tight_layout()

    fig.savefig(OUT / "sample470_first_cycle_05_vs_11_1p6_1p8_2p0.png", dpi=300)
    
    plt.show()


if __name__ == "__main__":
    main()
