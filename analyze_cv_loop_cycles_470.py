"""Compare loop-boundary CV cycles (1, 7, 13, ...) for sample 470."""

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
    "1.6": (ROOT / "inter_50_sccm_H2_CVs_1,6V_05_CV_C02.mpr", ROOT / "inter_50_sccm_H2_CVs_1,6V_11_CV_C02.mpr"),
    "1.8": (ROOT / "inter_50_sccm_H2_CVs_1,8V_05_CV_C02.mpr", ROOT / "inter_50_sccm_H2_CVs_1,8V_11_CV_C02.mpr"),
    "2.0": (ROOT / "inter_50_sccm_H2_CVs_2,0V_2_05_CV_C02.mpr", ROOT / "inter_50_sccm_H2_CVs_2,0V_2_11_CV_C02.mpr"),
}
VLOW, VHIGH = 1.4, 1.5
OUT = Path("results")


def load(path):
    df = we.read_file_safe(path, error_on_unknown_column=False, on_error="raise")
    if df is None or df.empty:
        raise RuntimeError(f"No data in {path}")
    df = df.copy()
    df["cell_voltage/V"] = df["Ewe/V"] - df["Ece/V"]
    df["current/mA"] = df["<I>/mA"]
    return df


def branches(df, cycle):
    g = df[df["cycle number"] == cycle].copy()
    if g.empty:
        return None
    v = g["cell_voltage/V"].to_numpy(float)
    split = int(np.nanargmax(v))
    return {"forward": g.iloc[: split + 1], "reverse": g.iloc[split:]}


def interpolate(group, grid):
    v = group["cell_voltage/V"].to_numpy(float)
    i = group["current/mA"].to_numpy(float)
    order = np.argsort(v)
    v, i = v[order], i[order]
    v, idx = np.unique(v, return_index=True)
    i = i[idx]
    return np.interp(grid, v, i)


def compare_cycle(df05, df11, cycle):
    b05, b11 = branches(df05, cycle), branches(df11, cycle)
    if b05 is None or b11 is None:
        return None
    vmax = min(b05["forward"]["cell_voltage/V"].max(), b11["forward"]["cell_voltage/V"].max())
    grid = np.linspace(VLOW, min(VHIGH, vmax), 301)
    result = {"cycle": cycle}
    for branch in ("forward", "reverse"):
        i05 = interpolate(b05[branch], grid)
        i11 = interpolate(b11[branch], grid)
        diff = i05 - i11
        result[branch] = {
            "mean_05_minus_11_mA": float(diff.mean()),
            "mean_abs_difference_mA": float(np.abs(diff).mean()),
            "max_abs_difference_mA": float(np.abs(diff).max()),
            "fraction_05_higher": float(np.mean(diff > 0)),
            "at_0.5V_mA": float(diff[0]),
            "at_0.65V_mA": float(diff[len(diff) // 2]),
            "at_0.8V_mA": float(diff[-1]),
        }
    return result


def main():
    OUT.mkdir(exist_ok=True)
    all_data = {}
    for voltage, (p05, p11) in FILES.items():
        df05, df11 = load(p05), load(p11)
        cycles = sorted(set(df05["cycle number"].unique()) & set(df11["cycle number"].unique()))
        selected = [int(c) for c in cycles if int(c) >= 1 and (int(c) - 1) % 6 == 5]
        all_data[voltage] = (df05, df11, selected)
        print(f"\nVOLTAGE {voltage} V; shared cycles={cycles[0]:g}-{cycles[-1]:g}; selected={selected}")
        for cycle in selected:
            print(f"cycle {cycle}: {compare_cycle(df05, df11, cycle)}")

    # Focused overlays: selected cycles, both branches, in the requested voltage interval.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, 6))
    for ax, (voltage, (df05, df11, selected)) in zip(axes, all_data.items()):
        for idx, cycle in enumerate(selected):
            for tech, df, ls in (("05", df05, "-"), ("11", df11, "--")):
                g = df[df["cycle number"] == cycle]
                g = g[(g["cell_voltage/V"] >= VLOW) & (g["cell_voltage/V"] <= VHIGH)]
                ax.plot(g["cell_voltage/V"], g["current/mA"], color=colors[idx], ls=ls, lw=1.0,
                        label=f"{tech}, cycle {cycle}" if voltage == "1.6" else None)
        ax.set_title(f"{voltage} V nominal")
        ax.set_xlabel("Cell voltage (V)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Current (mA)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Sample 470: loop-boundary CV cycles, 1.4–1.5 V")
    fig.tight_layout()
    fig.savefig(OUT / "sample470_loop_cycles_05_vs_11_1p4_1p5V_cycle6.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
