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
    "60": (ROOT / "inter_50_sccm_H2_CVs_2,0V_60cycles_05_CV_C02.mpr", ROOT / "inter_50_sccm_H2_CVs_2,0V_60cycles_11_CV_C02.mpr"),
    "120": (ROOT / "inter_50_sccm_H2_CVs_2,0V_120cycles_05_CV_C02.mpr", ROOT / "inter_50_sccm_H2_CVs_2,0V_120cycles_11_CV_C02.mpr"),
    "240": (ROOT / "inter_50_sccm_H2_CVs_2,0V_2_05_CV_C02.mpr", ROOT / "inter_50_sccm_H2_CVs_2,0V_2_11_CV_C02.mpr"),
    "360": (ROOT / "inter_50_sccm_H2_CVs_2,0V_360cycles_05_CV_C02.mpr", ROOT / "inter_50_sccm_H2_CVs_2,0V_360cycles_11_CV_C02.mpr"),
}
VLOW, VHIGH = 0.5,0.8
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
        selected = [int(c) for c in cycles if int(c) >= 1 and (int(c) - 1) % 6 == 0]
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
                        label=f"{tech}, cycle {cycle}" if voltage == "60" else None)
        ax.set_title(f"{voltage} V nominal")
        ax.set_xlabel("Cell voltage (V)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Current (mA)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Sample 470: loop-boundary CV cycles, 1.4–1.5 V")
    fig.tight_layout()
    fig.savefig(OUT / "sample470_loop_cycles_05_vs_11_1p4_1p5V_different_lenght.png", dpi=300)
    plt.close(fig)

    # Additional comparison: keep each CV cycle number fixed and overlay the
    # files with different numbers of preceding EIS cycles. The two columns
    # separate techniques 05 and 11 so the pre-EIS-history effect is visible.
    common_cycles = sorted(set.intersection(*(set(item[2]) for item in all_data.values())))
    file_colors = dict(zip(all_data.keys(), we.get_colors(len(all_data))))
    nrows = len(common_cycles)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, max(3.2 * nrows, 4.0)), squeeze=False, sharex=True)
    for row, cycle in enumerate(common_cycles):
        for col, tech_index in enumerate((0, 1)):
            ax = axes[row, col]
            for file_label, (df05, df11, _selected) in all_data.items():
                df = (df05, df11)[tech_index]
                g = df[df["cycle number"] == cycle]
                g = g[(g["cell_voltage/V"] >= VLOW) & (g["cell_voltage/V"] <= VHIGH)]
                ax.plot(g["cell_voltage/V"], g["current/mA"], color=file_colors[file_label], lw=1.1,
                        label=f"{file_label} preceding EIS cycles")
            ax.set_title(f"Technique {('05', '11')[tech_index]}, CV cycle {cycle}")
            ax.grid(alpha=0.25)
            if row == nrows - 1:
                ax.set_xlabel("Cell voltage (V)")
            if col == 0:
                ax.set_ylabel("Current (mA)")
            if row == 0 and col == 1:
                ax.legend(fontsize=8)
    fig.suptitle("Sample 470: same CV cycle across files with different preceding EIS counts")
    fig.tight_layout()
    fig.savefig(OUT / "sample470_crossfile_cycle_overlays_05_11_0p5_0p8V.png", dpi=300)
    plt.close(fig)

    # Single combined figure: all loop-boundary cycles on the same axes for
    # each technique. Colors identify the CV cycle; line styles identify the
    # number of preceding EIS cycles.
    combined_cycles = [2, 7, 13, 19]
    cycle_colors = dict(zip(combined_cycles, we.get_colors(len(combined_cycles))))
    file_styles = {"60": "-", "120": "--", "240": ":", "360": "-."}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for ax, tech_index in zip(axes, (0, 1)):
        for file_label, (df05, df11, _selected) in all_data.items():
            df = (df05, df11)[tech_index]
            for cycle in combined_cycles:
                g = df[df["cycle number"] == cycle]
                if g.empty:
                    continue
                g = g[(g["cell_voltage/V"] >= VLOW) & (g["cell_voltage/V"] <= VHIGH)]
                ax.plot(g["cell_voltage/V"], g["current/mA"], color=cycle_colors[cycle],
                        ls=file_styles[file_label], lw=1.2)
        ax.set_title(f"Technique {('05', '11')[tech_index]}")
        ax.set_xlabel("Cell voltage (V)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Current (mA)")
    from matplotlib.lines import Line2D
    cycle_handles = [Line2D([0], [0], color=cycle_colors[c], lw=2, label=f"CV cycle {c}") for c in combined_cycles]
    file_handles = [Line2D([0], [0], color="black", ls=file_styles[f], lw=1.5, label=f"{f} preceding EIS cycles") for f in file_styles]
    axes[1].legend(handles=cycle_handles + file_handles, fontsize=8)
    fig.suptitle("Sample 470: cycles 1, 7, 13, and 19 overlapped")
    fig.tight_layout()
    fig.savefig(OUT / "sample470_cycles_1_7_13_19_one_graph_05_11.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
