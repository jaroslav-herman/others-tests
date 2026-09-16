"""Plot the 40 C CV files for techniques 05 and 11."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wepy.basics as we

matplotlib.rcParams["text.usetex"] = False

ROOT = Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\interruptiony\470 CVs H2")
FILES = {
    "05": ROOT / "inter_50_sccm_H2_CVs_2,0V_40C_05_CV_C02.mpr",
    "11": ROOT / "inter_50_sccm_H2_CVs_2,0V_40C_11_CV_C02.mpr",
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


def main():
    OUT.mkdir(exist_ok=True)
    data = {tech: load(path) for tech, path in FILES.items()}
    for tech, df in data.items():
        cycles = sorted(df["cycle number"].unique())
        print(f"{tech}: rows={len(df)}, cycles={cycles[0]:g}-{cycles[-1]:g}, n_cycles={len(cycles)}")

    # Full CV range, all cycles.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = we.get_colors(max(len(df["cycle number"].unique()) for df in data.values()))
    for tech, df in data.items():
        cycles = sorted(df["cycle number"].unique())
        for idx, cycle in enumerate(cycles):
            g = df[df["cycle number"] == cycle]
            ax.plot(g["cell_voltage/V"], g["current/mA"], color=colors[idx], lw=0.9,
                    alpha=0.35 + 0.65 * (idx + 1) / len(cycles),
                    label=f"Technique {tech}" if idx == len(cycles) - 1 else None)
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Sample 470: 40 °C CVs, all cycles")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sample470_40C_CV_05_vs_11_all_cycles.png", dpi=300)
    plt.close(fig)

    # Loop-boundary cycles in the 0.5–0.8 V region.
    selected = [1, 7, 13, 19]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cycle_colors = dict(zip(selected, we.get_colors(len(selected))))
    for tech, df in data.items():
        for cycle in selected:
            g = df[df["cycle number"] == cycle]
            if g.empty:
                continue
            g = g[(g["cell_voltage/V"] >= 0.5) & (g["cell_voltage/V"] <= 0.8)]
            ax.plot(g["cell_voltage/V"], g["current/mA"], color=cycle_colors[cycle],
                    ls="-" if tech == "05" else "--", lw=1.2,
                    label=f"{tech}, cycle {cycle}")
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Sample 470: 40 °C loop-boundary CV cycles, 0.5–0.8 V")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "sample470_40C_CV_05_vs_11_cycles_1_7_13_19_0p5_0p8V.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
