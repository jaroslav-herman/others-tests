"""Compare sample 470 CVs measured at 40 C and 80 C."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wepy.basics as we

matplotlib.rcParams["text.usetex"] = False

ROOT = Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\interruptiony\470 CVs H2")
FILES = {
    "40 °C": {
        "05": ROOT / "inter_50_sccm_H2_CVs_2,0V_40C_05_CV_C02.mpr",
        "11": ROOT / "inter_50_sccm_H2_CVs_2,0V_40C_11_CV_C02.mpr",
    },
    "80 °C": {
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


def main():
    OUT.mkdir(exist_ok=True)
    data = {temp: {tech: load(path) for tech, path in paths.items()} for temp, paths in FILES.items()}
    for temp, techniques in data.items():
        for tech, df in techniques.items():
            cycles = sorted(df["cycle number"].unique())
            print(f"{temp}, technique {tech}: rows={len(df)}, cycles={cycles[0]:g}-{cycles[-1]:g}, n_cycles={len(cycles)}")

    temp_colors = dict(zip(data, we.get_colors(len(data))))

    # Full CV range, all cycles, with one panel per technique.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, tech in zip(axes, ("05", "11")):
        for temp, techniques in data.items():
            df = techniques[tech]
            cycles = sorted(df["cycle number"].unique())
            for idx, cycle in enumerate(cycles):
                g = df[df["cycle number"] == cycle]
                alpha = min(1.0, 0.2 + 0.8 * (idx + 1) / len(cycles))
                ax.plot(g["cell_voltage/V"], g["current/mA"], color=temp_colors[temp], alpha=alpha, lw=0.8,
                        label=temp if idx == len(cycles) - 1 else None)
        ax.set_title(f"Technique {tech}")
        ax.set_xlabel("Cell voltage (V)")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Current (mA)")
    fig.suptitle("Sample 470: CV comparison at 40 °C and 80 °C, all cycles")
    fig.tight_layout()
    fig.savefig(OUT / "sample470_CV_40C_vs_80C_all_cycles.png", dpi=300)
    plt.close(fig)

    # Loop-boundary cycles in the 0.5–0.8 V interval.
    selected = [1, 7, 13, 19]
    cycle_colors = dict(zip(selected, we.get_colors(len(selected))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, tech in zip(axes, ("05", "11")):
        for temp, techniques in data.items():
            df = techniques[tech]
            for cycle in selected:
                g = df[df["cycle number"] == cycle]
                if g.empty:
                    continue
                g = g[(g["cell_voltage/V"] >= 0.5) & (g["cell_voltage/V"] <= 0.8)]
                ax.plot(g["cell_voltage/V"], g["current/mA"], color=cycle_colors[cycle],
                        ls="-" if temp == "40 °C" else "--", lw=1.1,
                        label=f"{temp}, cycle {cycle}")
        ax.set_title(f"Technique {tech}")
        ax.set_xlabel("Cell voltage (V)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("Current (mA)")
    fig.suptitle("Sample 470: 40 °C vs 80 °C, cycles 1/7/13/19, 0.5–0.8 V")
    fig.tight_layout()
    fig.savefig(OUT / "sample470_CV_40C_vs_80C_cycles_1_7_13_19_0p5_0p8V.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
