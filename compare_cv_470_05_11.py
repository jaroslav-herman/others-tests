"""Compare the CVs measured as techniques 05 and 11 for sample 470."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wepy.basics as we
matplotlib.rcParams["text.usetex"] = False


DATA_DIR = Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\interruptiony\470 CVs H2")
FILES = {
    "05 (after previous sequence)": DATA_DIR / "inter_50_sccm_H2_CVs_1,6V_05_CV_C02.mpr",
    "11 (after previous sequence)": DATA_DIR / "inter_50_sccm_H2_CVs_1,6V_11_CV_C02.mpr",
}
OUT = Path("results")


def pick_column(df, *names):
    for name in names:
        if name in df.columns:
            return name
    return None


def load_cv(path):
    df = we.read_file_safe(path, error_on_unknown_column=False, on_error="raise")
    if df is None:
        raise RuntimeError(f"Could not read {path}")
    cycle_col = pick_column(df, "cycle number")
    current_col = pick_column(df, "<I>/mA", "I/mA")
    if cycle_col is None or current_col is None:
        raise RuntimeError(f"Missing cycle/current columns in {path}: {list(df.columns)}")
    if "Ewe/V" in df.columns and "Ece/V" in df.columns:
        df = df.copy()
        df["cell_voltage/V"] = df["Ewe/V"] - df["Ece/V"]
    else:
        voltage_col = pick_column(df, "control/V", "Ewe/V")
        if voltage_col is None:
            raise RuntimeError(f"Missing voltage columns in {path}: {list(df.columns)}")
        df = df.copy()
        df["cell_voltage/V"] = df[voltage_col]
    df["current/mA"] = df[current_col]
    return df


def cycle_metrics(group):
    v = group["cell_voltage/V"].to_numpy(float)
    i = group["current/mA"].to_numpy(float)
    valid = np.isfinite(v) & np.isfinite(i)
    v, i = v[valid], i[valid]
    order = np.argsort(v)
    vs, is_ = v[order], i[order]
    unique_v, idx = np.unique(vs, return_index=True)
    unique_i = is_[idx]
    return {
        "points": len(v),
        "v_min": float(np.min(v)),
        "v_max": float(np.max(v)),
        "i_min": float(np.min(i)),
        "i_max": float(np.max(i)),
        "i_at_0_8": float(np.interp(0.8, unique_v, unique_i))
        if unique_v[0] <= 0.8 <= unique_v[-1]
        else np.nan,
        "i_at_1_0": float(np.interp(1.0, unique_v, unique_i))
        if unique_v[0] <= 1.0 <= unique_v[-1]
        else np.nan,
        "i_at_1_2": float(np.interp(1.2, unique_v, unique_i))
        if unique_v[0] <= 1.2 <= unique_v[-1]
        else np.nan,
    }


def branch_metrics(group):
    """Interpolate forward/reverse currents without collapsing CV hysteresis."""
    v = group["cell_voltage/V"].to_numpy(float)
    i = group["current/mA"].to_numpy(float)
    split = int(np.nanargmax(v))
    branches = {"forward": (v[: split + 1], i[: split + 1]), "reverse": (v[split:], i[split:])}
    out = {}
    for direction, (vb, ib) in branches.items():
        order = np.argsort(vb)
        vs, is_ = vb[order], ib[order]
        vs, unique_idx = np.unique(vs, return_index=True)
        is_ = is_[unique_idx]
        for target in (1.0, 1.2, 1.4):
            out[f"{direction}_i@{target:.1f}V"] = float(np.interp(target, vs, is_))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    datasets = {label: load_cv(path) for label, path in FILES.items()}
    print("FILE SUMMARY")
    for label, df in datasets.items():
        cycles = list(pd.unique(df["cycle number"]))
        print(f"{label}: rows={len(df)}, n_cycles={len(cycles)}, cycles={cycles[0]:g}-{cycles[-1]:g}")
        print(f"  time range: {df['time/s'].min():.2f} to {df['time/s'].max():.2f} s" if "time/s" in df else "  time column unavailable")
        first_cycle = df[df["cycle number"] == cycles[0]]
        last_cycle = df[df["cycle number"] == cycles[-1]]
        print(f"  first cycle: {cycle_metrics(first_cycle)}")
        print(f"  final cycle: {cycle_metrics(last_cycle)}")
        print(f"  final branches: {branch_metrics(last_cycle)}")

    # Plot every cycle so changes caused by the preceding technique are visible.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = we.get_colors(max(len(df["cycle number"].unique()) for df in datasets.values()))
    for label, df in datasets.items():
        for n, (cycle, group) in enumerate(df.groupby("cycle number", sort=True)):
            alpha = 0.25 + 0.65 * (n + 1) / max(1, len(df["cycle number"].unique()))
            ax.plot(group["cell_voltage/V"], group["current/mA"], color=colors[n], alpha=alpha, lw=0.8,
                    label=label if n == len(df["cycle number"].unique()) - 1 else None)
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Sample 470: CV technique 05 vs 11")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "sample470_cv_05_vs_11_all_cycles.png", dpi=300)
    plt.close(fig)

    # Overlay the final cycle, which is usually the most stabilized CV.
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, df in datasets.items():
        cycle = sorted(pd.unique(df["cycle number"]))[-1]
        group = df[df["cycle number"] == cycle]
        ax.plot(group["cell_voltage/V"], group["current/mA"], lw=1.5, label=f"{label}, cycle {cycle}")
    ax.set_xlabel("Cell voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.set_title("Sample 470: final CV cycle comparison")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sample470_cv_05_vs_11_final_cycle.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
