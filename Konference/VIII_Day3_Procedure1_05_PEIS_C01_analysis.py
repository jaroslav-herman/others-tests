# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wepy.plots import apply_plot_style
from glob import glob
import os
import wepy.basics as we
import wepy.eis as weis

# %%
# DATA_FILE = Path(__file__).with_name("VIII_Day3_Procedure1_05_PEIS_C01_analysis.csv")

# Complete export: one spectrum per row.
fit_data = pd.read_csv(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\Konference\Munich operando\VIII_Day3_Procedure1_05_PEIS_C01_analysis.csv"
)

# Metadata describing each fitted spectrum.
metadata_columns = [
    "source_file",
    "source_path",
    "cycle",
    "circuit",
    "Ecell_V",
    "I_mA",
    "time/s",
    "total_points",
    "active_points",
    "fmin_Hz",
    "fmax_Hz",
    "fmin_act_Hz",
    "fmax_act_Hz",
    "Spectrum",
    "Working electrode potential (V)",
    "Counter electrode potential (V)",
    "Ecell_V",
]
metadata = fit_data.loc[:, metadata_columns].copy()

# Custom metadata imported from the clipboard.
custom_metadata_columns = [
    "Spectrum",
    "Working electrode potential (V)",
    "Counter electrode potential (V)",
    "Ecell_V",
]
custom_metadata = (
    fit_data.loc[:, custom_metadata_columns].copy()
    if custom_metadata_columns
    else pd.DataFrame(index=fit_data.index)
)

# Fitted values and their percentage errors.
parameter_columns = [
    "R0",
    "R0_e",
    "L0",
    "L0_e",
    "R1",
    "R1_e",
    "Q1",
    "Q1_e",
    "a1",
    "a1_e",
    "R2",
    "R2_e",
    "Q2",
    "Q2_e",
    "a2",
    "a2_e",
]
fit_parameters = fit_data.loc[:, parameter_columns].copy()

# Convenient spectrum-indexed tables for analysis and plotting.
index_columns = ["source_path", "cycle"]
if "Spectrum" in fit_data.columns:
    index_columns.append("Spectrum")
indexed_fit_data = fit_data.set_index(index_columns).sort_index()
parameter_values = indexed_fit_data.loc[
    :, [column for column in parameter_columns if not column.endswith("_e")]
]
parameter_errors_percent = indexed_fit_data.loc[
    :, [column for column in parameter_columns if column.endswith("_e")]
]
derived_columns = ["C1", "tau1", "C2", "tau2"]
derived_values = indexed_fit_data.loc[:, derived_columns].copy()

# print(f"Loaded {len(fit_data)} fitted spectra from {DATA_FILE.name}")
print(fit_data.head())
# %%
fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)


ax.axvline(0, color="k", ls="--")
ax.axhline(0, color="k", ls="--")
x = fit_data[(fit_data["cycle"] > 35) & (fit_data["cycle"] < 55)]["I_mA"] / 4.84
y = 1 / (fit_data[(fit_data["cycle"] > 35) & (fit_data["cycle"] < 55)]["R1"] * 4.84)
ax.plot(x, y, "x--",lw = 2,markersize = 10)
k, q = np.polyfit(x[8:], y[8:], 1)
ax.plot(x, k * x + q, "r-", lw = 4)
ax.set_xlabel(r"Current density  ($\mathrm{A \cdot cm^{-2}}$)")
ax.set_ylabel(r"$1/R_\mathrm{cat}$ ($\mathrm{S \cdot cm^{-2}}$)")
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\1overRcat.png")

plt.show()
# %%


