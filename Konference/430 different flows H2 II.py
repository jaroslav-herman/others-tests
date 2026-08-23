# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from wepy.plots import apply_plot_style

# %%
# DATA_FILE = Path(__file__).with_name('430 different flows H2 II.csv')

# Complete export: one spectrum per row.
fit_data = pd.read_csv(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\Konference\Munich operando\430 different flows H2 II.csv"
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
    "source_name",
    "relaxis_group",
    "Ecell_V",
    "Time",
    "I_mA",
    "FreeVariable2",
    "SrcFn",
]
metadata = fit_data.loc[:, metadata_columns].copy()

# Custom metadata imported from the clipboard.
custom_metadata_columns = [
    "source_name",
    "relaxis_group",
    "Ecell_V",
    "Time",
    "I_mA",
    "FreeVariable2",
    "SrcFn",
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

groups = fit_data.groupby("FreeVariable2")
qs = []
for flow, group in groups:
    x = group[(group["I_mA"] > 40) & (group["I_mA"] < 500)]["I_mA"].values
    y = 1 / group[(group["I_mA"] > 40) & (group["I_mA"] < 500)]["R1"].values
    plt.plot(x, y, "o-", label=f"Flow {flow}")
    k, q = np.polyfit(x, y, 1)
    qs.append(q)

plt.xlabel("Current (mA)")
plt.ylabel("R1 (Ω)")
plt.title("R1 vs Current for Different Flows")
plt.legend()
plt.show()


x = groups.groups.keys()
y = 1000*np.array(qs)*353*8.31/(2*96500)/4.84
fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(
    x,
    y,
    marker="o",
    linestyle=":",
    color='blue',
    linewidth=1.8,
    markersize=4,
    alpha=0.85,
)
ax.set_xlabel("Flow H2 (sccm)", size = 14)
ax.set_ylabel("Exchange curr. dens. ($\mathrm{mA \cdot cm^{-2}}$)", size = 14)
plt.ylim(140,320)
ax.axvline(0, color="k", ls="--")
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.97)
fig.savefig(Path(__file__).with_name("current_density_H2_flow.png"))
plt.show()


# %%
