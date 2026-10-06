# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wepy.basics as we

# %%
DATA_FILE = Path(__file__).with_name("453_eisfit_export.csv")

# Complete export: one spectrum per row.
fit_data = pd.read_csv(DATA_FILE)

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
    "Time",
    "Cycle mod 15",
]
metadata = fit_data.loc[:, metadata_columns].copy()

# Custom metadata imported from the clipboard.
custom_metadata_columns = [
    "Spectrum",
    "Working electrode potential (V)",
    "Counter electrode potential (V)",
    "Ecell_V",
    "Time",
    "Cycle mod 15",
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
    "R3",
    "R3_e",
    "Q3",
    "Q3_e",
    "a3",
    "a3_e",
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
derived_columns = ["C1", "tau1", "C2", "tau2", "C3", "tau3"]
derived_values = indexed_fit_data.loc[:, derived_columns].copy()

print(f"Loaded {len(fit_data)} fitted spectra from {DATA_FILE.name}")
print(fit_data.head())

# %%

s453_file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\453_eisfit_export.csv"
s457_file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\457_eisfit_export.csv"

s457 = we.read_file(s457_file,delimiter = ',', skiprows=0)
s453 = we.read_file(s453_file,delimiter = ',', skiprows=0)
                 
# %%
qs2 = []
for (time, group), color in zip(
    fit_data.groupby("Time"), we.get_colors(len(fit_data["Time"].unique()))
):
    x = group["I_mA"]
    y = 1 / group["R2"]
    plt.plot(x[:-10], y[:-10], "x", color=color)
    k, q = np.polyfit(x[9:-10], y[9:-10], 1)
    plt.plot(x[:-10], k * x[:-10] + q, c=color)
    qs2.append(q)


plt.show()

plt.plot(qs1)
plt.plot(qs2)
plt.show()


# %%
for sample in [s453,s457]:
    for (time, group), color in zip(
        sample.groupby("Time"), we.get_colors(len(sample["Time"].unique()))
    ):
        x = group["I_mA"]
        y = group["R2"]
        plt.plot(x, y, "x-", color=color)
#plt.yscale('log')



plt.show()
# %%
for time in range(1,30):
    for sample in [s453,s457]:
        x = sample[sample['Time'] == time]["I_mA"]
        y = sample[sample['Time'] == time]["R3"]
        plt.plot(x, y, "x-")
   # plt.yscale('log')



    plt.show()
# %%
