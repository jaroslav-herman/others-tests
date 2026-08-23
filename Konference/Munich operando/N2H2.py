# %%
import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
from wepy.plots import apply_plot_style

# %%

EIS_file = r"C:\Users\Herman\Python\WE\others-tests\Konference\Munich operando\plot_data_EIS.csv"
DRT_file = r"C:\Users\Herman\Python\WE\others-tests\Konference\Munich operando\plot_data_DRT.csv"
EIS_data = we.read_file(EIS_file, skiprows=4, delimiter=",")
DRT_data = we.read_file(DRT_file, skiprows=2, delimiter=",")

print(EIS_data.columns)
# %%
print(EIS_data["Series 2"])
# %%
EIS_data_H2 = (
    4840
    * EIS_data[
        EIS_data["Series 2"]
        == "PEIS_at_N2_flow_50_sccm_automated_01_PEIS.mpr [Cell] - cycle 11"
    ]
)
EIS_data_N2 = (
    4840
    * EIS_data[
        EIS_data["Series 2"]
        == "PEIS_at_N2_flow_50_sccm_automated_01_PEIS.mpr [Cell] - cycle 10"
    ]
)

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(
    EIS_data_H2["0.0"],
    EIS_data_H2["1"],
    lw=2,
    color="red",
    label=r"$\mathrm{50~sccm~H_2}$",
)
ax.plot(
    EIS_data_N2["0.0"],
    EIS_data_N2["1"],
    lw=2,
    color="green",
    label=r"$\mathrm{50~sccm~N_2}$",
)

ax.set_xlabel(r"$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \cdot cm^{2}\right) }$")
ax.set_ylabel(r"$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \cdot cm^{2} \right) }$")
ax.axhline(0, ls="--", c="black", lw=1)
ax.axvline(0, ls="--", c="black", lw=1)
ax.legend()
ax.set_aspect("equal")
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\H2N2_EISes.png")

plt.show()

# %%
DRT_data_H2 = (
    4840
    * DRT_data[
        DRT_data["Series 1"]
        == "PEIS_at_N2_flow_50_sccm_automated_01_PEIS.mpr [Cell] - cycle 11"
    ]
)
DRT_data_N2 = (
    4840
    * DRT_data[
        DRT_data["Series 1"]
        == "PEIS_at_N2_flow_50_sccm_automated_01_PEIS.mpr [Cell] - cycle 10"
    ]
)

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(
    DRT_data_H2["1"][10:-5],
    DRT_data_H2["0.0"][10:-5],
    lw=2,
    color="red",
    label=r"$\mathrm{50~sccm~H_2}$",
)
ax.plot(
    DRT_data_N2["1"][10:-5],
    DRT_data_N2["0.0"][10:-5],
    lw=2,
    color="green",
    label=r"$\mathrm{50~sccm~N_2}$",
)

ax.set_xlabel(r"$\tau$ (s)")
ax.axhline(0, ls="--", c="black", lw=1)
ax.legend()
ax.set_xscale("log")
ax.set_ylabel(r"$\gamma$ ($\mathrm{m\Omega \cdot cm^{2} \cdot s^{-1}}$)")

fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\H2N2_DRTs.png")

plt.show()
