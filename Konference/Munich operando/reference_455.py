# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wepy.basics as we
import wepy.eis as weis
import wepy.iv_curve as weiv
from wepy.plots import apply_plot_style

# %%

data = we.read_file(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\455_IV_cathode_etching_series_35min\IV_Day4_Procedure1_PEIS_20_sccm_H2_C02.mpr"
)
spectrum = data[
    (data["cycle number"] == 8) & (data["freq/Hz"] < 20000) & (data["freq/Hz"] > 5)
]

# %%
ReZwe = spectrum["Re(Z)/Ohm"][7:].to_numpy()
ImZwe = spectrum["-Im(Z)/Ohm"][7:].to_numpy()
fZwe = spectrum["freq/Hz"][7:].to_numpy()
ReZce = spectrum["Re(Zce)/Ohm"][:-10].to_numpy()
ImZce = spectrum["-Im(Zce)/Ohm"][:-10].to_numpy()
fZce = spectrum["freq/Hz"][:-10].to_numpy()
ReZwe_ce = spectrum["Re(Zwe-ce)/Ohm"].to_numpy()
ImZwe_ce = spectrum["-Im(Zwe-ce)/Ohm"].to_numpy()
fZwe_ce = spectrum["freq/Hz"].to_numpy()

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)

ax.plot(ReZwe, ImZwe, "-",c='red',lw = 4, label="Anode spectrum")
ax.plot(ReZce, ImZce, "-",c='blue',lw = 4, label="Cathode spectrum")
ax.plot(ReZwe_ce, ImZwe_ce, "-",c='green',lw = 4, label = 'Cell spectrum')

ax.set_xlabel(r"$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \cdot cm^{2}\right) }$")
ax.set_ylabel(r"$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \cdot cm^{2} \right) }$")
ax.axhline(0, ls="--", c="black", lw=1)
ax.axvline(0, ls="--", c="black", lw=1)
ax.set_aspect("equal")
ax.legend()

plt.savefig(r'C:\Users\Herman\Python\WE\others-tests\Konference\Munich operando/reference_455_EIS.png', dpi = 500)
plt.show()
# %%
fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ZweDRT = weis.get_drt(fZwe, (ReZwe - 1j * ImZwe), "ridge")
ZceDRT = weis.get_drt(fZce, (ReZce - 1j * ImZce), "ridge")
Zwe_ceDRT = weis.get_drt(fZwe_ce, (ReZwe_ce - 1j * ImZwe_ce), "ridge")

ax.plot(ZweDRT[0][10:-10], 4840*ZweDRT[1][10:-10], "-",lw=4, c = 'red', label = 'Anode DRT')
ax.plot(ZceDRT[0][10:], 4840*ZceDRT[1][10:], "-",lw=4, c = 'blue', label = 'Cathode DRT')
ax.plot(Zwe_ceDRT[0][10:-10], 4840*Zwe_ceDRT[1][10:-10], "-",lw=4, c = 'green', label = 'Cell DRT')
ax.set_xlabel(r"$\tau$ (s)")
ax.set_ylabel(r"$\gamma$ ($\mathrm{m\Omega \cdot cm^{2} \cdot s^{-1}}$)")
ax.set_xscale("log")
ax.legend()
plt.savefig(r'C:\Users\Herman\Python\WE\others-tests\Konference\Munich operando/reference_455_DRT.png', dpi = 500)
plt.show()
# %%
data_sv = we.read_file(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\455_IV_cathode_etching_series_35min\IV_Day4_Procedure1_03_SV_C02.mpr"
)

data_sv = data_sv[(data_sv["cycle number"] == 1) & (data_sv["control/V"] > 0)]
Ecell = data_sv["control/V"].values
mask = np.diff(Ecell) > 0
mask = np.append(mask, False)
fig, (ax_top, ax_bottom) = plt.subplots(2,1,figsize=(8, 5))
apply_plot_style(ax)
#ax.plot(data_sv["<I>/mA"][mask], data_sv["control/V"][mask], "o-")
#ax.plot(data_sv["<I>/mA"][mask], data_sv["Ewe/V"][mask], "o-")
#ax.plot(data_sv["<I>/mA"][mask], data_sv["Ece/V"][mask], "o-")




# Plot data
ax_top.plot(data_sv["<I>/mA"][mask]/4.84,data_sv["control/V"][mask],c='black',label = 'Cell')
ax_top.plot(data_sv["<I>/mA"][mask]/4.84,data_sv["Ewe/V"][mask],c='red',label = 'Anode')

ax_bottom.plot(data_sv["<I>/mA"][mask]/4.84,data_sv["Ece/V"][mask],c='blue',label = 'Cathode')

# Set y-limits (this defines the "gap")
ax_top.set_ylim(1.35, 2.05)
ax_bottom.set_ylim(-0.4, 0.05)

# Hide spines
ax_top.spines.bottom.set_visible(False)
ax_bottom.spines.top.set_visible(False)
ax_top.tick_params(labeltop=False)
ax_bottom.xaxis.tick_bottom()

# Diagonal break marks
d = 0.02
kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax_bottom.set_xlabel(r'Current density $\mathrm{ \left( mA~cm^{-2} \right) }$',size=16)
# ax_bottom.set_ylabel(r'Voltage $\mathrm{ \left( V \right) }$',size=16)


ax_top.tick_params(axis="y", labelsize=14)
ax_bottom.tick_params(axis="both", labelsize=14)
ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

fig.text(
    0.04,          # x-position (left margin)
    0.5,           # y-position (center of figure)
    r'Voltage $\mathrm{ \left( V \right) }$',
    va="center",
    rotation="vertical",
    fontsize=16
)
fig.legend(loc='lower left',bbox_to_anchor=(0.65,0.35),
        fontsize=13, ncol=1,handletextpad = 1,columnspacing = 0.5,borderpad = 0.4,
        title_fontsize = 13,alignment = 'left',edgecolor='0',framealpha=1)

plt.axhline(0,ls = '--',c = 'black')
plt.savefig(r'C:\Users\Herman\Python\WE\others-tests\Konference\Munich operando/reference_455_IVs.png', dpi = 500)
plt.show()






# %%
x = data_sv["<I>/mA"][mask][30:-10]
y = data_sv["Ece/V"][mask][30:-10]
plt.plot(x, y, "o-")
k, q = np.polyfit(x[:30], y[:30], 1)
plt.plot(x, k * x + q, "r-")
print(-1000 * k)
plt.show()
# %%

data_fit = we.read_file(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\Konference\Munich operando\WE-CE-data.csv",
    skiprows=4,
    delimiter=",",
)
I = data_fit[data_fit["Series 2"] == "R0 = y (CE)"]["0.0"].values/4.84
R0_CE = data_fit[data_fit["Seri" "es 2"] == "R0 = y (CE)"]["1"].values
R0_WE = data_fit[data_fit["Series 2"] == "R0 = y (WE)"]["1"].values
R1_CE = data_fit[data_fit["Series 2"] == "R1 = y (CE)"]["1"].values
R1_WE = data_fit[data_fit["Series 2"] == "R1 = y (WE)"]["1"].values
#plt.plot(I, R0_CE + R1_CE, "o-", label="R0 CE")
#plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
#print(R0_CE + R1_CE)
y1 = np.array(R0_CE) + np.array(R1_CE)
x1 = np.array(I)
E_CE_cumulative = np.array(
    [np.trapezoid(y1[: i + 1], x1[: i + 1]) for i in range(len(x1))]
)
ax.plot(x[:30]/4.84, y[:30], "-",c='blue',label = 'Measured')

ax.plot(I, -0.001 * E_CE_cumulative*4.84, "x--",lw=3,markersize = 3, label="Cathdoe EIS calculated",c='orange')
ax.axhline(0,ls = '--',c = 'black')
ax.axvline(0,ls = '--',c = 'black')
ax.set_xlabel(r'Current density $\mathrm{ \left( mA~cm^{-2} \right) }$')
ax.legend()
ax.set_ylabel(r'Cathode potential (V)')
plt.savefig(r'C:\Users\Herman\Python\WE\others-tests\Konference\Munich operando/reference_455_cathode_IVs.png', dpi = 500)

plt.show()

# %%
