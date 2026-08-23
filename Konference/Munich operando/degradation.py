# %%

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import wepy.eis as weis
from scipy.optimize import curve_fit
from wepy.plots import apply_plot_style

data_path = "C:/Users/Herman/Desktop/MFF/Diplomka/Data z PEMWE/"

from glob import glob
import os
import wepy.basics as we
import wepy.eis as weis
import wepy.iv_curve as weiv

# def custom_format(x, pos):
#    return r"${:,}$".format(int(x)).replace(',', r'\,')


# plt.rcParams['text.usetex'] = True
# plt.rcParams['axes.formatter.use_locale'] = True

# plt.rcParams.update({
#     "font.family": "serif",
#     "font.serif": "Times New Roman"
# })


s046 = {
    "name": "046",
    "cycles": 16,
    "loading": "14 nm Pt +  102 nm C",
    "label": "MEA 15Pt105C",
    "best_cycle": 24,
    "offset": 0,
    "label": "15",
}
s047 = {
    "name": "047",
    "cycles": 16,
    "loading": "14 nm Pt",
    "label": "MEA 15Pt",
    "label": "15",
}


def linear_func(x, a, b):
    return a * x + b


#
def linear_fit(x, k):
    return k * x


# *****************************************************************

cmap = plt.get_cmap("rainbow")
colors = cmap(np.linspace(0, 1, 16))
s047 = {"name": "047", "loading": 15}
s046 = {"name": "046", "loading": 15}
s028 = {"name": "028"}

MEA = [s046, s047]
weis.load_MEA_params(MEA)
cycle_plot = 19

for mea in MEA:
    ta = []
    tc1 = []
    tc2 = []
    t = []
    k1 = []
    k2 = []
    q2 = []
    folder_path = glob(
        os.path.join(
            "C:/Users/Herman/Desktop/MFF/Diplomka/Data z PEMWE/"
            + mea["name"]
            + "*/PEIS"
        )
    )[0]
    ignore_files = glob(os.path.join(folder_path, "nefitovat*"))
    ignore_c2 = np.loadtxt(ignore_files[2], skiprows=0)
    ignore_c1 = np.loadtxt(ignore_files[1], skiprows=0)
    ignore_a = np.loadtxt(ignore_files[0], skiprows=0)

    for cycle in range(0, mea["cycles"]):
        Ic2 = mea["I"][cycle]
        Ic1 = abs(mea["I"][cycle])
        Ia = abs(mea["I"][cycle])
        U_corr = mea["U"][cycle] - mea["I"][cycle] * mea["Ro"][cycle, :] / 1000

        yc2 = 1 / mea["Rc"][cycle, :]
        yc1 = 1 / mea["Rc"][cycle, :]
        ya = 1 / mea["Ra"][cycle, :]

        ignore_index_c2 = [0, 1, 2, 3, 4, 5]
        ignore_index_c1 = [14, 13, 12, 11, 10, 9, 8, 7]
        ignore_index_a = []
        for row, cycle_to_ignore in enumerate(ignore_c2[:, 0]):
            if int(cycle_to_ignore - 1) == cycle:
                ignore_index_c2.append(int(ignore_c2[row, 1]))

        for row, cycle_to_ignore in enumerate(ignore_c1[:, 0]):
            if int(cycle_to_ignore - 1) == cycle:
                ignore_index_c1.append(int(ignore_c1[row, 1]))

        for row, cycle_to_ignore in enumerate(ignore_a[:, 0]):
            if int(cycle_to_ignore - 1) == cycle:
                ignore_index_a.append(int(ignore_a[row, 1]))

        Ic2 = np.delete(Ic2, ignore_index_c2)
        yc2 = np.delete(yc2, ignore_index_c2)

        Ic1 = np.delete(Ic1, ignore_index_c1)
        yc1 = np.delete(yc1, ignore_index_c1)

        Ia = np.delete(Ia, ignore_index_a)
        ya = np.delete(ya, ignore_index_a)
        if mea == s046 and cycle == cycle_plot:
            plt.plot(mea["I"][cycle], 1 / mea["Rc"][cycle], "x", c="grey")
            plt.plot(Ic2, yc2, "x", c="blue")
            plt.plot(Ic1, yc1, "x", c="red")

        params, covariance = curve_fit(linear_fit, Ic1, yc1)

        k = params[0]
        tc1.append(2.3 / k)
        k1.append(k)
        if mea == s046 and cycle == cycle_plot:
            plt.plot(Ic1, k * Ic1, "--", label=mea["loading"])

        params, covariance = curve_fit(linear_func, Ic2, yc2)
        k = params[0]
        q = params[1]
        tc2.append(2.3 / k)
        k2.append(k)
        q2.append(q)
        if mea == s046 and cycle == cycle_plot:
            plt.plot(Ic2, k * Ic2 + q, "--", label=q)
            plt.plot(mea["I"][cycle], 1 / mea["Ra"][cycle], "x", c="grey")
            plt.plot(Ia, ya, "x", c="green")
        #
        # print(q)
        params, covariance = curve_fit(linear_fit, Ia, ya)
        k = params[0]
        ta.append(2.3 / k)
        if mea == s046 and cycle == cycle_plot:
            plt.plot(Ia, k * Ia, "--", label=2.3 / k)
            plt.legend()
            plt.title(" cycle " + str(cycle + 1))
            # # plt.savefig("C:/Users/Herman/Desktop/Elektrolyzéry výzkum/Článek/grafy/"+str(co)+' cycle '+str(cycle+1)+'.png',dpi = 300, format = 'png', bbox_inches = 'tight')
            plt.show()
    mea["ta"] = np.array(ta)
    mea["tc1"] = np.array(tc1)
    mea["tc2"] = np.array(tc2)
    # mea['tj'] = np.array(t)
    mea["k1"] = np.array(k1)
    mea["k2"] = np.array(k2)
    mea["q2"] = np.array(q2)


# %%


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

s046["color"] = "green"
s047["color"] = "red"
apply_plot_style(ax1)
apply_plot_style(ax2)

for mea in [s046, s047]:
    x = np.linspace(0, len(mea["q2"]), len(mea["q2"]))
    ax1.plot(
        x / 3,
        mea["q2"] * 353 * 8.31 / 96.500 / 2 / 4.84,
        "x--",
        c=mea["color"],
        lw=2,
        markersize=5,
    )

for mea in [s046, s047]:
    x = np.linspace(0, len(mea["q2"]), len(mea["q2"]))
    ax2.plot(x / 3, 1000 * mea["Cc"][:, 5], "x--", c=mea["color"], lw=2, markersize=5)

ax2.set_xlabel("Day")
ax1.set_ylabel(r"$j_\mathrm{c,0}$ ($\mathrm{mA \cdot cm^{-2}}$)")
ax2.set_ylabel(r"$C_\mathrm{c,0}$ ($\mathrm{mF \cdot cm^{-2}}$)")
ax1.set_xlim(-0.5, 8.8)
ax1.set_ylim(-10, 290)
ax2.set_ylim(-1, 23)
ax1.axhline(0, ls="--", c="black", lw=1)
ax2.axhline(0, ls="--", c="black", lw=1)
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\degradation_47_46.png")

plt.show()
# %%
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
apply_plot_style(ax1)
apply_plot_style(ax2)

for mea in [s047]:
    x = np.linspace(0, len(mea["q2"]), len(mea["q2"]))
    ax1.plot(
        x / 3,
        mea["q2"] * 353 * 8.31 / 96.500 / 2 / 4.84,
        "x--",
        c=mea["color"],
        lw=2,
        markersize=5,
    )

for mea in [s047]:
    x = np.linspace(0, len(mea["q2"]), len(mea["q2"]))
    ax2.plot(x / 3, 1000 * mea["Cc"][:, 5], "x--", c=mea["color"], lw=2, markersize=5)

ax2.set_xlabel("Day")
ax1.set_ylabel(r"$j_\mathrm{c,0}$ ($\mathrm{mA \cdot cm^{-2}}$)")
ax2.set_ylabel(r"$C_\mathrm{c,0}$ ($\mathrm{mF \cdot cm^{-2}}$)")
ax1.set_xlim(-0.5, 8.8)
ax1.set_ylim(-10, 290)
ax2.set_ylim(-1, 23)
ax1.axhline(0, ls="--", c="black", lw=1)
ax2.axhline(0, ls="--", c="black", lw=1)
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\degradation_47.png")

plt.show()
# %%

files = we.load_files(
    r"C:\Users\Herman\Desktop\MFF\Diplomka\Data z PEMWE\047_II_II_Ir etched star, Pt 14 nm no C etched star\Data z PEMWE",
    "PEIS.mpt",
)

# drt = hybdrt.models.DRT()

Zs = []
for file in files:
    cycles = [8, 23, 38]
    # try:
    df = we.read_file(file)

    for cycle in cycles:
        if "ay4_procedure2" in file:
            cycle += 1
        f, Z, E, I = weis.freq_and_Z(df, cycle, [5, 20000])
        # print(f,Z)
        # plt.plot(Z.real,-Z.imag)
        Zs.append(Z)

        E, Es = we.data_average(df, "<Ewe>/V", cycle)

# %%
fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
colors = we.get_colors(len(Zs))
for Z, color in zip(Zs, colors):
    ax.plot(1000 * 4.84 * Z.real, -1000 * 4.84 * Z.imag, c=color)

ax.set_xlabel(r"$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \cdot cm^{2}\right) }$")
ax.set_ylabel(r"$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \cdot cm^{2} \right) }$")
ax.axhline(0, ls="--", c="black", lw=1)
ax.axvline(0, ls="--", c="black", lw=1)
ax.set_xlim(-50, 1300)
ax.set_aspect("equal")
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\degradation_EIS.png")

plt.show()
# %%

files = we.load_files(
    r"C:\Users\Herman\Desktop\MFF\Diplomka\Data z PEMWE\047_II_II_Ir etched star, Pt 14 nm no C etched star\Data z PEMWE",
    "SV.mpt",
)

Es = []
Is = []
for file in files:
    data = we.read_file(file)
    Ex, Ix = weiv.IV_curves_data(data, norm=4.84)
    for E, I in zip(Ex, Ix):
        Es.append(E)
        Is.append(I)

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
colors = we.get_colors(len(Es))
for E, I, color in zip(Es, Is, colors):
    ax.plot(I[10:], E[10:], "-", c=color)


ax.set_xlabel(r"$j$ ($\mathrm{mA \cdot cm^{-2}}$)")
ax.set_ylabel(r"Cell Voltage (V)")
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\degradation_IVs.png")

plt.show()

# %%
file1 = r"C:\Users\Herman\Desktop\MFF\Diplomka\Data z PEMWE\047_II_II_Ir etched star, Pt 14 nm no C etched star\Data z PEMWE\II_Day2_procedure1_05_PEIS.mpr"
file2 = r"C:\Users\Herman\Desktop\MFF\Diplomka\Data z PEMWE\047_II_II_Ir etched star, Pt 14 nm no C etched star\Data z PEMWE\II_Day5_procedure1_05_PEIS.mpr"

f1, Z1, E1, I1 = weis.freq_and_Z(we.read_file(file1), 7)
f2, Z2, E2, I2 = weis.freq_and_Z(we.read_file(file2), 8)
print(I1, I2)
gamma1, drt1, R1 = weis.get_drt(f1, Z1, method="ridge")
gamma2, drt2, R2 = weis.get_drt(f2, Z2, method="ridge")
# %%
colors = we.get_colors(2)
fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(gamma1[15:-15], drt1[15:-15] * 4840, c=colors[0], label="Day 2", lw=4)
ax.plot(gamma2[15:-15], drt2[15:-15] * 4840, c=colors[1], label="Day 5", lw = 4)
ax.set_xscale("log")
ax.legend()
ax.set_xlabel(r"$\tau$ (s)")
ax.set_ylabel(r"$\gamma$ ($\mathrm{m\Omega \cdot cm^{2} \cdot s^{-1}}$)")

fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\degradation_DRTs.png")

plt.show()
