# %%

import numpy as np
import matplotlib.pyplot as plt
import os
from wepy.plots import apply_plot_style
import wepy.basics as we

freq = np.loadtxt(r"C:\Users\Herman\Desktop\WE\Konference\WDS\peis_bode.txt",skiprows = 2)

def cpe_impedance(Q, alpha, freq):
    omega = 2 * np.pi * freq
    return 1 / (Q * (1j * omega) ** alpha)

def r0_r1_p1_spectrum(R0, R1, Q1, alpha1, freq):
    Z_r0 = R0
    Z_r1_p1 = 1/(1/R1 + 1 / cpe_impedance(Q1, alpha1, freq))
    return Z_r0 + Z_r1_p1

def read_impedance_file(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' was not found. Please ensure the file exists.")
    try:
        # Attempt to read as comma-delimited
        data = np.loadtxt(filename, skiprows=2, delimiter=',')
        if data.shape[1] < 4:
            # Retry with whitespace delimiter
            data = np.loadtxt(filename, skiprows=2)
            if data.shape[1] < 4:
                raise ValueError(f"The file '{filename}' does not have the required 5 columns.")
    except Exception:
        # Fallback to whitespace delimiter
        data = np.loadtxt(filename, skiprows=2)
        if data.shape[1] < 4:
            raise ValueError(f"The file '{filename}' does not have the required 5 columns.")

    # freq = data[:, 0]
    Z_real_sample1 = data[:, 0]
    Z_imag_sample1 = -data[:, 1]
    Z_real_sample2 = data[:, 2]
    Z_imag_sample2 = -data[:, 3]
    return Z_real_sample1, Z_imag_sample1, Z_real_sample2, Z_imag_sample2

def plot_spectra(ax, Zr1, Zi1, Zr2, Zi2, title, Zsim1=None, Zsim2=None):
    ax.plot(Zr1, -Zi1, '-', label='Sample 1')
    ax.plot(Zr2, -Zi2, '-', label='Sample 2')
    if Zsim1 is not None:
        ax.plot(Zsim1.real, -Zsim1.imag, 'k--', label='Simulated 1')
    if Zsim2 is not None:
        ax.plot(Zsim2.real, -Zsim2.imag, 'r--', label='Simulated 2')
    ax.set_xlabel("Z'")
    ax.set_ylabel("-Z''")
    ax.set_title(title)
    ax.legend()
    ax.axvline(0,c='black')
    ax.axhline(0,c='black')
    # ax.grid(True)
    ax.axis('equal')
# %%

file_eis_anodes = r"C:\Users\Herman\Desktop\WE\Konference\WDS\peis_anodes.txt"
file_eis_cathodes =  r"C:\Users\Herman\Desktop\WE\Konference\WDS\peis_cathodes.txt"
file_eis_cells =  r"C:\Users\Herman\Desktop\WE\Konference\WDS\peis_cells.txt"


for f in [file_eis_anodes, file_eis_cathodes, file_eis_cells]:
    if not os.path.isfile(f):
        raise FileNotFoundError(f"Required data file '{f}' was not found in the working directory '{os.getcwd()}'.")

Zr1_a, Zi1_a, Zr2_a, Zi2_a = read_impedance_file(file_eis_anodes)
Zr1_c, Zi1_c, Zr2_c, Zi2_c = read_impedance_file(file_eis_cathodes)
Zr1_w, Zi1_w, Zr2_w, Zi2_w = read_impedance_file(file_eis_cells)

params1 = (0.0002, 0.1953, 0.1256, 0.931)
params2 = (0.0014, 0.2129, 0.1224, 0.93698)

Z_sim1_a = r0_r1_p1_spectrum(*params1, freq)
Z_sim2_a = r0_r1_p1_spectrum(*params2, freq)

params1 = (0.017, 0.02589, 0.0057476, 0.9670760)
params2 = (0.022, 0.02699, 0.0168, 0.917311)

Z_sim1_c = r0_r1_p1_spectrum(*params1, freq)
Z_sim2_c = r0_r1_p1_spectrum(*params2, freq)


fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(4840*Zr1_c,-4840*Zi1_c,lw = 4,label = 'Cathode spectrum')
ax.plot(4840*Z_sim1_c.real,-4840*Z_sim1_c.imag,'--',lw = 4, label='Calculated from cell spectrum')
ax.set_xlabel(r"$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \cdot cm^{2}\right) }$")
ax.set_ylabel(r"$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \cdot cm^{2} \right) }$")
ax.axhline(0, ls="--", c="black", lw=1)
ax.axvline(0, ls="--", c="black", lw=1)
ax.set_aspect("equal")
ax.legend()
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\cathode_EIS_1.png")
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(4840*Zr2_c,-4840*Zi2_c,lw = 4,label = 'Cathode spectrum')
ax.plot(4840*Z_sim2_c.real,-4840*Z_sim2_c.imag,'--',lw = 4, label='Calculated from cell spectrum')
ax.set_xlabel(r"$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \cdot cm^{2}\right) }$")
ax.set_ylabel(r"$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \cdot cm^{2} \right) }$")
ax.axhline(0, ls="--", c="black", lw=1)
ax.axvline(0, ls="--", c="black", lw=1)
ax.set_aspect("equal")
ax.legend()

fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\cathode_EIS_2.png")
plt.show()
# %%



data = we.read_file(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\455_IV_cathode_etching_series_35min\IV_Day4_Procedure1_PEIS_20_sccm_H2_C02.mpr"
)
spectrum = data[
    (data["cycle number"] == 7) & (data["freq/Hz"] < 20000) & (data["freq/Hz"] > 5)
]
params_CE = (0.019, 0.0111897, 0.0235726, 0.88478)
freq_CE = spectrum["freq/Hz"][:-10].to_numpy()
Z_sim = r0_r1_p1_spectrum(*params_CE, freq_CE)

ReZce = spectrum["Re(Zce)/Ohm"][:-10].to_numpy()
ImZce = spectrum["-Im(Zce)/Ohm"][:-10].to_numpy()

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(4840*ReZce,4840*ImZce,lw = 4,label = 'Cathode spectrum')
ax.plot(4840*Z_sim.real,-4840*Z_sim.imag,'--',lw = 4, label='Calculated from cell spectrum')
ax.set_xlabel(r"$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \cdot cm^{2}\right) }$")
ax.set_ylabel(r"$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \cdot cm^{2} \right) }$")
ax.axhline(0, ls="--", c="black", lw=1)
ax.axvline(0, ls="--", c="black", lw=1)
ax.set_aspect("equal")
ax.legend()
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\cathode_EIS_3.png")
plt.show()
# %%
print(Z_sim)