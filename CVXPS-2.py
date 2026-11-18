"""
Read all exported per-binding-energy region files (regions/*.dat),
each with two data columns: counts/s and ErrorBar (100 scans per energy),
and plot counts/s vs binding energy, one spectrum per scan number.

Only the first N_SCAN spectra (scan numbers 0 to 4) are shown.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import wepy.basics as we

OUT_DIR = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_5mVs_02"
PLOT_FILE = r"C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_5mVs_02\spectra_first100_scans.png"
N_SCANS = 160  # plot first 100 scan rows

plt.rcParams['text.usetex'] = False

# --- read all region files ------------------------------------------
files = sorted(glob.glob(os.path.join(OUT_DIR, "*.dat")))
colours = we.get_colors(N_SCANS)
energies, data = [], []   # data[i] = counts array (N_scans,) for energies[i]
for path in files:
    energy = None
    with open(path) as f:
        for line in f:
            if "# Binding Energy:" in line:
                m = re.search(r"([-\d.]+)", line.split(":", 1)[1])
                energy = float(m.group(1))
                break
    counts = np.loadtxt(path, comments="#", usecols=1)  # column 1 = counts/s
    energies.append(energy)
    data.append(counts[:N_SCANS])

energies = np.array(energies)
data = np.array(data)                 # shape: (N_energies, N_scans)
print(data)
print(f"Loaded {data.shape[0]} energies x {data.shape[1]} scans")

# --- plot the first N_SCANS spectra ---------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

for scan in range(min(N_SCANS, data.shape[1])):
    print(scan)
    ax.plot(energies, data[:, scan], "-", ms=4, label=f"Scan {scan}", color=colours[scan])

ax.set_xlabel("Binding energy (eV)")
ax.set_ylabel("counts / s")
ax.set_title("O1s spectra, one curve per scan number (first 100 scans)")
ax.invert_xaxis()   # XPS convention: binding energy decreases to the right
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(PLOT_FILE, dpi=150)
print(f"saved plot to {PLOT_FILE}")
plt.show()