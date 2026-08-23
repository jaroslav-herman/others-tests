# %%

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import wepy.eis as weis

from wepy.plots import apply_plot_style


from glob import glob
import os
import wepy.basics as we

import wepy.iv_curve as weiv

file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\438\VII_Day15_PEIS_klasik_2_C01.mpr"
cycles = np.linspace(5, 13, 9)
data = we.read_file(file)
colors = we.get_colors(len(cycles))

fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)

for cycle, color in zip(cycles, colors):
    f, Z, E, I = weis.freq_and_Z(data, cycle, [20000, 2])
    ax.plot(4840 * Z.real, -4840 * Z.imag, c=color, label=f"{E:.2f} V")

ax.legend()
ax.set_xlabel(r"$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \cdot cm^{2}\right) }$")
ax.set_ylabel(r"$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \cdot cm^{2} \right) }$")
ax.axhline(0, ls="--", c="black", lw=1)
ax.axvline(0, ls="--", c="black", lw=1)

ax.set_aspect("equal")
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\GDE_EIS.png")


plt.show()
