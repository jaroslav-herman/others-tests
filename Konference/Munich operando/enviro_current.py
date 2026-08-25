# %%
import warnings

import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from IPython import get_ipython

from wepy.plots import apply_plot_style

ip = get_ipython()
if ip is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
import re
import wepy.eis as weis
import pandas as pd

# %%

file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Proposal 20257155, Herman\MEA2, 5 min Pt on plain, IrOx, N212, small\07_nGDL_PEIS_2V_2.mpr"

data = we.read_file(file)

data1 = data[data["time/s"] < 7000]
fig, ax = plt.subplots(figsize=(8, 5))
apply_plot_style(ax)
ax.plot(
    data1["time/s"],
    data1["<I>/mA"],
    marker="o",
    linestyle=":",
    color='blue',
    linewidth=1.8,
    markersize=4,
    alpha=0.85,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Current (mA)")
fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.97)
fig.savefig(r"C:\Users\Herman\Python\WE\others-tests\Konference\enviro_current.png")
plt.show()
# %%
