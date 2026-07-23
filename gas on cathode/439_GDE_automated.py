# %%
import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')
import re

    # %%
files = we.load_files(r'\\ELECTROLYZER\PEM-WE_measurements\2026\439_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_etchedcathode_BDC929\PEISes',['CV','sccm_automated'],natural_sort=True)
files = we.load_files(r'\\ELECTROLYZER\PEM-WE_measurements\2026\430_VIII_VIII_IrOx(6,15)_60min50WTiCoverlayer,refel_N115_etchedcathode\Different H2 flows II',['CV','sccm_automated'],natural_sort=True)

print(files)
labels = [re.search(r'flow_(\d+)_sccm', file).group(1) for file in files]

# %%


colors = we.get_colors(len(files))
for file,color in zip(files,colors):

    match = re.search(r'flow_(\d+)_sccm', file)
    sccm_value = int(match.group(1)) if match else None
    # print(f'sccm: {sccm_value}')

    data = we.read_file(file)
    data_cycle = data[data['cycle number'] == 3]
    plt.plot(data_cycle['control/V'],data_cycle['<I>/mA'],c=color,label = sccm_value)
plt.legend()
plt.show()
# %%


for file,color,label in zip(files,colors,labels):
    data = we.read_file(file)
    data_cycle = data[data['cycle number'] == 3]
    dara_rising = data_cycle[data_cycle['control/V'].diff() > 0]
    plt.plot(dara_rising['control/V'],dara_rising['<I>/mA'],c=color,label = label  )
plt.ylim(-15,60)
plt.legend()
plt.show()
# %%
for file,color,label in zip(files,colors,labels):
    data = we.read_file(file)
    data_cycle = data[data['cycle number'] == 3]
    dara_rising = data_cycle[data_cycle['control/V'].diff() < 0]
    plt.plot(dara_rising['control/V'],dara_rising['<I>/mA'],c=color,label = label  )
plt.ylim(-15,60)
plt.legend()
plt.show()
# %%
