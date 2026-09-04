
# %%
import wepy.basics as we
import wepy.iv_curve as weiv
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')

# %%

folders = we.load_folders(r'\\ELECTROLYZER\PEM-WE_measurements\2026', ['444','443','439','438','436','433','432','431'],mode = 'any')
print(folders)
# %%

for folder in folders:
    files = we.load_files(folder, 'SV')

    for file in files:
        data = we.read_file(file)
        Es,Is = weiv.IV_curves_data(data)
        for E,I in zip(Es,Is):
            plt.plot(E,I)
    plt.show()