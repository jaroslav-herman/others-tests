
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

folders = we.load_folders(r'\\ELECTROLYZER\PEM-WE_measurements\2026','TiC')
print(folders)

for folder in folders:
    files = we.load_files(folder, 'SV',natural_sort=True)
    print(files)
    Es, Is = [],[]
    for file in files:
        try:
            data = we.read_file(file)
            E, I = weiv.IV_curves_data(data)
            for e in E:
                Es.append(e)
            for i in I:
                Is.append(i)
        except:
            pass
    colors = we.get_colors(len(Es))
    print(len(Es))
    for E,I,c in zip(Es,Is,colors):
        plt.plot(E,I,c=c)
    plt.show()
# %%
