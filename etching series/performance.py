# %%

from galvani import BioLogic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wepy.basics as we
import wepy.iv_curve as weiv

# %%
folders = we.load_folders(r'C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE','etching_series')
print(folders)
# %%

for folder in folders:
    Es = []
    Is = []
    files = we.load_files(folder, 'SV', '.mpr')
    for file in files:
        # ``files`` contains .mpr files, so use the MPRfile reader.

        #mpr_data = BioLogic.MPRfile(file).data
        #data = pd.DataFrame(mpr_data)
        data = we.read_file(file)
        E,I = weiv.IV_curves_data(data)
        for i in range(len(E)):
            Es.append(E[i])
            Is.append(I[i])
    
    colors = we.get_colors(len(Es))
    for E,I,color in zip(Es,Is,colors):
        plt.plot(E,I,c=color)
    plt.show()
# %%
