import numpy as np
import matplotlib.pyplot as plt
import wepy.basics as we
import wepy.iv_curve as weiv

samples = ['505','499','474']
labels = ['plain', 'etched','PTL']
colors = ['r','g','b']
for sample in samples:
    folder = we.load_folders(r'\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE',sample)
    print(folder)
    files = we.load_files(folder[0],['ay3','SV'],'.mpr',mode = 'all')
    print(files)
    data = we.read_file(files[0])
    E,I = weiv.IV_curves_data(data,4.84)
    plt.plot(E[0],I[0], color=colors[samples.index(sample)], label=labels[samples.index(sample)])
plt.xlim(1.45,2.02)
plt.xlabel('Cell Voltage (V)')
plt.ylabel('Current Density (mA cm$^{-2}$)')
plt.legend()
plt.savefig(r'C:\Users\Herman\OneDrive - Univerzita Karlova/plain_etched_PTL.png', dpi=300)
plt.show()