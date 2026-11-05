import numpy as np
import wepy.basics as we
import matplotlib.pyplot as plt

file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE\522_VI_Ni_Bekaert_PTL_no_sputtering_EIS_tests\CV-EIS\CVEIS_1,1-1,6V_10mVs_f_corrected_triple_PEIS.txt"
data = we.read_file(file, skiprows=0)
#print(data)
groups = data.groupby('Tech')
colors = we.get_colors(len(groups))
for index,group in groups:
    plt.plot(group['Ewe/V'],group['I/mA'],c=colors[index],label=index)
plt.legend()
plt.show()

for index,group in groups:
    plt.plot(group['time/s'],group['Ewe/V'],c=colors[index],label=index)
plt.legend()
plt.show()

groups = data.groupby('Loop')
colors = we.get_colors(len(groups))
all_freqs = []
all_Es = []
all_Is = []
all_Zres = []
all_Zims = []
all_indices = []
all_times = []
for index,group in groups:
    plt.plot(group['Re(Z)/Ohm'],group['-Im(Z)/Ohm'],c=colors[index])
    freq = group['freq/Hz'].values

    E = group['Ewe/V'].values
    I = group['I/mA'].values
    all_indices.append([index] * len(freq))
    all_freqs.append(freq)
    all_Es.append(E)
    all_Is.append(I)
    all_times.append(group['time/s'].values)
    all_Zres.append(group['Re(Z)/Ohm'].values)
    all_Zims.append(group['-Im(Z)/Ohm'].values)
print(all_freqs)
processed_data = np.column_stack([
    np.concatenate(all_indices),
    np.concatenate(all_freqs),
    np.concatenate(all_Es),
    np.concatenate(all_Is),
    np.concatenate(all_Zres),
    np.concatenate(all_Zims),
    np.concatenate(all_times)
])

np.savetxt(
    file[:-4] + '_processed.csv',
    processed_data,
    delimiter=',',
    header='cycle number,freq/Hz,<Ewe>/V,<I>/mA,Re(Z)/Ohm,-Im(Z)/Ohm,time/s',
    comments='',
)

plt.gca().set_aspect('equal')
plt.show()