# %%

import matplotlib.pyplot as plt
import numpy as np
import wepy.basics as we
 # %%
s467 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\467_eisfit_export.csv",'etching' : '20 min'}
s455 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\455_eisfit_export.csv",'etching' : '35 min'}
s453 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\453_eisfit_export.csv",'etching' : '140 min'}
s457 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\457_eisfit_export.csv",'etching' : '0 min'}
s468 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\468_eisfit_export.csv",'etching' : '210 min'}
s476 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\476_eisfit_export.csv",'etching' : '70 min'}
s480 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\480_eisfit_export.csv",'etching' : '210 min'}
samples = [s457,s467,s455,s453,s468,s476,s480]

for sample in samples:
    sample['data'] = we.read_file(sample['path'], skiprows=0, delimiter=",")
    sample['name'] = str(sample['path'].split('\\')[-1].split('_')[0]+' '+ sample['etching'])

# %%
for sample in samples:
    data = sample['data']
    sample['fit'] = []
    colors = we.get_colors(len(data['Time'].unique()))
    for time, group in data.groupby('Time'):
        
        x = group[group['I_mA'] < 1000]['I_mA']
        y = 1/group[group['I_mA'] < 1000]['R2']
        plt.plot(x, y,'x', label=sample['name'], color=colors[time-1])
        k,q = np.polyfit(x[8:], y[8:], 1)
        sample['fit'].append([k,q])
        plt.plot(x, k*x + q,'--', label=f"fit {sample['name']}: k={k:.2e}, q={q:.2e}", color=colors[time-1])
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    plt.title(f"Sample: {sample['name']}")
    sample['fit'] = np.array(sample['fit'])
    #plt.legend()
    plt.show()
# %%
for sample in samples:
    plt.plot(sample['fit'][:, 1], label=sample['name'])
plt.legend()
plt.show()
# # %%
# for sample in samples:
#     data = sample['data']
#     sample['fit_C2'] = []
#     colors = we.get_colors(len(data['Time'].unique()))
#     for time, group in data.groupby('Time'):
        
#         x = group[group['I_mA'] < 1000]['I_mA']
#         y = group[group['I_mA'] < 1000]['C2']
#         plt.plot(x, y,'x', color=colors[time-1])
#         k,q = np.polyfit(x[8:], y[8:], 1)
#         sample['fit_C2'].append([k,q])
#         plt.plot(x, k*x + q,'--',  color=colors[time-1])
#     plt.xlabel("Current [mA]")
#     plt.ylabel("1/R²")
#     sample['fit_C2'] = np.array(sample['fit_C2'])
#     #plt.legend()
#     plt.show()
# # %%
# for sample in samples:
#     plt.plot(sample['fit_C2'][:, 1], label=sample['name'])
# plt.legend()
# plt.show()
# # %%
# for sample in samples:
#     data = sample['data']
#     sample['fit_R1'] = []
#     colors = we.get_colors(len(data['Time'].unique()))
#     for time, group in data.groupby('Time'):
        
#         x = group[group['I_mA'] < 1000]['I_mA']
#         y = 1/group[group['I_mA'] < 1000]['R1']
#         plt.plot(x, y,'x', label=sample['name'], color=colors[time-1])
#         k,q = np.polyfit(x[8:], y[8:], 1)
#         sample['fit_R1'].append([k,q])
#         plt.plot(x, k*x + q,'--', label=f"fit {sample['name']}: k={k:.2e}, q={q:.2e}", color=colors[time-1])
#     plt.xlabel("Current [mA]")
#     plt.ylabel("1/R²")
#     sample['fit_R1'] = np.array(sample['fit_R1'])
#     #plt.legend()
#     plt.show()
# # %%
# for sample in samples:
#     plt.plot(2.3/sample['fit_R1'][:, 0], label=sample['name'])
# plt.legend()
# plt.show()

# %%
for time in range(20,27):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for sample in samples:
        data = sample['data']
        group = data[data['Time']==time]
        x = group[group['I_mA'] < 1000]['I_mA']
        y = 1/group[group['I_mA'] < 1000]['R1']
        ax1.plot(x, y,'-', label=sample['name'])
        ax2.plot(x, group[group['I_mA'] < 1000]['C1'],'-', label=sample['name'])

    ax1.set_xlabel("Current [mA]")
    ax1.set_ylabel("1/R²")
    ax1.set_title(f"Time: {time}")
    ax1.legend()
    ax2.set_xlabel("Current [mA]")
    ax2.set_ylabel("C1")
    ax2.set_title(f"Time: {time}")
    ax2.legend()
    plt.show()

# %%

for time in range(15,24):
    fig, ax1= plt.subplots(1, 1, figsize=(12, 5))
    for sample in samples:
        data = sample['data']
        group = data[data['Time']==time]
        x = group['I_mA']
        y = group['R3']
        ax1.plot(x, y,'-', label=sample['name'])

    ax1.set_xlabel("Current [mA]")
    ax1.set_ylabel("1/R²")
    ax1.set_title(f"Time: {time}")
    ax1.legend()
    ax1.set_yscale('log')

    plt.show()
# %%
