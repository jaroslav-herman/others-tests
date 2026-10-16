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

for sample in samples:
    plt.plot(sample['fit'][:, 1], label=sample['name'])
plt.legend()
plt.show()
# %%

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
for time in range(0,30):
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


# %%
colors = we.get_colors(len(samples))
for color, sample in zip(colors, samples):
    data = sample["data"]
    R1 = []
    for time, group in data.groupby("Time"):
        x = group["I_mA"].to_numpy()
        y = group["R2"].to_numpy()
        # Interpolate R1 with respect to I_mA, find R1 at I_mA = 200 mA

        interp_r1 = np.interp(500, x - x[0], y)
        R1.append(interp_r1)

    plt.plot(R1, label=sample["name"], color=color)
plt.xlabel("Time")
plt.ylabel("R1")
plt.yscale("log")
plt.legend()
plt.show()
# %%
# %%
colors = we.get_colors(len(samples))
for color, sample in zip(colors, samples):
    data = sample["data"]
    R1 = []
    for time, group in data.groupby("Time"):
        y = group["I_mA"].to_numpy()
        x = group["Ecell_V"].to_numpy()
        # Interpolate R1 with respect to I_mA, find R1 at I_mA = 200 mA

        interp_r1 = np.interp(1.55, x, y - y[0])
        R1.append(interp_r1)

    plt.plot(R1, label=sample["name"], color=color)
plt.xlabel("Time")
plt.ylabel("I/mA")

plt.legend()
plt.show()
# %%
# %%
for sample in samples:
    data = sample["data"]
    sample["fit_R0"] = []
    len_data = len(data["Time"].unique())
    colors = we.get_colors(3 * len_data)
    index = 0
    for (time, group), color in zip(data.groupby("Time"), colors):

        x = group["Ecell_V"]
        y = group["tau1"]
        plt.plot(x, y, "o-", color=colors[index])
        y = group["tau2"]
        plt.plot(x, y, "o-", color=colors[len_data + index])
        y = group["tau3"]
        plt.plot(x, y, "o-", color=colors[len_data * 2 + index])
        index += 1
    plt.xlabel("Cell voltage [V]")
    plt.ylabel("tau (s)")
    plt.yscale("log")
    plt.title(sample["name"])
    # plt.legend()
    plt.show()
# %%

'''Capacitance C1 of each sample vs. Current
Capacitance at I=0 time evolution for each sample'''    
for sample in samples:
    data = sample["data"]
    sample["fit"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for time, group in data.groupby("Time"):

        x = group[group["I_mA"] < 500]["I_mA"] - group[group["I_mA"] < 300]["I_mA"].iloc[0]
        y = group[group["I_mA"] < 500]["C1"]
        plt.plot(x, y, "x", label=sample["name"], color=colors[time - 1])
        k, q = np.polyfit(x[5:], y[5:], 1)
        sample["fit"].append([k, q])
        plt.plot(
            x,
            k * x + q,
            "--",
            label=f"fit {sample['name']}: k={k:.2e}, q={q:.2e}",
            color=colors[time - 1],
        )
    plt.xlabel("Current [mA]")
    plt.ylabel("C1")
    plt.title(f"Sample: {sample['name']}")
    sample["fit"] = np.array(sample["fit"])
    # plt.legend()
    plt.show()

colors = we.get_colors(len(samples))
for sample in samples:
    plt.plot(sample["fit"][:, 1], label=sample["name"],c=colors[samples.index(sample)])
plt.legend()

plt.show()
# %%
'''Ohmic resistance R0 of each sample vs. Current
R0 at I=0 time evolution for each sample'''   
for sample in samples:
    data = sample["data"]
    sample["fit_R0"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for (time, group), color in zip(data.groupby("Time"), colors):

        x = group[group["I_mA"] < 300]["I_mA"] - group[group["I_mA"] < 300]["I_mA"].iloc[0]
        y = group[group["I_mA"] < 300]["R0"]
        plt.plot(x, y, "x", color=color)
        k, q = np.polyfit(x[5:], y[5:], 1)
        sample["fit_R0"].append([k, q])
        plt.plot(x, k * x + q, "--", color=color)
    plt.xlabel("Current [mA]")
    plt.ylabel("R0")
    sample["fit_R0"] = np.array(sample["fit_R0"])
    # plt.legend()
    plt.show()
colors = we.get_colors(len(samples))
C1s = []
for sample in samples:
    plt.plot(
        sample["fit_R0"][:, 1], label=sample["name"], c=colors[samples.index(sample)]
    )
plt.yscale("log")
plt.legend()
plt.show()
for sample in samples:
    plt.plot(
        sample["fit_R0"][:, 1][5:], label=sample["name"], c=colors[samples.index(sample)]
    )

plt.legend()
plt.show()
# %%
'''Anode Tafel Slope of each sample'''  
for sample in samples:
    data = sample["data"]
    sample["fit_R1"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for (time, group), color in zip(data.groupby("Time"), colors):
        group = group[group["Ecell_V"] < 1.59]
        group = group[(group['I_mA'] - group["I_mA"].iloc[0] < 800) & (group['I_mA'] - group["I_mA"].iloc[0]> 10)]
        x = group["I_mA"] - group["I_mA"].iloc[0]
        y = 1 / group["R1"]
        plt.plot(x, y, "x", label=sample["name"], color=color)
        k, q = np.polyfit(x, y, 1)
        sample["fit_R1"].append([k, q])
        plt.plot(
            x,
            k * x + q,
            "--",
            label=f"fit {sample['name']}: k={k:.2e}, q={q:.2e}",
            color=color,
        )
    plt.title(f"Sample: {sample['name']}")
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R1")
    sample["fit_R1"] = np.array(sample["fit_R1"])
    # plt.legend()
    plt.show()
colors = we.get_colors(len(samples))
for sample,color in zip(samples,colors):
    plt.plot(2.3 / sample["fit_R1"][:, 0], label=sample["name"], c = color)
plt.legend()
plt.show()

# %%
