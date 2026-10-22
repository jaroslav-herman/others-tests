# %%
import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()

if ip is not None:
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')

# %%

s499 = {
    "path": "sample_499/499_fit_export.csv",
    "etching": "499",
}
s498 = {
    "path": "sample_498/498_fit_export.csv",
    "etching": "498",
}
s474 = {
    "path": "sample_474/474_fit_export.csv",
    "etching": "474",
}
samples = [s499, s498, s474]

for sample in samples:
    sample["data"] = we.read_file(sample["path"], skiprows=0, delimiter=",")
    sample["name"] = str(
        sample["path"].split("\\")[-1].split("_")[0] + " " + sample["etching"]
    )

# %%
'''Cell Voltage and Current and R1 vs Cell Voltage'''
for i in range(18):
    colors = we.get_colors(len(samples))
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    for sample,c in zip(samples, colors):
        data = sample["data"]
        data = data.sort_values(['Cycle mod', 'Time'])
        groups = data.groupby('Time', sort=True)
        # print(data.columns)
        for cycle, group in groups:
            
            if cycle == i:

                ax[0].plot(group[group['Ecell_V']<1.65]['Ecell_V'], group[group['Ecell_V']<1.65]['R1'],'-', c=c, )
                ax[1].plot(group[group['Ecell_V']<1.65]['I_mA']-group[group['Ecell_V']<1.65]['I_mA'].iloc[0], group[group['Ecell_V']<1.65]['Ecell_V'],'-',c=c,label=sample["name"])
                # Find index of Ecell_V closest to 1.6
                # idx_closest = (group['Ecell_V'] - 1.6).abs().idxmin()
                # print(group.loc[idx_closest, 'I_mA'])
        ax[0].set_yscale('log')
        # ax[0].set_xscale('log')
    # plt.ylim(0,25)
    # plt.xlim(0,500)
    ax[0].set_xlabel('Ecell (V)')
    ax[0].set_ylabel('R1 (Ohm)')
    ax[1].set_xlabel('Current (mA)')
    ax[1].set_ylabel('Cell Voltage (V)')
    ax[1].legend()
    plt.show()
# %%
'''Time Evolution of current and R1'''
for i in range(1,16):
    colors = we.get_colors(len(samples))
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    for sample,c in zip(samples, colors):
        data = sample["data"]
        data = data.sort_values(['Cycle mod', 'Time'])
        groups = data.groupby('Cycle mod', sort=True)
        # print(data.columns)
        for cycle, group in groups:
            if cycle == i:

                ax[0].plot(group['Time'], group['R1'],'-', c=c, )
                ax[1].plot(group['Time'], group['I_mA'],'-',c=c,label=sample["name"])


    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('R1 (Ohm)')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('I (mA)')
    ax[1].legend()
    plt.title(i)
    plt.show()

    # %%

'''Capacitance C1 of each sample vs. Current
Capacitance at I=0 time evolution for each sample'''    
for sample in samples:
    data = sample["data"]
    sample["fit"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for time, group in data.groupby("Time"):

        x = group[group["I_mA"] < 300]["I_mA"] - group[group["I_mA"] < 300]["I_mA"].iloc[0]
        y = group[group["I_mA"] < 300]["C1"]
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

plt.legend()
plt.show()

# %%
'''Anode Tafel Slope of each sample'''  
for sample in samples:
    data = sample["data"]
    sample["fit_R1"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for (time, group), color in zip(data.groupby("Time"), colors):
        group = group[(group['I_mA'] - group["I_mA"].iloc[0] < 200) & (group['I_mA'] - group["I_mA"].iloc[0]> 10)]
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
for sample in samples:
    sample["fit_R2"] = []

for time in range(0, 15):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    colors = we.get_colors(len(samples))
    for sample,c in zip(samples,colors):

        data = sample["data"]
        group = data[data["Time"] == time]
        try:
            x = group.loc[group["I_mA"] < 300, "I_mA"].to_numpy()
            y = group[group["I_mA"] < 300]["R2"]
            sample['fit_R2'].append(np.mean(group[group["I_mA"] < 300]["R2"]))
            ax1.plot(x - x[0], y, "-", label=sample["name"], c = c)
        except:
            pass

    ax1.set_xlabel("Current [mA]")
    ax1.set_title(f"Time: {time}")
    ax1.legend()
    ax1.set_yscale("log")

    plt.show()

for sample,c in zip(samples,colors):
    plt.plot(sample['fit_R2'],c=c,label=sample['name'])#
plt.legend()
plt.show()


# %%
'''All R in log scale'''

for sample in samples:
    data = sample["data"]
    sample["fit_R0"] = []
    len_data = len(data["Time"].unique())
    colors = we.get_colors(4 * len_data)
    index = 0
    for (time, group), color in zip(data.groupby("Time"), colors):

        x = group["Ecell_V"]
        y = group["R0"]
        plt.plot(x, y, "o-", color=colors[index])
        y = group["R1"]
        plt.plot(x, y, "o-", color=colors[len_data + index])
        y = group["R2"]
        plt.plot(x, y, "o-", color=colors[len_data + 2*index])
        y = group["R3"]
        plt.plot(x, y, "o-", color=colors[len_data * 3 + index])
        index += 1
    plt.xlabel("Cell Voltage [V]")
    plt.ylabel("R")
    plt.yscale("log")
    plt.title(sample["name"])
    # plt.legend()
    plt.show()
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
for sample in samples:
    data = sample["data"]
    sample["fit_R0"] = []
    len_data = len(data["Time"].unique())
    colors = we.get_colors(len_data)
    index = 0
    for (time, group), color in zip(data.groupby("Time"), colors):
        group = group[group['Ecell_V'] < 1.65]
        x = group["Ecell_V"]
        y = group["I_mA"]
        plt.plot(x, y, "o-", color=colors[index])

        index += 1
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [mA]")

    plt.title(sample["name"])
    # plt.legend()
    plt.show()
# %%
colors = we.get_colors(len(samples))
for color, sample in zip(colors, samples):
    data = sample["data"]
    I = []
    for time, group in data.groupby("Time"):
        I.append(
            group.loc[(group["Ecell_V"] - 1.65).abs().idxmin(), "I_mA"]
            - group.iloc[0]["I_mA"]
        )

    plt.plot(I, label=sample["name"], c=color)
plt.xlabel("Time")
plt.ylabel("Current [mA]")

plt.legend()
plt.show()
# %%
colors = we.get_colors(len(samples))
for color, sample in zip(colors, samples):
    data = sample["data"]
    R1 = []
    for time, group in data.groupby("Time"):
        x = group["I_mA"].to_numpy()
        y = group["R1"].to_numpy()
        # Interpolate R1 with respect to I_mA, find R1 at I_mA = 200 mA

        interp_r1 = np.interp(1000, x - x[0], y)
        R1.append(interp_r1)

    plt.plot(R1, label=sample["name"], color=color)
plt.xlabel("Time")
plt.ylabel("R1")
plt.yscale("log")
plt.legend()
plt.show()
# %%
colors = we.get_colors(len(samples))
for color, sample in zip(colors, samples):
    data = sample["data"]
    R1 = []
    for time, group in data.groupby("Time"):
        y = group["I_mA"].to_numpy()
        x = group["Ecell_V"].to_numpy()
        # Interpolate R1 with respect to I_mA, find R1 at I_mA = 200 mA

        interp_r1 = np.interp(1.85, x, y - y[0])
        R1.append(interp_r1)

    plt.plot(R1, label=sample["name"], color=color)
plt.xlabel("Time")
plt.ylabel("I/mA")

plt.legend()
plt.show()
# %%
