# %%

import matplotlib.pyplot as plt
import numpy as np
import wepy.basics as we

# %%
s323 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\323_exported.csv",
    "etching": "323",
}
s319 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\319_exported.csv",
    "etching": "319",
}
s318 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\318_exported.csv",
    "etching": "318",
}
s317 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\317_exported.csv",
    "etching": "317",
}
s101 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\101_exported.csv",
    "etching": "101",
}
s099 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\099_exported.csv",
    "etching": "099",
}
s090 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\090_exported.csv",
    "etching": "090",
}
s089 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\089_exported.csv",
    "etching": "089",
}
s088 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\088_exported.csv",
    "etching": "088",
}
s087 = {
    "path": r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\087_exported.csv",
    "etching": "087",
}
samples = [s323, s319, s318, s317, s101, s099, s090, s089, s088, s087]

for sample in samples:
    sample["data"] = we.read_file(sample["path"], skiprows=0, delimiter=",")
    sample["name"] = str(
        sample["path"].split("\\")[-1].split("_")[0] + " " + sample["etching"]
    )

# %%
for sample in samples:
    data = sample["data"]
    sample["fit"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for time, group in data.groupby("Time"):

        x = group[group["I_mA"] < 1000]["I_mA"]
        y = 1 / group[group["I_mA"] < 1000]["R2"]
        plt.plot(x, y, "x", label=sample["name"], color=colors[time - 1])
        k, q = np.polyfit(x[8:], y[8:], 1)
        sample["fit"].append([k, q])
        plt.plot(
            x,
            k * x + q,
            "--",
            label=f"fit {sample['name']}: k={k:.2e}, q={q:.2e}",
            color=colors[time - 1],
        )
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    plt.title(f"Sample: {sample['name']}")
    sample["fit"] = np.array(sample["fit"])
    # plt.legend()
    plt.show()
# %%
for sample in samples:
    plt.plot(sample["fit"][:, 1], label=sample["name"])
plt.legend()
plt.show()
# %%
for sample in samples:
    data = sample["data"]
    sample["fit_C1"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for (time, group), color in zip(data.groupby("Time"), colors):

        x = group[group["I_mA"] < 1000]["I_mA"]
        y = group[group["I_mA"] < 1000]["C1"]
        plt.plot(x, y, "x", color=color)
        k, q = np.polyfit(x[5:], y[5:], 1)
        sample["fit_C1"].append([k, q])
        plt.plot(x, k * x + q, "--", color=color)
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    sample["fit_C1"] = np.array(sample["fit_C1"])
    # plt.legend()
    plt.show()
# %%
colors = we.get_colors(len(samples))
C1s = []
for sample in samples:
    plt.plot(
        sample["fit_C1"][:, 1], label=sample["name"], c=colors[samples.index(sample)]
    )
plt.yscale("log")
plt.legend()
plt.show()

# %%
for sample in samples:
    data = sample["data"]
    sample["fit_R1"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for (time, group), color in zip(data.groupby("Time"), colors):

        x = group[group["I_mA"] < 500]["I_mA"]
        y = 1 / group[group["I_mA"] < 500]["R1"]
        plt.plot(x, y, "x", label=sample["name"], color=color)
        k, q = np.polyfit(x[2:], y[2:], 1)
        sample["fit_R1"].append([k, q])
        plt.plot(
            x,
            k * x + q,
            "--",
            label=f"fit {sample['name']}: k={k:.2e}, q={q:.2e}",
            color=color,
        )
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    sample["fit_R1"] = np.array(sample["fit_R1"])
    # plt.legend()
    plt.show()
# %%
for sample in samples:
    plt.plot(2.3 / sample["fit_R1"][:, 0], label=sample["name"])
plt.legend()
plt.show()

# %%
samples = [s129, s140, s150, s157, s159, s181]
for time in range(15, 30):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for sample in samples:
        data = sample["data"]
        group = data[data["Time"] == time]
        x = group[group["I_mA"] < 1000]["I_mA"]
        #  x = x-x.iloc[0]

        y = 1 / group[group["I_mA"] < 1000]["R1"]
        try:
            ax1.plot(x - x.iloc[0], y, "-", label=sample["name"])
            ax2.plot(
                x - x.iloc[0],
                group[group["I_mA"] < 1000]["C1"],
                "-",
                label=sample["name"],
            )
        except:
            pass

    ax1.set_xlabel("Current [mA]")
    ax1.set_ylabel("1/Ranode")
    ax1.set_title(f"Time: {time}")
    ax1.legend()
    ax2.set_xlabel("Current [mA]")
    ax2.set_ylabel("Canode")
    ax2.set_title(f"Time: {time}")
    ax2.legend()
    plt.show()

# %%
samples = [s129, s140, s150, s157, s159, s178, s181]
for time in range(20, 40):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    for sample in samples:

        data = sample["data"]
        group = data[data["Time"] == time]
        try:
            x = group.loc[group["I_mA"] < 4000, "I_mA"].to_numpy()
            y = group[group["I_mA"] < 4000]["R3"]
            ax1.plot(x - x[0], y, "-", label=sample["name"])
        except:
            pass

    ax1.set_xlabel("Current [mA]")
    ax1.set_ylabel("1/Ranode")
    ax1.set_title(f"Time: {time}")
    ax1.legend()
    ax1.set_yscale("log")

    plt.show()
# %%
samples = [s129, s140, s150, s157, s159, s178, s181]
colors = we.get_colors(7)
for cycle in range(20, 40):
    for sample, color in zip(samples, colors):
        data = sample["data"]
        group = data[data["Cycle mod 15"] == cycle]
        group.sort_values(by="Time", inplace=True)
        plt.plot(group["Time"], group["R3"], label=sample["name"], c=color)
    plt.title(cycle)
    # plt.xlim(0,20)
    plt.yscale("log")
    plt.legend()
    plt.show()

# %%
for sample in samples:
    plt.plot(sample["fit"][:, 1], label=sample["name"])
plt.legend()
plt.show()
# %%
for sample in samples:
    data = sample["data"]
    sample["fit_R0"] = []
    colors = we.get_colors(len(data["Time"].unique()))
    for (time, group), color in zip(data.groupby("Time"), colors):

        x = group[group["I_mA"] < 1000]["I_mA"]
        y = group[group["I_mA"] < 1000]["R0"]
        plt.plot(x, y, "x", color=color)
        k, q = np.polyfit(x[5:], y[5:], 1)
        sample["fit_R0"].append([k, q])
        plt.plot(x, k * x + q, "--", color=color)
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    sample["fit_R0"] = np.array(sample["fit_R0"])
    # plt.legend()
    plt.show()
# %%
colors = we.get_colors(len(samples))

for sample in samples:
    plt.plot(
        sample["fit_R0"][:, 1], label=sample["name"], c=colors[samples.index(sample)]
    )


plt.legend(loc="upper right")
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
        y = group["R1"]
        plt.plot(x, y, "o-", color=colors[index])
        y = group["R2"]
        plt.plot(x, y, "o-", color=colors[len_data + index])
        y = group["R3"]
        plt.plot(x, y, "o-", color=colors[len_data * 2 + index])
        index += 1
    plt.xlabel("Current [mA]")
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
    plt.xlabel("Current [mA]")
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
    colors = we.get_colors(len_data)
    index = 0
    for (time, group), color in zip(data.groupby("Time"), colors):

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
            group.loc[(group["Ecell_V"] - 1.586).abs().idxmin(), "I_mA"]
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

        interp_r1 = np.interp(200, x - x[0], y)
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

        interp_r1 = np.interp(1.6, x, y - y[0])
        R1.append(interp_r1)

    plt.plot(R1, label=sample["name"], color=color)
plt.xlabel("Time")
plt.ylabel("I/mA")

plt.legend()
plt.show()
# %%
