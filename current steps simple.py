# %%
import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

ip = get_ipython()
if ip is not None:
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "2")
# %%

# %%

file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE\417_VI_VI_NiAl_piperion60_plain_RDC1012\VI_Day3_test_PEIS_C01.mpt"

data = we.read_file(file)

groups = data.groupby("cycle number")
for cycle, group in list(groups)[:5]:
    plt.plot(group["time/s"][3:] + 5 * cycle, group["<I>/mA"][3:], c="black", lw=5)
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)
plt.savefig(
    r"C:\Users\Herman\Desktop\MFF\PhD\Škola\summer seminar 2026/fig_slide_3.png",
    dpi=500,
)
plt.show()
# %%

x = np.linspace(0, 6.28, 100)
y = np.sin(2*x)

plt.plot(x, y, c="black", lw=5)
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)

name =  r"sine2"
plt.savefig( name + ".svg",dpi=500)
plt.show()

# %%

x = np.linspace(0, 6.28, 100)
y = np.sin(2*x)

plt.plot(x, y, c="black", lw=5)
plt.xlabel('x')
plt.ylabel('sin (x)')
plt.title('Sinus function')

name =  r"sine2"
plt.savefig( name + ".svg",dpi=500)
plt.show()




# %%

x = np.linspace(0.02, 0.1, 1001)
y = np.sin(1/(x))


plt.plot(x, y, c="black", lw=5)
plt.xscale('log')
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)
plt.savefig(
    r"C:\Users\Herman\Desktop\MFF\PhD\Škola\summer seminar 2026/chirp.svg", dpi=500
)
plt.show()
# %%
file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE\417_VI_VI_NiAl_piperion60_plain_RDC1012\VI_Day3_test_CA_C01.mpt"

data = we.read_file(file)



plt.plot(data["time/s"][15:] , data["<I>/mA"][15:], c="black", lw=5)
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)
name =  r"C:\Users\Herman\Desktop\MFF\PhD\Škola\summer seminar 2026/fig_slide_4"
plt.savefig(
    name + ".png",
    dpi=500,
)
plt.savefig(
    name + ".svg",
    dpi=500,
)
plt.show()

# %%
file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\414_VIII_VIII_IrOx_5minPt_18nm_N212_plain_BDC903Ptwireasreference\trpeis_at_1,46V_for_30s_and_90s_OCV_1_PEIS.txt"

data = we.read_file(file, 0)
colors = we.get_colors(len(np.unique(data['freq/Hz'])))
for c,f in zip(colors,np.unique(data['freq/Hz'])):
    plt.plot(data.loc[data['freq/Hz'] == f, 'Re(Z)/Ohm'][1:],data.loc[data['freq/Hz'] == f, '-Im(Z)/Ohm'][1:],'x',c = c)
plt.axis("off")
plt.gca().set_aspect('equal')
fig = plt.gcf()
fig.patch.set_alpha(0)
name =  r"C:\Users\Herman\Desktop\MFF\PhD\Škola\summer seminar 2026/Zs shift 2"

plt.savefig(    name + ".png",    dpi=500,)
plt.savefig( name + ".svg",    dpi=500,)
plt.show()
# %%
file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\414_VIII_VIII_IrOx_5minPt_18nm_N212_plain_BDC903Ptwireasreference\trpeis_at_1,46V_for_30s_and_90s_OCV_1_PEIS.txt"

data = we.read_file(file, 0)
colors = we.get_colors(5)
for c,f in zip(colors,np.unique(data['freq/Hz'])[5:6]):
    plt.plot(data.loc[data['freq/Hz'] == f, 'Re(Z)/Ohm'][1:],data.loc[data['freq/Hz'] == f, '-Im(Z)/Ohm'][1:],'x',c = 'black')
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)
name =  r"C:\Users\Herman\Desktop\MFF\PhD\Škola\summer seminar 2026/Z shift"

plt.savefig(    name + ".png",    dpi=500,)
plt.savefig( name + ".svg",    dpi=500,)
plt.show()
# %%



# %%

"""
Example of the spectra evolution in time (2 cells)
"""

file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\414_VIII_VIII_IrOx_5minPt_18nm_N212_plain_BDC903Ptwireasreference\trpeis_at_1,46V_for_30s_and_90s_OCV_1_PEIS.txt"

data = we.read_file(file, 0)

groups = data.groupby("freq/Hz")
colors = we.get_colors(len(groups))
# for color, (f, group) in zip(colors, groups):
#     if f > 1000:
#         plt.plot(group["Re(Z)/Ohm"], group["-Im(Z)/Ohm"], "x", label=f, color=color)
#     # plt.plot(group['Re(Z)/Ohm'], group['-Im(Z)/Ohm'],'x', label=f, color=color)
# # plt.xlim(0,0.2)
# # plt.ylim(0,0.2)
# plt.legend()
# plt.show()

time_new = np.arange(0.4, 10, 0.1)
spectra = []
for f, group in groups:
    group_sorted = group.sort_values("time/s")
    
    interp_I = np.interp(time_new, group_sorted["time/s"] - 90, group_sorted["I/mA"])
    
    interp_ReZ = np.interp(
        time_new, group_sorted["time/s"] - 90, group_sorted["Re(Z)/Ohm"]
    )
    plt.plot(time_new, interp_ReZ)
    interp_ImZ = np.interp(
        time_new, group_sorted["time/s"] - 90, group_sorted["-Im(Z)/Ohm"]
    )
    spectra.append([interp_ReZ, interp_ImZ])

spectra = np.array(spectra)
plt.show()
# %%

colors = we.get_colors(len(spectra[0, 0, :]))
for color, (index, t) in zip(colors, enumerate(time_new)):
    if t > 0.6: 
        if t < 1:
            max_f = 3
        else:
            max_f = 1
        plt.plot(
            spectra[:, 0, index][max_f:],
            spectra[:, 1, index][max_f:],
            "-",
            c=color,
        )


# plt.ylim(0, 0.2)
# plt.xlim(0, 1)
plt.gca().set_aspect("equal")
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)
name =  r"C:\Users\Herman\Desktop\MFF\PhD\Škola\summer seminar 2026/spectra example"

plt.savefig(    name + ".png",    dpi=500,)
plt.savefig( name + ".svg",    dpi=500,)
plt.show()

# %%
