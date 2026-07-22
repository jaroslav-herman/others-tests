# %%
import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')
import re
import wepy.eis as weis
import pandas as pd

    # %%
files = we.load_files(r'\\ELECTROLYZER\PEM-WE_measurements\2026\430_VIII_VIII_IrOx(6,15)_60min50WTiCoverlayer,refel_N115_etchedcathode/Different N2 flows',['_PEIS','sccm_automated'],omit_string = ['_100_sccm','_7_sccm','1.5_sccm'],natural_sort=True)
print(files)

flows = [re.search(r'flow_(\d+)_sccm', file).group(1) for file in files]
colors = we.get_colors(len(files))

# %%



for file,color,label in zip(files,colors,flows):


    data = we.read_file(file)
    data = data[data['freq/Hz'] > 0]
    
    # Group data by 'cycle number' and average <Ewe>/V and <I>/mA for each cycle
    averaged = data.groupby('cycle number')[['Ewe-Ece/V', '<I>/mA']].mean().reset_index()
    plt.plot(averaged['Ewe-Ece/V'], averaged['<I>/mA'], '-', c=color, label=label)
plt.legend()
plt.show()
# %%


for file,label in zip(files,flows):


    data = we.read_file(file)
    data = data[data['freq/Hz'] > 0]
    
    groups = data.groupby('cycle number')
    Z_colors = we.get_colors(len(groups))
    for (cycle,group),color in zip(groups, Z_colors):
        plt.plot(group['Re(Z)/Ohm'],group['-Im(Z)/Ohm'], '-', c=color)
   # plt.legend()
    plt.show()
# %%
#files = list(reversed(files))
#flows = list(reversed(flows))
cycles = np.linspace(1,12,12)

params_all = []

for cycle in cycles:
    print(f"Cycle {cycle}")
    
    params = [0,0, 0.01,1e-8,  0.01, 0.01,  0.9, 0.3, 0.1,  0.9]
    bounds = (  [0,   1e-10, 0,   1e-5, 0.7, 0,   1e-2, 0.8],
                [0.05,1e-5,  2,   10,    1,  2,   10,   1])

    for file,color,flow in zip(files,colors,flows):


        data = we.read_file(file)
        data = data[data['freq/Hz'] > 0]



        group = data[data['cycle number']==cycle]



        f,Z,E,I = weis.freq_and_Z(group,cycle,[2,10000],control = 'Ewe-ce')
    #    print(E,I)
        init = params[2:]

        cir = "R0-L0-p(R1,CPE1)-p(R2,CPE2)"

        params,errors = weis.fit_spectrum(f, Z, cir="R0-L0-p(R1,CPE1)-p(R2,CPE2)", init=init, bounds=bounds, E=np.round(E,3), I=I, tau_sort=True)
        # print(params)
        f_model, Z_model = weis.show_fit(f,cir,params[2:],decades = (0,1), points = 100)
        plt.plot(Z.real, -Z.imag, 'x', c=color)
        plt.plot(Z_model.real, -Z_model.imag, '-', c=color)
        
        params_all.append(np.concatenate(([int(flow)],params)))
    # plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()

columns = ['flow', 'E', 'I', 'R0', 'L0', 'R1', 'Q1', 'a1', 'R2', 'Q2', 'a2']
params_all = pd.DataFrame(params_all, columns=columns)
# %%




plt.plot(params_all['flow'],params_all['R1'])
plt.xscale('log')
plt.show()
# %%
print(params_all)
print(params_all.columns)
# %%

params_flow = params_all.groupby('flow')
colors = we.get_colors(len(params_flow))
for (flow,group),c in zip(params_flow,colors):
#    plt.plot(group['I'],group['R1'],'-',c=c,label = flow)
    plt.plot(group['I'],1/group['R1'],'-',c=c,label = flow)
plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.ylim(1e-2,5)
plt.xlim(1e1,1000)
plt.show()
# %%
