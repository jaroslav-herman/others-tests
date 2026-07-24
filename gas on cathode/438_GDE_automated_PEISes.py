# %%
import warnings

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
from scipy.interpolate import interp1d
warnings.filterwarnings('ignore')  # Suppress all warnings

    # %%
folder = r'\\ELECTROLYZER\PEM-WE_measurements\2026\438_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_none_BDC929\Different H2 flows'    
# files = we.load_files(r'\\ELECTROLYZER\PEM-WE_measurements\2026\430_VIII_VIII_IrOx(6,15)_60min50WTiCoverlayer,refel_N115_etchedcathode/Different H2 flows II',['_PEIS','sccm_automated'],omit_string = ['_301_sccm',],natural_sort=True)
files = we.load_files(folder,['_PEIS','sccm_automated'],omit_string = ['_100_sccm','_340_sccm',],natural_sort=True)
flows = []
for file in files:
    match = re.search(r'(\d+)_sccm', file, re.IGNORECASE)
    if match:
        flows.append(match.group(1))
    else:
        flows.append('unknown')
colors = we.get_colors(len(files))
print(flows)
# %%



for file,color,label in zip(files,colors,flows):


    data = we.read_file(file)
    data = data[data['freq/Hz'] > 0]
    # print(data.columns)
    # Group data by 'cycle number' and average <Ewe>/V and <I>/mA for each cycle
    averaged = data.groupby('cycle number')[['Unnamed: 87', '<I>/mA']].mean().reset_index()
    # print(averaged)
    plt.plot(averaged['Unnamed: 87'], averaged['<I>/mA'], '-', c=color, label=label)
plt.legend()
plt.show()
# %%


for file,label in zip(files,flows):


    data = we.read_file(file)
    data = data[data['freq/Hz'] > 0]
    
    groups = data.groupby('cycle number')
    Z_colors = we.get_colors(len(groups))
    for (cycle,group),color in zip(groups, Z_colors):
        plt.plot(group['Re(Zwe-ce)/Ohm'],group['-Im(Zwe-ce)/Ohm'], '-', c=color)
   # plt.legend()
    # plt.xlim(0.02,0.05)
    # plt.ylim(-0.01,0.02)
    plt.gca().set_aspect('equal')
plt.show()
# %%
#files = list(reversed(files))
#flows = list(reversed(flows))
cycles = np.linspace(1,18,18)

params_all = []

for cycle in cycles:
    print(f"Cycle {cycle}")
    
    # params = [0,0, 0.01,1e-8,  0.01, 0.01,  0.9, 0.5, 0.1,  0.9]
    # bounds = (  [0,   1e-10, 0,   1e-5, 0.7, 0,   1e-5, 0.7],
    #             [0.05,1e-5,  2,   10,    1,  2,   10,   1])

    params = [0,0, 0.01,1e-8,  0.01, 0.01,  0.9]
    bounds = (  [0,   1e-10, 0,   1e-2, 0.7, ],
                [0.05,1e-5,  100,   1,    1, ])

    for file,color,flow in zip(files,colors,flows):


        data = we.read_file(file)
        data = data[data['freq/Hz'] > 0]

        group = data[data['cycle number']==cycle]

        f,Z,E,I = weis.freq_and_Z(group,cycle,[1,21000],control = 'Ewe-ce')
    #    print(E,I)
        init = params[2:]

        cir = "R0-L0-p(R1,CPE1)"
        f,Z = weis.remove_outliers(f,Z,threshold=0.5)
        params,errors = weis.fit_spectrum(f, Z, cir=cir, init=init, bounds=bounds, outliers=False, threshold=0.5, E=np.round(E,3), I=I, tau_sort=False)
        # print(params)
        f_model, Z_model = weis.show_fit(f,cir,params[2:],decades = (0,1), points = 100)
        # plt.plot(Z.real, -Z.imag, 'x', c=color)
        # plt.plot(Z_model.real, -Z_model.imag, '-', c=color)

        
        params_all.append(np.concatenate(([int(flow)],params)))
    # plt.legend()
    # plt.gca().set_aspect('equal')
    # plt.show()

columns = ['flow', 'E', 'I', 'R0', 'L0', 'R1', 'Q1', 'a1',]
# columns = ['flow', 'E', 'I', 'R0', 'L0', 'R1', 'Q1', 'a1', 'R2', 'Q2', 'a2']
params_all = pd.DataFrame(params_all, columns=columns)
params_all['C1'] = weis.capacitance(params_all['R1'], params_all['Q1'], params_all['a1'])
# params_all['C2'] = weis.capacitance(params_all['R2'], params_all['Q2'], params_all['a2'])

# %%




# plt.plot(params_all['flow'],params_all['R1'])
# plt.xscale('log')
# plt.show()
# %%
print(params_all)
print(params_all['I'].values)
print(params_all.columns)

# %%
params_all_hc = params_all[(params_all['I'] > 50) & (params_all['I'] < 2000)]
params_flow = params_all_hc.groupby('flow')
colors = we.get_colors(len(params_flow))
param_interp = []
I_interp = np.linspace(50, 2000, 200)
for (flow,group),c in zip(params_flow,colors):
#    plt.plot(group['I'],group['R1'],'-',c=c,label = flow)

    plt.plot(group['I'],1/group['R1'],'-',c=c,label = flow)
    # Interpolate 1/R1 as a function of I for this flow
    interp = interp1d(group['I'], 1/group['R1'], kind='cubic', fill_value='extrapolate')
    
    plt.plot(I_interp, interp(I_interp), '--', c=c)
    k,q = np.polyfit(group['I'],1/group['R1'],1)
    param_interp.append(interp(I_interp))
    # plt.plot(group['I'],k*group['I']+q,'--',c=c)
plt.xlabel('Current (mA)')

plt.legend()
# plt.xscale('log')
# plt.yscale('log')
plt.ylim(15,25)
plt.xlim(300,500)
plt.show()

param_interp = np.array(param_interp)

# %%
flows_array = np.array([int(flow) for flow in params_flow.groups.keys()])
for i in range(0, param_interp.shape[1], 10):
    plt.plot(flows_array, param_interp[:, i], 'o-', label=f'I={I_interp[i]:.0f} mA')
    plt.legend()
    # plt.grid()
    plt.xlabel('Flow (sccm)')
    plt.ylabel('1/R1 at I=0 (S)')
    plt.show()

# %%
