# %%

import matplotlib.pyplot as plt
import numpy as np
import wepy.basics as we
# %%

s456 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\456_export.csv",'etching' : 'Ti in 17Ar 0,1O2'}
s462 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\462_export.csv",'etching' : 'Ti_in_Red'}
s465 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\465_export.csv",'etching' : 'Ti_in_17Ar_2O2'}
s181 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\181_export.csv",'etching' : '520 nm Ti'}
s466 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\466_export.csv",'etching' : 'Ti_in_17Ar_angled'}
s473 = {'path' : r"C:\Users\Herman\OneDrive - Univerzita Karlova\Ti overlayer\473_export.csv",'etching' : 'Ti_240W'}
samples = [s456,s462,s465,s466,s473]

for sample in samples:
    sample['data'] = we.read_file(sample['path'], skiprows=0, delimiter=",")
    sample['name'] = str(sample['path'].split('\\')[-1].split('_')[0]+' '+ sample['etching'])

# %%
for sample in samples:
    data = sample['data']
    sample['fit'] = []
    colors = we.get_colors(len(data['Time'].unique()))
    for time, group in data.groupby('Time'):
        
        x = group[group['I_mA'] < 200]['I_mA']
        y = 1/group[group['I_mA'] < 200]['R1']
        plt.plot(x, y,'x', label=sample['name'], color=colors[time-1])
        k,q = np.polyfit(x[1:], y[1:], 1)
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
    plt.plot(2.3/sample['fit'][:, 0], label=sample['name'])
plt.legend()
plt.show()
# %%
for sample in samples:
    data = sample['data']
    sample['fit_C1'] = []
    colors = we.get_colors(len(data['Time'].unique()))
    for (time, group), color in zip(data.groupby('Time'), colors):
        
        x = group[group['I_mA'] < 300]['I_mA']
        y = group[group['I_mA'] < 300]['C1']
        plt.plot(x, y,'x', color=color)
        k,q = np.polyfit(x[4:], y[4:], 1)
        sample['fit_C1'].append([k,q])
        plt.plot(x, k*x + q,'--',  color=color)
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    sample['fit_C1'] = np.array(sample['fit_C1'])
    #plt.legend()
    plt.show()
# %%
colors = we.get_colors(len(samples))
C1s  = []
for sample in samples:
    plt.plot(sample['fit_C1'][:, 1], label=sample['name'],c=colors[samples.index(sample)])
    try:
        C1s.append(sample['fit_C1'][:, 1][20])
    except:
        pass
plt.legend(loc = 'lower right')
plt.show()

# %%
for sample in samples:
    data = sample['data']
    sample['fit_R1'] = []
    colors = we.get_colors(len(data['Time'].unique()))
    for (time, group), color in zip(data.groupby('Time'), colors):
        
        x = group[group['I_mA'] < 500]['I_mA']
        y = 1/group[group['I_mA'] < 500]['R1']
        plt.plot(x, y,'x', label=sample['name'], color=color)
        k,q = np.polyfit(x[2:], y[2:], 1)
        sample['fit_R1'].append([k,q])
        plt.plot(x, k*x + q,'--', label=f"fit {sample['name']}: k={k:.2e}, q={q:.2e}", color=color)
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    sample['fit_R1'] = np.array(sample['fit_R1'])
    #plt.legend()
    plt.show()
# %%
for sample in samples:
    plt.plot(2.3/sample['fit_R1'][:, 0], label=sample['name'])
plt.legend()
plt.show()

# %%
samples = [s129,s140,s150,s157,s159,s181]
for time in range(15,30):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for sample in samples:
        data = sample['data']
        group = data[data['Time']==time]
        x = group[group['I_mA'] < 1000]['I_mA']
      #  x = x-x.iloc[0]
   
        y = 1/group[group['I_mA'] < 1000]['R1']
        try:
            ax1.plot(x-x.iloc[0], y,'-', label=sample['name'])
            ax2.plot(x-x.iloc[0], group[group['I_mA'] < 1000]['C1'],'-', label=sample['name'])
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

for time in range(0,20):
    fig, ax1= plt.subplots(1, 1, figsize=(12, 5))
    colors = we.get_colors(len(samples))
    for sample,color in zip(samples,colors):
        
        data = sample['data']
        group = data[data['Time']==time]
        try:
            x = group.loc[group['I_mA']<500, 'I_mA'].to_numpy()
            y = group[group['I_mA']<500]['R0']
            ax1.plot(x-x[0], y,'-', label=sample['name'],c=color)
           # y = group[group['I_mA']<500]['R2']
          #  ax1.plot(x-x[0], y,'-',c=color)
        except:
            pass


    ax1.set_xlabel("Current [mA]")
    ax1.set_ylabel("1/Ranode")
    ax1.set_title(f"Time: {time}")
    ax1.legend()
    ax1.set_yscale('log')

    plt.show()
# %%

colors = we.get_colors(4)
for cycle in range(1,10):
    for sample,color in zip(samples,colors):
        data = sample['data']
        group = data[data['Cycle mod'] == cycle]
        group.sort_values(by='Time', inplace=True)
        plt.plot(group['Time'], group['C1'], label=sample['name'],c=color)
    plt.title(cycle)
    #plt.xlim(0,20)
    plt.yscale('log')
    plt.legend()
    plt.show()

# %%
for sample in samples:
    plt.plot(sample['fit'][:, 1], label=sample['name'])
plt.legend()
plt.show()
# %%
for sample in samples:
    data = sample['data']
    sample['fit_R0'] = []
    colors = we.get_colors(len(data['Time'].unique()))
    for (time, group), color in zip(data.groupby('Time'), colors):
        
        x = group[group['I_mA'] < 1000]['I_mA']
        y = group[group['I_mA'] < 1000]['R0']
        plt.plot(x, y,'x', color=color)
        k,q = np.polyfit(x[5:], y[5:], 1)
        sample['fit_R0'].append([k,q])
        plt.plot(x, k*x + q,'--',  color=color)
    plt.xlabel("Current [mA]")
    plt.ylabel("1/R²")
    sample['fit_R0'] = np.array(sample['fit_R0'])
    #plt.legend()
    plt.show()
# %%
colors = we.get_colors(len(samples))
R0s  = []
#samples = [s129,s140,s150,s157,s159,s181,s178]
samples = [s140,s150,s157,s159]
for sample in samples:
    plt.plot(sample['fit_R0'][:, 1], label=sample['name'],c=colors[samples.index(sample)])
    try:
        R0s.append(sample['fit_R0'][:, 1][20])
    except:
        pass
plt.legend(loc = 'upper right')
plt.show()
loading = [100,170,250,400,520]
loading = [100,170,250,400]
plt.plot(loading,R0s,'o-')
plt.xlabel('Ti loading in nm')
plt.ylabel('C anode, cycle 20')
plt.show()
# %%


for sample in samples:
    data = sample['data']
    sample['fit_R0'] = []
    len_data = len(data['Time'].unique())
    colors = we.get_colors(3*len_data)
    index = 0
    for (time, group), color in zip(data.groupby('Time'), colors):
        
        x = group['Ecell_V']
        y = group['R1']
        plt.plot(x, y,'o-', color=colors[index])
        y = group['R2']
        plt.plot(x, y,'o-', color=colors[len_data+index])
        y = group['R3']
        plt.plot(x, y,'o-', color=colors[len_data*2+index])
        index += 1
    plt.xlabel("Current [mA]")
    plt.ylabel("R")
    plt.yscale('log')
    plt.title(sample['name'])
    #plt.legend()
    plt.show()
# %%
for sample in samples:
    data = sample['data']
    sample['fit_R0'] = []
    len_data = len(data['Time'].unique())
    colors = we.get_colors(3*len_data)
    index = 0
    for (time, group), color in zip(data.groupby('Time'), colors):
        
        x = group['Ecell_V']
        y = group['tau1']
        plt.plot(x, y,'o-', color=colors[index])
        y = group['tau2']
        plt.plot(x, y,'o-', color=colors[len_data+index])
        y = group['tau3']
        plt.plot(x, y,'o-', color=colors[len_data*2+index])
        index += 1
    plt.xlabel("Current [mA]")
    plt.ylabel("R")
    plt.yscale('log')
    plt.title(sample['name'])
    #plt.legend()
    plt.show()

# %%
for sample in samples:
    data = sample['data']
    sample['fit_R0'] = []
    len_data = len(data['Time'].unique())
    colors = we.get_colors(len_data)
    index = 0
    for (time, group), color in zip(data.groupby('Time'), colors):
        
        x = group['Ecell_V']
        y = group['I_mA']
        plt.plot(x, y,'o-', color=colors[index])

        index += 1
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [mA]")

    plt.title(sample['name'])
    #plt.legend()
    plt.show()
# %%
colors = we.get_colors(len(samples))
for color,sample in zip(colors, samples):
    data = sample['data']
    I = []
    for (time, group) in (data.groupby('Time')):
        I.append(group.loc[(group['Ecell_V'] - 1.586).abs().idxmin(), 'I_mA'] - group.iloc[0]['I_mA'])


    plt.plot(I,label = sample['name'], c = color)
plt.xlabel("Time")
plt.ylabel("Current [mA]")

plt.legend()
plt.show()
# %%
colors = we.get_colors(len(samples))
for color,sample in zip(colors, samples):
    data = sample['data']
    R1 = []
    for (time, group) in data.groupby('Time'):
        x = group['I_mA'].to_numpy()
        y = group['R1'].to_numpy()
        # Interpolate R1 with respect to I_mA, find R1 at I_mA = 200 mA

        interp_r1 = np.interp(100, x-x[0], y)
        R1.append(interp_r1)


    plt.plot(R1,label = sample['name'], color=color)
plt.xlabel("Time")
plt.ylabel("R1")
plt.yscale('log')
plt.legend()
plt.show()
# %%
colors = we.get_colors(len(samples))
for color,sample in zip(colors, samples):
    data = sample['data']
    R1 = []
    for (time, group) in data.groupby('Time'):
        y = group['I_mA'].to_numpy()
        x = group['Ecell_V'].to_numpy()
        # Interpolate R1 with respect to I_mA, find R1 at I_mA = 200 mA

        interp_r1 = np.interp(1.6, x, y-y[0])
        R1.append(interp_r1)


    plt.plot(R1,label = sample['name'], color=color)
plt.xlabel("Time")
plt.ylabel("I/mA")

plt.legend()
plt.show()
# %%
