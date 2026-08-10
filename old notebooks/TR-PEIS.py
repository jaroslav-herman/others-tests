# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:33:48 2025

@author: Herman
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.optimize import curve_fit


def get_header_lines(filepath):
    with open(filepath, "r", encoding="latin-1") as f:
        # Skip first line, read second
        f.readline()  
        line2 = f.readline().strip()
    # line2 looks like: "Nb header lines : 76"
    # split by ":" and take the right part
    try:
        n_header = int(line2.split(":")[1])
    except Exception:
        raise ValueError(f"Could not parse header line: {line2}")
    return n_header

# \\ELECTROLYZER\PEM-WE_measurements\2025\215_III_III_IrOx on Ti, 8 nm plain Pt + 30 nm C cathode, GDL\PEIS looped 1,6 V

# MEA= glob(os.path.join('//ELECTROLYZER/PEM-WE_measurements/2025/'+'*O*'))
files = glob(os.path.join('//ELECTROLYZER/PEM-WE_measurements/2025/209_IV_V_Ti + IrOx Ar14 O1, plain,Ti400nm 0.5Pa Ir 25nm+20nmPt remake\impedance 1.9 V','*PEIS_C01.mpt'))

# \\ELECTROLYZER\PEM-WE_measurements\2025\209_IV_V_Ti + IrOx Ar14 O1, plain,Ti400nm 0.5Pa Ir 25nm+20nmPt remake\impedance 1.8 V

print(files)
V_query = np.full(len(files), 1.8)




Times = np.linspace(0.1,15.1,15)

cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0,1,len(Times)))




for j,t in enumerate(Times):
    Re_query = []
    Im_query= []
    I_query = []
    freq_query = []
    
    t_query = np.full(len(files), t)
    # for i,file in enumerate(files):

    #     header_lines = get_header_lines(file)
 
    #     with open(file, encoding='latin1') as f:
    #         data = np.loadtxt(f,skiprows = header_lines)
    #     Re = data[:,1]
    #     Im = data[:,2]
    #     freq = data[:,0]
    #     time = data[:,5]
    #     I = data[:,7]
    #     # plt.plot(time-time[0],I)
    #     # print(j)
    
    #     Re_query.append(np.interp(t, time-time[0], Re))
    #     Im_query.append(np.interp(t, time-time[0], Im))
    #     I_query.append(np.interp(t, time-time[0], I))
    #     freq_query.append(np.interp(t, time-time[0], freq))
        
        
    
        
    
    # plt.plot(Re_query,Im_query,'x',c=colors[j])
    
    # np.savetxt('C:/Users/Herman/Desktop/WE/PEM-WE measurements/209/loopeis/1.6 V/time '+str(round(t,2))+' s.txt',np.array([freq_query,Re_query,Im_query,t_query,V_query,I_query]   ).T,delimiter='\t',newline='\n', header='freq/Hz	Re(Z)/Ohm	-Im(Z)/Ohm	time/s	<E>/V	<I>/mA')
     
plt.show()

cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0,1,len(files)))
    
for i,file in enumerate(files):
    header_lines = get_header_lines(file)
    
     
    with open(file, encoding='latin1') as f:
        data = np.loadtxt(f,skiprows = header_lines)
    Re = data[:,1]
    Im = data[:,2]
    freq = data[:,0]
    time = data[:,5]
    I = data[:,7]
    plt.plot(time-time[0],I,c=colors[i])
    