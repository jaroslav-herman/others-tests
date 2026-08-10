

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.optimize import curve_fit


from matplotlib.ticker import LogLocator, ScalarFormatter, FuncFormatter

# Custom formatter function to specify decimal places
def format_with_precision(x, pos):
    return f"{x:.3f}"  # Change 3 to desired decimal places

def custom_format(x, pos):
    """Format ticks with specified values and no trailing zeros."""
    if x >= 1:
        return f"{int(x)}" if x.is_integer() else f"{x:.1f}".rstrip('0').rstrip('.')
    else:
        return f"{x:.10f}".rstrip('0').rstrip('.')  # For less than 1, use 2 decimal places

s225 = {'name':'225', "comp" : "15Ar"} # hotovo
s02 = {'name':'02', "comp" : "20O2"} # hotovo
s03 = {'name':'03', "comp" : "0.1O2,15Ar"} # hotovo
s04 = {'name':'04', "comp" : "15O2,3Ar"} # 
s05 = {'name':'05', "comp" : "0.2O2,15Ar"} # hotovo
s06 = {'name':'06', "comp" : "0.5O2,14.5Ar"} # hotovo
s07 = {'name':'07', "comp" : "5.0O2,10Ar"} # hotovo
s08 = {'name':'08', "comp" : "1.0O2,14Ar"} # hotovo
s09 = {'name':'09', "comp" : "2.0O2,13Ar"} # hotovo
# s091,s093,s103,s107,s120,s034


def load_MEA_params(MEA):
    for mea in MEA:
        mea['DRT_files'] = glob(os.path.join('C:/Users/Herman/Desktop/WE/PEM-WE measurements/225_V_V IrOx series 0.1 sccm O2 15 sccm Ar','*.csv'))
        
        # mea['cycles'] = len(mea['PEIS_files'])
        cycles = 15 # mea['cycles']
        print(cycles)
    
        mea['U'] = np.empty((cycles,24))
        mea['I'] = np.empty((cycles,24)) # Ohmic resistance
        mea['R1'] = np.empty((cycles,24)) # Ohmic resistance error
        mea['R2'] = np.empty((cycles,24)) #  resistance of inductance element
        mea['R3'] = np.empty((cycles,24)) # resistance of inductance element error
        mea['t1'] = np.empty((cycles,24)) # Ohmic resistance error
        mea['t2'] = np.empty((cycles,24)) #  resistance of inductance element
        mea['t3'] = np.empty((cycles,24)) # resistance of inductance element error

    
    
        for cycle, file in enumerate(mea['DRT_files']):  
            with open(file, encoding='latin1') as f:
                # print(f)
                d = np.loadtxt(f,delimiter =',',skiprows = 1)
            mea['U'][cycle] = d[:,1]
            mea['I'][cycle] = d[:,2]
            mea['R1'][cycle] = d[:,9]
            mea['R2'][cycle] = d[:,14]
            mea['R3'][cycle] = d[:,19]
            mea['t1'][cycle] = d[:,5]
            mea['t2'][cycle] = d[:,10]
            mea['t3'][cycle] = d[:,15]






# MEA = [s029,s041,s047,s044,s046,s055,s065,s076,s049,s050,s079,s092,s094,s096,s097,s098,s111,s109]
MEA = [s225]
load_MEA_params(MEA)
for mea in MEA:
   
    cmap = plt.get_cmap('rainbow')
    # colors = cmap(np.linspace(0,1,len(mea['DRT_files']))) 
    colors = cmap(np.linspace(0,1,24)) 
    for cycle in range(0,24):
        plt.plot(mea['U'][:,cycle],mea['R1'][:,cycle],'-',c=colors[cycle])
    plt.ylim(0,0.01)


    #         # I = data[cycle,2]
    #         # t2 = data[cycle,10]
    #         # p2 = data[cycle,14]
    #         # print(p2)
            
    #         # p1=np.array(p1)
    #         # p2=np.array(p2)
    #         # t2=np.array(t2)
    #         # plt.plot(I,t2/p2,'x',c=colors[cycle])
    #         # plt.plot(p2,label = str(file[-12:-6]),c = colors[cycle])
    #     # plt.show()
    #         # 
    #     # plt.ylim(0,400)
    #     # plt.legend()
    #     # plt.show()
    #     # print(U)
        
    #     I=np.array(I)
    #     # plt.plot(I[0:9],1/p2[0:9],'x',c='red')
    #     # k,q = np.polyfit(I[0:9],1/p2[0:9],1)
    #     # plt.plot(I[0:9],k*I[0:9]+q)
    #     # print(2.3/k,q)
        
        
    #     plt.plot(I,1/(p3),'x',c='blue')
    #     plt.plot(I,1/(p3+p2),'x',c='red')
    #     # k,q = np.polyfit(I,1/(p3+p1),1)
    #     # plt.plot(I,k*I+q)
    #     # print(2.3/k)
    #     # plt.plot(I[0:8],1/(p3[0:8]),'x')
    #     # plt.ylim(0,250)
    #     # plt.xlim(0,2000)
    #     # plt.plot(I,1/(p2+p1),'x')
    #     # plt.plot(I[0:7],np.array(p2[0:7])+np.array(p1[0:7]))
    # plt.show()
    # # plt.xlim(0,8000)
    # # plt.ylim(0,0.1)