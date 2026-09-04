# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 22:45:58 2024

@author: Herman
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:58:50 2024

@author: Herman
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy.optimize import curve_fit

MEA= glob(os.path.join('//ELECTROLYZER/PEM-WE_measurements/2025/'+'*O*'))
# MEA= glob(os.path.join('//NOTEB-NTB1-PEM/PEMWE research/2024_new_naming/'+'*O*'))

# print(MEA[0])

for mea in MEA:
    # print(mea)

    
    file_SV = glob(os.path.join(mea,'*SV*'+'*.mpt'))
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0,1,len(file_SV))) 
    
    maximum = 0
    for i,file in enumerate(file_SV):
        try:
            
            # print(file_SV)
            
                with open(file, encoding='latin1') as f:
                    data = np.loadtxt(f,skiprows = 75)
                    
                # print(data)
                plt.plot(data[:,7],data[:,8],c = colors[i])
                if max(data[:,8]) > maximum:
                    maximum = max(data[:,8])
                
            
        except:
            pass
    plt.title(mea[40:])
    
    plt.savefig('C:/Users/Herman/Desktop/WE/IrOx series/'+mea[40:]+'.png',dpi = 100, format = 'png', bbox_inches = 'tight')

    
    plt.show()
    print(maximum)
    print(mea[40:])
    
            
    # with open(file_SV[1],'r', encoding="latin-1" ) as my_file:
    #     for line in my_file:
    #         print(line)
    # # print(data)

# path = "\\\\ELECTROLYZER\\PEM-WE_measurements\\2025\\129_III_III_IrOx Ar14 O1, etched, Ir 25nm, BDC608\\III_Day3_procedure1_03_SV_C01.mpt"
# # os.startfile(path) 
# # with open(, "r", encoding="utf-8") as f:
# #     content = f.read()

# # print(content)
# print("Exists?", os.path.exists(path))
# print("Is file?", os.path.isfile(path))

# import traceback



# with open(path, "r", encoding="latin-1") as f:
#     print(f.read())
