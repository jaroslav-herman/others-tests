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



CV_N2 = we.read_file(r"\\ELECTROLYZER\PEM-WE_measurements\2026\439_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_etchedcathode_BDC929\PEISes\Procedure1_N2_2_02_CV_C01.mpt")
CV_H2 = we.read_file(r"\\ELECTROLYZER\PEM-WE_measurements\2026\439_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_etchedcathode_BDC929\PEISes\Procedure1_H2_from_I_2_02_CV_C01.mpt")
CV_H2_real = we.read_file(r"\\ELECTROLYZER\PEM-WE_measurements\2026\439_III_III_IrOxonPTL_150nm_Pt_500ug_Pressures_N115_etchedcathode_BDC929\PEISes\Procedure1_H2_from_I_3_CV_C01.mpt")
files = [CV_H2,CV_N2,CV_H2_real]
# %%
for data in files:
    E = data.loc[data['cycle number'] == 2, 'Ewe/V']
    I = data.loc[data['cycle number'] == 2, '<I>/mA']
    plt.plot(E,I)
plt.ylim(-30,50)
plt.show()

# %%