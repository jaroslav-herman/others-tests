# %%
import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')
import wepy.iv_curve as weiv

# %%

data = we.read_file(r"\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE\358_VI_VI AEA SinoTech, piperion plain, GDE, short activation\VI_Day2_Procedure4_03_SV_C01.mpt")
E,I = weiv.IV_curves_data(data,4.84)
plt.plot(E[0],I[0])
plt.xlabel('E (V)')
plt.ylabel('I (mA/cm2)')
plt.title('Sinotech AEA')
plt.grid()
plt.show()

# %%
