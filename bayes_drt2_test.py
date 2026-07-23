from bayes_drt2.inversion import Inverter
import wepy.basics as we
import numpy as np
import wepy.eis as weis
import matplotlib.pyplot as plt
import elchem.pemwe as pemwe


data = pemwe.read_file(r"\\ELECTROLYZER\PEM-WE_measurements\2026\430_VIII_VIII_IrOx(6,15)_60min50WTiCoverlayer,refel_N115_etchedcathode\Different H2 flows II\PEIS_at_N2_flow_80_sccm_automated_01_PEIS.mpt")
print(data.columns)
f,Z,E,I = weis.freq_and_Z(data,9,[0,15000],control = 'Ewe-ce')


plt.plot(Z.real,-Z.imag)
f,Z = weis.remove_outliers(f,Z,threshold=1)
plt.plot(Z.real,-Z.imag)

plt.show()
# Plot results
# axes = inv.plot_full_results()