import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
file_N2 = r"\\ELECTROLYZER\PEM-WE_measurements\2026\372_IV_IV IrOx, 5 min Pt, 30 min C, etched cathode\new station\SV N2.mpt"
file_H2 = r"\\ELECTROLYZER\PEM-WE_measurements\2026\372_IV_IV IrOx, 5 min Pt, 30 min C, etched cathode\new station\SV H2.mpt"
file_classic = r"\\ELECTROLYZER\PEM-WE_measurements\2026\372_IV_IV IrOx, 5 min Pt, 30 min C, etched cathode\new station\SV klasik_3.mpt"

df_N2 = we.read_file(file_N2)
df_H2 = we.read_file(file_H2)
df_classic = we.read_file(file_classic)
colors = ['green','black','red']

for df,label,color in zip([df_N2,df_classic,df_H2],['\mathrm{$N_2$} flushing','no flushing','$H_2$ flushing'],colors):
    unique_vals = np.unique(df["cycle number"].values)

    for val in unique_vals:
        Ecell = (df.loc[(df["cycle number"] == val), 'control/V'].values)
        mask = np.diff(Ecell) > 0
        mask = np.append(mask,False)

        I = (df.loc[(df["cycle number"] == val), '<I>/mA'].values)[mask]/4.84
        Ecell = Ecell[mask]
        plt.plot(I,Ecell,label = label,c=color)
plt.xlabel('Current Density (mA/cm²)', fontsize=14)
plt.ylabel('Ecell (V)', fontsize=14)
plt.title('Polarization Curves', fontsize=16)
plt.legend(loc = 'center left',fontsize=12)
plt.ylim(1.28,1.62)
# plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.8)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
#plt.savefig(r"C:\Users\Herman\Desktop\WE\Konference\WDS/compare_iv_gases.png",dpi = 500, format = 'png', bbox_inches = 'tight')
plt.show()