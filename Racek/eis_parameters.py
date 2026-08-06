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

# files = [
    
#      r"\\ELECTROLYZER\PEM-WE_measurements\2026\323_VII_VII Racek_2026_4, 14Ar1O2, GDE\323_exported.csv",
#      r"\\electrolyzer\PEM-WE_measurements\2024_new_naming\088_I_I_Ir_14_to_1_plain_etched_Ti\088_exported.csv",
#      r"\\ELECTROLYZER\PEM-WE_measurements\2026\319_VII_VII Racek_2026_3, 14Ar1O2, GDE\319_exported.csv",
#      r"\\ELECTROLYZER\PEM-WE_measurements\2024_new_naming\101_IV_IV_Ir_14_to_1_plain_racek_Ti-4\101_exported.csv",
#     r"\\electrolyzer\PEM-WE_measurements\2024_new_naming\087_IV_IV_Ir_14_to_1_plain\087_exported.csv",
#     r"\\ELECTROLYZER\PEM-WE_measurements\2024_new_naming\090_I_I_Ir_14_to_1_plain_Ti-2\090_exported.csv",
#          r"\\ELECTROLYZER\PEM-WE_measurements\2024_new_naming\099_IV_IV_Ir_14_to_1_plain_racek_Ti-3\099_exported.csv",
#          r"\\ELECTROLYZER\PEM-WE_measurements\2026\317_IV_IV Racek_2026_1, 14Ar1O2, GDE\317_exported.csv",
#         r"\\ELECTROLYZER\PEM-WE_measurements\2026\318_V_V Racek_2026_2, 14Ar1O2, GDE\318_exported.csv",
#          r"\\ELECTROLYZER\PEM-WE_measurements\2024_new_naming\089_IV_IV_Ir_14_to_1_plain_Ti-1\089_exported.csv",
         
#         ]

files = [
             r"C:\Users\Herman\OneDrive - Univerzita Karlova\Racek\087_exported.csv",
        ]
for i in range(18):
    colors = we.get_colors(len(files))
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    for file,c in zip(files, colors):
        data = we.read_file(file, skiprows=0, delimiter=',')
        data = data.sort_values(['Cycle mod 15', 'Time'])
        groups = data.groupby('Time', sort=True)
        # print(data.columns)
        for cycle, group in groups:
            
            if cycle == i:

                ax[0].plot(group['Ecell_V'], group['R2'],'-', c=c, )
                ax[1].plot(group['I_mA'], group['Ecell_V'],'-',c=c,label=file.split('\\')[-2])
                # Find index of Ecell_V closest to 1.6
                # idx_closest = (group['Ecell_V'] - 1.6).abs().idxmin()
                # print(group.loc[idx_closest, 'I_mA'])
        ax[0].set_yscale('log')
        # ax[0].set_xscale('log')
    # plt.ylim(0,25)
    # plt.xlim(0,500)
    ax[0].set_xlabel('Current (mA)')
    ax[0].set_ylabel('R1 (Ohm)')
    ax[1].set_xlabel('Current (mA)')
    ax[1].set_ylabel('Cell Voltage (V)')
    ax[1].legend()
    plt.show()
# %%
