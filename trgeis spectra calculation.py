import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import wepy.basics as we
# import wepy.loopeis as loop
import wepy.eis as weis
import pandas as pd
from impedance.models.circuits import CustomCircuit
from natsort import natsorted



time_start = 0.1
time_end = 500.1
points = 501

specta_all = []
Times = np.linspace(time_start,time_end,points)
colors = we.get_colors(len(Times))

spectra = []
plot = True

file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\480_VIII_cathode_etching_series_210min_more_C\TRGEIS\booster back\trgeis_at_-0,242A_for_599s_and_0s_OCV_GEIS.txt"
with open(file,'r', encoding='latin1') as f:
    df = pd.read_csv(f,skiprows = 0,delimiter = r'\t')

for j,t in enumerate(Times):
    Re_query = []
    Im_query= []
    I_query = []
    E_query = []
    freq_query = []
    t_query = np.full(len(np.unique(df['freq/Hz'])[1:]), t)
    # t_query = np.full(len(np.unique(df['Freq'])[1:]), t)
    # for f in np.unique(df['Freq'])[1:]:
    for f in np.unique(df['freq/Hz'])[1:]:
        Re = df.loc[(df["freq/Hz"] == f), 'Re(Z)/Ohm'].values
        Im = -df.loc[(df["freq/Hz"] == f), '-Im(Z)/Ohm'].values
        time = df.loc[(df["freq/Hz"] == f), 'time/s'].values
        I = df.loc[(df["freq/Hz"] == f), 'I/mA'].values
        E = df.loc[(df["freq/Hz"] == f), 'Ewe/V'].values
        freq = df.loc[(df["freq/Hz"] == f), 'freq/Hz'].values

        Re_query.append(np.interp(t, time-time[0], Re))
        Im_query.append(-np.interp(t, time-time[0], Im))
        I_query.append(np.interp(t, time-time[0], I))
        freq_query.append(np.interp(t, time-time[0], freq))
        E_query.append(np.interp(t, time-time[0], E))
    spectra.append([freq_query,np.array(Re_query) - 1j*np.array(Im_query),np.array(I_query),np.array(E_query)])      
    if plot:
        plt.plot(Re_query,np.array(Im_query),'-',c=colors[j])





if plot: 
    plt.xlim(0,0.3)
    plt.ylim(-0.1,0.3)
    plt.gca().set_aspect('equal')
    plt.show()


# Save all spectra to a single file
records = []
for j, t in enumerate(Times):
    freq_arr, Z_arr, I_arr, E_arr = spectra[j]
    for k in range(len(freq_arr)):
        records.append({
            'time/s': t,
            'freq/Hz': freq_arr[k],
            'Re(Z)/Ohm': Z_arr[k].real,
            '-Im(Z)/Ohm': -Z_arr[k].imag,
            '<I>/mA': I_arr[k],
            '<Ewe>/V': E_arr[k],
            'cycle number': j + 1
        })

out_df = pd.DataFrame(records)
out_file = os.path.splitext(file)[0] + '_spectra_all_times.csv'
out_df.to_csv(out_file, index=False, sep='\t')
print(f"Saved {len(out_df)} rows to {out_file}")