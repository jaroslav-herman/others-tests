# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:51:43 2025

@author: Herman
"""



import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import matplotlib
#Locale settings
import locale
# Set to German locale to get comma decimal separater
# locale.setlocale(locale.LC_NUMERIC, "cs_CZ")


plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.use_locale'] = True

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": "Times New Roman"
})


file = r"C:\Users\Herman\Python\WE\others-tests\Konference\Malta\Nyquist_AEM_example_data.csv"

# colors = ['orange','darkorange','red','darkred']


fig, ax = plt.subplots()

# Load data from CSV file and keep only the active (Included) Nyquist points
import pandas as pd
data = pd.read_csv(file)
active = data[data['Series'] == 'Included'].reset_index(drop=True)

Zre = 1000 * active['Re(Z) / Ohm'][::2]
Zim = 1000 * active['-Im(Z) / Ohm'][::2]

# Active points span log-spaced from 40 kHz (first point) down to 1 Hz (last point)
freq = np.logspace(np.log10(40000), np.log10(1), len(Zre))

# Plot Nyquist plot with points colored by frequency
# Fixed color scale so the inset uses the same colors as the main plot
logf = np.log10(freq)
vmin, vmax = logf.min(), logf.max()
im = ax.scatter(x=Zre, y=Zim, marker='x', c=logf, cmap='rainbow', vmin=vmin, vmax=vmax)

plt.xlabel(r'$Z_{\mathrm{Re}} ~ \mathrm{\left( m\Omega \right) }$',size=16)
plt.ylabel(r'$-Z_{\mathrm{Im}} ~ \mathrm{\left( m\Omega \right) }$',size=16,labelpad = -5)

plt.xticks(size=14)
plt.yticks(size=14)
plt.axhline(0, ls='--', c='black', lw=0.5)
plt.axvline(0, ls='--', c='black', lw=0.5)
plt.xlim(-10, 280)
plt.ylim(-30, 120)
plt.gca().set_aspect('equal')
#plt.text(20,-20,"Ohmic \n resistance",size = 15)
#plt.text(60,-25,"Inductive part",size = 15)
#plt.text(230,60,"Anodic \n reaction",size = 15)
#plt.text(30,40,"Cathodic \n reaction",size = 15)

cbar_ax = fig.add_axes([0.70, 0.32, 0.13, 0.03])
cbar = fig.colorbar(im, orientation="horizontal", ticks=[0, 2, 4.6], cax=cbar_ax)
cbar.ax.set_title("Frequency", fontsize=12)
cbar.set_ticks([0, 2, 4.6])
cbar.ax.set_xticklabels(["1 Hz", "100 Hz", "40 kHz"], fontsize=10)

for label, color in zip(cbar.ax.get_xticklabels(), [ 'indigo','lime','red']):
    label.set_color(color)

# Inset with the 20 highest-frequency points in more detail
N_HF = 20
hf = active.iloc[6:35][::2]
hf_re = 1000 * hf['Re(Z) / Ohm']
hf_im = 1000 * hf['-Im(Z) / Ohm']
hf_freq = np.logspace(np.log10(40000), np.log10(1), len(active))[6:35][::2]

axins = ax.inset_axes([0.20, 0.22, 0.72, 0.60])
axins.scatter(x=hf_re, y=hf_im, marker='x', c=np.log10(hf_freq), cmap='rainbow', vmin=vmin, vmax=vmax)
axins.axhline(0, ls='--', c='black', lw=0.5)
axins.set_aspect('equal')
# Leave a gap between the points and the left/right borders of the inset
x_pad = 0.08 * (hf_re.max() - hf_re.min())
axins.set_xlim(hf_re.min() - x_pad, hf_re.max() + x_pad)
axins.set_xticks([])
axins.set_yticks([])

ax.indicate_inset_zoom(axins, edgecolor='black', alpha=0.6)

plt.savefig(r'C:\Users\Herman\Python\WE\others-tests\Konference\Malta\PEIS_spectrum_example.png', dpi=1000, format='png', bbox_inches='tight')

plt.show()
