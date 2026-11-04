import wepy.basics as we
import numpy as np
import matplotlib.pyplot as plt
import wepy.iv_curve as weiv

files = we.load_files(r'\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE\499_V_Ni90nm(100)_on_etched_piperion_Piperion60_RDC1225_none', 'SV', '.mpr')

Es, Is = [],[]
fig, ax = plt.subplots()
for file in files[:6]:
    print(file)
  #  try:
    data = we.read_file_safe(file)
    Esx, Isx  = weiv.IV_curves_data(data, 4.84)
    print(len(Esx), len(Isx))

    for E,I in zip(Esx, Isx):
       # plt.plot(I,E)
       # plt.show()
        Es.append(E)
        Is.append(I)
  #  except:
  #      pass
colors = we.get_colors(len(Es))
for E,I,c in zip(Es, Is,colors):
    ax.plot(I, E,c=c)

ax.set_ylabel('Cell voltage (V)')
ax.set_xlabel(r'Current density $\mathrm{(mA cm^{-2})}$')
ax.set_ylim(1.4,2.02)

# Inset for current at E=1.7V per cycle
inset_rect = [0.57, 0.23, 0.3, 0.25]  # [left, bottom, width, height] in figure coords
inset_ax = fig.add_axes(inset_rect)

# Extract current at E=1.7V for each cycle
E_target = 1.7
I_at_E = []
for E, I in zip(Es, Is):
    # Find the index where E is closest to E_target
    idx = np.argmin(np.abs(E - E_target))
    I_at_E.append(I[idx])

# Plot current evolution
inset_ax.plot(range(1, len(I_at_E)+1), I_at_E, 'k-')
inset_ax.set_xlabel('Cycle', size = 15)
inset_ax.set_ylabel('Current (mA cm$^{-2}$)', size = 15)
inset_ax.set_title('Current at 1.7 V', size = 15)

# colorbar position/size in figure coordinates: [left, bottom, width, height]
cbar_rect = [0.25, 0.2, 0.2, 0.03]
fig = plt.gcf()
cbar_ax = fig.add_axes(cbar_rect)
sm = plt.cm.ScalarMappable(cmap='rainbow', norm=plt.Normalize(vmin=1, vmax=len(Es)))
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Cycle', labelpad=-10,size = 15)
cbar.set_ticks([1, len(Es)])

ax.set_title('Degradation IV Curves')
plt.savefig(r'C:\Users\Herman\Python\WE\others-tests\Konference\Malta\degradation_iv_curves.png', dpi=300)
plt.show()