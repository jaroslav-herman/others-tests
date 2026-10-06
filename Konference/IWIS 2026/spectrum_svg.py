import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import wepy.basics as we

file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\455_IV_cathode_etching_series_35min\IV_Day5_Procedure1_05_PEIS_C02.mpr"
file = r"C:\Users\Herman\OneDrive - Univerzita Karlova\katodovy oblouk\Data z PEMWE\453_VIII_cathode_etching_series_140min\VIII_Day3_Procedure1_05_PEIS_C01.mpr"
data = we.read_file(file)

print(data.columns)
cycle_data = data[data["cycle number"] == 16]
cycle_f_data = cycle_data[(cycle_data["freq/Hz"] < 30000) & (cycle_data["freq/Hz"] > 8)]
x = cycle_f_data["Re(Zwe-ce)/Ohm"].to_numpy()
y = cycle_f_data["-Im(Zwe-ce)/Ohm"].to_numpy()
window_length = min(11, len(x) if len(x) % 2 else len(x) - 1)
if window_length >= 5:
    x = savgol_filter(x, window_length=window_length, polyorder=3)
    y = savgol_filter(y, window_length=window_length, polyorder=3)
plt.plot(x, y, lw = 5, c="black")
plt.gca().set_aspect("equal")
plt.axis("off")
fig = plt.gcf()
fig.patch.set_alpha(0)
plt.savefig(
    r"C:\Users\Herman\OneDrive - Univerzita Karlova\Konference\IWIS\IWIS 2026/spectrum.svg",
    dpi=500,
)
plt.show()
