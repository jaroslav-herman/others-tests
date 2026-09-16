import wepy.basics as we
import wepy.iv_curve as weiv
import matplotlib.pyplot as plt

folders = we.load_folders(r"\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE", 'NiAl')

for folder in folders:
    file = we.load_files(folder, ['Day2','SV'])[0]
    data = we.read_file(file)
    voltages, currents = weiv.IV_curves_data(data,norm = 4.84)
    plt.plot(currents[0], voltages[0], label = we.get_sample_number(file))
plt.xlabel(r'Current Density ($\mathrm{mA~cm^{-2}}$)')
plt.ylabel(r'Cell Voltage (V)')
plt.legend(title = 'Sample Number')
# plt.savefig(r'C:\Users\Herman\Desktop\MFF\PhD\Škola\summer seminar 2026/AEM Iv curves for NiFe.png', dpi = 500)
plt.show()


