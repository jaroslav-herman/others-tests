import wepy.basics as we
import matplotlib.pyplot as plt

CV_const = r"\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode_etching_series_GDE\interruptions\inter_50_sccm_H2_CVs_2,0V_2_05_CV_C02.mpr"
CV_inter = r"\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode_etching_series_GDE\interruptions\inter_50_sccm_H2_CVs_2,0V_2_11_CV_C02.mpr"



data_const = we.read_file(CV_const, error_on_unknown_column=False)
data_inter = we.read_file(CV_inter, error_on_unknown_column=False)

datas = [data_const,data_inter]

for i in range(0,10):
    for data in datas:
        data = data[data['cycle number'] == i]
        plt.plot(data['Ewe/V']-data['Ece/V'],data['<I>/mA'])
    plt.show()
