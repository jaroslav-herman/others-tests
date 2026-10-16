import wepy.basics as we
import matplotlib.pyplot as plt
import numpy as np
files = we.load_files(r'\\ELECTROLYZER\PEM-WE_measurements\2026\480_VIII_cathode_etching_series_210min_more_C\TRGEIS\no booster/','_GEIS',extension='.txt')
print(files)
for file in files:
    print(file)
    #file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\480_VIII_cathode_etching_series_210min_more_C\TRGEIS\no booster\trgeis_at_0,9A_for_20s_and_10s_OCV4_GEIS.txt"
    data = we.read_file(file, skiprows=0, delimiter="\t")

    colors = we.get_colors(len(np.unique(data["Tech"])))
    groups = data.groupby("Tech")
    for i, (tech, group) in enumerate(groups):
        plt.plot(group['time/s'], group['Ewe/V'], color=colors[i], label=f'Tech {tech}')
    plt.xlabel('Time (s)')
    plt.ylabel('Ewe (V)')
    plt.show()