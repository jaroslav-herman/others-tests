import wepy.basics as we
import matplotlib.pyplot as plt
import wepy.iv_curve as weiv

files = we.load_files(r'\\ELECTROLYZER\PEM-WE_measurements\2026\476_VII_cathode_etching_series_70min','SV','.mpt', natural_sort=True)
print(files[:-3])
Es, Is = [],[]
for file in files:
    data = we.read_file(file)
    Eds, Ids = weiv.IV_curves_data(data)
    for E, I in zip(Eds,Ids):
        Es.append(E)
        Is.append(I)
colors = we.get_colors(len(Es))
for E, I, color in zip(Es,Is,colors):
    plt.plot(E,I, c = color)
plt.show()
