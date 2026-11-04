import numpy as np
import pandas as pd
import wepy.basics as we
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def interpolate_by_frequency(data, time_step=0.2):
    value_columns = ['Ewe/V', 'I/mA', 'Re(Z)/Ohm', '-Im(Z)/Ohm']
    required_columns = ['time/s', 'freq/Hz', *value_columns]
    samples = data[required_columns].dropna().copy()
    if samples.empty:
        raise ValueError('No complete EIS measurements available for interpolation.')
    if time_step <= 0:
        raise ValueError('time_step must be greater than zero.')

    start_time = samples['time/s'].min()
    end_time = samples['time/s'].max()
    n_steps = int(np.floor((end_time - start_time) / time_step + 1e-9))
    time_grid = start_time + np.arange(n_steps + 1) * time_step

    interpolated_groups = []
    for frequency, group in samples.groupby('freq/Hz', sort=True):
        group = (
            group.groupby('time/s', as_index=False)[value_columns]
            .mean()
            .sort_values('time/s')
        )
        observed_times = group['time/s'].to_numpy()
        valid_times = time_grid[
            (time_grid >= observed_times[0]) & (time_grid <= observed_times[-1])
        ]
        if valid_times.size == 0:
            continue

        interpolated_group = pd.DataFrame({
            'time/s': valid_times,
            'freq/Hz': frequency,
        })
        for column in value_columns:
            interpolated_group[column] = np.interp(
                valid_times, observed_times, group[column].to_numpy()
            )
        interpolated_groups.append(interpolated_group)

    if not interpolated_groups:
        raise ValueError(
            f'No frequency has measurements spanning a {time_step:g} s time grid.'
        )

    result = (
        pd.concat(interpolated_groups, ignore_index=True)
        .sort_values(['time/s', 'freq/Hz'])
        .reset_index(drop=True)
    )
    result.insert(
        0,
        'cycle number',
        pd.factorize(result['time/s'], sort=True)[0] + 1,
    )
    return result

file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\AEM-WE\522_VI_Ni_Bekaert_PTL_no_sputtering_EIS_tests\CV-EIS\CVEIS_1,2-1,5V_20mVs_corrected_2_PEIS.txt"
data = we.read_file(file, skiprows=0)
#print(data)
groups = data.groupby('Tech')
colors = we.get_colors(len(groups))
for index,group in groups:
    plt.plot(group['Ewe/V'],group['I/mA'],c=colors[index],label=index)
plt.legend()
plt.show()

for index,group in groups:
    plt.plot(group['time/s'],group['Ewe/V'],c=colors[index],label=index)
plt.legend()
plt.show()


interpolated_data = interpolate_by_frequency(data)
interpolated_data.to_csv(file[:-4] + '_processed.csv', index=False)

fig, ax = plt.subplots()
time_values = interpolated_data['time/s']
cmap = plt.get_cmap('rainbow')
norm = Normalize(vmin=time_values.min(), vmax=time_values.max())
for time, spectrum in interpolated_data.groupby('time/s', sort=True):
    spectrum = spectrum.sort_values('freq/Hz', ascending=False)
    ax.plot(
        spectrum['Re(Z)/Ohm'],
        spectrum['-Im(Z)/Ohm'],
        color=cmap(norm(time)),
    )
ax.set_xlabel('Re(Z)/Ohm')
ax.set_ylabel('-Im(Z)/Ohm')
ax.set_aspect('equal')
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=ax,
    label='time/s',
)
plt.show()