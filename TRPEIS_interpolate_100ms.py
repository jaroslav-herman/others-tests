import numpy as np
import wepy.basics as we
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
import os

# Disable LaTeX to avoid rendering issues
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

# Input file path
input_file = r"\\ELECTROLYZER\PEM-WE_measurements\2026\518_VII_Pt+200_nm_cathode\TR-PEIS\trpeis_at_1,46V_for_30s_and_60s_OCV_PEIS.txt"

# Output directory (local)
output_dir = r"C:\Users\Herman\Python\others-tests\TRPEIS_output"
os.makedirs(output_dir, exist_ok=True)

# Read the data
print("Reading data file...")
data = we.read_file(input_file, skiprows=0)

print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Understand the data structure
# The file contains TR-PEIS measurements with columns:
# time/s, Ewe/V, Ece/V, I/mA, freq/Hz, Re(Z)/Ohm, -Im(Z)/Ohm, Re(Zce)/Ohm, -Im(Zce)/Ohm, Loop, Tech, startTime/s

# Get unique frequencies and times
unique_freqs = np.sort(data['freq/Hz'].unique())
unique_times = np.sort(data['time/s'].unique())

print(f"\nUnique frequencies: {len(unique_freqs)} frequencies")
print(f"Frequency range: {unique_freqs.min():.2f} - {unique_freqs.max():.2f} Hz")
print(f"\nUnique times: {len(unique_times)} time points")
print(f"Time range: {unique_times.min():.2f} - {unique_times.max():.2f} s")

# The TR-PEIS procedure:
# Time-resolved EIS measures impedance at multiple frequencies at different times
# Each frequency is measured at multiple time points
# We need to interpolate to regular 100 ms intervals

# Create time grid with 100 ms spacing
time_min = data['time/s'].min()
time_max = data['time/s'].max()
time_grid = np.arange(np.floor(time_min * 10) / 10, np.ceil(time_max * 10) / 10 + 0.001, 0.1)

print(f"\nCreating time grid from {time_min:.2f} to {time_max:.2f} s with 100 ms spacing")
print(f"Number of interpolated time points: {len(time_grid)}")

# For TR-PEIS, we need to handle each frequency separately
# For each frequency, we interpolate across all time points

print("\nInterpolating impedance data for each frequency...")

# Dictionary to store interpolated data for each frequency
freq_data_dict = {}

for freq in unique_freqs:
    freq_data = data[data['freq/Hz'] == freq].sort_values('time/s')
    
    if len(freq_data) < 2:
        print(f"Warning: Only {len(freq_data)} data points for frequency {freq:.2f} Hz, skipping")
        continue
    
    times = freq_data['time/s'].values
    re_z = freq_data['Re(Z)/Ohm'].values
    im_z = freq_data['-Im(Z)/Ohm'].values
    ewe = freq_data['Ewe/V'].values
    current = freq_data['I/mA'].values
    
    # Linear interpolation with extrapolation
    interp_re = interp1d(times, re_z, kind='linear', fill_value='extrapolate')
    interp_im = interp1d(times, im_z, kind='linear', fill_value='extrapolate')
    interp_ewe = interp1d(times, ewe, kind='linear', fill_value='extrapolate')
    interp_i = interp1d(times, current, kind='linear', fill_value='extrapolate')
    
    # Interpolate at each time point in our grid
    re_interp = interp_re(time_grid)
    im_interp = interp_im(time_grid)
    ewe_interp = interp_ewe(time_grid)
    i_interp = interp_i(time_grid)
    
    # Store the interpolated data for this frequency
    freq_data_dict[freq] = {
        'time/s': time_grid,
        'Re(Z)/Ohm': re_interp,
        '-Im(Z)/Ohm': im_interp,
        'freq/Hz': np.full_like(time_grid, freq),
        'Ewe/V': ewe_interp,
        'I/mA': i_interp
    }

# Now create a DataFrame with all interpolated data
print("\nCreating DataFrame from interpolated data...")

# Collect all data
all_time = []
all_re = []
all_im = []
all_freq = []
all_ewe = []
all_i = []

for freq in freq_data_dict:
    fd = freq_data_dict[freq]
    all_time.extend(fd['time/s'])
    all_re.extend(fd['Re(Z)/Ohm'])
    all_im.extend(fd['-Im(Z)/Ohm'])
    all_freq.extend(fd['freq/Hz'])
    all_ewe.extend(fd['Ewe/V'])
    all_i.extend(fd['I/mA'])

df_interp = pd.DataFrame({
    'time/s': np.array(all_time),
    'Re(Z)/Ohm': np.array(all_re),
    '-Im(Z)/Ohm': np.array(all_im),
    'freq/Hz': np.array(all_freq),
    'Ewe/V': np.array(all_ewe),
    'I/mA': np.array(all_i)
})

print(f"Interpolated DataFrame shape: {df_interp.shape}")
print(df_interp.head(10))

# Save the interpolated data to local directory
output_csv_base = os.path.basename(input_file).replace('.txt', '_interpolated_100ms.csv')
output_csv = os.path.join(output_dir, output_csv_base)
print(f"\nSaving interpolated data to: {output_csv}")
df_interp.to_csv(output_csv, index=False)

# Now generate impedance spectra (Nyquist plots) for selected time points
print("\nGenerating Nyquist plots for selected time points...")

# Select time points to plot (e.g., every 1 second)
plot_times = np.arange(np.floor(time_min), np.ceil(time_max) + 1, 1.0)

# Create color map for the time points
colors = we.get_colors(len(plot_times))

plt.figure(figsize=(12, 8))

for i, plot_time in enumerate(plot_times):
    # Find data at this time point (or closest)
    time_mask = np.abs(df_interp['time/s'] - plot_time) < 0.05  # 50 ms window
    time_data = df_interp[time_mask]
    
    if len(time_data) == 0:
        continue
    
    # Sort by frequency (descending for Nyquist plot - typically high freq on right)
    time_data = time_data.sort_values('freq/Hz', ascending=False)
    
    # Plot Nyquist curve for this time point
    label = f't={plot_time:.1f} s'
    plt.plot(time_data['Re(Z)/Ohm'], time_data['-Im(Z)/Ohm'], 
             'o-', markersize=4, color=colors[i], 
             label=label, linewidth=1.5, markerfacecolor='none', markeredgewidth=1)

plt.xlabel('Re(Z) / Ohm', fontsize=12)
plt.ylabel('-Im(Z) / Ohm', fontsize=12)
plt.title('Nyquist Plots - Time-Resolved PEIS (100 ms interpolation)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.gca().set_aspect('equal')
plt.grid(True, alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=10)

# Save the plot
output_plot_base = os.path.basename(input_file).replace('.txt', '_nyquist_interpolated.png')
output_plot = os.path.join(output_dir, output_plot_base)
print(f"Saving Nyquist plot to: {output_plot}")
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.close()

# Also create a plot showing the time evolution of impedance at a specific frequency
print("\nCreating time evolution plots...")

# Select a characteristic frequency (e.g., the most common or a mid-range one)
# Use a frequency in the middle of the range
selected_freq = np.median(unique_freqs)

# Find the closest frequency in our data
freq_idx = np.argmin(np.abs(unique_freqs - selected_freq))
selected_freq = unique_freqs[freq_idx]

freq_data = df_interp[df_interp['freq/Hz'] == selected_freq].sort_values('time/s')

if len(freq_data) == 0:
    # Try the first frequency
    selected_freq = unique_freqs[0]
    freq_data = df_interp[df_interp['freq/Hz'] == selected_freq].sort_values('time/s')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(freq_data['time/s'], freq_data['Re(Z)/Ohm'], 'b-', label=f'Re(Z) at {selected_freq:.2f} Hz', linewidth=1.5)
ax1.set_ylabel('Re(Z) / Ohm', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.tick_params(axis='both', which='major', labelsize=10)

ax2.plot(freq_data['time/s'], freq_data['-Im(Z)/Ohm'], 'r-', label=f'-Im(Z) at {selected_freq:.2f} Hz', linewidth=1.5)
ax2.set_ylabel('-Im(Z) / Ohm', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)

ax3.plot(freq_data['time/s'], freq_data['Ewe/V'], 'g-', label=f'Ewe at {selected_freq:.2f} Hz', linewidth=1.5)
ax3.set_ylabel('Ewe / V', fontsize=12)
ax3.set_xlabel('Time / s', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.tick_params(axis='both', which='major', labelsize=10)

plt.suptitle(f'Impedance and Potential Time Evolution at {selected_freq:.2f} Hz', fontsize=14)
plt.tight_layout()

output_time_plot_base = os.path.basename(input_file).replace('.txt', f'_time_evolution_{selected_freq:.2f}Hz.png')
output_time_plot = os.path.join(output_dir, output_time_plot_base)
print(f"Saving time evolution plot to: {output_time_plot}")
plt.savefig(output_time_plot, dpi=300, bbox_inches='tight')
plt.close()

# Create Bode plots (magnitude and phase vs frequency) for selected time points
print("\nCreating Bode plots...")

# Select a subset of time points for clearer visualization
bode_times = plot_times[::2]  # Every other time point
bode_colors = we.get_colors(len(bode_times))

plt.figure(figsize=(14, 6))

# Plot for magnitude
plt.subplot(1, 2, 1)
for i, plot_time in enumerate(bode_times):
    time_mask = np.abs(df_interp['time/s'] - plot_time) < 0.05
    time_data = df_interp[time_mask].sort_values('freq/Hz')
    
    if len(time_data) == 0:
        continue
    
    # Calculate magnitude
    magnitude = np.sqrt(time_data['Re(Z)/Ohm']**2 + time_data['-Im(Z)/Ohm']**2)
    
    label = f't={plot_time:.1f} s'
    plt.semilogx(time_data['freq/Hz'], magnitude, '-', color=bode_colors[i], 
                 label=label, linewidth=1.5)

plt.xlabel('Frequency / Hz', fontsize=12)
plt.ylabel('|Z| / Ohm', fontsize=12)
plt.title('Bode Plot - Magnitude', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=10)

# Plot for phase
plt.subplot(1, 2, 2)
for i, plot_time in enumerate(bode_times):
    time_mask = np.abs(df_interp['time/s'] - plot_time) < 0.05
    time_data = df_interp[time_mask].sort_values('freq/Hz')
    
    if len(time_data) == 0:
        continue
    
    # Calculate phase in degrees
    phase = np.degrees(np.arctan2(-time_data['-Im(Z)/Ohm'], time_data['Re(Z)/Ohm']))
    
    label = f't={plot_time:.1f} s'
    plt.semilogx(time_data['freq/Hz'], phase, '-', color=bode_colors[i], 
                 label=label, linewidth=1.5)

plt.xlabel('Frequency / Hz', fontsize=12)
plt.ylabel('Phase / degrees', fontsize=12)
plt.title('Bode Plot - Phase', fontsize=14)
plt.grid(True, alpha=0.3, which='both')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tick_params(axis='both', which='major', labelsize=10)

plt.suptitle('Bode Plots - Time-Resolved PEIS', fontsize=16)
plt.tight_layout()

output_bode_plot_base = os.path.basename(input_file).replace('.txt', '_bode_interpolated.png')
output_bode_plot = os.path.join(output_dir, output_bode_plot_base)
print(f"Saving Bode plot to: {output_bode_plot}")
plt.savefig(output_bode_plot, dpi=300, bbox_inches='tight')
plt.close()

# Create a summary text file with processing information
print("\nCreating summary file...")
summary_content = f"""TR-PEIS Data Processing Summary
================================

Input File: {input_file}

Data Statistics:
- Total data points: {len(data)}
- Time range: {time_min:.2f} - {time_max:.2f} s
- Frequency range: {unique_freqs.min():.2f} - {unique_freqs.max():.2f} Hz
- Number of unique frequencies: {len(unique_freqs)}
- Number of unique times: {len(unique_times)}

Interpolation:
- Time grid spacing: 100 ms (0.1 s)
- Number of interpolated time points: {len(time_grid)}
- Interpolation method: Linear with extrapolation

Output Directory: {output_dir}

Output Files:
1. Interpolated data CSV: {output_csv_base}
   - Contains: time/s, Re(Z)/Ohm, -Im(Z)/Ohm, freq/Hz, Ewe/V, I/mA
   - Total rows: {len(df_interp)}

2. Nyquist plot: {output_plot_base}
   - Shows impedance spectra at selected time points

3. Time evolution plot: {output_time_plot_base}
   - Shows Re(Z), -Im(Z), and Ewe vs time at {selected_freq:.2f} Hz

4. Bode plot: {output_bode_plot_base}
   - Shows magnitude and phase vs frequency at selected time points

Processing completed successfully.
"""

summary_file = os.path.join(output_dir, os.path.basename(input_file).replace('.txt', '_processing_summary.txt'))
with open(summary_file, 'w') as f:
    f.write(summary_content)

print(f"Summary saved to: {summary_file}")

print("\nProcessing complete!")
print(f"Interpolated data saved to: {output_csv}")
print(f"Nyquist plot saved to: {output_plot}")
print(f"Time evolution plot saved to: {output_time_plot}")
print(f"Bode plot saved to: {output_bode_plot}")
print(f"Summary saved to: {summary_file}")

# List all output files
print("\nOutput files in directory:")
for f in os.listdir(output_dir):
    fpath = os.path.join(output_dir, f)
    size = os.path.getsize(fpath)
    print(f"  {f} ({size} bytes)")
