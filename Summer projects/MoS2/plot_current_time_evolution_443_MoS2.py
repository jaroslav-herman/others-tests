"""Time evolution of current at specific voltages for sample 443_III_MoS2_400nm_etched_cathode

Loads SV technique .mpr files, extracts current values at specific voltages,
and plots time evolution with wepy.get_colors for coloring.
"""

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import wepy
import matplotlib.pyplot as plt
import numpy as np

# Sample folder - use proper encoding
sample_folder = "C:/Users/Herman/OneDrive - Univerzita Karlova/Letní projekt/MoS2/443_III_MoS2_400nm_etched_cathode"

# Output folder (inside others-tests/Summer projects/MoS2)
output_folder = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_folder, exist_ok=True)

print(f"Sample folder: {sample_folder}")
print(f"Output folder: {output_folder}")

# Find all .mpr files with SV technique
print("\nFinding SV technique .mpr files...")
mpr_files = wepy.load_files(sample_folder, contains_string="SV", extension=".mpr")
print(f"Found {len(mpr_files)} SV files:")
for f in mpr_files:
    print(f"  - {os.path.basename(f)}")

if not mpr_files:
    print("Error: No SV technique .mpr files found!")
    sys.exit(1)

# Load all SV files
print("\nLoading data files...")
all_data = []
all_labels = []
colors = wepy.get_colors(len(mpr_files))

for i, file_path in enumerate(mpr_files):
    print(f"  Loading: {os.path.basename(file_path)}")
    data = wepy.read_file(file_path)
    all_data.append(data)
    
    # Generate label from filename
    basename = os.path.basename(file_path)
    # Extract day and procedure from filename
    label = basename.replace("_SV_C01.mpr", "").replace("III_Day", "Day ").replace("_Procedure", " Proc ")
    all_labels.append(label)

# Extract current values at specific voltages
print("\nExtracting current values at specific voltages...")
target_voltages = [1.6,1.8,2.0]  # V
voltage_data = {v: [] for v in target_voltages}

for i, data in enumerate(all_data):
    # Get time and voltage data
    time = data["time/s"]
    voltage = data["Ewe/V"]-data["Ece/V"]
    current = data["<I>/mA"]
    
    # Find indices where voltage is closest to target values
    for v in target_voltages:
        closest_idx = np.argmin(np.abs(voltage - v)) if len(voltage) > 0 else None
        if closest_idx is not None:
            voltage_data[v].append(current[closest_idx])

# Create time evolution plot
print("\nCreating time evolution plot...")
fig, ax = plt.subplots(figsize=(12, 8))

# Plot current values at each target voltage
for i, v in enumerate(target_voltages):
    color = colors[i % len(colors)] if len(colors) > 0 else None
    ax.plot(range(len(voltage_data[v])), voltage_data[v], 'o-', color=color, linewidth=2, label=f"{v} V")

# Customize plot
sample_name = os.path.basename(sample_folder)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Current / mA')
ax.set_title(f'Time Evolution of Current at Specific Voltages - {sample_name}')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')
plt.tight_layout()

# Save plot
output_path = os.path.join(output_folder, f"current_time_evolution_{sample_name.replace(' ', '_').replace('/', '_')}.png")
fig.savefig(output_path, dpi=300)
plt.close(fig)
print(f"\nPlot saved: file:///{output_path}")

print("\nDone!")