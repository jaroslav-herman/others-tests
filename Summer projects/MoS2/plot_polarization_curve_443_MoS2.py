"""
Polarization curve plot for sample 443_III_MoS2_400nm_etched_cathode

Loads SV technique .mpr files, extracts IV curves using wepy.iv_curve.IV_curves_data,
and plots polarization curves with wepy.get_colors for coloring.
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

# Extract IV curves from each dataset
print("\nExtracting IV curves...")
all_Ecells = []
all_Is = []
all_cycles = []

for i, data in enumerate(all_data):
    Ecells, Is = wepy.IV_curves_data(data)
    all_Ecells.append(Ecells)
    all_Is.append(Is)
    all_cycles.append([len(c) for c in Ecells])  # Number of points per cycle
    print(f"  File {i+1}: {len(Ecells)} cycles extracted")

# Create polarization curve plot
print("\nCreating polarization curve plot...")
fig, ax = plt.subplots(figsize=(12, 8))

for i, (Ecells, Is) in enumerate(zip(all_Ecells, all_Is)):
    color = colors[i % len(colors)] if len(colors) > 0 else None
    
    for cycle_idx, (Ecell, I) in enumerate(zip(Ecells, Is)):
        # Plot each cycle
        label = f"{all_labels[i]} - Cycle {cycle_idx + 1}" if len(Ecells) > 1 else all_labels[i]
        ax.plot(I, Ecell, color=color, linewidth=2, label=label, alpha=0.7)

# Customize plot
sample_name = os.path.basename(sample_folder)
ax.set_xlabel('Current / mA')
ax.set_ylabel('Cell Voltage / V')
ax.set_title(f'Polarization Curves - {sample_name}')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')
plt.tight_layout()

# Save plot
output_path = os.path.join(output_folder, f"polarization_curve_{sample_name.replace(' ', '_').replace('/', '_')}.png")
fig.savefig(output_path, dpi=300)
plt.close(fig)
print(f"\nPlot saved: file:///{output_path}")

print("\nDone!")
