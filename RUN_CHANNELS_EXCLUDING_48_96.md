# Plotting Channels Excluding 48-96 Range

To generate plots for ALL channels EXCEPT those in range 48-96 (i.e., channels 0-47 and 97-145), run one of the following:

## Option 1: Using the Python Script

```bash
python plot_channels_excluding_48_96.py
```

This will:
1. Process channel range 0-47 (all channels below 48)
2. Process channel range 97-145 (all channels above 96)
3. Generate correlation CSVs and heatmap PNGs for each range

## Option 2: Using the Batch File (Windows)

```cmd
plot_channels_excluding_48_96.bat
```

## Option 3: Manual Commands

If you prefer to run the commands manually:

### For channels 0-47:
```bash
python CVXPS-correlate-channel.py "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV.mpt" "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy" "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-47\12_CV_XPS_correlated.csv" 0 47
python CVXPS-heatmap-channel.py "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-47\12_CV_XPS_correlated.csv" "" ch0-47
```

### For channels 97-145:
```bash
python CVXPS-correlate-channel.py "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV.mpt" "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy" "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch97-145\12_CV_XPS_correlated.csv" 97 145
python CVXPS-heatmap-channel.py "C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch97-145\12_CV_XPS_correlated.csv" "" ch97-145
```

## Output

All results will be saved in:
- `C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch0-47\`
- `C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges\ch97-145\`

Each directory will contain:
- `12_CV_XPS_correlated.csv` - The correlated data
- 7 heatmap PNG files (auto bins, fixed bins, normalized, spectrum plots)

## Notes

- Channel range 0-47 includes 48 channels (0 through 47 inclusive)
- Channel range 97-145 includes 49 channels (97 through 145 inclusive)
- Total: 97 channels excluded from the 48-96 range
- The original 48-96 range has 49 channels (48 through 96 inclusive)
