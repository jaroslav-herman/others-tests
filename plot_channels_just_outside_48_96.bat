@echo off
REM Generate plots for channels just outside the 48-96 range
REM This script creates correlation CSVs and heatmaps for ch47-47 and ch97-97

set MPT_FILE=C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV.mpt
set XY_FILE=C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy
set CHANNEL_DIR=C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges

REM Create channel_ranges subdirectory for ch47-47
mkdir "%CHANNEL_DIR%\ch47-47" 2>nul
python CVXPS-correlate-channel.py "%MPT_FILE%" "%XY_FILE%" "%CHANNEL_DIR%\ch47-47\12_CV_XPS_correlated.csv" 47 47
python CVXPS-heatmap-channel.py "%CHANNEL_DIR%\ch47-47\12_CV_XPS_correlated.csv" "" ch47-47

REM Create channel_ranges subdirectory for ch97-97
mkdir "%CHANNEL_DIR%\ch97-97" 2>nul
python CVXPS-correlate-channel.py "%MPT_FILE%" "%XY_FILE%" "%CHANNEL_DIR%\ch97-97\12_CV_XPS_correlated.csv" 97 97
python CVXPS-heatmap-channel.py "%CHANNEL_DIR%\ch97-97\12_CV_XPS_correlated.csv" "" ch97-97

echo Done! Heatmaps generated in:
echo   %CHANNEL_DIR%\ch47-47
echo   %CHANNEL_DIR%\ch97-97
