@echo off
REM Generate plots for ALL channels EXCEPT those in range 48-96
REM This means processing:
REM   - Channel range 0-47 (all channels below 48)
REM   - Channel range 97-145 (all channels above 96)

set MPT_FILE=C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\12_CV.mpt
set XY_FILE=C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy
set CHANNEL_DIR=C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\channel_ranges

echo ======================================================================
echo Generating plots for ALL channels EXCEPT 48-96
echo ======================================================================

REM Create channel_ranges subdirectory for ch0-47
mkdir "%CHANNEL_DIR%\ch0-47" 2>nul
echo.
echo Processing ch0-47 (all channels below 48)...
python CVXPS-correlate-channel.py "%MPT_FILE%" "%XY_FILE%" "%CHANNEL_DIR%\ch0-47\12_CV_XPS_correlated.csv" 0 47
python CVXPS-heatmap-channel.py "%CHANNEL_DIR%\ch0-47\12_CV_XPS_correlated.csv" "" ch0-47

REM Create channel_ranges subdirectory for ch97-145
mkdir "%CHANNEL_DIR%\ch97-145" 2>nul
echo.
echo Processing ch97-145 (all channels above 96)...
python CVXPS-correlate-channel.py "%MPT_FILE%" "%XY_FILE%" "%CHANNEL_DIR%\ch97-145\12_CV_XPS_correlated.csv" 97 145
python CVXPS-heatmap-channel.py "%CHANNEL_DIR%\ch97-145\12_CV_XPS_correlated.csv" "" ch97-145

echo.
echo ======================================================================
echo Done! Heatmaps generated in:
echo   %CHANNEL_DIR%\ch0-47
echo   %CHANNEL_DIR%\ch97-145
echo ======================================================================
