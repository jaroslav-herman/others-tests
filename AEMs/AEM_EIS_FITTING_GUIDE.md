# AEM EIS Fitting Guide

This guide contains AEM-specific procedures, scripts, and parameter configurations for EIS data processing.

---

## Quick Reference

For any AEM-related EIS fitting request, follow the workflows described in this file. For general EIS processing, see the main `AGENTS.md` file.

---

## Primary Workflow

### Standard EIS Processing
Use `AEMs/scripts/process_eis_data.py` as the primary workflow for:
- Loading CSV files exported from eisyFIT
- Processing EIS fitting parameters (R0, R1, R2, R3, C1, C2, C3, C4, tau values)
- Generating standard plots (R vs I, C vs I, IV curves, time evolution)

**Always push changes to main branch** after creating or modifying processing scripts.

### Single Sample Processing Workflow
When a specific sample and parameter are requested:
1. **Identify the correct script** from `AEMs/scripts/` based on the parameter
2. **Temporarily modify the SAMPLES list** in the script to include ONLY the requested sample
3. **Run the script** - outputs will automatically be saved to `sample_{id}/{param_folder}/`
4. **Revert the SAMPLES list** back to its original state
5. **Push to main branch**

All analysis scripts now save their outputs directly into the sample folder, so no manual moving is required.

---

## Dedicated Linear Fit Analysis

For new samples, work in the respective sample folder (`AEMs/sample_XXX/`). Create the following analyses using the dedicated Python scripts from `AEMs/scripts/`:

### Scripts
1. **`scripts/c1_linear_fit_analysis.py`** - Capacitance C1 linear fit analysis
2. **`scripts/r0_linear_fit_analysis.py`** - Ohmic resistance R0 linear fit analysis
3. **`scripts/c2_r2_c3_r3_linear_fit_analysis.py`** - Combined analysis for C2, R2, C3, R3
4. **`scripts/tafel_analysis_all_samples.py`** - Tafel slope analysis

### Setup for New Samples
1. Add the new sample ID to the `SAMPLES` list in each script
2. Run the scripts
3. Ensure all output folders are created inside the sample's directory
4. Run `scripts/generate_polarization_graphs.py` to generate polarization curves and current evolution graphs for the new sample
5. Push all changes to main branch

---

## Always True Commands

1. **Push to main branch** - Always push changes to the main branch after processing
2. **Process only specified samples** - When a specific sample is requested for processing, run scripts ONLY for that sample, not for all samples in the SAMPLES list. Temporarily modify the SAMPLES list in the script to include only the requested sample, run it, then revert the SAMPLES list back to its original state.

---

## Parameter-Specific Current Limits

The `c2_r2_c3_r3_linear_fit_analysis.py` script applies parameter-specific current limits:

| Parameter | Current Range | Notes |
|-----------|---------------|-------|
| **R2** | 5-700 mA | Same as Tafel analysis |
| **R3** | 5-700 mA | Same as Tafel analysis |
| **C2** | 10 mA minimum, no upper limit | Original configuration |
| **C3** | 10 mA minimum, no upper limit | Original configuration |

**Important**: R0 analysis uses its own script (`r0_linear_fit_analysis.py`) with separate current limits and is **not affected** by the R2/R3 settings.

---

## Analysis Methodology

### Linear Fit Pattern
For each parameter, the following linear relationship is fitted:
```
param = k * I_corr + q
```

Where:
- `I_corr = I - I_short` (corrected for short circuit)
- `k` = slope
- `q` = intercept (tracked over Time)

### Tafel Analysis
- Uses only **reciprocal value of R1** (1/R1) for calculations
- Fit: `1/R1 = k * I + q`
- Tafel slope = `2.3 / k`

### Outlier Detection
- Z-score threshold: 2.5 on residuals
- Fit range: Middle 60% of data points (20%-80%)

---

## Column Distinction

### Time Columns in eisyFIT CSV Exports

| Column | Description | Usage |
|--------|-------------|-------|
| `Time` (capital T) | Time evolution of fitted parameters | Use for time evolution plots |
| `time/s` (lowercase with /s) | Individual measurement timestamps | Do NOT use for time evolution unless explicitly requested |

**Rule**: When a user requests "time evolution of parameters", always plot against the `Time` column unless explicitly stated otherwise.

---

## Sample Naming

### Google Sheets Reference
Use the Google Sheets file `electrolyzer_samples` as the primary source for sample names:
- **ID**: `1ycoCdaol3zYWx8PDTvrI30eZV0nuC7ooFpEvz9qOlds`
- **File**: [electrolyzer_samples](https://docs.google.com/spreadsheets/d/1ycoCdaol3zYWx8PDTvrI30eZV0nuC7ooFpEvz9qOlds/edit)

### Lookup Procedure
1. Look up the `Sample Number` column
2. Use the corresponding value from the `Sample Name` column
3. If `Sample Name` is empty or unavailable, use only the sample number

### Legend Format
In comparison plots between samples, include legends with both:
```
sample_number (sample_name)
```

Example: `474 (VI_Ni_on_Ni_PTL)`

---

## Folder Structure

### Root Level
- `AEM_EIS_FITTING_GUIDE.md` - This guide file (always kept at root)
- `scripts/` - All Python processing scripts
- `comparisons/` - Cross-sample comparison graphs and summary files
- `comparisons_XXX_YYY_ZZZ/` - Legacy comparison folders by sample numbers (e.g., `comparisons_474_499_505/`)
- If samples belong to a defined series (e.g., AEM Substrate Series), use the series name instead of sample numbers: `AEMs/AEM_Substrate_Series/`

### Sample Organization
Each sample has its own folder: `AEMs/sample_XXX/`

### Analysis Subfolders
Within each sample folder, analysis results are stored in subfolders:
- `c1_linear_XXX/` - C1 linear fit analysis
- `r0_linear_XXX/` - R0 (ohmic resistance) linear fit analysis
- `C2_linear_XXX/` - C2 linear fit analysis
- `R2_linear_XXX/` - R2 linear fit analysis
- `C3_linear_XXX/` - C3 linear fit analysis
- `R3_linear_XXX/` - R3 linear fit analysis
- `tafel_700ma_XXX/` - Tafel analysis

### Scripts Folder
All Python scripts are stored in `AEMs/scripts/` to keep the root clean.

---

## Output Files

### Per-Time Point Plots
Each analysis generates individual plots for each Time point:
- `{sample_id}_{param}_fit_time_{NN}.png` - Parameter vs Corrected Current with linear fit
- Includes residuals plot
- Highlights outliers and fit range

### Summary Files
- `{sample_id}_{param}_fit_summary.csv` - Summary statistics for each Time point
- `{sample_id}_{param}_fit_comparison.csv` - Comparison of results with and without outliers

### Time Evolution Plots
- `{sample_id}_{param}_intercept_time_evolution.png` - Intercept q vs Time
- `{sample_id}_{param}_slope_time_evolution.png` - Slope k vs Time
- `{sample_id}_{param}_outliers_vs_q_difference.png` - Outlier impact analysis

---

## Sample Series

### AEM Substrate Series
This series compares EIS results when catalyst is deposited on different substrates:
- **Plain membrane**
- **Etched membrane**
- **PTL (Porous Transport Layer)**

**Samples in this series**: 474, 499, 505

**Important**: Sample 498 is NOT part of this series.

This series is used for comparing how the substrate type affects electrochemical performance and EIS parameters.

#### Legend Format Rule
In comparison plots between samples, always use the format: `sample_number (sample_name)`
Example: `474 (VI_Ni_on_Ni_PTL)`

---

## Graph Delivery

When asked to show graphs, provide direct GitHub links:
- **Single graph**: `https://github.com/jaroslav-herman/others-tests/blob/main/AEMs/path/to/graph.png`
- **Folder**: `https://github.com/jaroslav-herman/others-tests/tree/main/AEMs/path/to/folder/`

This allows users to open them immediately in a browser or GitHub app.

**Always send GitHub links** when graphs are generated or requested.

---

## Script Configuration Details

### c1_linear_fit_analysis.py
- **Current range**: 10 mA minimum, no upper limit
- **Fit range**: Middle 60% (20%-80%)
- **Outlier Z-score**: 2.5

### r0_linear_fit_analysis.py
- **Current range**: 10 mA minimum, no upper limit
- **Fit range**: Middle 60% (20%-80%)
- **Outlier Z-score**: 2.5

### c2_r2_c3_r3_linear_fit_analysis.py
- **Current ranges**: Parameter-specific (see table above)
- **Fit range**: Middle 60% (20%-80%)
- **Outlier Z-score**: 2.5

### tafel_analysis_all_samples.py
- **Current range**: 5-700 mA
- **Uses**: Only reciprocal value of R1 (1/R1)
- **Fit**: `1/R1 = k * I + q` (forced zero intercept)
- **Tafel slope**: `2.3 / k`
- **Fit range**: Middle 60% (20%-80%)
- **Outlier Z-score**: 2.5
