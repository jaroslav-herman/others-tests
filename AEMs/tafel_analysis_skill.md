# Tafel Analysis Skill - EIS Data Processing for OER

## Overview

This document describes the optimized procedure for Tafel slope analysis from EIS fitting data (CSV files from eisyFIT program). The analysis focuses on the R1 parameter for OER (Oxygen Evolution Reaction) Tafel slope extraction.

## Procedure Summary

1. **Load CSV data** from eisyFIT export
2. **Correct for short circuit** by subtracting current at lowest voltage
3. **Filter data** to current range 5-700 mA
4. **Fit linear relationship**: 1/R1 = k * I (forced zero intercept, q=0)
5. **Calculate Tafel slope**: 2.3 / k
6. **Detect outliers** using Z-score on residuals (>2.5)
7. **Compare results** with and without outliers
8. **Generate plots** for visual verification
9. **Export results** to CSV

## Parameters

### Current Range
- **MIN_CURRENT**: 5 mA - Minimum current threshold
- **MAX_CURRENT**: 700 mA - Maximum current threshold (optimized from testing)
- **Rationale**: 5-700 mA provides optimal balance of data points and linear region for 1/R1 vs I relationship

### Fit Configuration
- **FORCE_ZERO_INTERCEPT**: True - Force line intersection with y-axis to 0 (q=0)
- **FIT_RANGE_START**: 0.2 - Start fit from 20% of in-range points
- **FIT_RANGE_END**: 0.8 - End fit at 80% of in-range points
- **Rationale**: Avoid edge effects by fitting middle portion of data

### Outlier Detection
- **OUTLIER_Z_SCORE**: 2.5 - Z-score threshold for outlier detection
- **Method**: Calculate residuals from fit line, standardize, flag points with |z-score| > 2.5
- **Impact**: Outliers systematically INCREASE Tafel slopes (mean difference: ~3.9 mV/dec)

## Mathematical Foundation

### Fit Equation
```
1/R1 = k * I + q
```

With forced zero intercept (q=0):
```
1/R1 = k * I
```

### Tafel Slope Calculation
```
Tafel slope = 2.3 / k  (in mV/dec)
```

Where:
- k = slope from linear fit of 1/R1 vs I
- 2.3 = conversion factor from natural log to base-10 log

### Short Circuit Correction
```
I_corrected = I_mA - I_short
```

Where I_short is the current at the lowest voltage (approximately short circuit current).

## Implementation

### Required Python Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
```

### Core Analysis Function

```python
def analyze_tafel_sample(csv_path, sample_id, 
                        current_min=5, current_max=700,
                        force_zero_intercept=True,
                        fit_start=0.2, fit_end=0.8,
                        outlier_z_score=2.5):
    """
    Perform Tafel analysis on a single EIS CSV file.
    
    Parameters:
    - csv_path: Path to CSV file from eisyFIT
    - sample_id: Sample identifier (e.g., '499')
    - current_min: Minimum current in mA
    - current_max: Maximum current in mA
    - force_zero_intercept: Force q=0 in fit
    - fit_start: Fraction of data to start fit
    - fit_end: Fraction of data to end fit
    - outlier_z_score: Z-score threshold for outlier detection
    
    Returns:
    - results_df: DataFrame with Tafel slopes for each Time point
    - Generates plots and CSV files in output directory
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Sort by current within each Time group
    df = df.sort_values(['Time', 'I_mA'])
    
    # Create output directory
    output_dir = Path(f"tafel_{current_max}ma_{sample_id}")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for time, group in df.groupby('Time'):
        # Filter to current range
        group = group[(group['I_mA'] >= current_min) & 
                      (group['I_mA'] <= current_max)]
        
        if len(group) < 5:
            continue
        
        # Short circuit correction
        i_short = group['I_mA'].iloc[0]
        I_corr = group['I_mA'].values - i_short
        R1_inv = 1 / group['R1'].values
        
        # Filter valid points
        valid = (R1_inv > 0) & (I_corr > 0)
        I_corr = I_corr[valid]
        R1_inv = R1_inv[valid]
        
        if len(I_corr) < 5:
            continue
        
        # Fit range
        start_idx = int(len(I_corr) * fit_start)
        end_idx = int(len(I_corr) * fit_end)
        
        I_fit = I_corr[start_idx:end_idx]
        R1_inv_fit = R1_inv[start_idx:end_idx]
        
        # Linear fit
        if force_zero_intercept:
            k = np.sum(I_fit * R1_inv_fit) / np.sum(I_fit ** 2)
            q = 0
        else:
            k, q = np.polyfit(I_fit, R1_inv_fit, 1)
        
        # Calculate Tafel slope
        tafel = 2.3 / k if k != 0 else np.nan
        
        # Outlier detection
        R1_inv_pred = k * I_corr + q
        residuals = R1_inv - R1_inv_pred
        std_res = np.std(residuals)
        outliers = np.abs(residuals / std_res) > outlier_z_score
        n_outliers = np.sum(outliers)
        
        # Fit without outliers
        if n_outliers > 0:
            I_no_out = I_corr[~outliers]
            R1_inv_no_out = R1_inv[~outliers]
            
            start_idx_no = int(len(I_no_out) * fit_start)
            end_idx_no = int(len(I_no_out) * fit_end)
            
            I_fit_no = I_no_out[start_idx_no:end_idx_no]
            R1_inv_fit_no = R1_inv_no_out[start_idx_no:end_idx_no]
            
            if force_zero_intercept:
                k_no_out = np.sum(I_fit_no * R1_inv_fit_no) / np.sum(I_fit_no ** 2)
            else:
                k_no_out, _ = np.polyfit(I_fit_no, R1_inv_fit_no, 1)
            
            tafel_no_out = 2.3 / k_no_out if k_no_out != 0 else np.nan
        else:
            k_no_out = k
            tafel_no_out = tafel
        
        results.append({
            'time': int(time),
            'tafel_with': tafel,
            'tafel_without': tafel_no_out,
            'difference': tafel_no_out - tafel,
            'k_with': k,
            'k_without': k_no_out,
            'n_in_range': len(I_corr),
            'n_outliers': n_outliers,
            'std_residuals': std_res
        })
    
    return pd.DataFrame(results)
```

## File Structure

### Input Files
- `AEMs/474_fit_export.csv` - Sample 474 EIS fitting data
- `AEMs/498_fit_export.csv` - Sample 498 EIS fitting data
- `AEMs/499_fit_export.csv` - Sample 499 EIS fitting data

### Output Files
For each sample, the analysis generates:
- `{sample}_tafel_summary.csv` - Summary statistics
- `{sample}_tafel_comparison.csv` - With vs without outliers comparison
- `{sample}_tafel_time_{XX}.png` - Individual plots for each Time point
- `{sample}_tafel_comparison.png` - Summary comparison plot
- `{sample}_tafel_outliers_vs_difference.png` - Outlier impact visualization

## Validation Results (Sample 499)

### 5-700 mA Range with Outlier Removal

| Metric | Value |
|--------|-------|
| Mean Tafel with outliers | 73.8 mV/dec |
| Mean Tafel without outliers | 69.9 mV/dec |
| Mean difference | -3.9 mV/dec |
| Max difference | +23.7 mV/dec (Time 18: 50.7→74.4) |
| Outliers detected | 0-1 per Time point |
| Conclusion | Outliers systematically INCREASE Tafel slopes |

### Key Findings
1. **Extended range (5-700 mA)** provides more data points than 5-300 mA
2. **Forced zero intercept (q=0)** is appropriate for this analysis
3. **Outlier removal** has significant impact (up to 23.7 mV/dec difference)
4. **Time 1** had insufficient data in initial 5-300 mA range
5. **All Time points** now have sufficient data with 5-700 mA range

## Usage Instructions

### For New Samples

1. Place CSV file from eisyFIT in `AEMs/` folder
2. Run the analysis script:
   ```bash
   cd AEMs
   python tafel_analysis_all_samples.py
   ```
3. Results will be saved in `tafel_700ma_{sample_id}/` folder

### Customizing Parameters

Modify the constants at the top of the script:
```python
CURRENT_MIN = 5      # Minimum current in mA
CURRENT_MAX = 700    # Maximum current in mA
FORCE_ZERO_INTERCEPT = True
FIT_RANGE_START = 0.2
FIT_RANGE_END = 0.8
OUTLIER_Z_SCORE = 2.5
```

## Best Practices

1. **Always use 5-700 mA range** for consistent comparison across samples
2. **Force zero intercept** (q=0) for physically meaningful fit
3. **Remove outliers** using Z-score > 2.5 on residuals
4. **Visually verify** each Time point plot before accepting results
5. **Report both** with and without outlier values for transparency

## References

- Original analysis pattern: `eis_parameters.py`
- First implementation: `tafel_analysis_499.py`
- Optimized implementation: `tafel_analysis_499_700ma.py` (in `tafel_700ma_499/`)
- Outlier analysis: `tafel_outlier_analysis_499.py`

## Version History

- **v1.0**: Initial procedure with 5-300 mA range
- **v2.0**: Extended to 5-500 mA, forced q=0
- **v3.0**: Extended to 5-700 mA, added outlier detection
- **v3.1**: Confirmed outlier impact, documented as skill

---

*Last updated: 2026-10-16*
*Optimized for samples: 474, 498, 499*
