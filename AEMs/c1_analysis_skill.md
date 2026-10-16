# C1 Anode Capacitance Analysis Skill

## Overview

This document describes the optimized procedure for analyzing anode capacitance (C1) from EIS fitting data (CSV files from eisyFIT program). The analysis follows the same pattern as the Tafel slope analysis for consistency.

## Procedure Summary

1. **Load CSV data** from eisyFIT export
2. **Correct for short circuit** by subtracting current at lowest voltage
3. **Filter data** to current range 10-700 mA
4. **Fit linear relationship**: C1 = k * I_corr + q
5. **Detect outliers** using Z-score on residuals (>2.5)
6. **Track intercept (q)** evolution over Time (represents C1 at zero current)
7. **Generate plots** for visual verification
8. **Export results** to CSV

## Parameters

### Current Range
- **CURRENT_MIN**: 10 mA - Minimum current threshold (changed from 5 mA to improve stability)
- **CURRENT_MAX**: 700 mA - Maximum current threshold (optimized from testing)
- **Rationale**: 10-700 mA provides optimal balance of data points and reduces noise from very low current region

### Fit Configuration
- **Fit equation**: C1 = k * I_corr + q
- **FIT_RANGE_START**: 0.2 - Start fit from 20% of in-range points
- **FIT_RANGE_END**: 0.8 - End fit at 80% of in-range points
- **Rationale**: Avoid edge effects by fitting middle portion of data

### Outlier Detection
- **OUTLIER_Z_SCORE**: 2.5 - Z-score threshold for outlier detection
- **Method**: Calculate residuals from fit line, standardize, flag points with |z-score| > 2.5

## Physical Interpretation

### Intercept q
- Represents **C1 at zero current** (short-circuit capacitance)
- This is the capacitance when no faradaic current flows
- Physical meaning: Double layer capacitance at the anode

### Slope k
- Represents **rate of change of C1 with current**
- Negative values indicate C1 decreases with increasing current
- Physical meaning: Possible pseudo-capacitance effects or surface coverage changes

## Mathematical Foundation

### Fit Equation
```
C1 = k * I_corr + q
```

Where:
- C1 = Anode capacitance (F)
- I_corr = Current corrected for short circuit (mA)
- k = Slope (F/mA)
- q = Intercept at I_corr=0 (F)

### Short Circuit Correction
```
I_corr = I_mA - I_short
```

Where I_short is the current at the lowest voltage (approximately short circuit current).

### Outlier Detection
```
residuals = C1 - (k * I_corr + q)
z_scores = |residuals| / std(residuals)
outliers = z_scores > 2.5
```

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
def analyze_c1_linear_sample(csv_path, sample_id,
                           current_min=10,
                           current_max=700,
                           fit_start=0.2,
                           fit_end=0.8,
                           outlier_z_score=2.5):
    """
    Perform C1 linear fit analysis on a single EIS CSV file.
    
    Parameters:
    - csv_path: Path to CSV file from eisyFIT
    - sample_id: Sample identifier (e.g., '499')
    - current_min: Minimum current in mA (default 10)
    - current_max: Maximum current in mA (default 700)
    - fit_start: Fraction of data to start fit (default 0.2)
    - fit_end: Fraction of data to end fit (default 0.8)
    - outlier_z_score: Z-score threshold for outlier detection (default 2.5)
    
    Returns:
    - results_df: DataFrame with q, k for each Time point
    - comparison_df: DataFrame with with/without outlier comparison
    - Generates plots and CSV files in output directory
    """
    # Load and prepare data
    df = pd.read_csv(csv_path)
    df = df.sort_values(['Time', 'I_mA'])
    
    # Create output directory
    output_dir = Path(f"c1_linear_{sample_id}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = []
    comparison_data = []
    
    for time, group in df.groupby('Time'):
        # Filter to current range
        group = group[(group['I_mA'] >= current_min) & 
                      (group['I_mA'] <= current_max)]
        
        if len(group) < 5:
            continue
        
        # Short circuit correction
        i_short = group['I_mA'].iloc[0]
        I_corr = group['I_mA'].values - i_short
        C1 = group['C1'].values
        
        # Filter valid points
        valid = (np.isfinite(C1)) & (C1 > 0) & (I_corr > 0)
        I_corr = I_corr[valid]
        C1 = C1[valid]
        
        if len(I_corr) < 5:
            continue
        
        # Fit range
        start_idx = int(len(I_corr) * fit_start)
        end_idx = int(len(I_corr) * fit_end)
        
        I_fit = I_corr[start_idx:end_idx]
        C1_fit = C1[start_idx:end_idx]
        
        # Linear fit: C1 = k * I_corr + q
        k, q = np.polyfit(I_fit, C1_fit, 1)
        
        # Outlier detection
        C1_pred = k * I_corr + q
        residuals = C1 - C1_pred
        std_res = np.std(residuals)
        
        z_scores = np.abs(residuals / std_res) if std_res > 0 else np.zeros_like(residuals)
        outliers = z_scores > outlier_z_score
        n_outliers = np.sum(outliers)
        
        # Fit without outliers
        if n_outliers > 0:
            I_no_out = I_corr[~outliers]
            C1_no_out = C1[~outliers]
            
            start_idx_no = int(len(I_no_out) * fit_start)
            end_idx_no = int(len(I_no_out) * fit_end)
            
            I_fit_no = I_no_out[start_idx_no:end_idx_no]
            C1_fit_no = C1_no_out[start_idx_no:end_idx_no]
            
            k_no_out, q_no_out = np.polyfit(I_fit_no, C1_fit_no, 1)
        else:
            k_no_out = k
            q_no_out = q
        
        # Store results
        results.append({
            'time': int(time),
            'k': k_no_out,
            'q': q_no_out,
            'n_points': len(I_corr),
            'n_fit_points': len(I_fit_no) if n_outliers > 0 else len(I_fit),
            'i_short': i_short,
            'std_residuals': std_res
        })
        
        comparison_data.append({
            'time': int(time),
            'k_with': k,
            'q_with': q,
            'k_without': k_no_out,
            'q_without': q_no_out,
            'q_difference': q_no_out - q,
            'n_in_range': len(I_corr),
            'n_outliers': n_outliers,
            'std_residuals': std_res
        })
    
    return pd.DataFrame(results), pd.DataFrame(comparison_data)
```

## File Structure

### Input Files
- `AEMs/474_fit_export.csv` - Sample 474 EIS fitting data
- `AEMs/498_fit_export.csv` - Sample 498 EIS fitting data
- `AEMs/499_fit_export.csv` - Sample 499 EIS fitting data

### Output Files
For each sample, the analysis generates:
- `{sample}_C1_fit_summary.csv` - Summary statistics (q, k for each Time)
- `{sample}_C1_fit_comparison.csv` - With vs without outliers comparison
- `{sample}_C1_fit_time_{XX}.png` - Individual plots for each Time point
- `{sample}_C1_intercept_time_evolution.png` - Time evolution of intercept q
- `{sample}_C1_slope_time_evolution.png` - Time evolution of slope k
- `{sample}_C1_outliers_vs_q_difference.png` - Outlier impact visualization

## Validation Results

### With 10-700 mA Range

| Sample | Mean q | Std q | Mean k | Outliers | Time Points |
|--------|--------|-------|---------|----------|-------------|
| 474 | 0.194 F | 0.043 | -3.09e-4 | 15 | 22 |
| 498 | 0.103 F | 0.018 | -1.50e-4 | 11 | 15 |
| 499 | 0.164 F | 0.020 | -3.04e-4 | 14 | 18 |

### Comparison: 5 mA vs 10 mA Lower Limit (Sample 499)

| Metric | 5 mA min | 10 mA min | Change |
|--------|----------|-----------|--------|
| Mean q | 0.186 F | 0.164 F | -0.022 F |
| Mean k | -7.72e-4 | -3.04e-4 | +4.68e-4 |
| Outliers | 19 | 14 | -5 |
| Time points | 18 | 18 | 0 |

**Conclusion**: 10 mA minimum provides more stable results with less noise.

## Usage Instructions

### For New Samples

1. Place CSV file from eisyFIT in `AEMs/` folder
2. Add sample to the SAMPLES list in `c1_linear_fit_analysis.py`
3. Run:
   ```bash
   cd AEMs
   python c1_linear_fit_analysis.py
   ```
4. Results will be saved in `c1_linear_{sample_id}/` folder

### Customizing Parameters

Modify the constants at the top of the script:
```python
CURRENT_MIN = 10     # Minimum current in mA
CURRENT_MAX = 700    # Maximum current in mA
FIT_RANGE_START = 0.2
FIT_RANGE_END = 0.8
OUTLIER_Z_SCORE = 2.5
```

## Best Practices

1. **Always use 10-700 mA range** for consistent comparison across samples
2. **Use linear fit C1 = k*I + q** (not forced zero intercept)
3. **Remove outliers** using Z-score > 2.5 on residuals
4. **Focus on intercept q** as the primary metric (C1 at zero current)
5. **Track time evolution** to observe capacitance changes during experiment
6. **Compare slopes** to understand current-dependent capacitance behavior

## Physical Interpretation Guidelines

### Intercept q (C1 at zero current)
- **0.05-0.1 F**: Low capacitance, possibly limited electrochemical surface area
- **0.1-0.2 F**: Typical double layer capacitance for many electrode materials
- **0.2-0.3 F**: High capacitance, may indicate pseudocapacitive behavior or large surface area
- **>0.3 F**: Very high, check for fitting artifacts or experimental issues

### Slope k
- **~0**: Capacitance independent of current (ideal double layer)
- **Negative**: Capacitance decreases with current (possible surface coverage effects)
- **Positive**: Capacitance increases with current (rare, may indicate measurement artifacts)

## References

- Original analysis pattern: `eis_parameters.py` (C1 vs Current plots)
- Tafel analysis procedure: `tafel_analysis_skill.md`
- First implementation: `capacitance_analysis.py` (simple C1 vs I plots)
- Optimized implementation: `c1_linear_fit_analysis.py`

## Version History

- **v1.0**: Initial C1 vs I plots (no fitting)
- **v2.0**: Linear fit with 5-700 mA range
- **v3.0**: Outlier detection added
- **v3.1**: Changed current minimum to 10 mA for improved stability
- **v3.2**: Documented as skill

---

*Last updated: 2026-10-16*
*Optimized for samples: 474, 498, 499*
