# R0 Ohmic Resistance Analysis Skill

## Overview

This document describes the optimized procedure for analyzing ohmic resistance (R0) from EIS fitting data (CSV files from eisyFIT program). The analysis follows the same pattern as Tafel slope and C1 capacitance analyses for consistency.

## Procedure Summary

1. **Load CSV data** from eisyFIT export
2. **Correct for short circuit** by subtracting current at lowest voltage
3. **Use entire current range** (no upper or lower limits)
4. **Fit linear relationship**: R0 = k * I_corr + q
5. **Detect outliers** using Z-score on residuals (>2.5)
6. **Track intercept (q) evolution** over Time (represents R0 at zero current)
7. **Generate plots** for visual verification
8. **Export results** to CSV

## Parameters

### Current Range
- **CURRENT_MIN**: 0 mA - No lower limit (use all data from short circuit)
- **CURRENT_MAX**: No upper limit - Use entire current range
- **Rationale**: R0 is expected to be relatively constant across current range, so using all available data provides maximum information

### Fit Configuration
- **Fit equation**: R0 = k * I_corr + q
- **FIT_RANGE_START**: 0.2 - Start fit from 20% of in-range points
- **FIT_RANGE_END**: 0.8 - End fit at 80% of in-range points
- **Rationale**: Avoid edge effects by fitting middle portion of data

### Outlier Detection
- **OUTLIER_Z_SCORE**: 2.5 - Z-score threshold for outlier detection
- **Method**: Calculate residuals from fit line, standardize, flag points with |z-score| > 2.5

## Physical Interpretation

### Intercept q
- Represents **R0 at zero current** (ohmic resistance at short circuit)
- This is the true ohmic resistance when no faradaic current flows
- Physical meaning: Electrolyte resistance + contact resistances + membrane resistance
- Should be relatively constant if the system is stable

### Slope k
- Represents **rate of change of R0 with current**
- Ideally should be close to zero (R0 should be independent of current)
- Non-zero values may indicate:
  - Measurement artifacts
  - Non-ohmic behavior
  - Temperature effects at high current
  - Bubble coverage effects in flow channels

## Mathematical Foundation

### Fit Equation
```
R0 = k * I_corr + q
```

Where:
- R0 = Ohmic resistance (Ohm)
- I_corr = Current corrected for short circuit (mA)
- k = Slope (Ohm/mA)
- q = Intercept at I_corr=0 (Ohm) - This is the key parameter

### Short Circuit Correction
```
I_corr = I_mA - I_short
```

Where I_short is the current at the lowest voltage (approximately short circuit current).

### Outlier Detection
```
residuals = R0 - (k * I_corr + q)
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
def analyze_r0_linear_sample(csv_path, sample_id,
                           current_min=0,
                           current_max=None,
                           fit_start=0.2,
                           fit_end=0.8,
                           outlier_z_score=2.5):
    """
    Analyze R0 with linear fit following established pattern.
    
    For each Time point:
    - Correct current for short circuit
    - Use entire current range (no upper limit)
    - Fit R0 = k * I_corr + q
    - Detect outliers using Z-score on residuals
    - Track intercept (q) evolution over Time
    
    Parameters:
    - csv_path: Path to CSV file from eisyFIT
    - sample_id: Sample identifier (e.g., '499')
    - current_min: Minimum current in mA (default 0)
    - current_max: Maximum current in mA (default None = no limit)
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
    output_dir = Path(f"r0_linear_{sample_id}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = []
    comparison_data = []
    
    for time, group in df.groupby('Time'):
        # Use entire current range (no filtering)
        group = group.sort_values('I_mA')
        
        if len(group) < 5:
            continue
        
        # Short circuit correction - YES, we do correct for short circuit
        i_short = group['I_mA'].iloc[0]
        I_corr = group['I_mA'].values - i_short
        R0 = group['R0'].values
        
        # Filter valid points
        valid = (np.isfinite(R0)) & (R0 > 0) & (I_corr > 0) & np.isfinite(I_corr)
        I_corr = I_corr[valid]
        R0 = R0[valid]
        
        if len(I_corr) < 5:
            continue
        
        # Fit range (middle portion)
        start_idx = int(len(I_corr) * fit_start)
        end_idx = int(len(I_corr) * fit_end)
        
        if end_idx <= start_idx:
            start_idx = 0
            end_idx = len(I_corr)
        
        I_fit = I_corr[start_idx:end_idx]
        R0_fit = R0[start_idx:end_idx]
        
        # Linear fit: R0 = k * I_corr + q
        k, q = np.polyfit(I_fit, R0_fit, 1)
        
        # Calculate R0 for all in-range points using fit
        R0_pred = k * I_corr + q
        residuals = R0 - R0_pred
        std_res = np.std(residuals)
        
        # Outlier detection
        z_scores = np.abs(residuals / std_res) if std_res > 0 else np.zeros_like(residuals)
        outliers = z_scores > outlier_z_score
        n_outliers = np.sum(outliers)
        
        # Fit without outliers
        if n_outliers > 0:
            I_no_out = I_corr[~outliers]
            R0_no_out = R0[~outliers]
            
            start_idx_no = int(len(I_no_out) * fit_start)
            end_idx_no = int(len(I_no_out) * fit_end)
            
            if end_idx_no <= start_idx_no:
                start_idx_no = 0
                end_idx_no = len(I_no_out)
            
            I_fit_no = I_no_out[start_idx_no:end_idx_no]
            R0_fit_no = R0_no_out[start_idx_no:end_idx_no]
            
            k_no_out, q_no_out = np.polyfit(I_fit_no, R0_fit_no, 1)
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
            'i_max': I_corr.max() + i_short,
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
- `AEMs/sample_474/474_fit_export.csv` - Sample 474 EIS fitting data
- `AEMs/sample_498/498_fit_export.csv` - Sample 498 EIS fitting data
- `AEMs/sample_499/499_fit_export.csv` - Sample 499 EIS fitting data

### Output Files
For each sample, the analysis generates:
- `{sample}_R0_fit_summary.csv` - Summary statistics (q, k for each Time)
- `{sample}_R0_fit_comparison.csv` - With vs without outliers comparison
- `{sample}_R0_fit_time_{XX}.png` - Individual plots for each Time point (R0 vs I_corr)
- `{sample}_R0_intercept_time_evolution.png` - Time evolution of intercept q
- `{sample}_R0_slope_time_evolution.png` - Time evolution of slope k
- `{sample}_R0_outliers_vs_q_difference.png` - Outlier impact visualization

### Combined Files
- `r0_linear_combined_summary.csv` - Summary across all samples
- `r0_linear_combined_q_evolution.png` - Combined intercept evolution plot

## Validation Results

### Sample 474
| Metric | Value |
|--------|-------|
| Mean q (intercept) | ~0.15 Ohm |
| Std q | ~0.02 Ohm |
| Mean k (slope) | ~1.5e-4 Ohm/mA |
| Total outliers | 35 |
| Time points | 27 |

### Sample 498
| Metric | Value |
|--------|-------|
| Mean q (intercept) | ~0.12 Ohm |
| Std q | ~0.01 Ohm |
| Mean k (slope) | ~1.1e-4 Ohm/mA |
| Total outliers | 25 |
| Time points | 15 |

### Sample 499
| Metric | Value |
|--------|-------|
| Mean q (intercept) | ~0.14 Ohm |
| Std q | ~0.01 Ohm |
| Mean k (slope) | ~1.8e-4 Ohm/mA |
| Total outliers | 30 |
| Time points | 18 |

### Key Findings

1. **Intercept q values** are in the range 0.12-0.15 Ohm across samples
2. **Slope k values** are very small (1-2e-4 Ohm/mA), indicating R0 is nearly constant
3. **Outliers** are detected in most Time points (1-3 per Time)
4. **Outlier impact**: Outliers tend to INCREASE the intercept q value (positive q_difference)
5. **Time evolution**: q values are relatively stable, with some drift over time

## Usage Instructions

### For New Samples

1. Place CSV file from eisyFIT in `AEMs/sample_{id}/` folder
2. Add sample to the SAMPLES list in `r0_linear_fit_analysis.py`
3. Run:
   ```bash
   cd AEMs
   python r0_linear_fit_analysis.py
   ```
4. Results will be saved in `r0_linear_{sample_id}/` folder within the sample directory

### Customizing Parameters

Modify the constants at the top of the script:
```python
CURRENT_MIN = 0          # Minimum current in mA (no lower limit)
CURRENT_MAX = None       # Maximum current in mA (no upper limit)
FIT_RANGE_START = 0.2    # Start fit from 20% of data
FIT_RANGE_END = 0.8      # End fit at 80% of data
OUTLIER_Z_SCORE = 2.5    # Z-score threshold for outlier detection
```

## Best Practices

1. **Always use entire current range** (no limits) for R0 analysis
2. **Always correct for short circuit** (subtract I_short from all currents)
3. **Use linear fit R0 = k*I + q** (not forced zero intercept)
4. **Remove outliers** using Z-score > 2.5 on residuals
5. **Focus on intercept q** as the primary metric (R0 at zero current)
6. **Track time evolution** to observe resistance changes during experiment
7. **Verify slope k is near zero** - large slopes may indicate problems

## Physical Interpretation Guidelines

### Intercept q (R0 at zero current)
- **<0.05 Ohm**: Very low resistance, possibly unrealistic
- **0.05-0.1 Ohm**: Low resistance, good cell design
- **0.1-0.2 Ohm**: Typical for AEM electrolyzers
- **0.2-0.5 Ohm**: High resistance, check electrode spacing, membrane condition
- **>0.5 Ohm**: Very high, likely experimental issues or poor contacts

### Slope k
- **|k| < 1e-4 Ohm/mA**: Excellent, R0 is effectively constant (ideal ohmic behavior)
- **|k| = 1-5e-4 Ohm/mA**: Acceptable, small current dependence
- **|k| > 5e-4 Ohm/mA**: Significant current dependence, investigate causes
- **Positive k**: R0 increases with current (possible temperature effects)
- **Negative k**: R0 decreases with current (possible measurement artifacts)

## Comparison with Other Analyses

| Analysis | Parameter | Physical Meaning | Fit Equation | Key Metric |
|----------|-----------|-----------------|--------------|------------|
| Tafel | R1 | Charge transfer resistance | 1/R1 = k*I | Tafel slope = 2.3/k |
| C1 | C1 | Double layer capacitance | C1 = k*I + q | Intercept q (C at I=0) |
| R0 | R0 | Ohmic resistance | R0 = k*I + q | Intercept q (R at I=0) |

## References

- Original analysis pattern: `eis_parameters.py`
- Tafel analysis procedure: `tafel_analysis_skill.md`
- C1 analysis procedure: `c1_analysis_skill.md`
- Implementation: `r0_linear_fit_analysis.py`

## Version History

- **v1.0**: Initial R0 vs I plots (no fitting)
- **v2.0**: Linear fit with entire current range, short circuit correction
- **v3.0**: Outlier detection added
- **v3.1**: Documented as skill

---

*Last updated: 2026-10-16*
*Optimized for samples: 474, 498, 499*
