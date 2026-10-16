# Agent Instructions and Rules

## Critical Rules

### 1. ALWAYS PUSH TO MAIN
**After every analysis, every script creation, every file modification:**
- Commit all changes
- Push to `origin main` branch
- Verify the push was successful

This ensures the user can always download the latest results from GitHub.

### 2. File Structure
- Store all sample-specific results in `AEMs/sample_{id}/` folders
- Keep scripts in `AEMs/` root for reuse
- Maintain consistent naming conventions

### 3. Analysis Patterns
- Follow established skills for Tafel, C1, R0 analyses
- Use consistent parameters (current ranges, outlier thresholds)
- Always correct for short circuit where applicable

## Skills Documentation

### Tafel Analysis (R1)
- **Script**: `tafel_analysis_all_samples.py`
- **Skill**: `tafel_analysis_skill.md`
- **Parameters**: 5-700 mA range, forced zero intercept, Z-score > 2.5 for outliers

### C1 Capacitance Analysis
- **Script**: `c1_linear_fit_analysis.py`
- **Skill**: `c1_analysis_skill.md`
- **Parameters**: 10-700 mA range, linear fit C1 = k*I + q, Z-score > 2.5 for outliers

### R0 Ohmic Resistance Analysis
- **Script**: `r0_linear_fit_analysis.py`
- **Skill**: `r0_analysis_skill.md`
- **Parameters**: No current limits, short circuit correction, linear fit R0 = k*I + q, Z-score > 2.5 for outliers

### C2, R2, C3, R3 Linear Fit Analysis
- **Script**: `c2_r2_c3_r3_linear_fit_analysis.py`
- **Parameters**: 10 mA minimum, no upper limit, short circuit correction, linear fit param = k*I + q, Z-score > 2.5 for outliers

## Workflow Checklist

1. [ ] Load data from eisyFIT CSV files
2. [ ] Apply appropriate analysis method (Tafel, C1, R0, etc.)
3. [ ] Generate plots with proper labels and legends
4. [ ] Save CSV summaries of results
5. [ ] Organize files in correct folder structure
6. [ ] **Commit all changes to git**
7. [ ] **Push to origin main**
8. [ ] Verify on GitHub that files are visible

## Last Updated
2026-10-16
