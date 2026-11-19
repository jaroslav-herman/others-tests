---
name: cvxps-processing
description: Load this skill when processing CVXPS (cyclic voltammetry + XPS) data — correlating .mpt CV files with channel-split .xy XPS files on absolute UTC time, deriving per-channel energy corrections from a reference measurement (Au or Ir at OCV), applying corrections to CV-XPS data, generating heatmaps (folded voltage axis, normalized, differential, smoothed), fitting spectra, and extracting voltage cuts. All scripts live in the others-tests repo root; data lives in C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\.
---

# CVXPS Data Processing

## Overview

CVXPS = time-resolved XPS measured during cyclic voltammetry. Data lives in
`C:\Users\Herman\OneDrive - Univerzita Karlova\CVXPS\`:

- `.mpt` — EC-Lab CV export (local time start + elapsed time/s per row)
- `.xy` — SpecsLab Prodigy XPS export with channel-split scans
  (`# Cycle: ..., Scan: N, Channel: C`), regions carry binding energy
  (`# Region: Ir 4f 60`), timestamps are UTC

Time correlation: .mpt local time is shifted by −1 h (CET) to UTC; .xy is already UTC.

**Computer time offset**: The EC computer clock is **64 seconds ahead** of the XPS computer clock. Use `-64` as the TIME_OFFSET_SECONDS parameter in `CVXPS-correlate-channel.py` to align the data properly.

## Scripts (repo root, run with `.venv\Scripts\python.exe`)

| Script | Purpose |
|--------|---------|
| `CVXPS-correlate-channel.py <mpt> <xy> <out_csv> <ch_min> <ch_max> [time_offset_s]` | Correlate CV (.mpt) with channel-filtered XPS (.xy) on UTC time; interpolate Ewe/I at each XPS scan; export CSV with `channel` column. Optional `time_offset_s` (default 0) adjusts for computer clock differences. Use `-64` for current EC/XPS setup. |
| `CVXPS-heatmap-channel.py <csv> <png> <tag>` | Heatmap from correlated CSV: folded y-axis (up 0.6→1.8 V, down 1.8→3.0), auto + fixed bins, row-normalized, spectrum-subtracted (sym-log) |
| `CVXPS-channel-spectra.py <csv> <png>` | Voltage-averaged spectra per channel, y-shifted, viridis color gradient |
| `CVXPS-xy-channel-spectra.py <xy> <png> [standard_xy]` | Same but from raw .xy (no CV correlation); optional standard-mode reference as background |
| `CVXPS-xy-channel-gaussfit.py <xy> <prefix>` | Batch 2-Gaussian fit of each channel spectrum (start middle, propagate outward); outputs fits overlay, params-vs-channel, per-param plots, CSV |
| `CVXPS-xy-channel-correct.py <xy> <fit_csv> <prefix>` | Linear-fit peak centers vs channel (peak2 all ch, peak1 ch>30); compute per-channel energy shift to literature BE; plot corrected spectra |
| `CVXPS-xy-channel-average.py <xy> <fit_csv> <corr_csv> <prefix>` | Cut low-amplitude channels (threshold = max(mean−2σ, p10)); average corrected spectra |
| `CVXPS-compare-corrected-vs-standard.py <avg_csv> <std_xy> <png>` | Compare corrected averaged spectrum with standard-mode reference |
| `CVXPS-xy-find-correction-xcorr.py <xy> <prefix> <be_lit>` | Find energy correction from a FixedEnergies .xy via cross-correlation with average spectrum (robust for weak signals) |
| `CVXPS-apply-correction-ir.py <ir_csv> <au_corr_csv> <prefix>` | Apply Au-derived per-channel energy correction to Ir 4f CV-XPS; folded heatmap (normal, row-normalized to highest 1 eV, differential, smoothed at 3/5/9/15 voltage bins); per-channel spectra; overall average |
| `CVXPS-ir-voltage-cuts.py <ir_csv> <au_corr_csv> <prefix>` | Extract voltage cuts: for 0.3 eV energy windows, plot counts/s vs folded potential |

## Workflow

### 1. Correlate CV + XPS
```bash
python CVXPS-correlate-channel.py ^
  "C:\...\CVXPS\12_CV.mpt" ^
  "C:\...\CVXPS\CV_5mbar_Fixed_energy_20mVs_channel_O1s.xy" ^
  "C:\...\CVXPS\channel_ranges\ch0-145\12_CV_ch0-145_XPS_correlated.csv" 0 145 -64
python CVXPS-heatmap-channel.py "<csv>" "" ch0-145
```
**Note**: The `-64` parameter accounts for the EC computer being 64 seconds ahead of the XPS computer.

### 2. Find energy correction from reference (Au or Ir at OCV)
```bash
# Au reference (strong signal, 2-Gaussian fit works well)
python CVXPS-xy-channel-gaussfit.py "C:\...\CVXPS\Au_ref_Fixed_energy.xy" "C:\...\CVXPS\Au_ref_Fixed_energy"
python CVXPS-xy-channel-correct.py "C:\...\CVXPS\Au_ref_Fixed_energy.xy" ^
  "C:\...\CVXPS\Au_ref_Fixed_energy_fit_params.csv" "C:\...\CVXPS\Au_ref_Fixed_energy"
# -> Au_ref_Fixed_energy_energy_corrections.csv  (per-channel shift, lit Au 4f 84.0/87.67 eV)

# Ir at OCV (weak signal, use xcorr)
python CVXPS-xy-find-correction-xcorr.py "C:\...\CVXPS\OCV_5mbar_Fixed_energy_Ir4f.xy" ^
  "C:\...\CVXPS\OCV_5mbar_Fixed_energy_Ir4f" 61.5
```

### 3. Apply correction to CV-XPS data
```bash
python CVXPS-apply-correction-ir.py ^
  "C:\...\CVXPS\channel_ranges\ch0-145\12_CV_ch0-145_XPS_correlated.csv" ^
  "C:\...\CVXPS\Au_ref_Fixed_energy_energy_corrections.csv" ^
  "C:\...\CVXPS\channel_ranges\ch0-145\12_CV_Ir4f_corrected"
```
Outputs: folded heatmap (normal / row-norm to highest 1 eV / differential / smoothed 3-5-9-15 bins), per-channel corrected spectra, overall average, amplitude cut plot.

### 4. Voltage cuts
```bash
python CVXPS-ir-voltage-cuts.py <ir_csv> <au_corr_csv> <prefix>
```

## Key conventions

- **Folded voltage axis**: up-sweep y=V (0.6→1.8), down-sweep y=2·1.8−V (continues 1.8→3.0); tick labels always show actual potential; white dashed line at 1.8 V.
- **Energy correction**: per-channel shift from reference; apply as `e_corrected = e_raw + shift[ch]`; interpolate onto common grid limited to original BE range.
- **Channel cutting**: threshold = max(mean−2σ, p10) of per-channel max amplitude.
- **Literature BE**: Au 4f 7/2=84.0, 4f 5/2=87.67 eV; Ir 4f 7/2≈60.8, 4f 5/2≈63.8 eV.
- **Float rounding bug**: always `round(round(y/V_BIN)*V_BIN, 2)` when binning to avoid empty heatmap rows from key mismatch.
- **OneDrive file locks**: use `save_fig()` with retry (5 attempts, 2 s sleep).
- **Computer time offset**: EC computer is **64 seconds ahead** of XPS computer. Use `TIME_OFFSET_SECONDS=-64` when correlating data.
- **Always commit and push new scripts to main** after creating them.
