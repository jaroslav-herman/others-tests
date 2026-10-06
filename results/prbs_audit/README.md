# PRBS processing and acquisition audit — 28 September 2026

Both the processing and the generated measurement procedure had problems. The
dominant demonstrated error in the previous fast-band result was interpolating
across missing data. The calculations previously delivered as reliable impedance
and the fitted 176 microsecond "delay" should not be used as calibrated results.

The source audited here is `VIII_Day9_PRBS_500mA_20mA_amp_C01.mpr`. All MPR data
are still loaded with `wepy.basics.read_file`. Original measurements and earlier
result files have not been overwritten.

## Evidence

| Check | Slow block | Fast block |
|---|---:|---:|
| Nominal sample interval | 0.2 ms | 0.2 ms |
| Missing samples relative to that grid | 42.80% | 42.48% |
| Typical long interval between records | 11.4 ms | 11.2 ms |
| Long intervals | 2,486 | 1,253 |
| Largest interval | 64.4 ms | 46.8 ms |
| Actual repeated block period | 5.149200 s | 4.194400 s |
| Repetitions used after discarding the first | 12 | 7 |
| Phase positions observed after pooling repeats | 99.24% | 97.56% |
| Largest remaining phase interval | 0.4 ms | 0.4 ms |
| Current variation between repeats, RMS / perturbation RMS | 0.192% | 0.174% |

An 11 ms gap spans almost three 4 ms PRBS bits. `np.interp` in the old processing
replaced the missing transitions with ramps. A median timestamp interval of
0.2 ms does not establish continuous 5 kHz acquisition. The MPR demonstrates
missing observations; it does not establish whether the cause was instrument
buffering, communications, host load or another recording limitation.

The old order-9 LFSR tap implementation has a 73-bit period, not 511 bits. The
stored 511-bit block is seven repeats of that short sequence. With observed
step overhead its underlying period is 0.7356 s (spectral spacing about 1.3594
Hz). Many of the old arbitrary CWT frequency points had negligible excitation.
The order-10 fast sequence does have 1023 bits. Both measured block periods
include an additional 0.2 ms per CP step: 196 steps in the old slow block and
512 in the fast block. Using nominal bit durations to infer the frequency axis
would therefore introduce another error.

The installed Galvani column map was also wrong for legacy IDs. Upstream maps
174 to `<Ewe>/V`, 178/179 to charges, 211 to an eight-byte charge and 212 to a
cycle counter. The old fallback's "unknown four bytes" were part of a real
charge field. The earlier `E1/V`/`E2/V` values were misdecoded charge/counter
data, not evidence of physically corrupted electrode channels. The numerical
voltage and current used before were at the correct offsets, so correcting
the names alone does not repair the spectrum. It does establish that voltage
was averaged, whereas the former explanation called it instantaneous Ewe.
[Upstream Galvani definitions](https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py)

The paper used measured voltage and current at 5 kHz and an order-16 PRBS,
with its CWT assumptions and bandwidth selected for those records. A matching
requested recording interval alone does not reproduce that experiment.
[Debenjak et al., paper](https://dsc.ijs.si/en/files/274/)

## Implemented recovery

1. Correct the legacy MPR data types while keeping the required wepy reader.
2. Infer blocks, repetitions and true periods from signed Ns differences and
   clean step timestamps. Check every recorded step phase for repeatability.
3. Discard the initial repetition. Exclude averaged voltage observations after
   long gaps; they represent an interval, not the endpoint waveform value.
4. Combine observations at the same phase in different repetitions. Remove
   additive drift and the CP-derived 0.251086 Hz heating harmonics jointly with
   the repeated voltage waveform. Do not fit voltage-only heating directly to
   unseparated PRBS response.
5. Permit only short residual interpolation after pooling. At 80 Hz the limit
   is 0.625 ms; this file has only 0.4 ms intervals left.
6. Compute measured E/I on excited Fourier lines. Require input amplitude above
   2% of the band's peak and input repeat SNR of at least 20 dB. Reject output
   lines with leave-one-period 10–90% sensitivity exceeding 3 degrees or 10%
   magnitude. Export all excited lines, including the four failed fast-band
   lines, separately for inspection.
7. Keep CWT available for continuous data, but refuse large gaps. Tests verify
   that the corrected Morlet convolution has the right phase orientation.

This is a periodic reconstruction/Fourier recovery, not a claim to have reproduced
the paper's CWT estimator. Repeated current waveforms are exceptionally consistent
in this file, which supports pooling, but voltage stationarity and acquisition
transfer effects still limit accuracy. No GEIS-derived phase correction, gain,
frequency adjustment, equivalent-circuit fit or smoothing is applied.

## Matched-frequency comparison

The old CWT result is interpolated onto the accepted recovered frequencies
within its original band, then both are compared to the same interpolated GEIS
reference. Slow: 21 lines, 2.719–29.908 Hz. Fast: 206 lines, approximately
30–79.868 Hz. This avoids comparing different point distributions.

| Metric | Slow: previous → recovered | Fast: previous → recovered |
|---|---:|---:|
| Median complex relative error | 11.09% → 3.30% | 23.17% → 6.94% |
| Phase RMS difference | 7.26° → 1.45° | 10.74° → 4.13° |

The recovery also provides a 20–30 Hz fast/slow overlap. Across all accepted
fast lines (20.027–79.868 Hz), median complex error is 6.47% and phase RMS is
3.85 degrees. There are 269 accepted lines in the final CSV.

Heating-on/off sensitivity is small for most lines: median complex change is
about 0.00033 ohm. A narrow region near 37 Hz is much more sensitive, consistent
with mixing of the nuisance baseline through the sampling mask; those unstable
points deserve caution. The broad fast-band error was present with both heating
choices, so heating removal alone could not fix it.

The remaining fast phase difference is systematic. Averaged Ewe versus the
recorded current, acquisition filtering/timing, residual reconstruction error
and changes in cell state remain candidates; this run cannot identify their
individual contributions. The GEIS DC values are about 499.923 mA and 1.523876 V.
The old fitted 176 microseconds was a slope against GEIS, not an independently
measured channel delay. Applying it as a correction would be unjustified.

The shaded bands show leave-one-period sensitivity, not an 80% confidence
interval for absolute impedance accuracy. They omit systematic errors and use
overlapping reconstructions. A short clean recording on a known dummy circuit
is needed to establish acquisition amplitude/phase accuracy.

The older ±10 mA `_3` file was checked too. Its fast block still has a 1.4 ms
unobserved phase interval after pooling and discarding the initial repetition;
the strict recovery correctly refuses it rather than hiding that gap.

## Files and reproduction

- `20mA_recovered.csv`: accepted impedance estimates.
- `20mA_recovered_all_excited_lines.csv`: includes repeatability-rejected points.
- `20mA_recovered_geis_comparison.csv`: GEIS residuals for accepted estimates.
- `20mA_recovered_audit.json`: timestamps, coverage, repeatability and limitations.
- `20mA_recovered_bode.png`, `20mA_recovered_nyquist.png`: old/new/GEIS comparison.
- `20mA_recovered_sampling.png`: raw gaps and input spectral excitation.
- `20mA_no_heating*`: the heating sensitivity diagnostic, before the final
  output-repeatability filter; use its all-point comparison accordingly.

From the project directory:

```powershell
uv run python calculate_prbs_impedance.py --mpr "C:\Users\Herman\OneDrive - Univerzita Karlova\PRBS\VIII_Day9_PRBS_500mA_20mA_amp_C01.mpr" --output results/prbs_audit/20mA_recovered --previous-results prbs_impedance_500mA_20mA_amp_cwt_corrected.csv
uv run python -m unittest discover -s tests -p test_prbs_recovery.py -v
```

Six tests pass: positive/negative CWT phase; maximal-length sequence states and
circular autocorrelation for all supported orders; rejection of large gaps;
fixed-width MPS rows; recovery of known negative/positive-phase networks with
simulated missing bursts and interval averages; and real-MPR charge consistency.
The synthetic recovery tests require median complex error below 0.5% and 90th
percentile below 2%. They validate the algorithm under their assumptions, not
the instrument's physical transfer function.

## Revised settings for a fresh measurement

`../../PRBS_500mA_20mA_Ewe_checked.mps` is one CP technique with 641 steps and
219,184 bytes. It preserves 500 ±20 mA and the existing 1 A range, bandwidth 4,
2 V potential limit, and Ewe-control header from the user's reference file.

- Stabilize at 500 mA for 30 s.
- Slow block: true PRBS-8, 255 bits at 10 ms, 13 passes. This uses 128 runs;
  it is longer than the faulty 73-bit sequence but keeps the settings compact.
- Fast block: true PRBS-10, 1023 bits at 4 ms, 8 passes, 512 runs.
- Record instantaneous `Ewe` every 1 ms. This lowers the requested data rate
  to 1000 points/s and still gives 12.5 samples per 80 Hz cycle.
- Disable the charge cutoff (`dQM = 0`). Step duration is controlled by time;
  preserve the separate voltage limit. Charge cutoff at the exact nominal
  step charge could otherwise truncate a step or alter loop execution.

```powershell
uv run python generate_prbs_cp_mps.py --slow-order 8 --fast-order 10 --amplitude-ma 20 --record-ms 1 --output PRBS_500mA_20mA_Ewe_checked.mps
```

The file's fixed widths, row counts, loop destinations and numerical fields
were checked. It has not been loaded in EC-Lab or run on the instrument here.
The 1 ms rate is a practical starting point, not a guarantee of uninterrupted
recording. Verify actual timestamps in a short pilot run before treating its
spectrum as quantitative. A continuous record supports the CWT route; it must
still use the observed current and timing. If the remaining phase discrepancy
persists on a clean dummy-circuit record, acquisition filtering/averaging and
channel timing need calibration rather than another fitted GEIS correction.
