"""Recover periodic PRBS spectra from complementary, incomplete repetitions.

GEIS is never used by this estimator. Actual Ns resets, timestamps, measured
current and Ewe determine the result. It requires reproducible periods and
near-complete phase coverage after pooling repeats. A non-periodic or badly
incomplete record must be measured again rather than filled across long gaps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sampling_audit(time: np.ndarray) -> dict:
    differences = np.diff(np.asarray(time, float))
    if not len(differences) or np.any(differences <= 0):
        raise ValueError("Timestamps must be finite, strictly increasing and unique")
    dt = float(np.median(differences))
    missing = np.maximum(np.rint(differences / dt).astype(int) - 1, 0)
    long_gaps = differences[differences > 2.1 * dt]
    return {
        "nominal_dt_s": dt,
        "n_samples": len(time),
        "missing_grid_fraction": float(missing.sum() / (missing.sum() + len(time))),
        "long_gap_count": len(long_gaps),
        "median_long_gap_s": float(np.median(long_gaps)) if len(long_gaps) else 0.0,
        "max_gap_s": float(differences.max()),
    }


def periodic_fill(values: np.ndarray, valid: np.ndarray, dt: float,
                  max_gap_s: float) -> tuple[np.ndarray, float]:
    """Only fill short residual holes AFTER combining complementary repeats."""
    indexes = np.flatnonzero(valid)
    if len(indexes) < 3:
        raise ValueError("Insufficient measured phase bins")
    largest_interval = float(np.diff(np.r_[indexes, indexes[0] + len(values)]).max() * dt)
    if largest_interval > max_gap_s * (1 + 1e-5):
        raise ValueError(
            f"Unobserved phase interval {largest_interval*1000:.3f} ms exceeds "
            f"{max_gap_s*1000:.3f} ms; repetitions cannot recover this record"
        )
    result = np.interp(np.arange(len(values)), indexes, values[indexes], period=len(values))
    return result, largest_interval


def infer_repeated_bands(data: pd.DataFrame) -> list[tuple[int, int]]:
    ns = data["Ns"].to_numpy(int)  # signed: uint16 differences would wrap!
    resets = np.flatnonzero(np.diff(ns) < 0) + 1
    starts = np.unique(ns[resets])
    if len(starts) != 2:
        raise ValueError(f"Expected two repeated Ns blocks; found reset targets {starts.tolist()}")
    return [(int(starts[0]), int(starts[1] - 1)), (int(starts[1]), int(ns.max()))]


def estimate_repeated_band(data: pd.DataFrame, voltage_column: str,
                          first_ns: int, last_ns: int, fmin: float, fmax: float,
                          band: str, discard_cycles: int = 1,
                          amplitude_floor_fraction: float = .02,
                          max_phase_gap_s: float = .001,
                          heating_frequency: float | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Estimate E/I only on excited Fourier lines of the measured repeat period.

    Percentiles describe leave-one-period-out sensitivity, not independent
    confidence limits or absolute accuracy. No phase/delay fit is applied.
    """
    time = data["time/s"].to_numpy(float)
    ns = data["Ns"].to_numpy(int)
    current = data["I/mA"].to_numpy(float) / 1000
    voltage = data[voltage_column].to_numpy(float)
    if not all(np.isfinite(a).all() for a in (time, current, voltage)):
        raise ValueError("Non-finite timestamps/current/voltage in MPR")
    in_band = (ns >= first_ns) & (ns <= last_ns)
    audit = sampling_audit(time[in_band])
    dt = audit["nominal_dt_s"]
    gap = np.r_[True, np.diff(time) > 2.1 * dt]
    transition = np.r_[False, np.diff(ns) != 0]
    reset = np.r_[False, np.diff(ns) < 0] & in_band
    cycle = np.cumsum(reset)
    cycle -= cycle[np.flatnonzero(in_band)[0]]
    starts = pd.DataFrame({"t": time, "ns": ns, "cycle": cycle})[in_band & transition & ~gap]
    period_candidates = []
    for _, group in starts.groupby("ns"):
        dc = np.diff(group.cycle.to_numpy())
        differences = np.diff(group.t.to_numpy())
        period_candidates.extend((differences[dc > 0] / dc[dc > 0]).tolist())
    if not period_candidates:
        raise ValueError("No repeat period can be inferred from measured step markers")
    period = float(np.median(period_candidates))
    nphase = int(round(period / dt))
    if not np.isclose(nphase * dt, period, rtol=1e-5):
        raise ValueError("Measured repeat period is incompatible with the sampling grid")
    phase_origins = starts[starts.ns == first_ns]
    if phase_origins.empty:
        raise ValueError("No directly recorded first-step transition for phase alignment")
    epoch = float(np.median(phase_origins.t - phase_origins.cycle * period))
    # Validate every step's phase across repeats, not just a fitted average T.
    step_phase = starts.t.to_numpy() - starts.cycle.to_numpy() * period
    phase_residuals = []
    for step in starts.ns.unique():
        values = step_phase[starts.ns.to_numpy() == step]
        phase_residuals.extend(abs(values - np.median(values)))
    max_jitter = float(np.max(phase_residuals))
    if max_jitter > .1 * dt:
        raise ValueError(f"Step timing is not repeatable: jitter {max_jitter:.6g} s")
    phase_float = (time - epoch - cycle * period) / dt
    phase = np.rint(phase_float).astype(int) % nphase
    if np.max(abs(phase_float[in_band] - np.rint(phase_float[in_band]))) > .05:
        raise ValueError("Observed samples do not align to the repeated phase grid")
    good = in_band & (cycle >= discard_cycles)
    is_averaged = voltage_column.startswith("<")
    if is_averaged:
        good &= ~gap  # average over a long gap is not an instantaneous endpoint
    used_cycles = np.unique(cycle[good])
    if len(used_cycles) < 4:
        raise ValueError("At least four retained repetitions are needed for a recovery audit")

    # Remove baseline/heating jointly with the measured periodic template. A
    # voltage-only harmonic fit to the PRBS would subtract some true response.
    # Alternating least squares avoids assigning repeated PRBS to the nuisance
    # baseline (still conditional on additive drift and a repeatable response).
    take = np.flatnonzero(good)
    tg = time[take]
    pg = phase[take]
    cg = cycle[take]
    centered_t = (tg - tg.mean()) / max(float(np.ptp(tg)), 1.)
    nuisance = [np.ones(len(tg)), centered_t, centered_t**2]
    if heating_frequency is not None:
        for h in (1, 2, 3):
            angle = 2 * np.pi * h * heating_frequency * tg
            nuisance.extend([np.sin(angle), np.cos(angle)])
    design = np.column_stack(nuisance)
    counts = np.bincount(pg, minlength=nphase)
    present = counts > 0
    voltage_good = voltage[take]
    coefficients = np.linalg.lstsq(design, voltage_good, rcond=None)[0]
    for _ in range(8):
        residual = voltage_good - design @ coefficients
        wave = np.bincount(pg, weights=residual, minlength=nphase) / np.maximum(counts, 1)
        coefficients = np.linalg.lstsq(design, voltage_good - wave[pg], rcond=None)[0]
    cleaned = voltage_good - design @ coefficients
    cycles_n = len(used_cycles)
    count_cycle = np.zeros((cycles_n, nphase))
    i_cycle = np.zeros_like(count_cycle)
    v_cycle = np.zeros_like(count_cycle)
    for row, value in enumerate(used_cycles):
        selected = cg == value
        count_cycle[row] = np.bincount(pg[selected], minlength=nphase)
        i_cycle[row] = np.bincount(pg[selected], weights=current[take[selected]], minlength=nphase)
        v_cycle[row] = np.bincount(pg[selected], weights=cleaned[selected], minlength=nphase)
    limit = min(max_phase_gap_s, 1 / (20 * fmax))

    def spectrum(rows):
        c = count_cycle[rows].sum(axis=0)
        measured = c > 0
        ii = i_cycle[rows].sum(axis=0) / np.maximum(c, 1)
        vv = v_cycle[rows].sum(axis=0) / np.maximum(c, 1)
        ii, widest = periodic_fill(ii, measured, dt, limit)
        vv, _ = periodic_fill(vv, measured, dt, limit)
        fi = np.fft.rfft(ii - ii.mean()) / nphase
        fv = np.fft.rfft(vv - vv.mean()) / nphase
        z = np.divide(fv, fi, out=np.full_like(fv, np.nan), where=abs(fi) > 1e-15)
        return z, fi, ii, vv, widest

    all_rows = np.arange(cycles_n)
    z, fi, ii, vv, widest = spectrum(all_rows)
    f = np.fft.rfftfreq(nphase, dt)
    jack_z, jack_i = [], []
    for omit in all_rows:
        try:
            zz, ff, _, _, _ = spectrum(all_rows[all_rows != omit])
        except ValueError:
            continue
        jack_z.append(zz)
        jack_i.append(ff)
    if len(jack_z) < 4:
        raise ValueError("Too few adequately sampled leave-one-period reconstructions")
    jack_z, jack_i = np.asarray(jack_z), np.asarray(jack_i)
    # Jackknife standard error of the coherent INPUT, used as a noise floor.
    input_se = np.sqrt((len(jack_i)-1) * np.mean(abs(jack_i - jack_i.mean(axis=0))**2, axis=0))
    snr_db = 20 * np.log10(np.maximum(abs(fi), 1e-20) / np.maximum(input_se, 1e-20))
    requested = (f >= fmin) & (f <= fmax)
    amplitude_limit = amplitude_floor_fraction * float(np.max(abs(fi[requested])))
    excited = (abs(fi) >= amplitude_limit) & (snr_db >= 20)
    valid = requested & excited & np.isfinite(z)
    # Input repeatability measures whether combining cycles is defensible.
    i_residual = current[take] - ii[pg]
    i_rms = np.sqrt(np.mean((current[take]-current[take].mean())**2))
    repeat_error = float(np.sqrt(np.mean(i_residual**2)) / i_rms)
    if repeat_error > .1:
        raise ValueError(f"Current waveform changes across periods ({repeat_error:.1%} RMS)")
    relative = np.divide(jack_z, z[None, :], out=np.full_like(jack_z, np.nan),
                         where=np.isfinite(z[None, :]) & (abs(z[None, :]) > 0))
    relative_phases = np.angle(relative, deg=True)
    phase_central = np.angle(z, deg=True)
    records = pd.DataFrame({
        "frequency_Hz": f[valid], "Re_Z_ohm": z.real[valid], "Im_Z_ohm": z.imag[valid],
        "minus_Im_Z_ohm": -z.imag[valid], "abs_Z_ohm": abs(z[valid]),
        "phase_deg": phase_central[valid],
        "abs_Z_p10_ohm": np.percentile(abs(jack_z[:, valid]), 10, axis=0),
        "abs_Z_p90_ohm": np.percentile(abs(jack_z[:, valid]), 90, axis=0),
        "phase_p10_deg": phase_central[valid] + np.percentile(relative_phases[:, valid], 10, axis=0),
        "phase_p90_deg": phase_central[valid] + np.percentile(relative_phases[:, valid], 90, axis=0),
        "input_peak_amplitude_A": 2*abs(fi[valid]), "input_repeat_snr_dB": snr_db[valid],
        "estimator": "periodic reconstruction FFT", "band": band,
        "n_periods": cycles_n, "period_s": period, "sample_rate_Hz": 1/dt,
        "phase_coverage": float(present.mean()), "max_phase_interval_s": widest,
        "uncertainty_kind": "10-90% leave-one-period sensitivity; excludes systematic errors",
        "voltage_channel": voltage_column, "applied_delay_correction_s": 0.,
    })
    # Flag poor repeatability without reference to GEIS. Keep these in the
    # all-lines export so rejection is auditable, not a cosmetic smoothing.
    records["quality_ok"] = (
        (records.phase_p90_deg-records.phase_p10_deg <= 3.)
        & ((records.abs_Z_p90_ohm-records.abs_Z_p10_ohm)/records.abs_Z_ohm <= .1)
    )
    records["quality_reason"] = np.where(records.quality_ok, "pass", "leave-one-period sensitivity exceeds 3 deg or 10 percent")
    audit.update({
        "band": band, "first_ns": first_ns, "last_ns": last_ns,
        "epoch_s": epoch, "period_s": period, "retained_periods": cycles_n,
        "discarded_initial_periods": discard_cycles, "phase_coverage": float(present.mean()),
        "max_remaining_phase_interval_s": widest, "step_repeat_jitter_s": max_jitter,
        "current_repeat_RMS_fraction": repeat_error,
        "voltage_repeat_residual_RMS_V": float(np.std(cleaned-vv[pg])),
        "accepted_frequencies": int(valid.sum()), "rejected_frequencies": int((requested & ~valid).sum()),
        "sensitivity_rejected_frequencies": int((~records.quality_ok).sum()),
        "leave_one_period_reconstructions": len(jack_z),
        "averaged_voltage": is_averaged, "excluded_postgap_averages": int((in_band & gap).sum()) if is_averaged else 0,
    })
    waveforms = {"phase_time_s": np.arange(nphase)*dt, "current_A": ii, "voltage_V": vv,
                 "counts": counts, "frequency_Hz": f, "input_amplitude_A": 2*abs(fi),
                 "accepted": valid}
    return records, audit, waveforms
