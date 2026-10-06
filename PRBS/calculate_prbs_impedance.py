"""Calculate impedance from a two-band PRBS PEM-electrolyzer experiment.

Default: reconstruct a repeated waveform from complementary observations,
validate repeat timing and coverage, then use E/I at excited Fourier lines.
This is conditional recovery for the gapped Day-9 records, not the paper's
CWT method. GEIS is used only for comparison and never to adjust the spectrum.

For continuous records, --estimator cwt retains the corrected Morlet method.
It now refuses large gaps instead of inventing ramps across missing pulses.

The input is loaded through ``wepy.basics.read_file``. A scoped correction to
the installed Galvani column map restores the documented legacy MPR IDs. In
particular ID 211 is an eight-byte charge, not a four-byte electrode voltage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft, signal
from scipy.ndimage import uniform_filter1d

import wepy.basics as we

plt.rcParams["text.usetex"] = False


DATA_ROOT = Path(r"C:\Users\Herman\OneDrive - Univerzita Karlova\PRBS")
DEFAULT_MPR_CANDIDATES = (
    DATA_ROOT / "VIII_Day9_PRBS_500mA_20mA_amp_C01.mpr",
    DATA_ROOT / "VIII_Day9_PRBS_500mA_3_C01.mpr",
    DATA_ROOT / "VIII_Day9_PRBS_500mA_2_C01.mpr",
    DATA_ROOT / "VIII_Day9_PRBS_500mA_C01.mpr",
)
DEFAULT_REFERENCE_MPR = DATA_ROOT / "VIII_Day9_CP_500mA_C01.mpr"
DEFAULT_GEIS_MPR = DATA_ROOT / "VIII_Day9_GEIS_500mA_C01.mpr"
DEFAULT_OUTPUT = Path("prbs_impedance_500mA_recovered")

PAPER_W0_LOW = 2.0 * np.pi
PAPER_W0_MIDDLE = 20.0 * np.pi
PAPER_W0_HIGH = 100.0 * np.pi


def resolve_default_prbs() -> Path:
    for candidate in DEFAULT_MPR_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_MPR_CANDIDATES[0]


def read_mpr(path: Path) -> pd.DataFrame:
    """Read through wepy using the upstream Galvani legacy column definitions.

    Source: https://github.com/echemdata/galvani/blob/master/galvani/BioLogic.py
    Restore the installed map even on failure; do not modify the dependency.
    Never pad records or infer a voltage channel from its numeric magnitude.
    """
    import galvani.BioLogic as biologic

    corrections = {
        174: ("<Ewe>/V", "<f4"),
        178: ("(Q-Qo)/C", "<f4"),
        179: ("dQ/C", "<f4"),
        211: ("Q charge/discharge/mA.h", "<f8"),
        212: ("half cycle", "<u4"),
        213: ("z cycle", "<u4"),
    }
    original_map = biologic.VMPdata_colID_dtype_map
    biologic.VMPdata_colID_dtype_map = {**original_map, **corrections}
    try:
        return we.read_file(str(path))
    finally:
        biologic.VMPdata_colID_dtype_map = original_map


def choose_column(data: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    raise KeyError(f"Could not find {label}. Available columns: {list(data.columns)}")


def get_ewe_column(data: pd.DataFrame, label: str) -> str:
    """Find a correctly decoded instantaneous or averaged Ewe channel."""
    if "Ewe/V" in data.columns:
        return "Ewe/V"
    if "<Ewe>/V" in data.columns:
        return "<Ewe>/V"
    raise KeyError(f"Could not find Ewe voltage for {label}. Available columns: {list(data.columns)}")


def paper_w0(frequency_hz: float) -> float:
    """Return the Morlet central-frequency parameter used in the paper."""
    if frequency_hz < 1.0:
        return PAPER_W0_LOW
    if frequency_hz < 100.0:
        return PAPER_W0_MIDDLE
    return PAPER_W0_HIGH


def wavelet_w0(frequency_hz: float, mode: str) -> float:
    """Select the wavelet width for a paper or compact PRBS measurement."""
    if mode == "compact":
        # The present order-9/repeated sequence has sparse spectral lines.
        # A broad wavelet is needed to average across those lines.
        return PAPER_W0_LOW
    if mode == "paper":
        return paper_w0(frequency_hz)
    raise ValueError(f"Unknown wavelet mode: {mode}")


def morlet_scale(frequency_hz: float, w0: float) -> float:
    """Convert target frequency to Morlet scale using the paper's Eq. (9)."""
    return (w0 + np.sqrt(2.0 + w0**2)) / (4.0 * np.pi * frequency_hz)


def morlet_coefficients(
    x: np.ndarray,
    sample_period: float,
    frequencies: np.ndarray,
    wavelet_mode: str = "compact",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate paper-convention Morlet CWT coefficients.

    For convolution, the CWT kernel is ``conj(psi(-t))``. The previous
    implementation used ``conj(psi(t))`` directly, which selected the
    negative-frequency analytic component and conjugated the impedance phase.
    For the Morlet wavelet, the correctly time-reversed/conjugated kernel has
    a positive carrier, as constructed below.
    """
    x = np.asarray(x, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if x.ndim != 1 or frequencies.ndim != 1 or not len(x) or not len(frequencies):
        raise ValueError("x and frequencies must be non-empty one-dimensional arrays")
    if np.any(frequencies <= 0):
        raise ValueError("CWT frequencies must be positive")

    w0_values = np.array([wavelet_w0(float(f), wavelet_mode) for f in frequencies])
    scales = np.array([morlet_scale(float(f), float(w0)) for f, w0 in zip(frequencies, w0_values)])
    support_samples = int(np.ceil(12.0 * np.max(scales) / sample_period))
    n_fft = fft.next_fast_len(len(x) + support_samples)
    x_fft = fft.fft(x, n_fft)
    time = (np.arange(n_fft) - n_fft // 2) * sample_period
    coefficients = np.empty((len(frequencies), len(x)), dtype=complex)

    for row, (scale, w0) in enumerate(zip(scales, w0_values)):
        correction = np.exp(-0.5 * w0**2)
        kernel_centered = (
            np.exp(-0.5 * (time / scale) ** 2)
            * (np.exp(1j * w0 * time / scale) - correction)
            * (np.pi ** -0.25)
            / np.sqrt(scale)
        )
        kernel = np.fft.ifftshift(kernel_centered)
        convolution = fft.ifft(x_fft * fft.fft(kernel, n_fft)) * sample_period
        coefficients[row] = convolution[: len(x)]
    return coefficients, scales, w0_values


def highpass_detrend(x: np.ndarray, sample_rate: float, cutoff_hz: float) -> np.ndarray:
    """Remove DC and slow operating-point drift without phase shift."""
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return signal.detrend(x, type="linear")
    sos = signal.butter(3, cutoff_hz / (sample_rate / 2), btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def interpolate_uniform(time: np.ndarray, *signals: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return uniformly sampled arrays required by FFT convolution."""
    time = np.asarray(time, dtype=float)
    signals = tuple(np.asarray(signal_value, dtype=float) for signal_value in signals)
    dt = float(np.median(np.diff(time)))
    differences = np.diff(time)
    if np.any(differences <= 0) or not np.isfinite(time).all():
        raise ValueError("CWT requires finite, increasing, unique timestamps")
    if differences.max() > 2.1 * dt:
        raise ValueError(
            f"CWT refused: timestamp gaps up to {differences.max()*1000:.3f} ms "
            f"at nominal dt={dt*1000:.3f} ms. Use --estimator repeated for "
            "repeatable sequences or acquire a continuous record."
        )
    if np.allclose(differences, dt, rtol=2e-4, atol=max(dt * 2e-4, 1e-12)):
        return (time, *signals)
    uniform_time = np.arange(time[0], time[-1] + 0.5 * dt, dt)
    return (uniform_time, *(np.interp(uniform_time, time, value) for value in signals))


def local_wavelet_coherence(
    voltage_coefficients: np.ndarray,
    current_coefficients: np.ndarray,
    scale: float,
    sample_period: float,
) -> np.ndarray:
    """Estimate local squared wavelet coherence for one frequency."""
    width = max(5, int(round(2.0 * scale / sample_period)))
    width = min(width, len(voltage_coefficients))
    if width % 2 == 0:
        width -= 1
    width = max(width, 3)
    cross = uniform_filter1d(
        voltage_coefficients * np.conj(current_coefficients), width, mode="nearest"
    )
    voltage_power = uniform_filter1d(np.abs(voltage_coefficients) ** 2, width, mode="nearest")
    current_power = uniform_filter1d(np.abs(current_coefficients) ** 2, width, mode="nearest")
    denominator = voltage_power * current_power
    return np.divide(np.abs(cross) ** 2, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def estimate_band(
    data: pd.DataFrame,
    time_column: str,
    voltage_column: str,
    current_column: str,
    start_s: float,
    end_s: float,
    f_min: float,
    f_max: float,
    n_frequencies: int,
    minimum_coherence: float,
    wavelet_mode: str = "compact",
) -> pd.DataFrame:
    selected = data[(data[time_column] >= start_s) & (data[time_column] < end_s)].copy()
    selected = selected.dropna(subset=[time_column, voltage_column, current_column])
    if len(selected) < 1000:
        raise ValueError(f"Only {len(selected)} samples available in {start_s:g}–{end_s:g} s")

    time, voltage, current = interpolate_uniform(
        selected[time_column].to_numpy(float),
        selected[voltage_column].to_numpy(float),
        selected[current_column].to_numpy(float) / 1000.0,
    )
    dt = float(np.median(np.diff(time)))
    sample_rate = 1.0 / dt
    frequencies = np.geomspace(f_min, f_max, n_frequencies)

    voltage = highpass_detrend(voltage, sample_rate, max(f_min / 3.0, 0.03))
    current = highpass_detrend(current, sample_rate, max(f_min / 3.0, 0.03))
    voltage_coefficients, scales, w0_values = morlet_coefficients(
        voltage, dt, frequencies, wavelet_mode
    )
    current_coefficients, _, _ = morlet_coefficients(current, dt, frequencies, wavelet_mode)

    records = []
    relative_time = time - time[0]
    for index, (frequency, scale, w0) in enumerate(zip(frequencies, scales, w0_values)):
        coi_edge_s = np.sqrt(2.0) * scale
        coi = (relative_time >= coi_edge_s) & (relative_time <= relative_time[-1] - coi_edge_s)
        if np.count_nonzero(coi) < 20:
            continue

        voltage_coefficient = voltage_coefficients[index]
        current_coefficient = current_coefficients[index]
        input_amplitude = np.abs(current_coefficient)
        amplitude_threshold = np.percentile(input_amplitude[coi], 10.0)
        input_amplitude_median = np.median(input_amplitude[coi])
        coherence = local_wavelet_coherence(voltage_coefficient, current_coefficient, scale, dt)
        z_local = np.divide(
            voltage_coefficient,
            current_coefficient,
            out=np.full_like(voltage_coefficient, np.nan + 1j * np.nan),
            where=np.abs(current_coefficient) > 0,
        )
        valid = (
            coi
            & np.isfinite(z_local)
            & (input_amplitude >= amplitude_threshold)
            & (coherence >= minimum_coherence)
        )
        z_local = z_local[valid]
        local_coherence = coherence[valid]
        if len(z_local) < 20:
            continue

        z_median = np.median(z_local.real) + 1j * np.median(z_local.imag)
        magnitudes = np.abs(z_local)
        phases = np.angle(z_local, deg=True)
        records.append(
            {
                "frequency_Hz": frequency,
                "Re_Z_ohm": z_median.real,
                "Im_Z_ohm": z_median.imag,
                "minus_Im_Z_ohm": -z_median.imag,
                "abs_Z_ohm": abs(z_median),
                "phase_deg": np.angle(z_median, deg=True),
                "abs_Z_p10_ohm": np.percentile(magnitudes, 10),
                "abs_Z_p90_ohm": np.percentile(magnitudes, 90),
                "phase_p10_deg": np.percentile(phases, 10),
                "phase_p90_deg": np.percentile(phases, 90),
                "coherence_median": np.median(local_coherence),
                "coherence_p10": np.percentile(local_coherence, 10),
                "input_coefficient_median_A": input_amplitude_median,
                "input_coefficient_p10_A": amplitude_threshold,
                "n_local_estimates": len(z_local),
                "coi_edge_s": coi_edge_s,
                "w0_rad": w0,
                "wavelet_mode": wavelet_mode,
                "sample_rate_Hz": sample_rate,
                "start_s": start_s,
                "end_s": end_s,
            }
        )
    return pd.DataFrame.from_records(records)


def load_geis(data: pd.DataFrame) -> pd.DataFrame:
    required = ("freq/Hz", "Re(Z)/Ohm", "-Im(Z)/Ohm")
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"GEIS file is missing {missing}; available columns: {list(data.columns)}")
    geis = data.loc[:, list(required)].dropna().copy()
    geis = geis[geis["freq/Hz"] > 0].sort_values("freq/Hz").drop_duplicates("freq/Hz")
    geis["Im_Z_ohm"] = -geis["-Im(Z)/Ohm"]
    geis["abs_Z_ohm"] = np.hypot(geis["Re(Z)/Ohm"], geis["Im_Z_ohm"])
    geis["phase_deg"] = np.degrees(np.arctan2(geis["Im_Z_ohm"], geis["Re(Z)/Ohm"]))
    return geis.reset_index(drop=True)


def compare_with_geis(results: pd.DataFrame, geis: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """Interpolate GEIS values and estimate any remaining linear phase delay."""
    geis_frequency = geis["freq/Hz"].to_numpy(float)
    log_geis_frequency = np.log(geis_frequency)
    overlap = results["frequency_Hz"].between(geis_frequency.min(), geis_frequency.max())
    comparison = results.loc[overlap].copy()
    if comparison.empty:
        return comparison, None

    log_frequency = np.log(comparison["frequency_Hz"].to_numpy(float))
    for source, target in (
        ("Re(Z)/Ohm", "geis_Re_Z_ohm"),
        ("Im_Z_ohm", "geis_Im_Z_ohm"),
        ("abs_Z_ohm", "geis_abs_Z_ohm"),
        ("phase_deg", "geis_phase_deg"),
    ):
        comparison[target] = np.interp(log_frequency, log_geis_frequency, geis[source].to_numpy(float))
    comparison["Re_residual_ohm"] = comparison["Re_Z_ohm"] - comparison["geis_Re_Z_ohm"]
    comparison["Im_residual_ohm"] = comparison["Im_Z_ohm"] - comparison["geis_Im_Z_ohm"]
    comparison["phase_residual_deg"] = comparison["phase_deg"] - comparison["geis_phase_deg"]

    delay_mask = comparison["frequency_Hz"].between(10.0, min(150.0, geis_frequency.max()))
    delay_data = comparison.loc[delay_mask].sort_values("frequency_Hz")
    delay_s = None
    if len(delay_data) >= 4:
        phase_residual = np.unwrap(np.radians(delay_data["phase_residual_deg"].to_numpy(float)))
        angular_frequency = 2.0 * np.pi * delay_data["frequency_Hz"].to_numpy(float)
        slope, _ = np.polyfit(angular_frequency, phase_residual, 1)
        delay_s = float(-slope)
    return comparison, delay_s


def detect_band_windows(
    data: pd.DataFrame,
    time_column: str,
    current_column: str,
) -> tuple[float, float, float, float]:
    """Detect stabilization, slow PRBS and fast PRBS intervals from current."""
    time = data[time_column].to_numpy(float)
    current = data[current_column].to_numpy(float)
    edges = np.arange(time[0], time[-1] + 1.0, 1.0)
    bin_index = np.clip(np.digitize(time[:-1], edges) - 1, 0, len(edges) - 1)
    transitions = np.bincount(
        bin_index,
        weights=(np.abs(np.diff(current)) > 2.0),
        minlength=len(edges),
    )
    active = transitions >= 10.0
    active_indices = np.flatnonzero(active)
    if len(active_indices) < 3:
        raise ValueError("Could not detect two PRBS bands from measured current")
    first = int(active_indices[0])
    fast_candidates = active_indices[(active_indices > first + 2) & (transitions[active_indices] >= 80.0)]
    if len(fast_candidates) == 0:
        raise ValueError("Could not detect the fast PRBS band from measured current")
    fast_start_bin = int(fast_candidates[0])
    last = int(active_indices[-1])
    slow_start = float(edges[first])
    slow_end = float(edges[fast_start_bin])
    fast_start = slow_end
    fast_end = min(float(edges[last] + 1.0), float(time[-1]))
    if fast_end <= fast_start or slow_end <= slow_start:
        raise ValueError("Detected PRBS band windows are invalid")
    return slow_start, slow_end, fast_start, fast_end


def make_plots(results: pd.DataFrame, geis: pd.DataFrame, output_stem: Path,
               previous: pd.DataFrame | None = None) -> None:
    geis = geis[geis["freq/Hz"].between(results.frequency_Hz.min(), results.frequency_Hz.max())]
    fig, axes = plt.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    for band, group in results.groupby("band"):
        group = group.sort_values("frequency_Hz")
        axes[0].loglog(group["frequency_Hz"], group["abs_Z_ohm"], ".-", label=band)
        axes[0].fill_between(
            group["frequency_Hz"], group["abs_Z_p10_ohm"], group["abs_Z_p90_ohm"], alpha=0.15
        )
        axes[1].semilogx(group["frequency_Hz"], group["phase_deg"], ".-", label=band)
        axes[1].fill_between(group["frequency_Hz"], group["phase_p10_deg"], group["phase_p90_deg"], alpha=.15)
    if previous is not None:
        for number, (_, group) in enumerate(previous.groupby("band")):
            axes[0].loglog(group.frequency_Hz, group.abs_Z_ohm, "--", color="0.65", label="Previous CWT" if number==0 else None)
            axes[1].semilogx(group.frequency_Hz, group.phase_deg, "--", color="0.65")
    axes[0].loglog(geis["freq/Hz"], geis["abs_Z_ohm"], "k.", ms=4, label="GEIS reference")
    axes[1].semilogx(geis["freq/Hz"], geis["phase_deg"], "k.", ms=4, label="GEIS reference")
    axes[0].set_ylabel("|Z| / ohm")
    axes[1].set_ylabel("phase / degree")
    axes[1].set_xlabel("frequency / Hz")
    axes[0].legend()
    axes[1].legend()
    if "uncertainty_kind" in results:
        fig.suptitle("Shading: 10-90% leave-one-period sensitivity; not total accuracy", fontsize=10)
    fig.savefig(output_stem.with_name(output_stem.name + "_bode.png"), dpi=250)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for band, group in results.groupby("band"):
        group = group.sort_values("frequency_Hz")
        ax.plot(group["Re_Z_ohm"], group["minus_Im_Z_ohm"], "." if len(group)>80 else ".-",
                ms=3 if len(group)>80 else 5, label=band)
    if previous is not None:
        for number, (_, group) in enumerate(previous.groupby("band")):
            ax.plot(group.Re_Z_ohm, group.minus_Im_Z_ohm, "--", color="0.65", label="Previous CWT" if number==0 else None)
    ax.plot(geis["Re(Z)/Ohm"], geis["-Im(Z)/Ohm"], "k.", ms=4, label="GEIS reference")
    ax.set_xlabel("Re(Z) / ohm")
    ax.set_ylabel("-Im(Z) / ohm")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(output_stem.with_name(output_stem.name + "_nyquist.png"), dpi=250)
    plt.close(fig)


def run_repeated_analysis(data, voltage_column, geis, heating_frequency, args, prbs_path):
    from prbs_repeated_spectrum import infer_repeated_bands, estimate_repeated_band

    blocks = infer_repeated_bands(data)
    spectra, audits, waves = [], [], []
    for (lo, hi), (fmin, fmax), name in zip(
        blocks, ((2., 30.), (20., args.fast_fmax)), ("slow PRBS", "fast PRBS")
    ):
        spectrum, audit, wave = estimate_repeated_band(
            data, voltage_column, lo, hi, fmin, fmax, name,
            heating_frequency=None if args.no_heating else heating_frequency,
        )
        spectra.append(spectrum)
        audits.append(audit)
        waves.append(wave)
    results = pd.concat(spectra, ignore_index=True)
    results.to_csv(args.output.with_name(args.output.name+"_all_excited_lines.csv"), index=False)
    results = results[results.quality_ok].copy()
    comparison, _ = compare_with_geis(results, geis)
    comparison["complex_relative_error_pct"] = 100*np.hypot(comparison.Re_residual_ohm, comparison.Im_residual_ohm)/comparison.geis_abs_Z_ohm
    results.to_csv(args.output.with_suffix(".csv"), index=False)
    comparison.to_csv(args.output.with_name(args.output.name+"_geis_comparison.csv"), index=False)
    previous = pd.read_csv(args.previous_results) if args.previous_results else None
    make_plots(results, geis, args.output, previous)
    metrics = []
    for name, group in comparison.groupby("band"):
        metric = {"band": name,
                  "median_complex_error_pct": float(group.complex_relative_error_pct.median()),
                  "phase_RMS_deg": float(np.sqrt(np.mean(group.phase_residual_deg**2))),
                  "frequency_min_Hz": float(group.frequency_Hz.min()),
                  "frequency_max_Hz": float(group.frequency_Hz.max())}
        metrics.append(metric)
    report = {"mpr": str(prbs_path), "voltage_channel": voltage_column,
              "current_channel": "I/mA", "heating_frequency_Hz": heating_frequency,
              "heating_removed": not args.no_heating, "bands": audits, "geis_comparison": metrics,
              "limitations": ["Conditional recovery assuming repeatable response across periods.",
                              "Residual systematic phase/gain error is not included in leave-one-period spread.",
                              "No GEIS-derived delay, gain, frequency warp or model was applied.",
                              "Recorded <Ewe> is averaged; aperture/channel transfer has not been calibrated."]}
    args.output.with_name(args.output.name+"_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, (audit, wave) in enumerate(zip(audits, waves)):
        start = audit["epoch_s"] + 2*audit["period_s"] + .5
        selected = data[data["time/s"].between(start, start+.12)]
        axes[0,col].plot(selected["time/s"]-start, selected["I/mA"], ".-", ms=2, lw=.7, label="Recorded I (lines cross gaps)")
        axes[0,col].set_title(f"{audit['band']}: {audit['missing_grid_fraction']:.1%} missing grid samples")
        axes[0,col].set_xlabel("Elapsed time / s")
        axes[0,col].set_ylabel("Current / mA")
        f = wave["frequency_Hz"]
        sel = (f>=2) & (f<=args.fast_fmax)
        axes[1,col].semilogy(f[sel],wave["input_amplitude_A"][sel]*1000,".",ms=2,label="Measured harmonic amplitude")
        sel = wave["accepted"] & np.isin(f, results.loc[results.band==audit["band"], "frequency_Hz"])
        axes[1,col].semilogy(f[sel],wave["input_amplitude_A"][sel]*1000,".",label="Accepted impedance lines")
        axes[1,col].set_xlabel("Frequency / Hz")
        axes[1,col].set_ylabel("Current peak amplitude / mA")
        axes[1,col].legend(fontsize=8)
    fig.savefig(args.output.with_name(args.output.name+"_sampling.png"), dpi=180)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(results)} accepted impedance lines, comparisons, plots and audit to {args.output.resolve()}")


def periodic_design(time: np.ndarray, frequency: float, harmonics: int, polynomial_degree: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return slow polynomial and harmonic design matrices."""
    scale = max(float(np.max(np.abs(time))), 1.0)
    u = time / scale
    polynomial = np.column_stack([u**degree for degree in range(polynomial_degree + 1)])
    harmonic = np.column_stack(
        [component for k in range(1, harmonics + 1) for component in (
            np.sin(2 * np.pi * k * frequency * time),
            np.cos(2 * np.pi * k * frequency * time),
        )]
    )
    return polynomial, harmonic


def fit_reference_periodic_pattern(
    time: np.ndarray,
    voltage: np.ndarray,
    harmonics: int = 3,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Find the CP oscillation frequency and fit its harmonic waveform."""
    detrended = signal.detrend(voltage, type="linear")
    frequencies, power = signal.periodogram(detrended, fs=1 / np.median(np.diff(time)))
    band = (frequencies >= 0.15) & (frequencies <= 0.35)
    if not np.any(band):
        raise ValueError("Reference CP file has no resolvable frequency bins in 0.15–0.35 Hz")
    coarse_frequency = frequencies[band][np.argmax(power[band])]
    candidates = np.linspace(max(0.15, coarse_frequency - 0.015), min(0.35, coarse_frequency + 0.015), 121)

    best = None
    for frequency in candidates:
        polynomial, harmonic = periodic_design(time, frequency, harmonics)
        design = np.column_stack((polynomial, harmonic))
        coefficient, _, _, _ = np.linalg.lstsq(design, voltage, rcond=None)
        residual = voltage - design @ coefficient
        score = float(np.dot(residual, residual))
        if best is None or score < best[0]:
            best = (score, frequency, polynomial, harmonic, coefficient)
    _, frequency, polynomial, harmonic, coefficient = best
    trend = polynomial @ coefficient[: polynomial.shape[1]]
    pattern = harmonic @ coefficient[polynomial.shape[1] :]
    return frequency, trend, pattern, coefficient


def fit_prbs_periodic_correction(
    time: np.ndarray,
    voltage: np.ndarray,
    frequency: float,
    harmonics: int = 3,
    start_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the CP-derived periodic frequencies to PRBS Ewe, allowing phase."""
    mask = time >= start_s
    polynomial, harmonic = periodic_design(time[mask], frequency, harmonics)
    design = np.column_stack((polynomial, harmonic))
    coefficient, _, _, _ = np.linalg.lstsq(design, voltage[mask], rcond=None)
    full_polynomial, full_harmonic = periodic_design(time, frequency, harmonics)
    fitted_periodic = full_harmonic @ coefficient[full_polynomial.shape[1] :]
    return fitted_periodic, coefficient


def make_heating_plot(
    reference_time: np.ndarray,
    reference_voltage: np.ndarray,
    reference_trend: np.ndarray,
    reference_pattern: np.ndarray,
    prbs_time: np.ndarray,
    prbs_voltage: np.ndarray,
    prbs_pattern: np.ndarray,
    corrected_voltage: np.ndarray,
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=False, constrained_layout=True)
    axes[0].plot(reference_time, reference_voltage, ".", ms=2, alpha=0.5, label="500 mA CP Ewe")
    axes[0].plot(reference_time, reference_trend, lw=2, label="CP slow trend")
    axes[0].plot(reference_time, reference_trend + reference_pattern, lw=1, label="trend + periodic pattern")
    axes[0].set_xlabel("reference time / s")
    axes[0].set_ylabel("Ewe / V")
    axes[0].legend()
    axes[1].plot(prbs_time, prbs_pattern * 1e3, lw=0.8, label="fitted PRBS periodic component")
    axes[1].set_xlabel("PRBS time / s")
    axes[1].set_ylabel("periodic component / mV")
    axes[1].legend()
    axes[2].plot(prbs_time, prbs_voltage, lw=0.5, alpha=0.5, label="PRBS Ewe")
    axes[2].plot(prbs_time, corrected_voltage, lw=0.5, label="Ewe minus periodic component")
    axes[2].set_xlabel("PRBS time / s")
    axes[2].set_ylabel("voltage / V")
    axes[2].legend()
    fig.savefig(output_stem.with_name(output_stem.name + "_heating_correction.png"), dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpr", type=Path, default=None, help="PRBS MPR; defaults to the newest Day-9 file")
    parser.add_argument("--reference-mpr", type=Path, default=DEFAULT_REFERENCE_MPR)
    parser.add_argument("--geis-mpr", type=Path, default=DEFAULT_GEIS_MPR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--settle-s", type=float, default=None, help="Override detected PRBS start time")
    parser.add_argument("--slow-duration-s", type=float, default=None, help="Override detected slow-band duration")
    parser.add_argument("--fast-duration-s", type=float, default=None, help="Override detected fast-band duration")
    parser.add_argument("--minimum-coherence", type=float, default=0.5)
    parser.add_argument("--estimator", choices=("repeated", "cwt"), default="repeated",
                        help="Recover repeated waveforms, or CWT for continuous records only")
    parser.add_argument("--previous-results", type=Path, help="Previous CSV to overlay as a diagnostic")
    parser.add_argument("--no-heating", action="store_true", help="Diagnostic: omit heating nuisance regression")
    parser.add_argument(
        "--fast-fmax",
        type=float,
        default=80.0,
        help="Upper fast-band frequency in Hz; use about 80 Hz for 4 ms PRBS segments",
    )
    parser.add_argument(
        "--wavelet-mode",
        choices=("compact", "paper"),
        default="compact",
        help="Use a broad wavelet for the present short PRBS or paper bandwidths for order-16 PRBS",
    )
    args = parser.parse_args()

    prbs_path = args.mpr or resolve_default_prbs()
    for path in (prbs_path, args.reference_mpr, args.geis_mpr):
        if not path.exists():
            raise FileNotFoundError(f"MPR file not found: {path}")
    if not 0.0 <= args.minimum_coherence <= 1.0:
        raise ValueError("--minimum-coherence must be between 0 and 1")
    if args.fast_fmax <= 30.0:
        raise ValueError("--fast-fmax must be greater than the slow-band upper frequency (30 Hz)")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = read_mpr(prbs_path)
    time_column = choose_column(data, ("time/s", "Time/s"), "time")
    ewe_column = get_ewe_column(data, "PRBS measurement")
    current_column = choose_column(data, ("I/mA", "<I>/mA"), "measured current")
    data = data.sort_values(time_column).reset_index(drop=True)

    reference = read_mpr(args.reference_mpr)
    reference_time_column = choose_column(reference, ("time/s", "Time/s"), "reference time")
    reference_ewe_column = get_ewe_column(reference, "500 mA CP reference")
    reference = reference.sort_values(reference_time_column).reset_index(drop=True)

    geis = load_geis(read_mpr(args.geis_mpr))
    reference_time = reference[reference_time_column].to_numpy(float)
    reference_voltage = reference[reference_ewe_column].to_numpy(float)
    periodic_frequency, reference_trend, reference_pattern, _ = fit_reference_periodic_pattern(
        reference_time, reference_voltage
    )

    if args.estimator == "repeated":
        run_repeated_analysis(data, ewe_column, geis, periodic_frequency, args, prbs_path)
        return

    prbs_time = data[time_column].to_numpy(float)
    prbs_voltage = data[ewe_column].to_numpy(float)
    prbs_pattern, _ = fit_prbs_periodic_correction(
        prbs_time, prbs_voltage, periodic_frequency, start_s=args.settle_s or 5.0
    )
    if args.no_heating:
        prbs_pattern[:] = 0
    voltage_column = "Ewe_corrected/V"
    data[voltage_column] = prbs_voltage - prbs_pattern

    detected_slow_start, detected_slow_end, detected_fast_start, detected_fast_end = detect_band_windows(
        data, time_column, current_column
    )
    slow_start = args.settle_s if args.settle_s is not None else detected_slow_start
    slow_end = slow_start + args.slow_duration_s if args.slow_duration_s is not None else detected_slow_end
    fast_start = slow_end
    fast_end = fast_start + args.fast_duration_s if args.fast_duration_s is not None else detected_fast_end

    slow = estimate_band(
        data,
        time_column,
        voltage_column,
        current_column,
        slow_start,
        slow_end,
        1.0,
        30.0,
        45,
        args.minimum_coherence,
        args.wavelet_mode,
    )
    slow["band"] = "slow PRBS"
    fast = estimate_band(
        data,
        time_column,
        voltage_column,
        current_column,
        fast_start,
        fast_end,
        30.0,
        args.fast_fmax,
        55,
        args.minimum_coherence,
        args.wavelet_mode,
    )
    fast["band"] = "fast PRBS"
    results = pd.concat([slow, fast], ignore_index=True)
    comparison, delay_s = compare_with_geis(results, geis)

    csv_path = args.output.with_suffix(".csv")
    comparison_path = args.output.with_name(args.output.name + "_geis_comparison.csv")
    results.to_csv(csv_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    make_plots(results, geis, args.output)
    make_heating_plot(
        reference_time,
        reference_voltage,
        reference_trend,
        reference_pattern,
        prbs_time,
        prbs_voltage,
        prbs_pattern,
        data[voltage_column].to_numpy(float),
        args.output,
    )

    print(f"Loaded {len(data):,} samples from {prbs_path}")
    print(f"Measured channels: voltage={ewe_column!r}, current={current_column!r}")
    print(f"Reference CP: {args.reference_mpr}")
    print(f"GEIS reference: {args.geis_mpr}")
    print(f"Detected CP heating frequency: {periodic_frequency:.6f} Hz (period {1 / periodic_frequency:.3f} s)")
    print(
        "PRBS windows: "
        f"slow {slow_start:.3f}–{slow_end:.3f} s, "
        f"fast {fast_start:.3f}–{fast_end:.3f} s"
    )
    print(f"Morlet wavelet mode: {args.wavelet_mode}")
    print(f"Wrote {len(results)} impedance points to {csv_path.resolve()}")
    print(f"Wrote {len(comparison)} GEIS-overlap points to {comparison_path.resolve()}")
    if delay_s is not None:
        print(f"Estimated residual phase-delay diagnostic: {delay_s * 1e6:.1f} microseconds")
    else:
        print("Estimated residual phase-delay diagnostic: unavailable")
    print(f"Wrote {args.output.with_name(args.output.name + '_bode.png').resolve()}")
    print(f"Wrote {args.output.with_name(args.output.name + '_nyquist.png').resolve()}")
    print(f"Wrote {args.output.with_name(args.output.name + '_heating_correction.png').resolve()}")


if __name__ == "__main__":
    main()
