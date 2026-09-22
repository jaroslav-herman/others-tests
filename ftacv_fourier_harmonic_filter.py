"""Fourier-domain isolation of FTACV harmonics from BioLogic MPR files.

The workflow follows the paper description: the measured total current is
transformed with an FFT, each harmonic is retained with a narrow band-pass
mask, and the filtered spectrum is transformed back with an inverse FFT.

Running this file directly from VS Code recursively processes every ``.mpr``
file under ``DEFAULT_INPUT`` when ``DEFAULT_FILE_NAMES`` is empty.
The output files are written directly beside that data file.  The FFT uses all
trimmed measured points and edge-pads only to the next power of two.
Use ``--harmonics`` and the other command-line options to override defaults.
"""

from __future__ import annotations

import argparse
import re
from time import perf_counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = False
import numpy as np
import pandas as pd

import wepy.basics as we

# ``wepy`` may load a local Matplotlib style, so enforce a headless,
# non-TeX renderer after all project imports.
plt.rcParams["text.usetex"] = False


DEFAULT_INPUT = Path(
    r"\\ELECTROLYZER\PEM-WE_measurements\2026\470_IV_cathode_etching_series_GDE\FTacV\parameter range 3"
)
# Leave empty to process every .mpr file recursively under DEFAULT_INPUT.
# Add explicit full paths here only when a selected subset is needed.
DEFAULT_FILE_NAMES: list[str] = []
DEFAULT_FUNDAMENTAL_HZ = 500
# Number of fundamental cycles removed from each endpoint before FFT
# processing. The number of samples is calculated from the measured dt and f0.
DEFAULT_TRIM_CYCLES = 10
# Manual half-width of every harmonic window, in FFT bins. The automatic
# bandwidth fraction is used by default; this value applies only when the
# automatic bandwidth is disabled.
DEFAULT_BANDWIDTH_BINS = 5
DEFAULT_BANDWIDTH_FRACTION = 0.05
DEFAULT_MAX_HARMONIC = 9
MAX_FUNDAMENTAL_HZ = 500.0
# Extra cycles removed after inverse-FFT reconstruction to suppress
# frequency-filter ringing at the finite-record boundaries.
DEFAULT_EDGE_GUARD_CYCLES = 10.0
FREQUENCY_FOLDER_PATTERN = re.compile(
    r"(?<![\d.])(?P<frequency>\d+(?:\.\d+)?)\s*Hz\b",
    re.IGNORECASE,
)
OUTPUT_SUFFIXES = (
    "_harmonic_envelopes.csv",
    "_summary.csv",
    "_power_spectrum.csv",
    "_filtered_dc_current.csv",
    "_ten_panel.png",
)


def largest_allowed_length(length: int) -> tuple[int, int]:
    """Return the next (2**N, N), constrained to the paper's N=14..20 range."""
    n = max(14, int(np.ceil(np.log2(length))))
    if n > 22:
        raise ValueError(f"{length} points require 2**{n}; maximum supported is 2**22")
    return 2**n, n


def harmonic_filter(
    current: np.ndarray,
    dt: float,
    fundamental_hz: float,
    harmonic_number: int,
    bandwidth_bins: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the real reconstructed harmonic and its FFT mask.

    The positive and negative frequency bins are both retained, which makes
    the inverse transform real and preserves the measured phase.
    """
    n = current.size
    spectrum = np.fft.fft(current)
    frequencies = np.fft.fftfreq(n, dt)
    target = harmonic_number * fundamental_hz
    mask = np.zeros(n, dtype=bool)
    bin_width = 1.0 / (n * dt)
    for signed_target in (target, -target):
        mask |= np.abs(frequencies - signed_target) <= bandwidth_bins * bin_width
    filtered = np.where(mask, spectrum, 0.0)
    return np.fft.ifft(filtered).real, mask


def lowpass_filter(signal: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    """Return the zero-phase FFT low-pass reconstruction."""
    spectrum = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal.size, dt)
    filtered = np.where(np.abs(frequencies) <= cutoff_hz, spectrum, 0.0)
    return np.fft.ifft(filtered).real


def harmonic_envelope(
    time: np.ndarray,
    dc_voltage: np.ndarray,
    harmonic: np.ndarray,
    fundamental_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one positive peak envelope point per fundamental cycle."""
    cycle_numbers = np.floor((time - time[0]) * fundamental_hz).astype(int)
    unique_cycles = np.unique(cycle_numbers)
    envelope_voltage = []
    envelope_current = []
    for cycle in unique_cycles:
        selected = cycle_numbers == cycle
        if not np.any(selected):
            continue
        envelope_voltage.append(float(np.mean(dc_voltage[selected])))
        envelope_current.append(float(np.max(np.abs(harmonic[selected]))))
    return np.asarray(envelope_voltage), np.asarray(envelope_current)


def frequency_from_folder(path: Path) -> float | None:
    """Return the frequency from the nearest parent folder containing ``... Hz``."""
    for folder in (path.parent, *path.parents):
        match = FREQUENCY_FOLDER_PATTERN.search(folder.name)
        if match:
            return float(match.group("frequency"))
    return None


def output_files_for(path: Path) -> tuple[Path, ...]:
    """Return the complete set of files proving that one input was processed."""
    output_folder = path.resolve().parent
    return tuple(output_folder / f"{path.stem}{suffix}" for suffix in OUTPUT_SUFFIXES)


def process_file(
    path: Path,
    fundamental_hz: float,
    max_harmonic: int,
    bandwidth_bins: int | None,
    bandwidth_fraction: float | None,
    trim_cycles: float,
    edge_guard_cycles: float,
) -> None:
    process_started = perf_counter()
    data = we.read_file_safe(str(path))
    if data is None or data.empty:
        print(f"Skipping unreadable/empty file: {path.name}")
        return

    required = {"time/s", "I/mA"}
    missing = required - set(data.columns)
    if missing:
        print(f"Skipping {path.name}; missing columns: {sorted(missing)}")
        return

    time_all = data["time/s"].to_numpy(dtype=float)
    current_all = data["I/mA"].to_numpy(dtype=float)
    voltage_column = "control/V" if "control/V" in data.columns else "Ewe/V"
    voltage_all = data[voltage_column].to_numpy(dtype=float)
    dt = float(np.median(np.diff(time_all)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid time spacing in {path.name}: {dt}")
    sampling_hz = 1.0 / dt
    nyquist_hz = sampling_hz / 2.0
    if fundamental_hz >= nyquist_hz:
        raise ValueError(
            f"Fundamental frequency {fundamental_hz:g} Hz is not below the "
            f"Nyquist frequency {nyquist_hz:g} Hz in {path.name}"
        )

    trim_count = int(round(trim_cycles / (fundamental_hz * dt)))
    if 2 * trim_count >= len(data) - 2:
        raise ValueError("Endpoint trimming removes the entire measurement")
    time = time_all[trim_count : len(data) - trim_count]
    current_all = current_all[trim_count : len(data) - trim_count]
    voltage_all = voltage_all[trim_count : len(data) - trim_count]
    original_count = len(time)
    sample_count, exponent = largest_allowed_length(original_count)
    # Retain every measured point. Only the shortfall to the next power of two
    # is edge-padded for the FFT; padded points are never plotted/exported.
    time_block = time
    current = current_all
    voltage = voltage_all
    current_centered = current - np.mean(current)
    pad_count = sample_count - original_count
    if bandwidth_bins is None and bandwidth_fraction is not None:
        bandwidth_bins = int(
            np.ceil(bandwidth_fraction * fundamental_hz * sample_count * dt)
        )
    if bandwidth_bins is None:
        bandwidth_bins = DEFAULT_BANDWIDTH_BINS
    bandwidth_hz = bandwidth_bins / (sample_count * dt)
    highest_harmonic_hz = max_harmonic * fundamental_hz
    if highest_harmonic_hz + bandwidth_hz >= nyquist_hz:
        raise ValueError(
            f"The {max_harmonic}th harmonic band reaches "
            f"{highest_harmonic_hz + bandwidth_hz:g} Hz, above the "
            f"Nyquist frequency {nyquist_hz:g} Hz in {path.name}"
        )
    current_fft = np.pad(current_centered, (0, pad_count), mode="edge")
    current_for_dc_fft = np.pad(current, (0, pad_count), mode="edge")
    voltage_for_dc_fft = np.pad(voltage, (0, pad_count), mode="edge")

    # The excitation frequency is supplied by the experiment.  Do not infer
    # it from the voltage trace: the slow ACV ramp can dominate that spectrum.
    harmonics: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    for harmonic in range(1, max_harmonic + 1):
        harmonics[harmonic], masks[harmonic] = harmonic_filter(
            current_fft, dt, fundamental_hz, harmonic, bandwidth_bins
        )
        harmonics[harmonic] = harmonics[harmonic][:original_count]
    dc_current = lowpass_filter(current_for_dc_fft, dt, fundamental_hz / 2.0)[:original_count]
    dc_voltage = lowpass_filter(voltage_for_dc_fft, dt, fundamental_hz / 2.0)[:original_count]
    # A top-hat frequency filter has a sinc-like impulse response. Remove a
    # post-filter guard region so ringing is not shown as a real envelope at
    # either boundary.
    edge_guard_count = int(round(edge_guard_cycles / (fundamental_hz * dt)))
    if 2 * edge_guard_count >= original_count - 2:
        raise ValueError("Edge guard removes the entire trimmed measurement")
    valid = slice(edge_guard_count, original_count - edge_guard_count)
    time_block = time[valid]
    current = current[valid]
    voltage = voltage[valid]
    current_centered = current_centered[valid]
    dc_current = dc_current[valid]
    dc_voltage = dc_voltage[valid]
    harmonics = {
        harmonic: signal[valid] for harmonic, signal in harmonics.items()
    }
    original_count = len(time_block)
    envelopes = {
        harmonic: harmonic_envelope(time_block, dc_voltage, signal, fundamental_hz)
        for harmonic, signal in harmonics.items()
    }

    stem = path.stem
    # Keep all generated files directly beside the source MPR file.
    file_output = path.resolve().parent

    envelope_table = pd.DataFrame(
        {"dc_potential/V": envelopes[1][0]}
    )
    for harmonic, (envelope_potential, envelope_current) in envelopes.items():
        if harmonic == 1:
            envelope_table["dc_potential/V"] = envelope_potential
        envelope_table[f"harmonic_{harmonic:02d}_envelope/mA"] = envelope_current
    envelope_table.to_csv(file_output / f"{stem}_harmonic_envelopes.csv", index=False)

    frequencies = np.fft.rfftfreq(sample_count, dt)
    current_spectrum = np.abs(np.fft.rfft(current_fft))
    power_spectrum_path = file_output / f"{stem}_power_spectrum.csv"
    pd.DataFrame(
        {
            "frequency_Hz": frequencies,
            "fft_magnitude_mA": current_spectrum,
            "power_mA2": current_spectrum**2,
        }
    ).to_csv(power_spectrum_path, index=False)

    filtered_dc_path = file_output / f"{stem}_filtered_dc_current.csv"
    pd.DataFrame(
        {
            "time_s": time_block,
            "dc_potential_V": dc_voltage,
            "filtered_dc_current_mA": dc_current,
        }
    ).to_csv(filtered_dc_path, index=False)

    summary_rows = []
    for harmonic, reconstructed in harmonics.items():
        summary_rows.append(
            {
                "harmonic": harmonic,
                "frequency_Hz": harmonic * fundamental_hz,
                "rms_mA": float(np.sqrt(np.mean(reconstructed**2))),
                "peak_to_peak_mA": float(np.ptp(reconstructed)),
            }
        )
    pd.DataFrame(summary_rows).to_csv(file_output / f"{stem}_summary.csv", index=False)

    # Ten-panel paper-style figure: raw data, power spectrum, DC current,
    # and harmonic envelopes 1--7 versus DC potential.
    multi_fig, multi_axes = plt.subplots(5, 2, figsize=(13, 20), constrained_layout=True)
    ax = multi_axes[0, 0]
    ax.plot(time_block, current, color="tab:blue", lw=0.55, label="i")
    ax.set_xlabel("Time / s")
    ax.set_ylabel("i / mA", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.2)
    ax_e = ax.twinx()
    ax_e.plot(time_block, voltage, color="tab:red", lw=0.65, label="E")
    ax_e.set_ylabel("E / V", color="tab:red")
    ax_e.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title("a) Raw FTacV data")

    ax = multi_axes[0, 1]
    power = np.maximum(current_spectrum**2, np.finfo(float).tiny)
    ax.plot(frequencies, power, color="0.2", lw=0.7)
    spectrum_limit = max(8 * fundamental_hz, 40.0)
    ax.set_xlim(0, spectrum_limit)
    ax.set_yscale("log")
    half_width = bandwidth_bins / (sample_count * dt)
    for harmonic in range(1, min(max_harmonic, int(spectrum_limit // fundamental_hz)) + 1):
        center = harmonic * fundamental_hz
        ax.axvspan(center - half_width, center + half_width, color="tab:red", alpha=0.22)
        ax.text(
            center,
            0.94,
            f"{center - half_width:.3f}\u2013{center + half_width:.3f}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=7,
            color="tab:red",
        )
    ax.set_xlabel("Frequency / Hz")
    ax.set_ylabel("Power, |FFT(i)|² (log scale)")
    ax.set_title("b) Fourier power spectrum and harmonic windows")
    ax.grid(alpha=0.2)

    ax = multi_axes[1, 0]
    ax.plot(dc_voltage, dc_current, color="tab:green", lw=0.8)
    ax.set_xlabel("DC potential, E / V")
    ax.set_ylabel("Filtered DC current / mA")
    ax.set_title("c) Filtered DC current")
    ax.grid(alpha=0.2)

    harmonic_axes = {
        1: multi_axes[1, 1],
        2: multi_axes[2, 0],
        3: multi_axes[2, 1],
        4: multi_axes[3, 0],
        5: multi_axes[3, 1],
        6: multi_axes[4, 0],
        7: multi_axes[4, 1],
    }
    ordinal = {
        1: "1st",
        2: "2nd",
        3: "3rd",
        4: "4th",
        5: "5th",
        6: "6th",
        7: "7th",
    }
    for panel_index, harmonic in enumerate((1, 2, 3, 4, 5, 6, 7), start=1):
        ax = harmonic_axes[harmonic]
        envelope_potential, envelope_current = envelopes[harmonic]
        ax.plot(
            envelope_potential,
            envelope_current,
            color=f"C{panel_index}",
            lw=1.0,
            marker=".",
            ms=2,
        )
        ax.set_xlabel("DC potential, E / V")
        ax.set_ylabel("Envelope current / mA")
        ax.set_title(
            f"{chr(99 + panel_index)}) {ordinal[harmonic]} harmonic envelope "
            f"({harmonic * fundamental_hz:g} Hz)"
        )
        ax.grid(alpha=0.2)

    multi_figure_path = file_output / f"{stem}_ten_panel.png"
    multi_fig.savefig(multi_figure_path, dpi=220)
    plt.close(multi_fig)

    print(
        f"{path.name}: trimmed={trim_count} points/edge, N={exponent}, "
        f"data={original_count}, FFT={sample_count}, "
        f"f0={fundamental_hz:.6g} Hz, bandwidth=±{bandwidth_hz:.6g} Hz "
        f"({bandwidth_bins} bins); "
        f"output_folder={file_output}; saved "
        f"{file_output / f'{stem}_harmonic_envelopes.csv'}, "
        f"{file_output / f'{stem}_summary.csv'}, {power_spectrum_path}, "
        f"{filtered_dc_path}, and {multi_figure_path}; "
        f"exported harmonics=1–{DEFAULT_MAX_HARMONIC}; "
        f"processing time={perf_counter() - process_started:.2f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--frequency",
        type=float,
        default=None,
        help=(
            "Override the folder-derived fundamental frequency in Hz. "
            f"If omitted, folder names are used; fallback is {DEFAULT_FUNDAMENTAL_HZ:g} Hz."
        ),
    )
    parser.add_argument("--harmonics", type=int, default=DEFAULT_MAX_HARMONIC)
    parser.add_argument(
        "--trim-cycles",
        type=float,
        default=DEFAULT_TRIM_CYCLES,
        help=f"Fundamental cycles removed at each endpoint (default: {DEFAULT_TRIM_CYCLES})",
    )
    parser.add_argument(
        "--edge-guard-cycles",
        type=float,
        default=DEFAULT_EDGE_GUARD_CYCLES,
        help=(
            "Cycles removed after inverse FFT to suppress edge ringing "
            f"(default: {DEFAULT_EDGE_GUARD_CYCLES})"
        ),
    )
    parser.add_argument(
        "--bandwidth-bins",
        type=int,
        default=None,
        help="Manual half-width of each FFT band in bins",
    )
    parser.add_argument(
        "--bandwidth-fraction",
        type=float,
        default=DEFAULT_BANDWIDTH_FRACTION,
        help="Half-width as a fraction of the fundamental (default: 0.1)",
    )
    parser.add_argument(
        "--file-name",
        action="append",
        help=(
            "Process this MPR file; repeat the option for multiple files. "
            "By default, every MPR file under --input is processed when "
            "DEFAULT_FILE_NAMES is empty."
        ),
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Process every .mpr file found recursively under --input",
    )
    parser.add_argument(
        "--recalculate-all",
        action="store_true",
        help="Recalculate files even when all expected output files already exist",
    )
    args = parser.parse_args()
    if (
        (args.frequency is not None and args.frequency <= 0)
        or (args.frequency is not None and args.frequency > MAX_FUNDAMENTAL_HZ)
        or args.harmonics != DEFAULT_MAX_HARMONIC
        or (args.bandwidth_bins is not None and args.bandwidth_bins < 0)
        or (args.bandwidth_fraction is not None and args.bandwidth_fraction < 0)
        or args.trim_cycles < 0
        or args.edge_guard_cycles < 0
    ):
        raise SystemExit(
            f"--frequency must be > 0 and <= {MAX_FUNDAMENTAL_HZ:g} Hz; "
            f"--harmonics must be exactly "
            f"{DEFAULT_MAX_HARMONIC}; "
            "--bandwidth-bins and --trim-cycles must be >= 0"
        )

    files = sorted(args.input.rglob("*.mpr"))
    if args.all_files and args.file_name:
        raise SystemExit("Use either --all-files or --file-name, not both")
    process_all = args.all_files or (not args.file_name and not DEFAULT_FILE_NAMES)
    if not process_all:
        requested_names = args.file_name or DEFAULT_FILE_NAMES
        selected_files: list[Path] = []
        for requested in requested_names:
            requested_path = Path(requested).expanduser()
            if requested_path.is_file():
                selected_files.append(requested_path.resolve())
                continue
            selected_files.extend(
                file
                for file in files
                if file.name == requested_path.name
                or file.stem == requested_path.stem
            )
        files = sorted(set(selected_files))
    if not files:
        raise FileNotFoundError(f"No matching .mpr files found under {args.input}")
    batch_started = perf_counter()
    failed_count = 0
    skipped_count = 0
    for file in files:
        expected_outputs = output_files_for(file)
        if not args.recalculate_all and all(output.exists() for output in expected_outputs):
            skipped_count += 1
            print(f"SKIP completed: {file} (all outputs already exist)")
            continue
        try:
            folder_frequency = frequency_from_folder(file)
            fundamental_hz = (
                args.frequency
                if args.frequency is not None
                else folder_frequency or DEFAULT_FUNDAMENTAL_HZ
            )
            if args.frequency is None and folder_frequency is not None:
                print(f"{file.name}: using {fundamental_hz:g} Hz from folder {file.parent.name!r}")
            elif args.frequency is None:
                print(
                    f"{file.name}: no frequency in parent folders; "
                    f"using fallback {fundamental_hz:g} Hz"
                )
            process_file(
                file,
                fundamental_hz,
                args.harmonics,
                args.bandwidth_bins,
                args.bandwidth_fraction,
                args.trim_cycles,
                args.edge_guard_cycles,
            )
        except ValueError as error:
            failed_count += 1
            print(f"FAILED {file}: {error}")
    print(
        f"Found {len(files)} file(s); skipped={skipped_count}; "
        f"failures={failed_count}; "
        f"elapsed={perf_counter() - batch_started:.2f} s total."
    )


if __name__ == "__main__":
    main()
