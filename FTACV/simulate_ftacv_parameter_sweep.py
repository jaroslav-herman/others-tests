"""Generate ten-panel FTACV plots for a scan-rate/frequency/amplitude sweep."""

from __future__ import annotations

from itertools import product
from pathlib import Path

from ftacv_fourier_harmonic_filter import OUTPUT_SUFFIXES, process_file
from ftacv_simulator import (
    RandlesBVParameters,
    TriangularFTACVProfile,
    save_simulation,
    simulate_ftacv,
)


OUTPUT = Path(__file__).resolve().parent / "results" / "ftacv_parameter_sweep"
SCAN_RATES_MV_S = (100, 50, 20, 10)
FREQUENCIES_HZ = (5, 10, 20, 50)
AMPLITUDES_MV = (10, 50, 100, 200)
DT_S = 1e-3

# These are the circuit parameters currently used by simulate_ftacv_demo.py.
PARAMETERS = RandlesBVParameters(
    Rs_ohm=0.02,
    Cdl_F=1e-1,
    E0_V=1.2,
    I0_A=1e-8,
    tafel_slope_V_dec=0.05,
    n=4,
    temperature_K=353,
)


def safe_number(value: int | float) -> str:
    return f"{value:g}".replace(".", "p")


def folder_name(frequency_hz: int, amplitude_mv: int) -> str:
    return f"{frequency_hz:g} Hz, {amplitude_mv:g} mV amp"


def stem(scan_rate_mv_s: int) -> str:
    """Use the naming convention expected by compare_ftacv_parameter_range2.py."""
    return f"FTACV_{scan_rate_mv_s:02d}_ACV"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for existing in OUTPUT.rglob("*"):
        if existing.is_file():
            existing.unlink()
    total = len(SCAN_RATES_MV_S) * len(FREQUENCIES_HZ) * len(AMPLITUDES_MV)
    for number, (scan_rate, frequency, amplitude) in enumerate(
        product(SCAN_RATES_MV_S, FREQUENCIES_HZ, AMPLITUDES_MV), start=1
    ):
        output_folder = OUTPUT / folder_name(frequency, amplitude)
        output_folder.mkdir(parents=True, exist_ok=True)
        file_stem = stem(scan_rate)
        csv_path = save_simulation(
            simulate_ftacv(
                TriangularFTACVProfile(
                    start_voltage_V=1.2,
                    vertex_voltage_V=1.7,
                    scan_rate_V_s=scan_rate / 1000.0,
                    frequency_Hz=float(frequency),
                    amplitude_V=amplitude / 1000.0,
                ),
                PARAMETERS,
                duration_s=2.0 * (1.7 - 1.2) / (scan_rate / 1000.0),
                dt_s=DT_S,
            ),
            output_folder / f"{file_stem}.csv",
        )
        process_file(
            csv_path,
            fundamental_hz=float(frequency),
            max_harmonic=7,
            bandwidth_bins=None,
            bandwidth_fraction=0.05,
            trim_cycles=5.0,
            edge_guard_cycles=5.0,
        )
        keep_suffixes = {"_ten_panel.png", "_harmonic_envelopes.csv", "_filtered_dc_current.csv"}
        for suffix in OUTPUT_SUFFIXES:
            generated = output_folder / f"{file_stem}{suffix}"
            if not any(generated.name.endswith(keep) for keep in keep_suffixes) and generated.exists():
                generated.unlink()
        input_csv = output_folder / f"{file_stem}.csv"
        if input_csv.exists():
            input_csv.unlink()
        print(f"[{number}/{total}] {file_stem}")

    remaining = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
    allowed = ("_ten_panel.png", "_harmonic_envelopes.csv", "_filtered_dc_current.csv")
    if any(not path.name.endswith(allowed) for path in remaining):
        raise RuntimeError("Sweep output contains an unexpected file")
    png_count = sum(path.name.endswith("_ten_panel.png") for path in remaining)
    envelope_count = sum(path.name.endswith("_harmonic_envelopes.csv") for path in remaining)
    dc_count = sum(path.name.endswith("_filtered_dc_current.csv") for path in remaining)
    print(f"Created {png_count} PNG plots, {envelope_count} harmonic CSVs, and {dc_count} filtered-DC CSVs in {OUTPUT}")


if __name__ == "__main__":
    main()
