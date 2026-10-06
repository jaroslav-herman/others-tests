"""Simulate a fixed FTACV waveform across Cdl and series-resistance values."""

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


OUTPUT = Path(__file__).resolve().parent / "results" / "ftacv_c_r_sweep"
CAPACITANCES_F = (0.01, 0.05, 0.1, 0.2)
RESISTANCES_OHM = (0.01, 0.02, 0.04, 0.08)
FREQUENCY_HZ = 20.0
AMPLITUDE_V = 0.1
SCAN_RATE_V_S = 0.05
START_V = 1.2
VERTEX_V = 1.7
DT_S = 1e-3
DURATION_S = 2.0 * (VERTEX_V - START_V) / SCAN_RATE_V_S


def safe_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def folder_name(capacitance_F: float, resistance_ohm: float) -> str:
    return f"C_{safe_number(capacitance_F)}F_R_{safe_number(resistance_ohm)}ohm"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for existing in OUTPUT.rglob("*"):
        if existing.is_file():
            existing.unlink()

    profile = TriangularFTACVProfile(
        start_voltage_V=START_V,
        vertex_voltage_V=VERTEX_V,
        scan_rate_V_s=SCAN_RATE_V_S,
        frequency_Hz=FREQUENCY_HZ,
        amplitude_V=AMPLITUDE_V,
    )
    for capacitance_F, resistance_ohm in product(CAPACITANCES_F, RESISTANCES_OHM):
        parameters = RandlesBVParameters(
            Rs_ohm=resistance_ohm,
            Cdl_F=capacitance_F,
            E0_V=1.2,
            I0_A=1e-8,
            tafel_slope_V_dec=0.05,
            n=4,
            temperature_K=353,
        )
        folder = OUTPUT / folder_name(capacitance_F, resistance_ohm)
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = save_simulation(
            simulate_ftacv(profile, parameters, duration_s=DURATION_S, dt_s=DT_S),
            folder / "simulation.csv",
        )
        process_file(
            csv_path,
            fundamental_hz=FREQUENCY_HZ,
            max_harmonic=7,
            bandwidth_bins=None,
            bandwidth_fraction=0.05,
            trim_cycles=5.0,
            edge_guard_cycles=5.0,
        )
        keep_suffixes = {"_ten_panel.png", "_harmonic_envelopes.csv", "_filtered_dc_current.csv"}
        for suffix in OUTPUT_SUFFIXES:
            generated = folder / f"simulation{suffix}"
            if not any(generated.name.endswith(keep) for keep in keep_suffixes) and generated.exists():
                generated.unlink()
        csv_path.unlink()
        print(f"C={capacitance_F:g} F, R={resistance_ohm:g} ohm")

    files = [path for path in OUTPUT.rglob("*") if path.is_file()]
    allowed = ("_ten_panel.png", "_harmonic_envelopes.csv", "_filtered_dc_current.csv")
    if any(not path.name.endswith(allowed) for path in files):
        raise RuntimeError("Sweep output contains an unexpected file")
    print(f"Created {len(CAPACITANCES_F) * len(RESISTANCES_OHM)} simulations in {OUTPUT}")


if __name__ == "__main__":
    main()

