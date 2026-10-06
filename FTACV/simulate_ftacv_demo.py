"""Generate one FTACV simulation and process it with the Fourier workflow."""

from __future__ import annotations

from pathlib import Path

from ftacv_simulator import (
    FTACVProfile,
    RandlesBVParameters,
    save_simulation,
    simulate_ftacv,
)
from ftacv_fourier_harmonic_filter import process_file

OUTPUT = Path(__file__).resolve().parent / "results" / "ftacv_simulation_demo"


def main() -> None:
    frequency_hz = 10.0
    profile = FTACVProfile(
        start_voltage_V=1.2,
        scan_rate_V_s=0.05,
        amplitude_V=0.1,
        frequency_Hz=frequency_hz,
    )
    parameters = RandlesBVParameters(
        Rs_ohm=0.01,
        Cdl_F=1e-2,
        E0_V=1.2,
        I0_A=1e-8,
        tafel_slope_V_dec=0.05,
        n=4,
        temperature_K=353,
    )
    csv_path = save_simulation(
        simulate_ftacv(profile, parameters, duration_s=10, dt_s=1e-4),
        OUTPUT / "ftacv_simulation.csv",
    )
    process_file(
        csv_path,
        fundamental_hz=frequency_hz,
        max_harmonic=7,
        bandwidth_bins=None,
        bandwidth_fraction=0.05,
        trim_cycles=5,
        edge_guard_cycles=5,
    )
    print(f"Wrote simulation and Fourier outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
