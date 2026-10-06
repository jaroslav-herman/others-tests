"""Fixed-step FTACV simulation for a Randles circuit with Butler--Volmer kinetics.

The simulator uses SI units internally and returns a pandas DataFrame with the
column names used by the project's Fourier-analysis workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


FARADAY_CONSTANT = 96485.33212
GAS_CONSTANT = 8.314462618


@dataclass(frozen=True)
class FTACVProfile:
    """A linear potential ramp with a sinusoidal AC perturbation."""

    start_voltage_V: float
    scan_rate_V_s: float
    amplitude_V: float
    frequency_Hz: float
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.start_voltage_V,
            self.scan_rate_V_s,
            self.amplitude_V,
            self.frequency_Hz,
            self.phase_rad,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("FTACV profile parameters must be finite")
        if self.frequency_Hz <= 0:
            raise ValueError("frequency_Hz must be positive")

    def __call__(self, time_s: float | np.ndarray) -> float | np.ndarray:
        time = np.asarray(time_s, dtype=float)
        voltage = (
            self.start_voltage_V
            + self.scan_rate_V_s * time
            + self.amplitude_V
            * np.sin(2.0 * np.pi * self.frequency_Hz * time + self.phase_rad)
        )
        return float(voltage) if voltage.ndim == 0 else voltage


@dataclass(frozen=True)
class TriangularFTACVProfile:
    """A forward-and-reverse triangular DC ramp with an AC perturbation."""

    start_voltage_V: float
    vertex_voltage_V: float
    scan_rate_V_s: float
    frequency_Hz: float
    amplitude_V: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.start_voltage_V,
            self.vertex_voltage_V,
            self.scan_rate_V_s,
            self.frequency_Hz,
            self.amplitude_V,
            self.phase_rad,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Triangular FTACV profile parameters must be finite")
        if self.vertex_voltage_V <= self.start_voltage_V:
            raise ValueError("vertex_voltage_V must exceed start_voltage_V")
        if self.scan_rate_V_s <= 0:
            raise ValueError("scan_rate_V_s must be positive")
        if self.frequency_Hz <= 0:
            raise ValueError("frequency_Hz must be positive")

    @property
    def one_way_duration_s(self) -> float:
        return (self.vertex_voltage_V - self.start_voltage_V) / self.scan_rate_V_s

    @property
    def duration_s(self) -> float:
        return 2.0 * self.one_way_duration_s

    def __call__(self, time_s: float | np.ndarray) -> float | np.ndarray:
        time = np.asarray(time_s, dtype=float)
        forward_time = np.minimum(np.maximum(time, 0.0), self.one_way_duration_s)
        reverse_time = np.minimum(
            np.maximum(time - self.one_way_duration_s, 0.0), self.one_way_duration_s
        )
        base = np.where(
            time <= self.one_way_duration_s,
            self.start_voltage_V + self.scan_rate_V_s * forward_time,
            self.vertex_voltage_V - self.scan_rate_V_s * reverse_time,
        )
        voltage = base + self.amplitude_V * np.sin(
            2.0 * np.pi * self.frequency_Hz * time + self.phase_rad
        )
        return float(voltage) if voltage.ndim == 0 else voltage


VoltageProfile = Callable[[float | np.ndarray], float | np.ndarray]


@dataclass(frozen=True)
class RandlesBVParameters:
    """Parameters for Rs in series with Cdl || Butler--Volmer."""

    Rs_ohm: float
    Cdl_F: float
    E0_V: float
    I0_A: float | None = None
    exchange_current_density_A_cm2: float | None = None
    area_cm2: float = 1.0
    alpha: float = 0.5
    tafel_slope_V_dec: float | None = None
    n: int = 1
    temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if self.I0_A is not None and self.exchange_current_density_A_cm2 is not None:
            raise ValueError("Set either I0_A or exchange_current_density_A_cm2, not both")
        if self.I0_A is None and self.exchange_current_density_A_cm2 is None:
            raise ValueError("Set I0_A or exchange_current_density_A_cm2")
        for name in ("Rs_ohm", "Cdl_F", "E0_V", "area_cm2", "alpha", "temperature_K"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.Rs_ohm < 0:
            raise ValueError("Rs_ohm must be non-negative")
        if self.Cdl_F <= 0:
            raise ValueError("Cdl_F must be positive")
        if self.area_cm2 <= 0:
            raise ValueError("area_cm2 must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.tafel_slope_V_dec is not None and (
            not np.isfinite(self.tafel_slope_V_dec) or self.tafel_slope_V_dec <= 0
        ):
            raise ValueError("tafel_slope_V_dec must be positive and finite")
        if self.n <= 0:
            raise ValueError("n must be positive")
        if self.temperature_K <= 0:
            raise ValueError("temperature_K must be positive")
        if self.I0_A is not None and (not np.isfinite(self.I0_A) or self.I0_A < 0):
            raise ValueError("I0_A must be finite and non-negative")
        if self.exchange_current_density_A_cm2 is not None and (
            not np.isfinite(self.exchange_current_density_A_cm2)
            or self.exchange_current_density_A_cm2 < 0
        ):
            raise ValueError("exchange_current_density_A_cm2 must be finite and non-negative")

    @property
    def exchange_current_A(self) -> float:
        if self.I0_A is not None:
            return self.I0_A
        return float(self.exchange_current_density_A_cm2 * self.area_cm2)

    @property
    def effective_alpha(self) -> float:
        """Return alpha, deriving it from the anodic Tafel slope when supplied.

        ``tafel_slope_V_dec`` is the anodic slope in V/decade.  The conversion
        follows ``b = 2.303 * R * T / (alpha * n * F)``.
        """
        if self.tafel_slope_V_dec is None:
            return self.alpha
        alpha = 2.303 * GAS_CONSTANT * self.temperature_K / (
            self.n * FARADAY_CONSTANT * self.tafel_slope_V_dec
        )
        if not 0 < alpha < 1:
            raise ValueError(
                "tafel_slope_V_dec implies a Butler–Volmer alpha outside (0, 1); "
                "check slope, temperature, and n"
            )
        return float(alpha)

    def faradaic_current_A(self, electrode_voltage_V: float) -> float:
        """Return Butler--Volmer current at an electrode potential."""
        eta = electrode_voltage_V - self.E0_V
        factor = self.n * FARADAY_CONSTANT / (GAS_CONSTANT * self.temperature_K)
        alpha = self.effective_alpha
        anodic = np.exp(np.clip(alpha * factor * eta, -700.0, 700.0))
        cathodic = np.exp(np.clip(-(1.0 - alpha) * factor * eta, -700.0, 700.0))
        return float(self.exchange_current_A * (anodic - cathodic))

    def faradaic_derivative_A_per_V(self, electrode_voltage_V: float) -> float:
        eta = electrode_voltage_V - self.E0_V
        factor = self.n * FARADAY_CONSTANT / (GAS_CONSTANT * self.temperature_K)
        alpha = self.effective_alpha
        anodic = np.exp(np.clip(alpha * factor * eta, -700.0, 700.0))
        cathodic = np.exp(np.clip(-(1.0 - alpha) * factor * eta, -700.0, 700.0))
        return float(self.exchange_current_A * factor * (alpha * anodic + (1.0 - alpha) * cathodic))


def _solve_monotonic_step(
    applied_voltage_V: float,
    previous_electrode_V: float,
    dt_s: float,
    parameters: RandlesBVParameters,
) -> tuple[float, float, float]:
    """Solve one backward-Euler electrode state using safeguarded Newton steps."""
    resistance = parameters.Rs_ohm
    capacitance = parameters.Cdl_F
    guess = previous_electrode_V
    for _ in range(80):
        faradaic = parameters.faradaic_current_A(guess)
        capacitive = capacitance * (guess - previous_electrode_V) / dt_s
        residual = guess + resistance * (faradaic + capacitive) - applied_voltage_V
        derivative = 1.0 + resistance * (
            parameters.faradaic_derivative_A_per_V(guess) + capacitance / dt_s
        )
        correction = residual / derivative
        next_guess = guess - correction
        if abs(residual) <= 1e-12 * max(1.0, abs(applied_voltage_V)):
            total = faradaic + capacitive
            return guess, total, faradaic
        if not np.isfinite(next_guess) or abs(next_guess - guess) > 2.0:
            next_guess = guess - np.sign(residual) * min(abs(correction), 0.1)
        guess = next_guess
    raise RuntimeError("Implicit FTACV step did not converge")


def _steady_state_electrode(applied_voltage_V: float, parameters: RandlesBVParameters) -> float:
    electrode = applied_voltage_V
    for _ in range(100):
        current = parameters.faradaic_current_A(electrode)
        residual = electrode + parameters.Rs_ohm * current - applied_voltage_V
        derivative = 1.0 + parameters.Rs_ohm * parameters.faradaic_derivative_A_per_V(electrode)
        correction = residual / derivative
        electrode -= correction
        if abs(residual) <= 1e-12 * max(1.0, abs(applied_voltage_V)):
            return float(electrode)
    raise RuntimeError("Initial FTACV operating point did not converge")


def simulate_ftacv(
    profile: VoltageProfile,
    parameters: RandlesBVParameters,
    duration_s: float,
    dt_s: float,
    *,
    initial_electrode_voltage_V: float | None = None,
) -> pd.DataFrame:
    """Simulate a uniformly sampled FTACV trace."""
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite")
    sample_count = int(round(duration_s / dt_s)) + 1
    if sample_count < 2 or not np.isclose((sample_count - 1) * dt_s, duration_s, rtol=1e-10, atol=1e-12):
        raise ValueError("duration_s must be an integer multiple of dt_s")

    time = np.arange(sample_count, dtype=float) * dt_s
    applied = np.asarray(profile(time), dtype=float)
    if applied.shape != time.shape or not np.all(np.isfinite(applied)):
        raise ValueError("profile must return one finite voltage value per time sample")

    electrode = np.empty(sample_count, dtype=float)
    total_current = np.empty(sample_count, dtype=float)
    faradaic_current = np.empty(sample_count, dtype=float)
    capacitive_current = np.empty(sample_count, dtype=float)
    electrode[0] = (
        _steady_state_electrode(applied[0], parameters)
        if initial_electrode_voltage_V is None
        else float(initial_electrode_voltage_V)
    )
    faradaic_current[0] = parameters.faradaic_current_A(electrode[0])
    capacitive_current[0] = 0.0
    total_current[0] = faradaic_current[0]

    for index in range(1, sample_count):
        electrode[index], total_current[index], faradaic_current[index] = _solve_monotonic_step(
            applied[index], electrode[index - 1], dt_s, parameters
        )
        capacitive_current[index] = total_current[index] - faradaic_current[index]

    return pd.DataFrame(
        {
            "time/s": time,
            "control/V": applied,
            "Ewe/V": applied,
            "I/mA": total_current * 1e3,
            "I_faradaic/mA": faradaic_current * 1e3,
            "I_capacitive/mA": capacitive_current * 1e3,
            "E_electrode/V": electrode,
        }
    )


def save_simulation(data: pd.DataFrame, path: str | Path) -> Path:
    """Validate and write simulator output as CSV."""
    required = {"time/s", "control/V", "I/mA"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Simulation data is missing columns: {sorted(missing)}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return output
