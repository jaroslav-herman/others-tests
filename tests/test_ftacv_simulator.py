from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from ftacv_simulator import (
    FTACVProfile,
    RandlesBVParameters,
    TriangularFTACVProfile,
    save_simulation,
    simulate_ftacv,
)


class FTACVSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = RandlesBVParameters(Rs_ohm=2.0, Cdl_F=1e-3, E0_V=0.5, I0_A=1e-3)

    def test_profile_values_and_period(self) -> None:
        profile = FTACVProfile(0.5, 0.1, 0.02, 10.0, phase_rad=np.pi / 2)
        self.assertAlmostEqual(profile(0.0), 0.52)
        self.assertAlmostEqual(profile(0.1) - profile(0.0), 0.01, places=12)

    def test_triangular_profile_reaches_vertex_and_returns(self) -> None:
        profile = TriangularFTACVProfile(1.2, 1.7, 0.01, 5.0)
        self.assertAlmostEqual(profile.one_way_duration_s, 50.0)
        self.assertAlmostEqual(profile.duration_s, 100.0)
        self.assertAlmostEqual(profile(0.0), 1.2)
        self.assertAlmostEqual(profile(50.0), 1.7)
        self.assertAlmostEqual(profile(100.0), 1.2)

    def test_zero_overpotential_has_zero_bv_current(self) -> None:
        self.assertAlmostEqual(self.parameters.faradaic_current_A(0.5), 0.0, places=15)

    def test_components_sum_to_total(self) -> None:
        profile = FTACVProfile(0.45, 0.0, 0.02, 5.0)
        data = simulate_ftacv(profile, self.parameters, 0.2, 0.001)
        np.testing.assert_allclose(
            data["I/mA"], data["I_faradaic/mA"] + data["I_capacitive/mA"], rtol=1e-12, atol=1e-12
        )
        self.assertEqual(len(data), 201)
        np.testing.assert_allclose(np.diff(data["time/s"]), 0.001)

    def test_invalid_parameters_and_duration(self) -> None:
        with self.assertRaises(ValueError):
            FTACVProfile(0.5, 0.0, 0.1, 0.0)
        with self.assertRaises(ValueError):
            RandlesBVParameters(1.0, 0.0, 0.5, I0_A=1e-3)
        with self.assertRaises(ValueError):
            simulate_ftacv(lambda t: np.zeros_like(t), self.parameters, 0.11, 0.03)

    def test_density_convenience(self) -> None:
        parameters = RandlesBVParameters(
            Rs_ohm=0.0, Cdl_F=1e-3, E0_V=0.5, exchange_current_density_A_cm2=2e-3, area_cm2=3.0
        )
        self.assertAlmostEqual(parameters.exchange_current_A, 6e-3)

    def test_tafel_slope_defines_transfer_coefficient(self) -> None:
        parameters = RandlesBVParameters(
            Rs_ohm=0.0,
            Cdl_F=1e-3,
            E0_V=0.5,
            I0_A=1e-3,
            tafel_slope_V_dec=2.303 * 8.314462618 * 298.15 / 96485.33212 / 0.5,
        )
        self.assertAlmostEqual(parameters.effective_alpha, 0.5, places=6)

    def test_stiff_rc_case_remains_finite(self) -> None:
        parameters = RandlesBVParameters(Rs_ohm=1e4, Cdl_F=1e-6, E0_V=0.5, I0_A=1e-6)
        data = simulate_ftacv(FTACVProfile(0.5, 0.0, 0.01, 100.0), parameters, 0.02, 1e-5)
        self.assertTrue(np.isfinite(data.to_numpy()).all())

    def test_saved_csv_has_analysis_columns(self) -> None:
        data = simulate_ftacv(FTACVProfile(0.5, 0.0, 0.01, 10.0), self.parameters, 0.1, 0.001)
        with TemporaryDirectory() as folder:
            path = save_simulation(data, Path(folder) / "simulation.csv")
            self.assertTrue(path.is_file())
            self.assertEqual(set(("time/s", "control/V", "I/mA")) - set(data.columns), set())


if __name__ == "__main__":
    unittest.main()
