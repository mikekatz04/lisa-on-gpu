"""JAX-vs-C++ parity tests for ``SOBBHTDIonTheFly``.

Same shape as ``test_jax_vs_cpp_gb.py``, but for the SOBBH source.
Both the JAX path (via ``sobbhtaylert3.py``) and the C++ path (via
``SOBBHTDIonTheFly`` in ``TDIonTheFly.cu``) are ports of the same PN
expressions, so the per-pixel disagreement should sit at the last-bit
floor. Tolerances match the GB tests.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

import fastlisaresponse
from fastlisaresponse.tdionfly import SOBBHTDIonTheFly


# ----------------------------------------------------------------
# A few SOBBH parameter sets at sensible LISA-band frequencies.
# Avoid masses / f_lows that put us within seconds of coalescence
# during the 30-day test window (we'd then be comparing the |M|=0
# branch in both backends, which trivially agrees and tells us
# nothing).
# ----------------------------------------------------------------
_SOBBH_CASES = [
    # name, (m1, m2, s1, s2, distance_pc, f_low_Hz, phi_c, inc, psi, lam, beta)
    ("30-30_spinless", (30.0, 30.0, 0.0, 0.0, 1.0e9, 0.005, 0.0, 0.7, 0.3, 1.2, 0.4)),
    ("36-29_aligned", (36.0, 29.0, 0.3, 0.2, 8.0e8, 0.01, 0.5, 1.0, 0.4, 0.5, -0.2)),
    ("50-40_low_f0", (50.0, 40.0, 0.1, -0.2, 1.5e9, 0.003, 1.1, 0.5, 0.6, 2.7, 0.9)),
]


def _ensure_paths():
    if "SOBBH_TAYLOR_T3_PATH" not in os.environ:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        candidate = os.path.join(repo_root, "sobbhtaylert3.py")
        if os.path.isfile(candidate):
            os.environ["SOBBH_TAYLOR_T3_PATH"] = candidate


def _make_orbits():
    from lisatools.detector import EqualArmlengthOrbits
    o = EqualArmlengthOrbits()
    o.configure(linear_interp_setup=True)
    return o


def _run_one(backend: str, t_arr, T, t_ref, params):
    o = _make_orbits()
    sobbh = SOBBHTDIonTheFly(
        t_arr, T, t_ref,
        sampling_frequency=1.0 / (t_arr[1] - t_arr[0]),
        num_sub=1,
        orbits=o,
        force_backend=backend,
    )
    m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta = params
    out = sobbh(
        np.array([m1]), np.array([m2]), np.array([s1]), np.array([s2]),
        np.array([distance]), np.array([f_low]), np.array([phi_c]),
        np.array([inc]), np.array([psi]), np.array([lam]), np.array([beta]),
        return_spline=False,
    )
    return (np.asarray(out.tdi_amp[0]),
            np.asarray(out.tdi_phase[0]),
            np.asarray(out.phase_ref[0]))


class TestJaxVsCppSOBBH(unittest.TestCase):

    AMP_RTOL = 1e-10
    AMP_ATOL = 1e-14    # SOBBH amps are ~1e-25; atol must be well below
    # SOBBH GW phase accumulates from a 3.5PN polynomial in x with
    # negative-fractional-power log terms; for ~1e6-rad accumulated
    # phase the FMA-reordering floor between two independent ports of
    # the same expressions is ~1e-6 to 1e-7 rad. The C++ uses libm
    # ``pow``/``log``; JAX uses XLA's lowering. We assert 1e-5 rad,
    # which is at the noise floor for SOBBH but still well below any
    # physically meaningful phase scale (1 cycle = 6.28 rad).
    PHASE_ATOL = 1e-5
    PREF_ATOL = 1e-5

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        cls.T = 3600.0 * 24 * 30
        cls.N = 256
        cls.t = np.linspace(0.0, cls.T, cls.N)
        cls.t_ref = 0.0

        cls.has_cpp = fastlisaresponse.has_backend("cpu")
        cls.has_jax = fastlisaresponse.has_backend("jax")
        if not cls.has_jax:
            raise unittest.SkipTest("JAX backend not registered.")

    def test_jax_runs(self):
        """JAX SOBBH path produces finite amp/phase on every case."""
        for name, params in _SOBBH_CASES:
            with self.subTest(case=name):
                amp_j, ph_j, pr_j = _run_one("jax", self.t, self.T, self.t_ref, params)
                self.assertTrue(np.all(np.isfinite(amp_j)), f"{name}: amp NaN")
                self.assertTrue(np.all(np.isfinite(ph_j)), f"{name}: phase NaN")
                self.assertTrue(np.all(np.isfinite(pr_j)), f"{name}: pref NaN")
                self.assertGreater(np.max(np.abs(amp_j)), 0.0,
                                   f"{name}: amp identically zero")

    def test_parity_with_cpp(self):
        """JAX and C++ SOBBH outputs match to rtol=1e-10 per pixel."""
        if not self.has_cpp:
            self.skipTest("C++ 'cpu' backend not installed -- parity skipped.")

        for name, params in _SOBBH_CASES:
            with self.subTest(case=name):
                amp_c, ph_c, pr_c = _run_one("cpu", self.t, self.T, self.t_ref, params)
                amp_j, ph_j, pr_j = _run_one("jax", self.t, self.T, self.t_ref, params)
                for ch in range(amp_c.shape[0]):
                    np.testing.assert_allclose(
                        amp_j[ch], amp_c[ch],
                        rtol=self.AMP_RTOL, atol=self.AMP_ATOL,
                        err_msg=f"{name}: tdi_amp[ch={ch}] mismatch",
                    )
                    diff = ph_j[ch] - ph_c[ch]
                    diff_mod = np.remainder(diff + np.pi, 2 * np.pi) - np.pi
                    self.assertLess(
                        float(np.max(np.abs(diff_mod))),
                        self.PHASE_ATOL,
                        msg=f"{name}: tdi_phase[ch={ch}] disagrees by "
                            f"{float(np.max(np.abs(diff_mod))):.3e}",
                    )
                np.testing.assert_allclose(
                    pr_j, pr_c, rtol=0.0, atol=self.PREF_ATOL,
                    err_msg=f"{name}: phase_ref mismatch",
                )


if __name__ == "__main__":
    unittest.main()
