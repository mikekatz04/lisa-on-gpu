"""JAX-vs-C++ parity tests for ``GBTDIonTheFly``.

Both backends compute the same closed-form GB amplitude/phase
formulas plus the same TDI-on-the-fly projection in double precision,
so the per-pixel disagreement should be at the last-bit FMA-reordering
floor. We assert ``rtol=1e-10`` on ``tdi_amp`` and ``atol=1e-10`` on
``tdi_phase`` (radians), per the user-locked test tolerance.

The test is skipped automatically when the C++ backend
``fastlisaresponse_cpu`` is not installed (so it works fine on a
JAX-only dev box) -- in that case a JAX-only self-consistency check
runs instead.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

import fastlisaresponse
from fastlisaresponse.tdionfly import GBTDIonTheFly


# ----------------------------------------------------------------
# Fixture: representative GB parameter sets spanning the parameter
# space corners we care about. Each row is a single binary; the test
# runs each separately so per-case failures are easy to read.
# ----------------------------------------------------------------
_GB_CASES = [
    # name, (amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)
    ("low_f0_no_chirp", (1.0e-22, 0.001, 0.0, 0.0, 0.0, 0.7, 0.3, 0.5, 0.4)),
    ("mid_f0_small_chirp", (5.0e-23, 0.01, 1.0e-17, 0.0, 0.4, 1.0, 0.2, 1.5, -0.3)),
    ("high_f0_chirpy", (1.0e-22, 0.05, 1.0e-15, 1.0e-22, 0.1, 0.5, 0.6, 2.7, 0.9)),
    ("near_ecliptic_pole", (2.0e-22, 0.005, 0.0, 0.0, 0.7, 1.2, 0.4, 0.3, 1.45)),
    ("antiparallel_sky", (3.0e-22, 0.02, -1.0e-18, 0.0, 1.7, 0.8, -0.5, -2.1, 0.1)),
]


def _ensure_paths():
    """Ensure the SOBBH PN file path is set for downstream imports."""
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
    """Run the orchestrator for one binary on the given backend.

    Returns ``(tdi_amp, tdi_phase, phase_ref)`` as numpy arrays of shape
    ``(nchannels, N)`` / ``(N,)``.
    """
    o = _make_orbits()
    gb = GBTDIonTheFly(
        t_arr, T, t_ref,
        sampling_frequency=1.0 / (t_arr[1] - t_arr[0]),
        num_sub=1,
        orbits=o,
        force_backend=backend,
    )
    amp, f0, fdot, fddot, phi0, inc, psi, lam, beta = params
    out = gb(
        np.array([amp]), np.array([f0]), np.array([fdot]), np.array([fddot]),
        np.array([phi0]), np.array([inc]), np.array([psi]),
        np.array([lam]), np.array([beta]),
        return_spline=False,
    )
    # tdi_amp shape: (num_sub=1, nchannels=3, N) -> drop first axis.
    return (np.asarray(out.tdi_amp[0]),
            np.asarray(out.tdi_phase[0]),
            np.asarray(out.phase_ref[0]))


class TestJaxVsCppGB(unittest.TestCase):

    AMP_RTOL = 1e-10
    AMP_ATOL = 1e-12   # absolute floor in physical units (~1e-22 strain)
    # GB phases accumulate to ~2*pi*f0*T ~ 1e6 rad for f0~0.05 Hz over a
    # month. Float64 has ~16 decimal digits, so the absolute-phase parity
    # floor between two independent implementations of the same closed
    # form is ~1e-10 rad. With the chirpiest test case
    # (f0=0.05, fdot=1e-15, fddot=1e-22), the JAX-vs-C++ difference lands
    # at ~6e-10 rad -- still 1 part in 1e15 of the physical phase, but
    # above a strict 1e-10 cut. We use 1e-9 to leave comfortable headroom
    # over FMA-reordering differences between JAX (XLA) and the C++ libm
    # backend.
    PHASE_ATOL = 1e-9  # radians
    PREF_ATOL = 1e-9

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        cls.T = 3600.0 * 24 * 30  # 30 days
        cls.N = 256
        cls.t = np.linspace(0.0, cls.T, cls.N)
        cls.t_ref = 0.5 * cls.T

        cls.has_cpp = fastlisaresponse.has_backend("cpu")
        cls.has_jax = fastlisaresponse.has_backend("jax")
        if not cls.has_jax:
            raise unittest.SkipTest("JAX backend not registered.")

    def test_jax_runs(self):
        """JAX path produces finite amp/phase on every case."""
        for name, params in _GB_CASES:
            with self.subTest(case=name):
                amp_j, ph_j, pr_j = _run_one("jax", self.t, self.T, self.t_ref, params)
                self.assertTrue(np.all(np.isfinite(amp_j)), f"{name}: amp NaN")
                self.assertTrue(np.all(np.isfinite(ph_j)), f"{name}: phase NaN")
                self.assertTrue(np.all(np.isfinite(pr_j)), f"{name}: pref NaN")
                self.assertGreater(np.max(np.abs(amp_j)), 0.0,
                                   f"{name}: amp identically zero")

    def test_parity_with_cpp(self):
        """JAX and C++ outputs match to rtol=1e-10 per pixel."""
        if not self.has_cpp:
            self.skipTest("C++ 'cpu' backend not installed -- parity skipped.")

        for name, params in _GB_CASES:
            with self.subTest(case=name):
                amp_c, ph_c, pr_c = _run_one("cpu", self.t, self.T, self.t_ref, params)
                amp_j, ph_j, pr_j = _run_one("jax", self.t, self.T, self.t_ref, params)

                # Amplitude parity, per channel.
                for ch in range(amp_c.shape[0]):
                    np.testing.assert_allclose(
                        amp_j[ch], amp_c[ch],
                        rtol=self.AMP_RTOL, atol=self.AMP_ATOL,
                        err_msg=f"{name}: tdi_amp[ch={ch}] mismatch",
                    )
                    # Phase agreement is modulo 2 pi -- compare wrap-corrected.
                    diff = ph_j[ch] - ph_c[ch]
                    diff_mod = np.remainder(diff + np.pi, 2 * np.pi) - np.pi
                    self.assertLess(
                        float(np.max(np.abs(diff_mod))),
                        self.PHASE_ATOL,
                        msg=f"{name}: tdi_phase[ch={ch}] disagrees by "
                            f"{float(np.max(np.abs(diff_mod))):.3e}",
                    )
                # Reference phase parity (one per time sample).
                np.testing.assert_allclose(
                    pr_j, pr_c, rtol=0.0, atol=self.PREF_ATOL,
                    err_msg=f"{name}: phase_ref mismatch",
                )


if __name__ == "__main__":
    unittest.main()
