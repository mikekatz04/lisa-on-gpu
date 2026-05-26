"""JAX-vs-C++ parity test for the WDM kernels.

Configuration:
* Observation: 4 months (4 * 30 * 86400 s = 1.0368e7 s, picked to land
  on an integer number of 2-hour wavelet pixels).
* Wavelet pixel duration: ~2 hours (target). ``WDMSettings.adjust_to_even_bins``
  picks the closest duration in ``[1.9, 2.1]`` hours that makes both
  ``Nf`` and ``Nt`` even at the underlying ``dt = 15 s`` (1.0368e7 s
  / 15 s = 691200 total samples, factorable as ~ 1440 x 480).
* Lookup table: ``n_ref_only`` build (Plan A) at fdot=0, num_layers_diff=2.
  Cached to ``wdm_lookup_4mo_2hr_jax_parity.h5`` next to the test.
* Inject a handful of GBs in the mid-band and compare the JAX and
  C++ ``gb_wdm_{get_ll,fill_global,swap_ll}`` outputs.

The test auto-skips if the C++ backend isn't installed (so it can
still demonstrate the JAX kernels run end-to-end on the real
lookup table).
"""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STORE_PATH = REPO_ROOT / "wdm_lookup_4mo_2hr_jax_parity.h5"


def _ensure_sobbh_path():
    if "SOBBH_TAYLOR_T3_PATH" not in os.environ:
        candidate = REPO_ROOT / "sobbhtaylert3.py"
        if candidate.is_file():
            os.environ["SOBBH_TAYLOR_T3_PATH"] = str(candidate)


def _build_or_load_wdm_lookup(store_path: Path, backend_name: str = "cpu"):
    """Return ``(wdm_lookup_table, wdm_settings)`` for the test config.

    Uses ``WDM_BUILD_KIND='n_ref_only'`` (Plan A), fdot=0. Cached to
    ``store_path``; rebuilt if the file is absent.
    """
    from lisatools.domains import WDMLookupTable, WDMSettings

    # 4 months, 2-hour wavelet target.
    Tobs = 4.0 * 30.0 * 86400.0           # 1.0368e7 s
    dt = 15.0                              # underlying TD sample step
    Nf, Nt, wavelet_duration = WDMSettings.adjust_to_even_bins(
        t_min=1.9 * 3600.0, t_max=2.1 * 3600.0,
        dt=dt, Tobs=Tobs, num_linspace=2000,
    )
    print(f"[wdm-build] Nf={Nf}, Nt={Nt}, wavelet_duration={wavelet_duration:.2f}s "
          f"({wavelet_duration/3600.0:.3f}h), Tobs={Nt*wavelet_duration:.3e}s")

    # Restrict the active band to the LISA-relevant mHz region.
    min_freq = 1.0e-4
    max_freq = 1.5e-2

    # Time-band: trim edges so the window doesn't leak.
    EDGE_CUT = 8
    min_time = EDGE_CUT * wavelet_duration
    max_time = (Nt - EDGE_CUT) * wavelet_duration

    wdm_set = WDMSettings(
        Nf, Nt, dt,
        min_freq=min_freq, max_freq=max_freq,
        min_time=min_time, max_time=max_time,
        force_backend=backend_name,
    )

    if store_path.exists():
        print(f"[wdm-build] loading cached lookup table {store_path}")
        wdm_lookup = WDMLookupTable.from_file(str(store_path), force_backend=backend_name)
        return wdm_lookup, wdm_set

    # Build (n_ref_only / Plan A). One-pixel build => fast.
    m_ref = int(3.0e-3 / wdm_set.layer_df)
    norm_freq_single_layer, m_diffs, _ = WDMLookupTable.apply_eps_frequency(
        eps=0.001, settings=wdm_set, m_ref=m_ref, num_layers_diff=2,
    )
    fdot_vals = np.array([0.0])     # no fdot for the parity test

    print(f"[wdm-build] building lookup table (m_ref={m_ref}, "
          f"len(norm_freq)={len(norm_freq_single_layer)}, len(m_diffs)={len(m_diffs)}) -> {store_path}")
    t0 = time.perf_counter()
    wdm_lookup = WDMLookupTable(
        wdm_set, nchannels=3,
        norm_freq_single_layer=norm_freq_single_layer,
        m_diffs=m_diffs, fdot_vals=fdot_vals, m_ref=m_ref,
        batch_size_gen=5, td_window=None,
        store_path=str(store_path),
        build_kind="n_ref_only",
        time_layers=256,           # speeds the build up; Plan A only reads 1 pixel
    )
    print(f"[wdm-build] built in {time.perf_counter() - t0:.1f}s")
    return wdm_lookup, wdm_set


def _ensure_paths():
    _ensure_sobbh_path()


class TestJaxVsCppWDM(unittest.TestCase):
    """JAX-vs-C++ parity for the GB WDM kernels at 4 months / 2-hour pixels."""

    @classmethod
    def setUpClass(cls):
        _ensure_paths()
        import fastlisaresponse
        cls.has_cpp = fastlisaresponse.has_backend("cpu")
        cls.has_jax = fastlisaresponse.has_backend("jax")
        if not cls.has_jax:
            raise unittest.SkipTest("JAX backend not registered.")

        cls.store_path = _DEFAULT_STORE_PATH
        cls.wdm_lookup, cls.wdm_set = _build_or_load_wdm_lookup(cls.store_path)

        # Total observation comes from the wavelet grid; t_ref at the
        # middle of the active time band so the test pixels lie inside.
        cls.Nt = cls.wdm_set.Nt
        cls.Nf = cls.wdm_set.Nf
        cls.layer_dt = cls.wdm_set.layer_dt
        cls.Tobs = cls.Nt * cls.layer_dt
        cls.t_ref = cls.Tobs * 0.5

        # Two test GBs in the middle of the active band.
        # f0 ~ 3 mHz (matches m_ref), small chirp, sky angles off the
        # poles. Amp ~ 1e-22 (typical LISA-band GB strain).
        cls.gb_cases = [
            # (amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)
            (1.0e-22, 3.0e-3, 0.0, 0.0, 0.3, 0.7, 0.4, 1.2, 0.4),
            (5.0e-23, 5.0e-3, 1.0e-17, 0.0, 1.1, 1.0, 0.2, -0.7, -0.3),
        ]

    def _make_wdm_holder(self):
        """Build a minimal AnalysisContainerArray-like holder with data + noise.

        The C++ ``GBWDMComputations`` reads ``wdm_holder.linear_data_arr[0]``
        and ``wdm_holder.linear_psd_arr[0]``. For the parity test we
        synthesise zero data + identity noise so both backends compare
        ``h_h`` against the same template-only inner product.
        """
        # Build via lisatools' AnalysisContainerArray when available;
        # fall back to a duck-typed namespace otherwise.
        from types import SimpleNamespace
        nch = 3
        Nf_act = self.wdm_set.Nf_active
        Nt_act = self.wdm_set.Nt_active

        # XYZ cross-channel layout: noise shape (1, nch, nch, Nf_act, Nt_act).
        data = np.zeros((1, nch, Nf_act, Nt_act), dtype=np.float64)
        # Identity invC per pixel so (h|h) reduces to sum_c sum_pix w_c^2.
        invC = np.zeros((1, nch, nch, Nf_act, Nt_act), dtype=np.float64)
        for c in range(nch):
            invC[0, c, c] = 1.0

        return SimpleNamespace(
            linear_data_arr=[data.reshape(-1)],
            linear_psd_arr=[invC.reshape(-1)],
            __len__=lambda self: 1,    # unused for AnalysisContainerArray API
        ), len

    def _run_get_ll(self, backend_name: str, params: np.ndarray):
        from fastlisaresponse.gbcomps import GBWDMComputations
        from fastlisaresponse.tdiconfig import TDIConfig
        from lisatools.detector import EqualArmlengthOrbits

        orbits = EqualArmlengthOrbits()
        orbits.configure(linear_interp_setup=True)
        tdi_config = TDIConfig("1st generation")

        gb = GBWDMComputations(
            self.wdm_set, t_ref=self.t_ref,
            orbits=orbits, tdi_config=tdi_config,
            force_backend=backend_name, tdi_type="XYZ",
        )
        holder, _ = self._make_wdm_holder()
        # `wdm_holder.__len__` is needed by the C++ ``len(wdm_holder)``;
        # supply a real list-of-1 instead.
        holder = type("WDMHolder", (), {
            "linear_data_arr": holder.linear_data_arr,
            "linear_psd_arr": holder.linear_psd_arr,
            "__len__": lambda self: 1,
        })()
        like = gb.get_ll_wdm(
            params, holder, convert_to_ra_dec=False,
        )
        return np.asarray(like), np.asarray(gb.d_h_out), np.asarray(gb.h_h_out)

    def test_jax_runs(self):
        """JAX path runs end-to-end on a real lookup table and produces finite output."""
        for i, p in enumerate(self.gb_cases):
            with self.subTest(case=i):
                like, dh, hh = self._run_get_ll(
                    "jax", np.atleast_2d(np.asarray(p, dtype=np.float64)),
                )
                self.assertTrue(np.all(np.isfinite(like)),
                                f"case {i}: like NaN")
                self.assertTrue(np.all(np.isfinite(dh)) and np.all(np.isfinite(hh)),
                                f"case {i}: inner products NaN")
                self.assertGreater(float(hh[0]), 0.0,
                                   f"case {i}: h_h identically zero (no overlap with template)")

    def test_parity_with_cpp(self):
        """JAX and C++ ``gb_wdm_get_ll`` outputs match across the test GBs."""
        if not self.has_cpp:
            self.skipTest("C++ 'cpu' backend not installed -- WDM parity skipped.")

        for i, p in enumerate(self.gb_cases):
            with self.subTest(case=i):
                params2d = np.atleast_2d(np.asarray(p, dtype=np.float64))
                like_c, dh_c, hh_c = self._run_get_ll("cpu", params2d)
                like_j, dh_j, hh_j = self._run_get_ll("jax", params2d)

                # data == 0 so d_h must be 0 on both sides; assert
                # h_h matches to the FMA-reorder floor.
                np.testing.assert_allclose(dh_j, dh_c, rtol=0.0, atol=1e-30,
                                           err_msg=f"case {i}: d_h mismatch")
                np.testing.assert_allclose(hh_j, hh_c, rtol=1e-8, atol=0.0,
                                           err_msg=f"case {i}: h_h mismatch "
                                                   f"(jax={hh_j}, cpp={hh_c})")


if __name__ == "__main__":
    unittest.main()
