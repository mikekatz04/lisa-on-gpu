"""JAX SOBBH (stellar-origin black-hole binary) source.

Thin class wrapper around the module-level ``amplitude(t, params)`` /
``gw_phase(t, params)`` functions exposed by the prototype
``sobbhtaylert3.py`` at the repo root. That file is the single source
of truth for the PN expressions; the C++ ``SOBBHTDIonTheFly`` kernel in
``TDIonTheFly.cu`` was ported from the same place, so the JAX path and
the C++ path stay in sync as long as both load the same PN code.

Parameter layout matches the C++ class header
(``TDIonTheFly.hh:205-216``):

    params[0]  = m1            (solar masses)
    params[1]  = m2            (solar masses)
    params[2]  = s1            (dimensionless spin, primary)
    params[3]  = s2            (dimensionless spin, secondary)
    params[4]  = distance      (parsecs)
    params[5]  = f_low         (Hz, GW frequency at t_ref)
    params[6]  = phi_c         (rad, reference orbital phase)
    params[7]  = inc           (rad, inclination)
    params[8]  = psi           (rad, polarization)
    params[9]  = lam           (rad, ecliptic longitude)
    params[10] = beta          (rad, ecliptic latitude)
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp

from ..base import JaxAmpPhaseSource


def _load_sobbh_pn():
    """Locate and import ``sobbhtaylert3.py`` from the repo root.

    The file lives outside the installed package (it is the reference
    prototype kept at the repo root). We resolve it relative to the
    package install location by walking up parents looking for the
    file. The env var ``SOBBH_TAYLOR_T3_PATH`` overrides this.
    """
    env = os.environ.get("SOBBH_TAYLOR_T3_PATH")
    if env and Path(env).is_file():
        target = Path(env)
    else:
        here = Path(__file__).resolve()
        target = None
        for parent in here.parents[:6]:
            candidate = parent / "sobbhtaylert3.py"
            if candidate.is_file():
                target = candidate
                break
        if target is None:
            raise FileNotFoundError(
                "Could not locate sobbhtaylert3.py. Set SOBBH_TAYLOR_T3_PATH "
                "to point at it explicitly."
            )

    spec = importlib.util.spec_from_file_location("_sobbh_pn", str(target))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pn = None


def _pn_mod():
    global _pn
    if _pn is None:
        _pn = _load_sobbh_pn()
    return _pn


class JaxSOBBHSource(JaxAmpPhaseSource):
    """Stellar-origin binary black-hole TaylorT3-style PN source.

    Args:
        t_ref: Reference time at which ``f_low`` is the GW frequency
            (seconds). Same as ``SOBBHTDIonTheFly::t_ref`` in C++.
    """

    n_params: int = 11
    param_names: Tuple[str, ...] = (
        "m1", "m2", "s1", "s2",
        "distance", "f_low", "phi_c",
        "inc", "psi", "lam", "beta",
    )
    # Sky-angle indices for SOBBH (shifted by +2 vs UCB).
    inc_index: int = 7
    psi_index: int = 8
    lam_index: int = 9
    beta_index: int = 10

    def __init__(self, t_ref: float):
        self.t_ref = float(t_ref)

    def amplitude(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Delegate to ``sobbhtaylert3.amplitude(t, params)``.

        Note: ``t_ref`` is implicit in the SOBBH PN model -- ``f_low``
        is the GW frequency at the time origin. The C++ kernel uses
        ``t`` directly (not ``t - t_ref``), and so do we. Setting
        ``t_ref != 0`` shifts the user's time origin but does not enter
        the amplitude formula.
        """
        return _pn_mod().amplitude(t, params)

    def phase(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Delegate to ``sobbhtaylert3.gw_phase(t, params)``."""
        return _pn_mod().gw_phase(t, params)

    def f(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Delegate to ``sobbhtaylert3.gw_frequency_t(t, params)``."""
        return _pn_mod().gw_frequency_t(t, params)
