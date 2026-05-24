"""JAX UCB (galactic-binary) source.

1:1 JAX port of ``GBTDIonTheFly::ucb_{amplitude,phase,f,fdot}`` from
``TDIonTheFly.cu``. Parameter layout matches the C++ class:

    params[0] = amplitude  (A0)
    params[1] = f0         (Hz, at t_ref)
    params[2] = fdot       (Hz/s)
    params[3] = fddot      (Hz/s^2)
    params[4] = phi0       (rad)
    params[5] = inc        (rad)
    params[6] = psi        (rad)
    params[7] = lam        (rad, ecliptic longitude)
    params[8] = beta       (rad, ecliptic latitude)

Same formulas as ``GBTDIonTheFly::ucb_*`` in
``TDIonTheFly.cu:4471-4514``. ``t_ref`` is fixed at construction.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from ..base import JaxAmpPhaseSource


class JaxUCBSource(JaxAmpPhaseSource):
    """Galactic-binary monochromatic-with-chirp source.

    Args:
        t_ref: GB phase reference time (seconds). Same as
            ``GBTDIonTheFly::t_ref`` in C++.
    """

    n_params: int = 9
    param_names: Tuple[str, ...] = (
        "amp", "f0", "fdot", "fddot", "phi0",
        "inc", "psi", "lam", "beta",
    )
    inc_index: int = 5
    psi_index: int = 6
    lam_index: int = 7
    beta_index: int = 8

    # C++ indices
    amplitude_index = 0
    f0_index = 1
    fdot0_index = 2
    fddot0_index = 3
    phi0_index = 4

    def __init__(self, t_ref: float):
        self.t_ref = float(t_ref)

    def amplitude(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        # Mirrors TDIonTheFly.cu:4488-4495.
        A0 = params[self.amplitude_index]
        f0 = params[self.f0_index]
        fdot = params[self.fdot0_index]
        t_diff = t - self.t_ref
        return A0 * (1.0 + (2.0 / 3.0) * (fdot / f0) * t_diff)

    def phase(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        # Mirrors TDIonTheFly.cu:4471-4484. Note the LDC ``-phi0``.
        f0 = params[self.f0_index]
        phi0 = params[self.phi0_index]
        fdot = params[self.fdot0_index]
        fddot = params[self.fddot0_index]
        t_diff = t - self.t_ref
        return (
            -phi0
            + 2.0 * jnp.pi * (
                f0 * t_diff
                + 0.5 * fdot * t_diff * t_diff
                + (1.0 / 6.0) * fddot * t_diff * t_diff * t_diff
            )
        )

    # ---- convenience helpers (not required by projection) ----
    def f(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        f0 = params[self.f0_index]
        fdot = params[self.fdot0_index]
        fddot = params[self.fddot0_index]
        t_diff = t - self.t_ref
        return f0 + fdot * t_diff + 0.5 * fddot * t_diff * t_diff

    def fdot(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        fdot = params[self.fdot0_index]
        fddot = params[self.fddot0_index]
        t_diff = t - self.t_ref
        return fdot + fddot * t_diff
