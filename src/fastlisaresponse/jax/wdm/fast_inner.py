"""JAX port of ``fast_wdm_inner`` (TDIonTheFly.cu:1306-1424).

Per-time-pixel projection + central-difference frequency computation,
returning the convention ``M' = conj(M * exp(-i pi/2))`` that
:func:`WaveletLookupTableWrapJAX.get_w_mn_lookup` expects.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from ..base import JaxAmpPhaseSource
from ..orbits import OrbitsWrapJAX
from ..projection import get_phase_ref, get_sky_vectors, get_tdi_Xf_single
from ..tdi_config import TDIConfigWrapJAX


def _project_at(t: jnp.ndarray, params: jnp.ndarray,
                source: JaxAmpPhaseSource,
                orbits: OrbitsWrapJAX,
                tdi_config: TDIConfigWrapJAX,
                k: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray
                ) -> jnp.ndarray:
    """Project at a single ``t``, returning complex (nchannels,) array."""
    return get_tdi_Xf_single(t, params, source, orbits, tdi_config, k, u, v)


def fast_wdm_inner_jax(tn: jnp.ndarray, params: jnp.ndarray,
                       source: JaxAmpPhaseSource,
                       orbits: OrbitsWrapJAX,
                       tdi_config: TDIConfigWrapJAX,
                       k: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
                       deriv_delta_t: float = 500.0
                       ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Mirror of the C++ ``fast_wdm_inner``.

    Returns
    -------
    M_prime : complex (nchannels,)
        The rotated TDI value ``conj(M * exp(-i pi/2))`` --
        the convention the WDM lookup expects.
    f : real (nchannels,)
        Per-channel instantaneous frequency from the central-difference
        phase derivative.
    fdot : real (nchannels,)
        Per-channel fdot (currently held at 0 to match the C++ default;
        the C++ has the analytic-fdot path commented out at
        ``TDIonTheFly.cu:1399-1403``).
    phase_ref : real scalar
        Reference phase at ``tn`` (for diagnostics; the kernels here
        don't use it -- the rotation has already absorbed it).
    """
    M_mid = _project_at(tn, params, source, orbits, tdi_config, k, u, v)
    M_up = _project_at(tn + deriv_delta_t, params, source, orbits, tdi_config, k, u, v)
    M_dn = _project_at(tn - deriv_delta_t, params, source, orbits, tdi_config, k, u, v)

    phase_ref_mid = get_phase_ref(tn, params, source, orbits)
    phase_ref_up = get_phase_ref(tn + deriv_delta_t, params, source, orbits)
    phase_ref_dn = get_phase_ref(tn - deriv_delta_t, params, source, orbits)

    residual_frequency = (phase_ref_up - phase_ref_dn) / (2.0 * deriv_delta_t) / (2.0 * jnp.pi)

    # Per-channel TDI phases include the reference phase (-arg(M * exp(i phi_ref))).
    # ``M * exp(i phi)`` because we want the phase of the combined signal in the
    # convention the C++ uses.
    def tdi_phase_at(Mc, phi):
        rot = jnp.exp(1j * phi)
        return -jnp.angle(Mc * rot)

    phi_mid = tdi_phase_at(M_mid, phase_ref_mid)
    phi_up = tdi_phase_at(M_up, phase_ref_up)
    phi_dn = tdi_phase_at(M_dn, phase_ref_dn)

    # Unwrap each side against the mid phase before differencing (the
    # C++ unwrap-around-mid trick, TDIonTheFly.cu:1384-1389).
    dphi_up = phi_up - phi_mid
    dphi_up = jnp.where(dphi_up > jnp.pi, dphi_up - 2.0 * jnp.pi, dphi_up)
    dphi_up = jnp.where(dphi_up < -jnp.pi, dphi_up + 2.0 * jnp.pi, dphi_up)
    dphi_dn = phi_dn - phi_mid
    dphi_dn = jnp.where(dphi_dn > jnp.pi, dphi_dn - 2.0 * jnp.pi, dphi_dn)
    dphi_dn = jnp.where(dphi_dn < -jnp.pi, dphi_dn + 2.0 * jnp.pi, dphi_dn)

    tdi_freq = (dphi_up - dphi_dn) / (2.0 * deriv_delta_t) / (2.0 * jnp.pi)
    f = residual_frequency + tdi_freq
    fdot = jnp.zeros_like(f)

    # Rotate to the M' convention.
    M_prime = jnp.conj(M_mid * jnp.exp(-1j * jnp.pi / 2.0))

    # When the C++ kernel finds M_mid == 0 (out of orbit bounds) it
    # bails. We propagate a zero M' so the lookup yields 0.
    bad = (jnp.abs(M_mid[0]) == 0.0) | (jnp.abs(M_up[0]) == 0.0) | (jnp.abs(M_dn[0]) == 0.0)
    M_prime = jnp.where(bad, jnp.zeros_like(M_prime), M_prime)

    return M_prime, f, fdot, phase_ref_mid
