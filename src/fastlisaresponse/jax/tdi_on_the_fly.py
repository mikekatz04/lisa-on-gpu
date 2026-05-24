"""Functional JAX kernels mirroring ``gb_run_wave_tdi_wrap`` and
``sobbh_run_wave_tdi_wrap`` from the C++ side.

These are *pure* functions of (params, t_arr, source, orbits,
tdi_config) -- they return new arrays rather than mutating buffers, so
``jax.grad`` / ``jax.jit`` / ``jax.vmap`` work without surprises. The
``GBTDIonTheFlyWrapJAX`` / ``SOBBHTDIonTheFlyWrapJAX`` thin objects in
``wrappers.py`` adapt this to the C++ wrapper API.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .amp_phase_extract import extract_and_unwrap
from .base import JaxAmpPhaseSource
from .orbits import OrbitsWrapJAX
from .projection import get_phase_ref, get_sky_vectors, get_tdi_Xf_single
from .tdi_config import TDIConfigWrapJAX


def _run_single_binary(params: jnp.ndarray, t_arr: jnp.ndarray,
                       source: JaxAmpPhaseSource,
                       orbits: OrbitsWrapJAX,
                       tdi_config: TDIConfigWrapJAX
                       ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate the (M, tdi_amp, tdi_phase, phase_ref) block for one source.

    Returns:
        M: complex (nchannels, N) -- the raw complex TDI samples.
        tdi_amp: real (nchannels, N) -- signed amplitudes.
        tdi_phase: real (nchannels, N) -- unwrapped phases.
        phase_ref: real (N,) -- the SC1 reference phase.
    """
    # Sky vectors (per-source, constant in t).
    k, u, v = get_sky_vectors(params, source)

    # Project at every time. The projection returns shape (N, nchannels).
    proj = get_tdi_Xf_single(t_arr, params, source, orbits, tdi_config, k, u, v)
    # Re-shape to (nchannels, N) to match the C++ memory layout used by
    # the WDM / spline downstream kernels.
    M = jnp.transpose(proj, (1, 0))

    # Reference phase at spacecraft 1.
    phase_ref = get_phase_ref(t_arr, params, source, orbits)

    # Extract per-channel (amp, unwrapped phase).
    tdi_amp, tdi_phase = extract_and_unwrap(M, phase_ref)
    return M, tdi_amp, tdi_phase, phase_ref


def run_wave_tdi(params: jnp.ndarray, t_arr: jnp.ndarray,
                 source: JaxAmpPhaseSource,
                 orbits: OrbitsWrapJAX,
                 tdi_config: TDIConfigWrapJAX
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Functional analog of ``LISATDIonTheFly::run_wave_tdi``.

    ``params``: shape ``(num_bin, n_params)`` OR flat ``(num_bin * n_params,)``.
    ``t_arr``: shape ``(num_bin, N)``  OR ``(N,)`` (will be broadcast).

    Returns ``(M, tdi_amp, tdi_phase, phase_ref)`` with shapes
        M         : complex (num_bin, nchannels, N)
        tdi_amp   : real    (num_bin, nchannels, N)
        tdi_phase : real    (num_bin, nchannels, N)
        phase_ref : real    (num_bin, N)
    """
    n_params = source.n_params
    params = jnp.asarray(params)
    if params.ndim == 1:
        assert params.shape[0] % n_params == 0
        params = params.reshape(-1, n_params)
    elif params.ndim == 2:
        assert params.shape[1] == n_params
    else:
        raise ValueError(f"params must be 1D or 2D, got shape {params.shape}.")
    num_bin = params.shape[0]

    t_arr = jnp.asarray(t_arr)
    if t_arr.ndim == 1:
        t_arr = jnp.broadcast_to(t_arr, (num_bin, t_arr.shape[0]))
    elif t_arr.ndim == 2:
        if t_arr.shape[0] == 1:
            t_arr = jnp.broadcast_to(t_arr, (num_bin, t_arr.shape[1]))
    else:
        raise ValueError(f"t_arr must be 1D or 2D, got shape {t_arr.shape}.")
    assert t_arr.shape[0] == num_bin

    # ``vmap`` over the binary axis. The projection itself is
    # constructed at trace time (TDIConfig unrolled), so vmap turns the
    # whole pipeline into one fused computation.
    def per_bin(p, t):
        return _run_single_binary(p, t, source, orbits, tdi_config)

    Ms, amps, phases, prefs = jax.vmap(per_bin)(params, t_arr)
    return Ms, amps, phases, prefs


# --------------------------------------------------------------------
# Public source-specific convenience kernels matching C++ names.
# --------------------------------------------------------------------
def gb_run_wave_tdi(params: jnp.ndarray, t_arr: jnp.ndarray,
                    t_ref: float,
                    orbits: OrbitsWrapJAX,
                    tdi_config: TDIConfigWrapJAX
                    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX analog of ``gb_run_wave_tdi_wrap``.

    Parameters
    ----------
    params : array (num_bin, 9) or (9*num_bin,)
        UCB parameters per binary; layout from
        :class:`JaxUCBSource`.
    t_arr : array (num_bin, N) or (N,)
        Times (seconds, SSB).
    t_ref : float
        UCB reference time.
    orbits, tdi_config : JAX-side handles.

    Returns ``(M, tdi_amp, tdi_phase, phase_ref)`` -- same shape contract
    as :func:`run_wave_tdi`.
    """
    from .sources.ucb import JaxUCBSource
    source = JaxUCBSource(t_ref=t_ref)
    return run_wave_tdi(params, t_arr, source, orbits, tdi_config)


def sobbh_run_wave_tdi(params: jnp.ndarray, t_arr: jnp.ndarray,
                       t_ref: float,
                       orbits: OrbitsWrapJAX,
                       tdi_config: TDIConfigWrapJAX
                       ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX analog of ``sobbh_run_wave_tdi_wrap``.

    See :func:`gb_run_wave_tdi`; the only difference is the source
    plugin and parameter count (11 instead of 9).
    """
    from .sources.sobbh import JaxSOBBHSource
    source = JaxSOBBHSource(t_ref=t_ref)
    return run_wave_tdi(params, t_arr, source, orbits, tdi_config)
