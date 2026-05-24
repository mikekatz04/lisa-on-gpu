"""JAX port of ``LISATDIonTheFly::get_tdi_Xf_single`` and friends.

Goal: byte-for-byte parity (modulo last-bit FMA reorderings) with the
C++ projection in ``TDIonTheFly.cu:3605-3910``. The structure here
mirrors that file 1:1; comments cite C++ line numbers where useful.

The functions take a :class:`OrbitsWrapJAX`, a
:class:`TDIConfigWrapJAX`, and a source object inheriting from
:class:`JaxAmpPhaseSource`. The TDI configuration is consumed as
*Python* lists/arrays (not jnp) so the per-unit / per-sub-delay loops
unroll at trace time -- ``num_units`` is small (24 for 1st-gen, 48 for
2nd-gen) and unrolling is the simplest way to keep ``jax.jit`` and
``jax.grad`` happy without resorting to ``jax.lax.fori_loop`` over
heterogeneous link indices.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .base import JaxAmpPhaseSource
from .orbits import OrbitsWrapJAX
from .tdi_config import TDIConfigWrapJAX

# Mirrors ``C_inv = 3.3356409519815204e-09`` in TDIonTheFly.cu (the
# reciprocal of the speed of light in s/m).
C_INV = 3.3356409519815204e-09


# --------------------------------------------------------------------
# Geometry helpers (TDIonTheFly.cu:3605-3635)
# --------------------------------------------------------------------
def get_sky_vectors(params: jnp.ndarray, source: JaxAmpPhaseSource
                    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (k, u, v) sky basis from (lam, beta) in ``params``.

    Identical to ``LISATDIonTheFly::get_sky_vectors``
    (``TDIonTheFly.cu:3605-3625``).
    """
    inc_i, psi_i, lam_i, beta_i = source.get_sky_indices()
    beta = params[beta_i]
    lam = params[lam_i]
    cb, sb = jnp.cos(beta), jnp.sin(beta)
    cl, sl = jnp.cos(lam), jnp.sin(lam)
    v = jnp.array([-sb * cl, -sb * sl, cb])
    u = jnp.array([sl, -cl, 0.0])
    k = jnp.array([-cb * cl, -cb * sl, -sb])
    return k, u, v


def xi_projections(u: jnp.ndarray, v: jnp.ndarray, n: jnp.ndarray
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """``(xi_p, xi_c)`` for arm normal ``n``.

    Mirrors ``LISATDIonTheFly::xi_projections``
    (``TDIonTheFly.cu:3628-3635``).
    """
    u_dot_n = jnp.sum(u * n, axis=-1)
    v_dot_n = jnp.sum(v * n, axis=-1)
    xi_p = 0.5 * (u_dot_n * u_dot_n - v_dot_n * v_dot_n)
    xi_c = u_dot_n * v_dot_n
    return xi_p, xi_c


# --------------------------------------------------------------------
# get_hp_hc  (TDIonTheFly.cu:3886-3910)
# --------------------------------------------------------------------
def get_hp_hc(t: jnp.ndarray, params: jnp.ndarray,
              source: JaxAmpPhaseSource, phase_change: float
              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Polarization tensor projection at time(s) ``t``.

    Matches the C++ ``LISATDIonTheFly::get_hp_hc``: the un-projected
    source polarizations are
        hSp = -cos(phase + phase_change) * amp * (1 + cos^2(iota))
        hSc = -sin(phase + phase_change) * 2 * amp * cos(iota)
    and the rotation by ``psi`` gives
        hp = hSp * cos(2psi) - hSc * sin(2psi)
        hc = hSp * sin(2psi) + hSc * cos(2psi)
    """
    inc_i, psi_i, _, _ = source.get_sky_indices()
    amp = source.amplitude(t, params)
    phase = source.phase(t, params)
    psi = params[psi_i]
    inc = params[inc_i]

    cos2psi = jnp.cos(2.0 * psi)
    sin2psi = jnp.sin(2.0 * psi)
    cosi = jnp.cos(inc)

    hSp = -jnp.cos(phase + phase_change) * amp * (1.0 + cosi * cosi)
    hSc = -jnp.sin(phase + phase_change) * 2.0 * amp * cosi

    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi
    return hp, hc


# --------------------------------------------------------------------
# Single-time-sample TDI projection (TDIonTheFly.cu:3699-3834)
# --------------------------------------------------------------------
def get_tdi_Xf_single(t: jnp.ndarray, params: jnp.ndarray,
                      source: JaxAmpPhaseSource,
                      orbits: OrbitsWrapJAX,
                      tdi_config: TDIConfigWrapJAX,
                      k: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray
                      ) -> jnp.ndarray:
    """Project the source onto LISA arms and form the TDI variables at ``t``.

    Returns complex array of shape ``t.shape + (num_channels,)``
    (typically ``(N, 3)`` for one binary, or scalar last axis for one
    time sample).

    The implementation unrolls the ``num_units`` outer loop at Python
    trace time (the TDIConfig arrays are concrete numpy ints). The
    inner per-time work is a single ``jnp`` expression so JAX can fuse
    everything via ``jit`` / ``vmap``.
    """
    num_units = tdi_config.num_units
    num_channels = tdi_config.num_channels

    # Force ``t`` to at least 1D so the reductions broadcast cleanly.
    t = jnp.asarray(t)

    # Per-channel accumulators (complex). Build as a list and stack
    # so we can selectively add into specific channels per unit.
    accum_per_channel = [jnp.zeros_like(t, dtype=jnp.complex128)
                         for _ in range(num_channels)]

    for unit_i in range(num_units):
        unit_start = int(tdi_config.unit_starts[unit_i])
        unit_length = int(tdi_config.unit_lengths[unit_i])
        base_link = int(tdi_config.tdi_base_links[unit_i])
        channel = int(tdi_config.channels[unit_i])
        sign = float(tdi_config.tdi_signs[unit_i])

        # Sum of LTTs of all combination links applied as delays before
        # the base-link projection. Matches TDIonTheFly.cu:3737-3756 --
        # each LTT is evaluated at the original ``t``, not chained.
        total_delay = jnp.zeros_like(t)
        for sub_i in range(unit_length):
            ci = unit_start + sub_i
            comb_link = int(tdi_config.tdi_link_combinations[ci])
            if comb_link == -11:
                # ``-11`` is the C++ sentinel for "no delay in this slot".
                continue
            total_delay = total_delay + orbits.get_light_travel_time(
                t, comb_link
            )

        time_eval = t - total_delay
        time_rec = time_eval

        # Base-link LTT and the receiver/emitter spacecraft for ``base_link``.
        L = orbits.get_light_travel_time(time_rec, base_link)
        time_em = time_rec - L

        bli = orbits.get_link_ind(base_link)
        sc_r = int(orbits.sc_r[bli])
        sc_e = int(orbits.sc_e[bli])

        x_rec = orbits.get_pos(time_rec, sc_r)
        x_em = orbits.get_pos(time_em, sc_e)
        n = x_rec - x_em
        n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)

        k_dot_n = jnp.sum(k * n, axis=-1)
        k_dot_x_rec = jnp.sum(k * x_rec, axis=-1)
        k_dot_x_em = jnp.sum(k * x_em, axis=-1)

        pre_factor = 1.0 / (1.0 - k_dot_n)

        delay_rec = time_rec - k_dot_x_rec * C_INV
        delay_em = time_em - k_dot_x_em * C_INV

        xi_p, xi_c = xi_projections(u, v, n)

        # Real part of the projection: phase_change = 0.
        hp_rec_r, hc_rec_r = get_hp_hc(delay_rec, params, source, 0.0)
        hp_em_r, hc_em_r = get_hp_hc(delay_em, params, source, 0.0)
        large_real = (hp_em_r - hp_rec_r) * xi_p + (hc_em_r - hc_rec_r) * xi_c

        # Imag part: phase_change = pi/2.
        hp_rec_i, hc_rec_i = get_hp_hc(delay_rec, params, source, jnp.pi / 2.0)
        hp_em_i, hc_em_i = get_hp_hc(delay_em, params, source, jnp.pi / 2.0)
        large_imag = (hp_em_i - hp_rec_i) * xi_p + (hc_em_i - hc_rec_i) * xi_c

        contrib = sign * pre_factor * (large_real + 1j * large_imag)
        accum_per_channel[channel] = accum_per_channel[channel] + contrib

    return jnp.stack(accum_per_channel, axis=-1)


# --------------------------------------------------------------------
# get_phase_ref (TDIonTheFly.cu:4406-4420)
# --------------------------------------------------------------------
def get_phase_ref(t: jnp.ndarray, params: jnp.ndarray,
                  source: JaxAmpPhaseSource,
                  orbits: OrbitsWrapJAX) -> jnp.ndarray:
    """Reference-phase at spacecraft 1.

    Mirrors ``LISATDIonTheFly::get_phase_ref`` exactly: evaluate the
    source phase at the SSB-shifted time
    ``t - k . x_rec(t, sc=1) * C_inv``.
    """
    k, _, _ = get_sky_vectors(params, source)
    x_rec = orbits.get_pos(t, 1)  # spacecraft 1
    k_dot_x_rec = jnp.sum(k * x_rec, axis=-1)
    t_sc = t - k_dot_x_rec * C_INV
    return source.phase(t_sc, params)
