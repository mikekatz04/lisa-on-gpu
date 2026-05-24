"""JAX analogues of ``gb_wdm_het_{fill_global,get_ll,swap_ll}``.

These mirror the C++ kernels in
``../../cutils/TDIonthefly.cu`` (per the chunked-heterodyne pipeline
laid out in ``CHUNKED_HET_DESIGN_NOTES.md``). The kernel-shaped code
becomes a ``jax.lax.scan`` over chunks (outer) and a ``jax.vmap`` over
binaries (inner) -- functionally equivalent to the CUDA outer-block /
inner-loop pattern, JIT-friendly, and ``jax.grad``-able.

Validated algorithm reference lives in ``check_shortened_wdm.py``
(``chunked_get_ll_python_reference`` -> ``Test L`` passes to ~5e-16).
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp

from ..base import JaxAmpPhaseSource
from ..orbits import OrbitsWrapJAX
from ..projection import get_sky_vectors
from ..tdi_config import TDIConfigWrapJAX

from .fast_inner_heterodyne import (
    ALPHA_AUTO,
    fast_wdm_inner_heterodyne_jax,
    gb_chunk_fd_to_wdm_jax,
)


def _build_chunk_wdm(chunk_t_start, params, source, orbits, tdi_config,
                     k_sky, u_sky, v_sky,
                     T_chunk, N_sparse, Nf, Nt_sub, dt, wdm_window,
                     tukey_alpha):
    """Heterodyne FD -> chunk WDM (one (chunk, binary))."""
    N_chunk_td = Nf * Nt_sub
    chunk_fd, _k_f0 = fast_wdm_inner_heterodyne_jax(
        chunk_t_start=chunk_t_start, T_chunk=T_chunk,
        N_sparse=N_sparse, N_chunk_td=N_chunk_td,
        params=params, source=source, orbits=orbits, tdi_config=tdi_config,
        k_sky=k_sky, u_sky=u_sky, v_sky=v_sky, tukey_alpha=tukey_alpha,
    )
    w_chunk = gb_chunk_fd_to_wdm_jax(
        chunk_fd, wdm_window, Nf=Nf, Nt_sub=Nt_sub, data_dt=dt,
    )
    return w_chunk                                  # (nch, Nf, Nt_sub) real


def gb_wdm_het_fill_global_jax(
    params_batch: jnp.ndarray,                      # (num_bin, 9)
    factors:      jnp.ndarray,                      # (num_bin,)
    chunk_t_starts: jnp.ndarray,                    # (n_chunks,)
    chunk_keep_lo: jnp.ndarray, chunk_keep_hi: jnp.ndarray,
    chunk_n_global_lo: jnp.ndarray,
    source: JaxAmpPhaseSource, orbits: OrbitsWrapJAX, tdi_config: TDIConfigWrapJAX,
    wdm_window: jnp.ndarray,                        # (Nt_sub,)
    Nf: int, Nt: int, Nt_sub: int, N_sparse: int,
    dt: float, T_chunk: float,
    tukey_alpha: float = ALPHA_AUTO,
) -> jnp.ndarray:
    """Accumulate per-binary stitched WDM into a global template.

    Returns ``template`` of shape ``(nchannels, Nf, Nt)``.
    """
    num_bin = params_batch.shape[0]

    # Pre-compute sky vectors per binary (functions of lam, beta only).
    k_sky, u_sky, v_sky = jax.vmap(
        lambda p: get_sky_vectors(p, source)
    )(params_batch)
    # k_sky, u_sky, v_sky: (num_bin, 3)

    template = jnp.zeros((3, Nf, Nt))

    # Outer loop: chunks. Inner: vmap over binaries.
    def chunk_body(template_state, j):
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        keep_len = khi - klo

        def _one_bin(p, f, k_s, u_s, v_s):
            w_chunk = _build_chunk_wdm(
                chunk_t_start, p, source, orbits, tdi_config,
                k_s, u_s, v_s, T_chunk, N_sparse, Nf, Nt_sub, dt,
                wdm_window, tukey_alpha,
            )
            # restrict to chunk's keep band
            return f * jax.lax.dynamic_slice(
                w_chunk, (0, 0, klo), (3, Nf, keep_len),
            )                                          # (3, Nf, keep_len)
        per_bin = jax.vmap(_one_bin)(
            params_batch, factors, k_sky, u_sky, v_sky,
        )                                              # (num_bin, 3, Nf, keep_len)
        per_chunk_sum = jnp.sum(per_bin, axis=0)       # (3, Nf, keep_len)
        new_template = jax.lax.dynamic_update_slice(
            template_state, per_chunk_sum, (0, 0, n_lo),
        )
        # Note: dynamic_update_slice REPLACES that block; to accumulate
        # additively we'd need to read+add+write. Use jnp.add via gather.
        existing = jax.lax.dynamic_slice(
            template_state, (0, 0, n_lo), (3, Nf, keep_len),
        )
        new_template = jax.lax.dynamic_update_slice(
            template_state, existing + per_chunk_sum, (0, 0, n_lo),
        )
        return new_template, None

    n_chunks = chunk_t_starts.shape[0]
    template, _ = jax.lax.scan(chunk_body, template, jnp.arange(n_chunks))
    return template


def gb_wdm_het_get_ll_jax(
    params_batch: jnp.ndarray,                      # (num_bin, 9)
    data_d: jnp.ndarray, invC: jnp.ndarray,         # (3, Nf, Nt) each
    chunk_t_starts: jnp.ndarray,
    chunk_keep_lo: jnp.ndarray, chunk_keep_hi: jnp.ndarray,
    chunk_n_global_lo: jnp.ndarray,
    source: JaxAmpPhaseSource, orbits: OrbitsWrapJAX, tdi_config: TDIConfigWrapJAX,
    wdm_window: jnp.ndarray,
    Nf: int, Nt: int, Nt_sub: int, N_sparse: int,
    dt: float, T_chunk: float,
    tukey_alpha: float = ALPHA_AUTO,
) -> Sequence[jnp.ndarray]:
    """Per-binary ``<d|h>``, ``<h|h>`` via chunked-heterodyne.

    Outer = chunks, inner = vmap over binaries. PSD + data slabs are
    sliced once per chunk (the analogue of the CUDA shared-memory
    load).
    """
    num_bin = params_batch.shape[0]
    k_sky, u_sky, v_sky = jax.vmap(
        lambda p: get_sky_vectors(p, source)
    )(params_batch)

    def chunk_body(carry, j):
        d_h_acc, h_h_acc = carry
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        keep_len = khi - klo

        # Per-chunk data + invC slabs (sliced once for all binaries below).
        data_slab = jax.lax.dynamic_slice(data_d, (0, 0, n_lo), (3, Nf, keep_len))
        invC_slab = jax.lax.dynamic_slice(invC,   (0, 0, n_lo), (3, Nf, keep_len))

        def _one_bin(p, k_s, u_s, v_s):
            w_chunk = _build_chunk_wdm(
                chunk_t_start, p, source, orbits, tdi_config,
                k_s, u_s, v_s, T_chunk, N_sparse, Nf, Nt_sub, dt,
                wdm_window, tukey_alpha,
            )
            w_keep = jax.lax.dynamic_slice(w_chunk, (0, 0, klo),
                                            (3, Nf, keep_len))
            dh = jnp.sum(data_slab * w_keep * invC_slab)
            hh = jnp.sum(w_keep * w_keep * invC_slab)
            return dh, hh
        dh_arr, hh_arr = jax.vmap(_one_bin)(
            params_batch, k_sky, u_sky, v_sky,
        )                                              # (num_bin,)
        return (d_h_acc + dh_arr, h_h_acc + hh_arr), None

    n_chunks = chunk_t_starts.shape[0]
    init = (jnp.zeros(num_bin), jnp.zeros(num_bin))
    (d_h, h_h), _ = jax.lax.scan(chunk_body, init, jnp.arange(n_chunks))
    return d_h, h_h


def gb_wdm_het_swap_ll_jax(
    params_add: jnp.ndarray, params_rem: jnp.ndarray,    # (num_bin, 9) each
    data_d: jnp.ndarray, invC: jnp.ndarray,
    chunk_t_starts: jnp.ndarray,
    chunk_keep_lo: jnp.ndarray, chunk_keep_hi: jnp.ndarray,
    chunk_n_global_lo: jnp.ndarray,
    source: JaxAmpPhaseSource, orbits: OrbitsWrapJAX, tdi_config: TDIConfigWrapJAX,
    wdm_window: jnp.ndarray,
    Nf: int, Nt: int, Nt_sub: int, N_sparse: int,
    dt: float, T_chunk: float,
    tukey_alpha: float = ALPHA_AUTO,
) -> Sequence[jnp.ndarray]:
    """5-way swap-ll accumulator (chunked-heterodyne)."""
    num_bin = params_add.shape[0]
    k_a, u_a, v_a = jax.vmap(lambda p: get_sky_vectors(p, source))(params_add)
    k_r, u_r, v_r = jax.vmap(lambda p: get_sky_vectors(p, source))(params_rem)

    def chunk_body(carry, j):
        d_h_a, d_h_r, aa, rr, ar = carry
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        keep_len = khi - klo
        data_slab = jax.lax.dynamic_slice(data_d, (0, 0, n_lo), (3, Nf, keep_len))
        invC_slab = jax.lax.dynamic_slice(invC,   (0, 0, n_lo), (3, Nf, keep_len))

        def _one(p_a, p_r, ka, ua, va, kr, ur, vr):
            w_a = _build_chunk_wdm(chunk_t_start, p_a, source, orbits, tdi_config,
                                    ka, ua, va, T_chunk, N_sparse, Nf, Nt_sub,
                                    dt, wdm_window, tukey_alpha)
            w_r = _build_chunk_wdm(chunk_t_start, p_r, source, orbits, tdi_config,
                                    kr, ur, vr, T_chunk, N_sparse, Nf, Nt_sub,
                                    dt, wdm_window, tukey_alpha)
            w_a = jax.lax.dynamic_slice(w_a, (0, 0, klo), (3, Nf, keep_len))
            w_r = jax.lax.dynamic_slice(w_r, (0, 0, klo), (3, Nf, keep_len))
            return (
                jnp.sum(data_slab * w_a * invC_slab),
                jnp.sum(data_slab * w_r * invC_slab),
                jnp.sum(w_a * w_a * invC_slab),
                jnp.sum(w_r * w_r * invC_slab),
                jnp.sum(w_a * w_r * invC_slab),
            )
        dha, dhr, aa_v, rr_v, ar_v = jax.vmap(_one)(
            params_add, params_rem, k_a, u_a, v_a, k_r, u_r, v_r,
        )
        return (d_h_a + dha, d_h_r + dhr,
                aa + aa_v, rr + rr_v, ar + ar_v), None

    n_chunks = chunk_t_starts.shape[0]
    init = (jnp.zeros(num_bin), jnp.zeros(num_bin),
            jnp.zeros(num_bin), jnp.zeros(num_bin), jnp.zeros(num_bin))
    (d_h_a, d_h_r, aa, rr, ar), _ = jax.lax.scan(
        chunk_body, init, jnp.arange(n_chunks),
    )
    return d_h_a, d_h_r, aa, rr, ar


__all__ = [
    "gb_wdm_het_fill_global_jax",
    "gb_wdm_het_get_ll_jax",
    "gb_wdm_het_swap_ll_jax",
]
