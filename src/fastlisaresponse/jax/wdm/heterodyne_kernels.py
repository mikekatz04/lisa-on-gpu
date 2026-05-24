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

    keep_len = keep_hi - keep_lo VARIES per chunk (the first / last chunks
    are partial because of n_pad), so the per-chunk write region has
    dynamic SHAPE -- jax.lax.dynamic_slice / dynamic_update_slice require
    static shapes. We resolve this by always operating on a length-Nt_sub
    block (static) and masking out the pad cells (mask = 0 outside [klo,
    khi)). The dynamic offset into the global template is
    ``n_lo - klo``, which places the kept cells at global indices
    ``[n_lo, n_lo + keep_len)``; the pad cells either fall off the end
    or land on previous-chunk keep regions where the zero mask makes
    them no-ops in ``existing + masked``.
    """
    num_bin = params_batch.shape[0]

    # Pre-compute sky vectors per binary (functions of lam, beta only).
    k_sky, u_sky, v_sky = jax.vmap(
        lambda p: get_sky_vectors(p, source)
    )(params_batch)
    # k_sky, u_sky, v_sky: (num_bin, 3)

    template = jnp.zeros((3, Nf, Nt))
    n_local = jnp.arange(Nt_sub)

    # Outer loop: chunks. Inner: vmap over binaries.
    def chunk_body(template_state, j):
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        mask = ((n_local >= klo) & (n_local < khi)).astype(template_state.dtype)
        # mask shape (Nt_sub,) -> broadcast to (1, 1, Nt_sub) for w_chunk
        mask_3d = mask[None, None, :]

        def _one_bin(p, f, k_s, u_s, v_s):
            w_chunk = _build_chunk_wdm(
                chunk_t_start, p, source, orbits, tdi_config,
                k_s, u_s, v_s, T_chunk, N_sparse, Nf, Nt_sub, dt,
                wdm_window, tukey_alpha,
            )
            return f * w_chunk * mask_3d                  # (3, Nf, Nt_sub)
        per_bin = jax.vmap(_one_bin)(
            params_batch, factors, k_sky, u_sky, v_sky,
        )                                                  # (num_bin, 3, Nf, Nt_sub)
        per_chunk_sum = jnp.sum(per_bin, axis=0)           # (3, Nf, Nt_sub)
        offset = (n_lo - klo).astype(jnp.int32)
        zero32 = jnp.int32(0)
        existing = jax.lax.dynamic_slice(
            template_state, (zero32, zero32, offset), (3, Nf, Nt_sub),
        )
        new_template = jax.lax.dynamic_update_slice(
            template_state, existing + per_chunk_sum, (zero32, zero32, offset),
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

    n_local = jnp.arange(Nt_sub)

    def chunk_body(carry, j):
        d_h_acc, h_h_acc = carry
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        mask = ((n_local >= klo) & (n_local < khi)).astype(d_h_acc.dtype)
        mask_3d = mask[None, None, :]

        # Per-chunk data + invC slabs at the FULL Nt_sub width, indexed
        # at offset (n_lo - klo) so that local k=klo lines up with global
        # k=n_lo. Cells outside [klo, khi) are gated by the mask.
        offset = (n_lo - klo).astype(jnp.int32)
        zero32 = jnp.int32(0)
        data_slab = jax.lax.dynamic_slice(data_d, (zero32, zero32, offset),
                                            (3, Nf, Nt_sub))
        invC_slab = jax.lax.dynamic_slice(invC,   (zero32, zero32, offset),
                                            (3, Nf, Nt_sub))

        def _one_bin(p, k_s, u_s, v_s):
            w_chunk = _build_chunk_wdm(
                chunk_t_start, p, source, orbits, tdi_config,
                k_s, u_s, v_s, T_chunk, N_sparse, Nf, Nt_sub, dt,
                wdm_window, tukey_alpha,
            )
            w_keep = w_chunk * mask_3d
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

    n_local = jnp.arange(Nt_sub)

    def chunk_body(carry, j):
        d_h_a, d_h_r, aa, rr, ar = carry
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        mask = ((n_local >= klo) & (n_local < khi)).astype(d_h_a.dtype)
        mask_3d = mask[None, None, :]
        offset = (n_lo - klo).astype(jnp.int32)
        zero32 = jnp.int32(0)
        data_slab = jax.lax.dynamic_slice(data_d, (zero32, zero32, offset),
                                            (3, Nf, Nt_sub))
        invC_slab = jax.lax.dynamic_slice(invC,   (zero32, zero32, offset),
                                            (3, Nf, Nt_sub))

        def _one(p_a, p_r, ka, ua, va, kr, ur, vr):
            w_a = _build_chunk_wdm(chunk_t_start, p_a, source, orbits, tdi_config,
                                    ka, ua, va, T_chunk, N_sparse, Nf, Nt_sub,
                                    dt, wdm_window, tukey_alpha)
            w_r = _build_chunk_wdm(chunk_t_start, p_r, source, orbits, tdi_config,
                                    kr, ur, vr, T_chunk, N_sparse, Nf, Nt_sub,
                                    dt, wdm_window, tukey_alpha)
            w_a = w_a * mask_3d
            w_r = w_r * mask_3d
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


def gb_wdm_het_get_ll_grad_jax(
    params_batch: jnp.ndarray,                      # (num_bin, nparams)
    data_d: jnp.ndarray, invC: jnp.ndarray,
    chunk_t_starts: jnp.ndarray,
    chunk_keep_lo: jnp.ndarray, chunk_keep_hi: jnp.ndarray,
    chunk_n_global_lo: jnp.ndarray,
    source: JaxAmpPhaseSource, orbits: OrbitsWrapJAX, tdi_config: TDIConfigWrapJAX,
    wdm_window: jnp.ndarray,
    Nf: int, Nt: int, Nt_sub: int, N_sparse: int,
    dt: float, T_chunk: float,
    tukey_alpha: float = ALPHA_AUTO,
) -> jnp.ndarray:
    """JAX-autograd gradient of L = sum_i (<d|h_i> - 0.5 <h_i|h_i>)
    w.r.t. ``params_batch``.

    Returns:
        grad: ``(num_bin, nparams)`` -- same shape as ``params_batch``.

    Per-binary L_i depends only on params_batch[i, :], so the Jacobian
    of sum_i L_i w.r.t. params_batch is block-diagonal and ``jax.grad``
    of the scalar sum recovers the per-binary gradients in one pass.
    """
    def scalar_loss(p):
        d_h, h_h = gb_wdm_het_get_ll_jax(
            p, data_d, invC,
            chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_lo,
            source, orbits, tdi_config, wdm_window,
            Nf=Nf, Nt=Nt, Nt_sub=Nt_sub, N_sparse=N_sparse,
            dt=dt, T_chunk=T_chunk, tukey_alpha=tukey_alpha,
        )
        return jnp.sum(d_h - 0.5 * h_h)
    return jax.grad(scalar_loss)(params_batch)


def gb_wdm_het_swap_ll_grad_jax(
    params_add: jnp.ndarray, params_rem: jnp.ndarray,    # (num_bin, nparams) each
    data_d: jnp.ndarray, invC: jnp.ndarray,
    chunk_t_starts: jnp.ndarray,
    chunk_keep_lo: jnp.ndarray, chunk_keep_hi: jnp.ndarray,
    chunk_n_global_lo: jnp.ndarray,
    source: JaxAmpPhaseSource, orbits: OrbitsWrapJAX, tdi_config: TDIConfigWrapJAX,
    wdm_window: jnp.ndarray,
    Nf: int, Nt: int, Nt_sub: int, N_sparse: int,
    dt: float, T_chunk: float,
    tukey_alpha: float = ALPHA_AUTO,
):
    """JAX-autograd gradients of all 5 swap_ll terms w.r.t. theta_add,
    plus the combined "add-on-clean-data" likelihood gradient

        L_add = <d - h_rem | h_add> - 0.5 <h_add|h_add>
              = dh_add - ar - 0.5 * aa,

    matching the Python FD wrapper :meth:`GBWDMHeterodyne.swap_ll_grad`.

    Returns a dict with the same keys as the Python FD method:
        grad_dh_add, grad_dh_rem, grad_aa, grad_rr, grad_ar, grad_L_add.
    """
    common_args = (data_d, invC, chunk_t_starts, chunk_keep_lo, chunk_keep_hi,
                    chunk_n_global_lo, source, orbits, tdi_config, wdm_window)
    common_kwargs = dict(Nf=Nf, Nt=Nt, Nt_sub=Nt_sub, N_sparse=N_sparse,
                          dt=dt, T_chunk=T_chunk, tukey_alpha=tukey_alpha)

    def term_sum(p_add, term_index):
        """Return sum_i (term_index'th output of swap_ll_jax) -- scalar."""
        outs = gb_wdm_het_swap_ll_jax(p_add, params_rem, *common_args, **common_kwargs)
        return jnp.sum(outs[term_index])

    grad_dh_add = jax.grad(lambda p: term_sum(p, 0))(params_add)
    grad_dh_rem = jax.grad(lambda p: term_sum(p, 1))(params_add)
    grad_aa     = jax.grad(lambda p: term_sum(p, 2))(params_add)
    grad_rr     = jax.grad(lambda p: term_sum(p, 3))(params_add)
    grad_ar     = jax.grad(lambda p: term_sum(p, 4))(params_add)
    grad_L_add  = grad_dh_add - grad_ar - 0.5 * grad_aa
    return dict(
        grad_dh_add=grad_dh_add, grad_dh_rem=grad_dh_rem,
        grad_aa=grad_aa, grad_rr=grad_rr, grad_ar=grad_ar,
        grad_L_add=grad_L_add,
    )


__all__ = [
    "gb_wdm_het_fill_global_jax",
    "gb_wdm_het_get_ll_jax",
    "gb_wdm_het_swap_ll_jax",
    "gb_wdm_het_get_ll_grad_jax",
    "gb_wdm_het_swap_ll_grad_jax",
]
