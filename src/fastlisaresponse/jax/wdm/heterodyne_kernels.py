"""JAX analogues of ``gb_wdm_het_{fill_global,get_ll,swap_ll}``.

These mirror the C++ kernels in
``../../cutils/TDIonthefly.cu`` (per the chunked-heterodyne pipeline
laid out in ``CHUNKED_HET_DESIGN_NOTES.md``). The kernel-shaped code
becomes a ``jax.lax.scan`` over chunks (outer) and a ``jax.vmap`` over
binaries (inner) -- functionally equivalent to the CUDA outer-block /
inner-loop pattern, JIT-friendly, and ``jax.grad``-able.

Per the sprint-wide rule "JAX may diverge internally, must match C++
inner-product outputs": this file is not a literal translation of the
CUDA kernels -- it follows the C++ output contract, not its code
structure. Specifically:

* ``data_d`` is taken at active-band layout ``(nchannels, Nf_active,
  Nt_active)`` -- the same layout the C++ kernels now consume.
* ``invC`` is taken at ``(nchannels, Nf_active, Nt_active)`` for
  ``tdi_type == TDI_AET / TDI_AE`` (diagonal noise) and at
  ``(nchannels, nchannels, Nf_active, Nt_active)`` for
  ``tdi_type == TDI_XYZ`` (full cross-channel Hermitian Sigma^-1).
* The inner product is dispatched on ``tdi_type``:
  * AET / AE: ``sum_c d[c] * h[c] * invC[c]``
  * XYZ:     ``sum_{c1,c2} d[c1] * h[c2] * invC[c1, c2]``

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

# Mirror TDIonTheFly.hh:29-31.
TDI_XYZ = 1
TDI_AET = 2
TDI_AE  = 3


def _inner_product_diag(data_slab, w_keep, invC_slab):
    """AET / AE diagonal inner product: sum_c d[c] * w[c] * invC[c]."""
    return jnp.sum(data_slab * w_keep * invC_slab)


def _inner_product_xyz(data_slab, w_keep, invC_slab):
    """XYZ cross-channel inner product:
    sum_{c1,c2,m,n} d[c1,m,n] * w[c2,m,n] * invC[c1,c2,m,n].
    """
    return jnp.einsum('imn,jmn,ijmn->', data_slab, w_keep, invC_slab)


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
    """fill_global accumulates into the FULL ``(nch, Nf, Nt)`` template
    (NOT active-band). The active-band layout only applies to data /
    invC on the inner-product paths (``get_ll`` / ``swap_ll``)."""
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
    data_d: jnp.ndarray, invC: jnp.ndarray,         # active-band layout
    chunk_t_starts: jnp.ndarray,
    chunk_keep_lo: jnp.ndarray, chunk_keep_hi: jnp.ndarray,
    chunk_n_global_lo: jnp.ndarray,
    source: JaxAmpPhaseSource, orbits: OrbitsWrapJAX, tdi_config: TDIConfigWrapJAX,
    wdm_window: jnp.ndarray,
    Nf: int, Nt: int, Nt_sub: int, N_sparse: int,
    dt: float, T_chunk: float,
    ind_min_f: int, ind_min_t: int,
    Nf_active: int, Nt_active: int,
    tdi_type: int = TDI_XYZ,
    tukey_alpha: float = ALPHA_AUTO,
) -> Sequence[jnp.ndarray]:
    """Per-binary ``<d|h>``, ``<h|h>`` via chunked-heterodyne.

    Outer = chunks, inner = vmap over binaries. PSD + data slabs are
    sliced once per chunk (the analogue of the CUDA shared-memory load).

    Data / invC layout (matches the C++ contract -- ``AnalysisContainerArray``
    natural output sliced to ``[ind_min_t, ind_max_t] x [ind_min_f,
    ind_max_f]``):

    * ``data_d`` : ``(nchannels, Nf_active, Nt_active)`` real.
    * ``invC``   : ``(nchannels, nchannels, Nf_active, Nt_active)``
      when ``tdi_type == TDI_XYZ`` (full Hermitian Sigma^-1 across
      channels), else ``(nchannels, Nf_active, Nt_active)``
      diagonal.

    Pixels with (m, n) outside the active band are skipped automatically
    by the mask -- they correspond to template energy in spectral tails
    that the active band intentionally truncates (matches C++
    behaviour exactly).

    ``tdi_type`` is treated as static metadata; the dispatch happens at
    trace time, so a JIT'd function specialises to one branch per
    backend choice (just like the C++ ``#ifdef`` switch).
    """
    num_bin = params_batch.shape[0]
    nchannels = data_d.shape[0]
    k_sky, u_sky, v_sky = jax.vmap(
        lambda p: get_sky_vectors(p, source)
    )(params_batch)

    n_local = jnp.arange(Nt_sub)
    # Active-band frequency slice runs over [ind_min_f, ind_min_f +
    # Nf_active). Done by static jnp.s_[]; w_chunk is built at full Nf.
    f_start = int(ind_min_f)
    f_stop  = int(ind_min_f) + int(Nf_active)

    # Choose the inner-product reducer at trace time.
    if int(tdi_type) == TDI_XYZ:
        ip = _inner_product_xyz
    else:
        ip = _inner_product_diag

    def chunk_body(carry, j):
        d_h_acc, h_h_acc = carry
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        # Active-band offset: a local index k inside [klo, khi) maps to
        # the global pixel g = (n_lo - klo) + k, which in active-band
        # coords is a = g - ind_min_t = (n_lo - klo - ind_min_t) + k.
        # The data / invC slab is sliced at offset
        # (n_lo - klo - ind_min_t) and length Nt_sub; cells with a
        # outside [0, Nt_active) are gated by the active mask.
        offset = (n_lo - klo - jnp.int32(ind_min_t)).astype(jnp.int32)
        zero32 = jnp.int32(0)
        a_local = offset + n_local
        active_mask = ((a_local >= 0) & (a_local < jnp.int32(Nt_active)))
        keep_mask = (n_local >= klo) & (n_local < khi)
        mask = (active_mask & keep_mask).astype(d_h_acc.dtype)
        mask_3d = mask[None, None, :]                        # (1,1,Nt_sub)

        if int(tdi_type) == TDI_XYZ:
            invC_slab = jax.lax.dynamic_slice(
                invC, (zero32, zero32, zero32, offset),
                (nchannels, nchannels, Nf_active, Nt_sub),
            )
        else:
            invC_slab = jax.lax.dynamic_slice(
                invC, (zero32, zero32, offset),
                (nchannels, Nf_active, Nt_sub),
            )
        data_slab = jax.lax.dynamic_slice(
            data_d, (zero32, zero32, offset),
            (nchannels, Nf_active, Nt_sub),
        )

        def _one_bin(p, k_s, u_s, v_s):
            w_chunk = _build_chunk_wdm(
                chunk_t_start, p, source, orbits, tdi_config,
                k_s, u_s, v_s, T_chunk, N_sparse, Nf, Nt_sub, dt,
                wdm_window, tukey_alpha,
            )                                                # (nch, Nf, Nt_sub)
            # Active-band slice in frequency (static start/stop).
            w_active = jax.lax.dynamic_slice(
                w_chunk, (zero32, jnp.int32(f_start), zero32),
                (nchannels, Nf_active, Nt_sub),
            )
            w_keep = w_active * mask_3d
            dh = ip(data_slab, w_keep, invC_slab)
            hh = ip(w_keep,    w_keep, invC_slab)
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
    ind_min_f: int, ind_min_t: int,
    Nf_active: int, Nt_active: int,
    tdi_type: int = TDI_XYZ,
    tukey_alpha: float = ALPHA_AUTO,
) -> Sequence[jnp.ndarray]:
    """5-way swap-ll accumulator (chunked-heterodyne).

    Same data / invC layout, ``tdi_type`` dispatch and active-band
    handling as :func:`gb_wdm_het_get_ll_jax` -- see its docstring.
    """
    num_bin = params_add.shape[0]
    nchannels = data_d.shape[0]
    k_a, u_a, v_a = jax.vmap(lambda p: get_sky_vectors(p, source))(params_add)
    k_r, u_r, v_r = jax.vmap(lambda p: get_sky_vectors(p, source))(params_rem)

    n_local = jnp.arange(Nt_sub)
    f_start = int(ind_min_f)

    if int(tdi_type) == TDI_XYZ:
        ip = _inner_product_xyz
    else:
        ip = _inner_product_diag

    def chunk_body(carry, j):
        d_h_a, d_h_r, aa, rr, ar = carry
        chunk_t_start = chunk_t_starts[j]
        klo = chunk_keep_lo[j]
        khi = chunk_keep_hi[j]
        n_lo = chunk_n_global_lo[j]
        offset = (n_lo - klo - jnp.int32(ind_min_t)).astype(jnp.int32)
        zero32 = jnp.int32(0)
        a_local = offset + n_local
        active_mask = ((a_local >= 0) & (a_local < jnp.int32(Nt_active)))
        keep_mask = (n_local >= klo) & (n_local < khi)
        mask = (active_mask & keep_mask).astype(d_h_a.dtype)
        mask_3d = mask[None, None, :]

        if int(tdi_type) == TDI_XYZ:
            invC_slab = jax.lax.dynamic_slice(
                invC, (zero32, zero32, zero32, offset),
                (nchannels, nchannels, Nf_active, Nt_sub),
            )
        else:
            invC_slab = jax.lax.dynamic_slice(
                invC, (zero32, zero32, offset),
                (nchannels, Nf_active, Nt_sub),
            )
        data_slab = jax.lax.dynamic_slice(
            data_d, (zero32, zero32, offset),
            (nchannels, Nf_active, Nt_sub),
        )

        def _one(p_a, p_r, ka, ua, va, kr, ur, vr):
            w_a = _build_chunk_wdm(chunk_t_start, p_a, source, orbits, tdi_config,
                                    ka, ua, va, T_chunk, N_sparse, Nf, Nt_sub,
                                    dt, wdm_window, tukey_alpha)
            w_r = _build_chunk_wdm(chunk_t_start, p_r, source, orbits, tdi_config,
                                    kr, ur, vr, T_chunk, N_sparse, Nf, Nt_sub,
                                    dt, wdm_window, tukey_alpha)
            w_a_active = jax.lax.dynamic_slice(
                w_a, (zero32, jnp.int32(f_start), zero32),
                (nchannels, Nf_active, Nt_sub),
            )
            w_r_active = jax.lax.dynamic_slice(
                w_r, (zero32, jnp.int32(f_start), zero32),
                (nchannels, Nf_active, Nt_sub),
            )
            w_a_keep = w_a_active * mask_3d
            w_r_keep = w_r_active * mask_3d
            return (
                ip(data_slab, w_a_keep, invC_slab),
                ip(data_slab, w_r_keep, invC_slab),
                ip(w_a_keep,  w_a_keep, invC_slab),
                ip(w_r_keep,  w_r_keep, invC_slab),
                ip(w_a_keep,  w_r_keep, invC_slab),
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
    ind_min_f: int, ind_min_t: int,
    Nf_active: int, Nt_active: int,
    tdi_type: int = TDI_XYZ,
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
            dt=dt, T_chunk=T_chunk,
            ind_min_f=ind_min_f, ind_min_t=ind_min_t,
            Nf_active=Nf_active, Nt_active=Nt_active,
            tdi_type=tdi_type, tukey_alpha=tukey_alpha,
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
    ind_min_f: int, ind_min_t: int,
    Nf_active: int, Nt_active: int,
    tdi_type: int = TDI_XYZ,
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
                          dt=dt, T_chunk=T_chunk,
                          ind_min_f=ind_min_f, ind_min_t=ind_min_t,
                          Nf_active=Nf_active, Nt_active=Nt_active,
                          tdi_type=tdi_type, tukey_alpha=tukey_alpha)

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


def gb_wdm_het_hessian_jax(
    params_batch: jnp.ndarray,                      # (num_bin, nparams)
    data_d: jnp.ndarray, invC: jnp.ndarray,
    chunk_t_starts: jnp.ndarray,
    chunk_keep_lo: jnp.ndarray, chunk_keep_hi: jnp.ndarray,
    chunk_n_global_lo: jnp.ndarray,
    source: JaxAmpPhaseSource, orbits: OrbitsWrapJAX, tdi_config: TDIConfigWrapJAX,
    wdm_window: jnp.ndarray,
    Nf: int, Nt: int, Nt_sub: int, N_sparse: int,
    dt: float, T_chunk: float,
    ind_min_f: int, ind_min_t: int,
    Nf_active: int, Nt_active: int,
    tdi_type: int = TDI_XYZ,
    tukey_alpha: float = ALPHA_AUTO,
) -> jnp.ndarray:
    """JAX-autograd per-binary Hessian of L_i = <d|h_i> - 0.5 <h_i|h_i>
    w.r.t. ``params_batch[i, :]``.

    Returns ``(num_bin, nparams, nparams)`` -- the per-binary Hessian
    block. Cross-binary entries are zero by construction (L_i depends
    only on params_batch[i, :]), so we ``vmap(hessian(per_binary_loss))``
    rather than ``hessian(sum_loss)`` to avoid materialising the
    block-diagonal full Hessian.
    """
    def per_binary_loss(p_single):
        d_h, h_h = gb_wdm_het_get_ll_jax(
            p_single[None, :], data_d, invC,
            chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_lo,
            source, orbits, tdi_config, wdm_window,
            Nf=Nf, Nt=Nt, Nt_sub=Nt_sub, N_sparse=N_sparse,
            dt=dt, T_chunk=T_chunk,
            ind_min_f=ind_min_f, ind_min_t=ind_min_t,
            Nf_active=Nf_active, Nt_active=Nt_active,
            tdi_type=tdi_type, tukey_alpha=tukey_alpha,
        )
        return (d_h[0] - 0.5 * h_h[0])
    return jax.vmap(jax.hessian(per_binary_loss))(params_batch)


__all__ = [
    "gb_wdm_het_fill_global_jax",
    "gb_wdm_het_get_ll_jax",
    "gb_wdm_het_swap_ll_jax",
    "gb_wdm_het_get_ll_grad_jax",
    "gb_wdm_het_swap_ll_grad_jax",
    "gb_wdm_het_hessian_jax",
]
