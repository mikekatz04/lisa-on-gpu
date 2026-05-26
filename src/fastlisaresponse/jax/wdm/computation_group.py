"""``GBComputationGroupWrapJAX`` -- JAX-side mirror of the C++
``GBComputationGroupWrap`` class.

Exposes the same method names that
:class:`fastlisaresponse.gbcomps.GBWDMComputations` calls
(``gb_wdm_get_ll``, ``gb_wdm_fill_global``, ``gb_wdm_swap_ll``), with
signatures matching the C++ side. Because JAX arrays are immutable,
the in-place buffer mutation contract of the C++ kernels is
replaced by a hybrid:

* ``gb_wdm_get_ll`` / ``gb_wdm_swap_ll`` write into the pre-allocated
  ``d_h_out`` / ``h_h_out`` arrays *if those are numpy arrays* (cheap
  host-side copy from JAX), so existing call sites keep working
  unmodified.
* ``gb_wdm_fill_global`` does the same with the ``templates`` buffer.

This compromise keeps the orchestration code in
:mod:`fastlisaresponse.gbcomps` backend-agnostic; the JAX path
internally runs the pure-functional kernels and copies results back.
For autograd users who want to differentiate through, the underlying
:mod:`fastlisaresponse.jax.wdm.kernels` functions are the proper
entry point.

The source plug-in defaults to :class:`JaxUCBSource`; callers can
override via the constructor (the C++ side hard-codes the source
class per ``GB*TDIonTheFly``, so this matches).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import JaxAmpPhaseSource
from ..orbits import OrbitsWrapJAX
from ..tdi_config import TDIConfigWrapJAX
from .kernels import (
    DEFAULT_NUM_DIFF,
    gb_wdm_fill_global_jax,
    gb_wdm_get_ll_jax,
    gb_wdm_swap_ll_jax,
)
from .wavelet_lookup import WaveletLookupTableWrapJAX
from .wdm_domain import WDMDomainWrapJAX


def _copy_into(buf, value):
    """Copy a JAX result into a numpy-like buffer (preserves dtype)."""
    arr = np.asarray(value).astype(buf.dtype, copy=False).reshape(buf.shape)
    buf[...] = arr


class GBComputationGroupWrapJAX:
    """Pure-Python analog of the C++ ``GBComputationGroupWrap``.

    The C++ side instantiates this with no arguments and then calls
    methods on it. We mirror that. The JAX-specific source plugin
    (default :class:`fastlisaresponse.jax.sources.JaxUCBSource`) is
    held on the instance and built lazily on first use because it
    needs ``t_ref`` -- which comes in via the method signatures.
    """

    def __init__(self, source_cls: Optional[type] = None,
                 num_diff: int = DEFAULT_NUM_DIFF):
        """Args:
            source_cls: Subclass of :class:`JaxAmpPhaseSource` used to
                evaluate the per-time amplitude/phase. Default
                :class:`JaxUCBSource`.
            num_diff: Width of the per-(n, c) m-window. Each pixel
                scatters into ``2*num_diff + 1`` neighbouring layers
                around the anchor ``int(f / layer_df)``. Default
                matches the C++ instantiation ``<2, 5>`` -- i.e.
                ``num_diff = 2``, five layers per pixel.
        """
        self._source_cls = source_cls   # default UCB; lazily resolved
        self.num_diff = int(num_diff)

    def _resolve_source(self, t_ref: float) -> JaxAmpPhaseSource:
        cls = self._source_cls
        if cls is None:
            from ..sources.ucb import JaxUCBSource
            cls = JaxUCBSource
        return cls(t_ref=float(t_ref))

    # ----------------------------------------------------------------
    # gb_wdm_get_ll
    # ----------------------------------------------------------------
    def gb_wdm_get_ll(
        self, d_h_out, h_h_out,
        cpp_orbits, cpp_tdi_config, cpp_wdm_lookup, cpp_wdm,
        params_in, data_index, noise_index,
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t,
    ):
        source = self._resolve_source(t_ref)
        # Re-shape the flat params buffer to (num_bin, nparams).
        p2d = np.asarray(params_in).reshape(num_bin, nparams)
        d_h_arr, h_h_arr = gb_wdm_get_ll_jax(
            p2d, np.asarray(data_index), np.asarray(noise_index),
            source, cpp_orbits, cpp_tdi_config, cpp_wdm_lookup, cpp_wdm,
            float(T), float(t_ref), int(tdi_type),
            deriv_delta_t=float(deriv_delta_t),
            num_diff=self.num_diff,
        )
        _copy_into(d_h_out, d_h_arr)
        _copy_into(h_h_out, h_h_arr)

    # ----------------------------------------------------------------
    # gb_wdm_fill_global
    # ----------------------------------------------------------------
    def gb_wdm_fill_global(
        self, templates,
        cpp_orbits, cpp_tdi_config, cpp_wdm_lookup, cpp_wdm,
        params_in, data_index, factors,
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t,
    ):
        source = self._resolve_source(t_ref)
        p2d = np.asarray(params_in).reshape(num_bin, nparams)
        # Start from the existing templates value (read once into JAX
        # space), accumulate, copy back to the caller's buffer.
        tmpl_in = np.asarray(templates).copy()  # snapshot current state
        import jax.numpy as jnp
        tmpl_jax = jnp.asarray(tmpl_in)
        out = gb_wdm_fill_global_jax(
            tmpl_jax, p2d, np.asarray(data_index), np.asarray(factors),
            source, cpp_orbits, cpp_tdi_config, cpp_wdm_lookup, cpp_wdm,
            float(T), float(t_ref), int(tdi_type),
            deriv_delta_t=float(deriv_delta_t),
            num_diff=self.num_diff,
        )
        _copy_into(templates, out)

    # ----------------------------------------------------------------
    # gb_wdm_swap_ll
    # ----------------------------------------------------------------
    def gb_wdm_swap_ll(
        self,
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        cpp_orbits, cpp_tdi_config, cpp_wdm_lookup, cpp_wdm,
        params_add_in, params_remove_in, data_index, noise_index,
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t,
    ):
        source = self._resolve_source(t_ref)
        pa = np.asarray(params_add_in).reshape(num_bin, nparams)
        pr = np.asarray(params_remove_in).reshape(num_bin, nparams)
        d_h_a, d_h_r, aa, rr, ar = gb_wdm_swap_ll_jax(
            pa, pr, np.asarray(data_index), np.asarray(noise_index),
            source, cpp_orbits, cpp_tdi_config, cpp_wdm_lookup, cpp_wdm,
            float(T), float(t_ref), int(tdi_type),
            deriv_delta_t=float(deriv_delta_t),
            num_diff=self.num_diff,
        )
        _copy_into(d_h_add_out, d_h_a)
        _copy_into(d_h_remove_out, d_h_r)
        _copy_into(add_add_out, aa)
        _copy_into(remove_remove_out, rr)
        _copy_into(add_remove_out, ar)

    # ----------------------------------------------------------------
    # Spline / gradient variants -- not yet implemented in JAX.
    # ----------------------------------------------------------------
    def _unimplemented(self, name):
        raise NotImplementedError(
            f"{name} is not implemented in the JAX backend yet. "
            "Use force_backend='cpu' (or one of the CUDA backends) for "
            "this path. The JAX path covers the direct gb_wdm_get_ll / "
            "gb_wdm_fill_global / gb_wdm_swap_ll kernels only at this "
            "point."
        )

    def gb_wdm_spline_get_ll(self, *a, **k):
        self._unimplemented("gb_wdm_spline_get_ll")
    def gb_wdm_spline_fill_global(self, *a, **k):
        self._unimplemented("gb_wdm_spline_fill_global")
    def gb_wdm_spline_swap_ll(self, *a, **k):
        self._unimplemented("gb_wdm_spline_swap_ll")
    def gb_wdm_get_ll_grad(self, *a, **k):
        self._unimplemented(
            "gb_wdm_get_ll_grad (use jax.grad on gb_wdm_get_ll_jax instead)"
        )
    def gb_wdm_swap_ll_grad(self, *a, **k):
        self._unimplemented(
            "gb_wdm_swap_ll_grad (use jax.grad on gb_wdm_swap_ll_jax instead)"
        )
    def gb_wdm_spline_get_ll_grad(self, *a, **k):
        self._unimplemented("gb_wdm_spline_get_ll_grad")

    # FD-domain methods (gb_fd_*) similarly punt until the FD JAX
    # kernel lands.
    def gb_fd_get_ll(self, *a, **k):
        self._unimplemented("gb_fd_get_ll")
    def gb_fd_fill_global(self, *a, **k):
        self._unimplemented("gb_fd_fill_global")
    def gb_fd_swap_ll(self, *a, **k):
        self._unimplemented("gb_fd_swap_ll")
    def gb_fd_get_ll_grad(self, *a, **k):
        self._unimplemented("gb_fd_get_ll_grad")
    def gb_fd_swap_ll_grad(self, *a, **k):
        self._unimplemented("gb_fd_swap_ll_grad")

    # ----------------------------------------------------------------
    # Chunked-heterodyne JAX kernels (gb_wdm_het_*)
    #
    # Each method below accepts the C++-style argument list packed by
    # :class:`fastlisaresponse.gbcomps.GBWDMComputations`, then adapts
    # internally to the pure-functional standalone JAX kernel:
    #
    # * flat ``params`` -> ``(num_bin, nparams)``
    # * flat ``data`` / ``invC`` -> ``(nchannels, Nf, Nt)``
    # * ``cpp_orbits`` / ``cpp_tdi_config`` are already
    #   :class:`OrbitsWrapJAX` / :class:`TDIConfigWrapJAX` (built by
    #   :class:`FastLISAResponseJaxBackend`).
    # * the per-binary :class:`JaxAmpPhaseSource` is resolved lazily
    #   from ``t_ref``.
    # * scratch integers the JAX path doesn't need (``log2_Nt_sub``,
    #   ``log2_N_sparse``, ``n_rfft_chunk``, ``grid_dim``, ``n_chunks``,
    #   ``N_cp_sig``, ``N_cp_orbit``, layer-group arrays) are accepted
    #   for ABI parity and ignored.
    # ----------------------------------------------------------------

    @staticmethod
    def _reshape_params(params_in, num_bin, nparams):
        return np.asarray(params_in).reshape(int(num_bin), int(nparams))

    @staticmethod
    def _reshape_grid(arr, nchannels, Nf, Nt):
        a = np.asarray(arr)
        if a.ndim == 1:
            a = a.reshape(int(nchannels), int(Nf), int(Nt))
        else:
            assert a.shape == (int(nchannels), int(Nf), int(Nt)), (
                f"expected ({nchannels}, {Nf}, {Nt}), got {a.shape}")
        return a

    def gb_wdm_het_fill_global(
        self,
        templates,
        cpp_orbits, cpp_tdi_config,
        params_in, factors,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_lo,
        wdm_window,
        n_chunks, num_bin, nparams,
        Nf, Nt, Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tukey_alpha, grid_dim,
        N_cp_sig, N_cp_orbit,
    ):
        """C++-style chunked-het fill_global -> JAX standalone kernel.

        Mutates ``templates`` in place (must be numpy)."""
        from .heterodyne_kernels import gb_wdm_het_fill_global_jax
        source = self._resolve_source(t_ref)
        p2d = self._reshape_params(params_in, num_bin, nparams)
        import jax.numpy as jnp
        out = gb_wdm_het_fill_global_jax(
            params_batch=jnp.asarray(p2d),
            factors=jnp.asarray(np.asarray(factors)),
            chunk_t_starts=jnp.asarray(np.asarray(chunk_t_starts)),
            chunk_keep_lo=jnp.asarray(np.asarray(chunk_keep_lo)),
            chunk_keep_hi=jnp.asarray(np.asarray(chunk_keep_hi)),
            chunk_n_global_lo=jnp.asarray(np.asarray(chunk_n_global_lo)),
            source=source, orbits=cpp_orbits, tdi_config=cpp_tdi_config,
            wdm_window=jnp.asarray(np.asarray(wdm_window)),
            Nf=int(Nf), Nt=int(Nt), Nt_sub=int(Nt_sub),
            N_sparse=int(N_sparse),
            dt=float(dt), T_chunk=float(T_chunk),
            tukey_alpha=float(tukey_alpha),
        )
        # Accumulate into the caller's buffer (matches C++ contract).
        tmpl_np = np.asarray(templates)
        out_np  = np.asarray(out).reshape(tmpl_np.shape)
        tmpl_np += out_np

    def gb_wdm_het_get_ll(
        self,
        d_h_out, h_h_out,
        cpp_orbits, cpp_tdi_config,
        params_in, data_index, noise_index,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_lo,
        wdm_window,
        data_d, invC,
        n_chunks, num_bin, nparams,
        Nf, Nt, Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tukey_alpha, grid_dim,
        N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
    ):
        """C++-style chunked-het get_ll -> JAX standalone kernel."""
        from .heterodyne_kernels import gb_wdm_het_get_ll_jax
        source = self._resolve_source(t_ref)
        p2d = self._reshape_params(params_in, num_bin, nparams)
        d_grid = self._reshape_grid(data_d, nchannels, Nf, Nt)
        c_grid = self._reshape_grid(invC,   nchannels, Nf, Nt)
        import jax.numpy as jnp
        d_h, h_h = gb_wdm_het_get_ll_jax(
            params_batch=jnp.asarray(p2d),
            data_d=jnp.asarray(d_grid), invC=jnp.asarray(c_grid),
            chunk_t_starts=jnp.asarray(np.asarray(chunk_t_starts)),
            chunk_keep_lo=jnp.asarray(np.asarray(chunk_keep_lo)),
            chunk_keep_hi=jnp.asarray(np.asarray(chunk_keep_hi)),
            chunk_n_global_lo=jnp.asarray(np.asarray(chunk_n_global_lo)),
            source=source, orbits=cpp_orbits, tdi_config=cpp_tdi_config,
            wdm_window=jnp.asarray(np.asarray(wdm_window)),
            Nf=int(Nf), Nt=int(Nt), Nt_sub=int(Nt_sub),
            N_sparse=int(N_sparse),
            dt=float(dt), T_chunk=float(T_chunk),
            tukey_alpha=float(tukey_alpha),
        )
        _copy_into(d_h_out, d_h)
        _copy_into(h_h_out, h_h)

    def gb_wdm_het_swap_ll(
        self,
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        cpp_orbits, cpp_tdi_config,
        params_add_in, params_remove_in, data_index, noise_index,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_lo,
        wdm_window,
        data_d, invC,
        n_chunks, num_bin, nparams,
        Nf, Nt, Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tukey_alpha, grid_dim,
        N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b,
    ):
        """C++-style chunked-het swap_ll -> JAX standalone kernel."""
        from .heterodyne_kernels import gb_wdm_het_swap_ll_jax
        source = self._resolve_source(t_ref)
        pa = self._reshape_params(params_add_in,    num_bin, nparams)
        pr = self._reshape_params(params_remove_in, num_bin, nparams)
        d_grid = self._reshape_grid(data_d, nchannels, Nf, Nt)
        c_grid = self._reshape_grid(invC,   nchannels, Nf, Nt)
        import jax.numpy as jnp
        d_h_a, d_h_r, aa, rr, ar = gb_wdm_het_swap_ll_jax(
            params_add=jnp.asarray(pa), params_rem=jnp.asarray(pr),
            data_d=jnp.asarray(d_grid), invC=jnp.asarray(c_grid),
            chunk_t_starts=jnp.asarray(np.asarray(chunk_t_starts)),
            chunk_keep_lo=jnp.asarray(np.asarray(chunk_keep_lo)),
            chunk_keep_hi=jnp.asarray(np.asarray(chunk_keep_hi)),
            chunk_n_global_lo=jnp.asarray(np.asarray(chunk_n_global_lo)),
            source=source, orbits=cpp_orbits, tdi_config=cpp_tdi_config,
            wdm_window=jnp.asarray(np.asarray(wdm_window)),
            Nf=int(Nf), Nt=int(Nt), Nt_sub=int(Nt_sub),
            N_sparse=int(N_sparse),
            dt=float(dt), T_chunk=float(T_chunk),
            tukey_alpha=float(tukey_alpha),
        )
        _copy_into(d_h_add_out,       d_h_a)
        _copy_into(d_h_remove_out,    d_h_r)
        _copy_into(add_add_out,       aa)
        _copy_into(remove_remove_out, rr)
        _copy_into(add_remove_out,    ar)

    def gb_wdm_het_get_ll_grad(
        self,
        grad_out,
        cpp_orbits, cpp_tdi_config,
        params_in, data_index, noise_index,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_lo,
        wdm_window,
        data_d, invC,
        n_chunks, num_bin, nparams,
        Nf, Nt, Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tukey_alpha, grid_dim,
        N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
    ):
        """C++-style gradient of L = sum_i (<d|h_i> - 0.5 <h_i|h_i>)."""
        from .heterodyne_kernels import gb_wdm_het_get_ll_grad_jax
        source = self._resolve_source(t_ref)
        p2d = self._reshape_params(params_in, num_bin, nparams)
        d_grid = self._reshape_grid(data_d, nchannels, Nf, Nt)
        c_grid = self._reshape_grid(invC,   nchannels, Nf, Nt)
        import jax.numpy as jnp
        grad = gb_wdm_het_get_ll_grad_jax(
            params_batch=jnp.asarray(p2d),
            data_d=jnp.asarray(d_grid), invC=jnp.asarray(c_grid),
            chunk_t_starts=jnp.asarray(np.asarray(chunk_t_starts)),
            chunk_keep_lo=jnp.asarray(np.asarray(chunk_keep_lo)),
            chunk_keep_hi=jnp.asarray(np.asarray(chunk_keep_hi)),
            chunk_n_global_lo=jnp.asarray(np.asarray(chunk_n_global_lo)),
            source=source, orbits=cpp_orbits, tdi_config=cpp_tdi_config,
            wdm_window=jnp.asarray(np.asarray(wdm_window)),
            Nf=int(Nf), Nt=int(Nt), Nt_sub=int(Nt_sub),
            N_sparse=int(N_sparse),
            dt=float(dt), T_chunk=float(T_chunk),
            tukey_alpha=float(tukey_alpha),
        )
        _copy_into(grad_out, grad)

    def gb_wdm_het_swap_ll_grad(
        self,
        grad_add_out, grad_remove_out,
        cpp_orbits, cpp_tdi_config,
        params_add_in, params_remove_in, data_index, noise_index,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_lo,
        wdm_window,
        data_d, invC,
        n_chunks, num_bin, nparams,
        Nf, Nt, Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref,
        tukey_alpha, grid_dim,
        N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b,
    ):
        """C++-style swap_ll gradient -> JAX autograd kernel.

        The standalone JAX swap_ll_grad returns a dict with gradients
        of each of the five terms w.r.t. ``params_add``. We forward
        ``grad_L_add = dh_add - ar - 0.5 * aa`` into ``grad_add_out``,
        and re-evaluate with add/remove swapped to fill
        ``grad_remove_out`` symmetrically.
        """
        from .heterodyne_kernels import gb_wdm_het_swap_ll_grad_jax
        source = self._resolve_source(t_ref)
        pa = self._reshape_params(params_add_in,    num_bin, nparams)
        pr = self._reshape_params(params_remove_in, num_bin, nparams)
        d_grid = self._reshape_grid(data_d, nchannels, Nf, Nt)
        c_grid = self._reshape_grid(invC,   nchannels, Nf, Nt)
        import jax.numpy as jnp
        common = dict(
            data_d=jnp.asarray(d_grid), invC=jnp.asarray(c_grid),
            chunk_t_starts=jnp.asarray(np.asarray(chunk_t_starts)),
            chunk_keep_lo=jnp.asarray(np.asarray(chunk_keep_lo)),
            chunk_keep_hi=jnp.asarray(np.asarray(chunk_keep_hi)),
            chunk_n_global_lo=jnp.asarray(np.asarray(chunk_n_global_lo)),
            source=source, orbits=cpp_orbits, tdi_config=cpp_tdi_config,
            wdm_window=jnp.asarray(np.asarray(wdm_window)),
            Nf=int(Nf), Nt=int(Nt), Nt_sub=int(Nt_sub),
            N_sparse=int(N_sparse),
            dt=float(dt), T_chunk=float(T_chunk),
            tukey_alpha=float(tukey_alpha),
        )
        out_add = gb_wdm_het_swap_ll_grad_jax(
            params_add=jnp.asarray(pa), params_rem=jnp.asarray(pr),
            **common,
        )
        out_rem = gb_wdm_het_swap_ll_grad_jax(
            params_add=jnp.asarray(pr), params_rem=jnp.asarray(pa),
            **common,
        )
        _copy_into(grad_add_out,    out_add["grad_L_add"])
        _copy_into(grad_remove_out, out_rem["grad_L_add"])

    def gb_wdm_het_hessian(self, *args, **kwargs):
        """JAX-autograd per-binary Hessian -- standalone signature
        passthrough (no C++ analog yet). See
        :func:`gb_wdm_het_hessian_jax` for the arg list.
        """
        from .heterodyne_kernels import gb_wdm_het_hessian_jax
        return gb_wdm_het_hessian_jax(*args, **kwargs)


class SOBBHComputationGroupWrapJAX(GBComputationGroupWrapJAX):
    """Stellar-origin BBH analog of :class:`GBComputationGroupWrapJAX`.

    Identical chunked-heterodyne kernel infrastructure -- the only
    differences are:

    * The default JAX source is :class:`JaxSOBBHSource` (provides the
      SOBBH PN amplitude / phase that the C++ side gets from
      ``SOBBHTDIonTheFly``).
    * The kernel methods are exposed under the ``sobbh_wdm_het_*``
      names that match the C++ ``SOBBHComputationGroupWrap`` and that
      :class:`fastlisaresponse.gbcomps.SOBBHWDMComputations` dispatches
      to. The bodies are aliased to the corresponding ``gb_wdm_het_*``
      methods on the parent class -- the per-binary parameter count
      (11 instead of 9) and the ``f_low`` carrier index (5 instead of
      1) are handled on the caller side by
      :class:`SOBBHWDMComputations`.
    """

    def __init__(self, source_cls: Optional[type] = None,
                 num_diff: int = DEFAULT_NUM_DIFF):
        if source_cls is None:
            from ..sources.sobbh import JaxSOBBHSource
            source_cls = JaxSOBBHSource
        super().__init__(source_cls=source_cls, num_diff=num_diff)

    # Alias the chunked-het kernel methods under the SOBBH naming
    # convention so the C++-style dispatch in SOBBHWDMComputations
    # resolves cleanly via ``self._kernel("...")``.
    sobbh_wdm_het_fill_global  = GBComputationGroupWrapJAX.gb_wdm_het_fill_global
    sobbh_wdm_het_get_ll       = GBComputationGroupWrapJAX.gb_wdm_het_get_ll
    sobbh_wdm_het_swap_ll      = GBComputationGroupWrapJAX.gb_wdm_het_swap_ll
    sobbh_wdm_het_get_ll_grad  = GBComputationGroupWrapJAX.gb_wdm_het_get_ll_grad
    sobbh_wdm_het_swap_ll_grad = GBComputationGroupWrapJAX.gb_wdm_het_swap_ll_grad
    sobbh_wdm_het_hessian      = GBComputationGroupWrapJAX.gb_wdm_het_hessian
