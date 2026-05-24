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
