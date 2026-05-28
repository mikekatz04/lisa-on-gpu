"""JAX port of the WDM-domain kernels in ``TDIonTheFly.cu``.

Mirrors the C++ ``GBComputationGroup`` methods used by
:class:`fastlisaresponse.gbcomps.GBWDMComputations`:

* ``gb_wdm_get_ll``    -- per-binary ``(d|h), (h|h)``.
* ``gb_wdm_fill_global`` -- accumulate per-binary WDM template into a
  global template buffer.
* ``gb_wdm_swap_ll``   -- RJMCMC swap proposal: 5-way accumulators
  ``(d|h_add), (d|h_rem), (h_add|h_add), (h_rem|h_rem), (h_add|h_rem)``.

All routines are pure JAX (``jit`` / ``grad`` / ``vmap`` compatible).
The data containers ``WaveletLookupTableWrapJAX`` and
``WDMDomainWrapJAX`` mirror the C++ wrapper APIs so they slot
unchanged into the existing
:class:`fastlisaresponse.gbcomps.GBWDMComputations` plumbing.

Scope (current cut):

* TDI type: ``XYZ`` (cross-channel noise), ``AET`` and ``AE``
  (diagonal noise).
* Lookup table kind: both ``PER_N`` and ``N_REF_ONLY``.
* Single-m per channel (no per-pixel m-window expansion, i.e.
  ``num_diff=0`` in the C++ kernel template). This is the typical
  GB-search case.

Out-of-scope (raises ``NotImplementedError``):

* The spline-path WDM variants (``gb_wdm_spline_*``); use the C++
  backend for those.
* Per-parameter gradient kernels (``gb_wdm_*_grad_*``); JAX users
  should call :func:`jax.grad` on the likelihood directly instead.
"""
from __future__ import annotations

from .wavelet_lookup import WaveletLookupTableWrapJAX
from .wdm_settings import WDMSettingsWrapJAX
from .wdm_domain import WDMDomainWrapJAX
from .computation_group import GBComputationGroupWrapJAX, SOBBHComputationGroupWrapJAX

__all__ = [
    "WaveletLookupTableWrapJAX",
    "WDMSettingsWrapJAX",
    "WDMDomainWrapJAX",
    "GBComputationGroupWrapJAX",
    "SOBBHComputationGroupWrapJAX",
]
