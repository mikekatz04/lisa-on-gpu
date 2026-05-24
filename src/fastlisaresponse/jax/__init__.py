"""JAX backend for fastlisaresponse TDI-on-the-fly.

This subpackage mirrors what ``fastlisaresponse.cutils`` exposes for the
CPU / CUDA C++ backends, but in pure ``jax.numpy``. It is meant to live
alongside the C++ backends -- the C++ path remains the reference; the JAX
path enables ``jax.grad`` / ``jax.jit`` over the same physics.

Public entry points:

* :class:`FastLISAResponseJaxBackend` -- backend object plugged into
  :mod:`gpubackendtools` via the top-level ``fastlisaresponse.__init__``.
* :class:`JaxAmpPhaseSource` -- base for source classes that provide
  ``amplitude(t, params)`` and ``phase(t, params)``.
* :class:`JaxUCBSource` -- 1:1 match for the C++ ``GBTDIonTheFly`` UCB
  amplitude/phase.
* :class:`JaxSOBBHSource` -- 1:1 match for the C++ ``SOBBHTDIonTheFly``
  PN amplitude/phase. The PN expansions are imported from
  ``sobbhtaylert3.py`` at the repo root (already JAX).
* :func:`gb_run_wave_tdi` / :func:`sobbh_run_wave_tdi` -- functional kernels
  returning ``(tdi_amp, tdi_phase, phase_ref)`` for downstream caller adapters.

The submodule import is gated on ``import jax`` so that users without JAX
can still use the C++ backends without paying any import cost.
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    _HAS_JAX = False

if _HAS_JAX:
    from .backend import FastLISAResponseJaxBackend
    from .base import JaxAmpPhaseSource
    from .sources.ucb import JaxUCBSource
    from .sources.sobbh import JaxSOBBHSource
    from .wrappers import (
        GBTDIonTheFlyWrapJAX,
        SOBBHTDIonTheFlyWrapJAX,
        OrbitsWrapJAX,
        TDIConfigWrapJAX,
    )
    from .tdi_on_the_fly import gb_run_wave_tdi, sobbh_run_wave_tdi

    __all__ = [
        "FastLISAResponseJaxBackend",
        "JaxAmpPhaseSource",
        "JaxUCBSource",
        "JaxSOBBHSource",
        "GBTDIonTheFlyWrapJAX",
        "SOBBHTDIonTheFlyWrapJAX",
        "OrbitsWrapJAX",
        "TDIConfigWrapJAX",
        "gb_run_wave_tdi",
        "sobbh_run_wave_tdi",
    ]
else:
    __all__ = []
