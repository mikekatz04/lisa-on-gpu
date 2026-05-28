"""``FastLISAResponseJaxBackend`` -- the JAX entry into the backend registry.

Slots into ``fastlisaresponse.cutils.FastLISAResponseBackendMethods`` so
that orchestration code (``GBTDIonTheFly``, ``GBWDMComputations``, ...)
can fan out to JAX via ``force_backend='jax'`` without any per-backend
branching above the wrapper layer.

The C++ backends each populate ``BackendMethods.xp`` with ``numpy`` or
``cupy`` and load native wrapper classes from a per-CUDA-version plugin
module (``fastlisaresponse_backend_cpu``, ``fastlisaresponse_backend_cuda12x``,
...). The JAX backend instead exposes ``jax.numpy`` for ``xp`` and
points the wrapper slots at the pure-Python classes defined in
``fastlisaresponse.jax.wrappers``.

The "native module installed" check that ``CpuBackend.__init__`` does
doesn't apply here, so we subclass ``Backend`` directly rather than
``CpuBackend`` and skip the importlib check (substituted by a single
``import jax`` at backend construction).
"""
from __future__ import annotations

from typing import Optional

from gpubackendtools.exceptions import BackendUnavailableException
# ``MissingDependency`` is the actual class name in
# ``gpubackendtools.exceptions``; ``gpubackendtools.gpubackendtools``
# exposes a same-named ``MissingDependencies`` alias for legacy reasons,
# but the public path is the singular one.
from gpubackendtools.exceptions import MissingDependency as MissingDependencies
from gpubackendtools.gpubackendtools import Backend

from ..cutils import FastLISAResponseBackend, FastLISAResponseBackendMethods


def _check_jax() -> "module":  # type: ignore[name-defined]
    """Return the ``jax.numpy`` module or raise a clean error."""
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as e:
        raise MissingDependencies(
            "'jax' backend requires jax (with jaxlib)",
            pip_deps=["jax", "jaxlib"],
            conda_deps=["jax"],
        ) from e
    # Required: float64 must be enabled so amp/phase parity with the C++
    # double-precision kernel is reachable. We turn it on at module load
    # time, the same way ``sobbhtaylert3.py`` does.
    from jax import config
    config.update("jax_enable_x64", True)
    return jnp


def _jax_methods_loader() -> FastLISAResponseBackendMethods:
    """Build the ``FastLISAResponseBackendMethods`` for the JAX backend.

    Orbit + cubic-spline wrappers come from the upstream JAX backends
    in ``lisatools.jax`` and ``gpubackendtools.jax`` -- single source
    of truth, so a fix to the JAX orbit interpolation lands once and
    every downstream package picks it up.

    The slots that the JAX path doesn't (yet) implement are populated
    with classes that raise ``NotImplementedError`` on first use.
    """
    jnp = _check_jax()

    from .tdi_config import TDIConfigWrapJAX
    from .wrappers import GBTDIonTheFlyWrapJAX, SOBBHTDIonTheFlyWrapJAX
    from .wdm import (
        GBComputationGroupWrapJAX,
        SOBBHComputationGroupWrapJAX,
        WaveletLookupTableWrapJAX,
        WDMSettingsWrapJAX,
        WDMDomainWrapJAX,
    )
    from lisatools.jax import OrbitsWrapJAX
    from gpubackendtools.jax import CubicSplineWrapJAX

    def _unimplemented_slot(name):
        class _NotImpl:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError(
                    f"{name} is not implemented in the JAX backend yet. "
                    "Use the C++ backend (force_backend='cpu' or 'cuda12x' / "
                    "'cuda13x') for this path."
                )
        _NotImpl.__name__ = name
        return _NotImpl

    # ``TDITypeDict`` is normally pulled from the compiled C++ module
    # (it carries the int enum tags TDI_XYZ, TDI_AET, TDI_AE used by
    # the WDM/FD code paths). For the JAX backend we use the same int
    # values as ``TDIonTheFly.hh:29-31``.
    TDI_TYPE_DICT = {"XYZ": 1, "AET": 2, "AE": 3}

    return FastLISAResponseBackendMethods(
        TDSplineTDIWaveformWrap=_unimplemented_slot("TDSplineTDIWaveformWrap"),
        FDSplineTDIWaveformWrap=_unimplemented_slot("FDSplineTDIWaveformWrap"),
        GBTDIonTheFlyWrap=GBTDIonTheFlyWrapJAX,
        SOBBHTDIonTheFlyWrap=SOBBHTDIonTheFlyWrapJAX,
        LISAResponseWrap=_unimplemented_slot("LISAResponseWrap"),
        LISAResponse=_unimplemented_slot("LISAResponse"),
        OrbitsWrap=OrbitsWrapJAX,                   # from lisatools.jax
        TDIConfig=_unimplemented_slot("TDIConfig"),
        TDIConfigWrap=TDIConfigWrapJAX,
        CubicSplineWrap=CubicSplineWrapJAX,         # from gpubackendtools.jax
        WDMSettingsWrap=WDMSettingsWrapJAX,
        WDMDomainWrap=WDMDomainWrapJAX,
        FDDomainWrap=_unimplemented_slot("FDDomainWrap"),
        WaveletLookupTableWrap=WaveletLookupTableWrapJAX,
        GBComputationGroupWrap=GBComputationGroupWrapJAX,
        SOBBHComputationGroupWrap=SOBBHComputationGroupWrapJAX,
        TDITypeDict=TDI_TYPE_DICT,
        xp=jnp,
    )


class FastLISAResponseJaxBackend(Backend, FastLISAResponseBackend):
    """JAX backend object plugged into the ``Globals().backends_manager``.

    Mirrors :class:`FastLISAResponseCpuBackend` /
    :class:`FastLISAResponseCuda12xBackend` but uses ``jax.numpy`` for
    ``xp`` and pure-Python wrapper classes (no compiled plugin module).
    """

    _backend_name: str = "fastlisaresponse_backend_jax"  # nominal name
    _name = "fastlisaresponse_jax"

    def __init__(self, *args, **kwargs):
        methods = _jax_methods_loader()
        Backend.__init__(
            self,
            name=self._name,
            methods=methods,
            features=Backend.Feature.NUMPY,  # closest match: pure-Python, host-side
        )
        FastLISAResponseBackend.__init__(self, methods)

    # Skip the importlib check that ``CpuBackend._check_module_installed``
    # would do -- there is no native plugin module; the JAX dependency
    # is verified by ``_check_jax()`` instead.
    @staticmethod
    def _check_module_installed(*args, **kwargs):
        return None
