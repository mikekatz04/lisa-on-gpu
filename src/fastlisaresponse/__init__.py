"""FastLISAResponse."""

# ruff: noqa: E402
try:
    from fastlisaresponse._version import (  # pylint: disable=E0401,E0611
        __version__,
        __version_tuple__,
    )

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False

# Phase 3H (2026-06-02): top-level deprecation notice. The `fastlisaresponse`
# package is being absorbed into the sprint reorg -- generic LISA-response
# code lives in `lisatools.response`, GB-specific code in `gbgpu`, and
# SOBBH-specific code in `bbhx`. Each individual moved symbol still works
# via per-module deprecation shims (and emits its own DeprecationWarning on
# first import), but this notice lets users see the broader picture at
# `import fastlisaresponse` time.
import warnings as _warnings
_warnings.warn(
    "`fastlisaresponse` is being deprecated in favor of `lisatools.response` "
    "(generic LISA response/TDI), `gbgpu` (GB-specific kernels) and `bbhx` "
    "(SOBBH-specific kernels). The native modules still work and existing "
    "imports keep functioning via shims, but every shim emits its own "
    "DeprecationWarning telling you where the new home is. Plan for a one-"
    "release-cycle deprecation window before the package is retired.",
    DeprecationWarning,
    stacklevel=2,
)
del _warnings

from . import cutils, utils

from gpubackendtools import Globals
from .cutils import FastLISAResponseCpuBackend, FastLISAResponseCuda11xBackend, FastLISAResponseCuda12xBackend, FastLISAResponseCuda13xBackend


# Backend registry. The cutils-side (C++/CUDA) backends are always
# registered. The JAX backend is added on top if jax is importable, so
# users without jax can keep using the C++ path without paying any cost.
add_backends = {
    "fastlisaresponse_cpu": FastLISAResponseCpuBackend,
    "fastlisaresponse_cuda11x": FastLISAResponseCuda11xBackend,
    "fastlisaresponse_cuda12x": FastLISAResponseCuda12xBackend,
    "fastlisaresponse_cuda13x": FastLISAResponseCuda13xBackend,
}

try:
    # Importing the .jax subpackage gates on `import jax`; the symbol
    # is None if jax is missing.
    from .jax import FastLISAResponseJaxBackend as _FastLISAResponseJaxBackend
    if _FastLISAResponseJaxBackend is not None:
        add_backends["fastlisaresponse_jax"] = _FastLISAResponseJaxBackend
except (ImportError, ModuleNotFoundError):
    pass

Globals().backends_manager.add_backends(add_backends)



from gpubackendtools import get_backend as _get_backend
from gpubackendtools import has_backend as _has_backend
from gpubackendtools import get_first_backend as _get_first_backend
from gpubackendtools.gpubackendtools import Backend


def get_backend(backend: str) -> Backend:
    __doc__ = _get_backend.__doc__
    if "fastlisaresponse_" not in backend:
        return _get_backend("fastlisaresponse_" + backend)
    else:
        return _get_backend(backend)


def has_backend(backend: str) -> Backend:
    __doc__ = _has_backend.__doc__
    if "fastlisaresponse_" not in backend:
        return _has_backend("fastlisaresponse_" + backend)
    else:
        return _has_backend(backend)

        
def get_first_backend(backend: str) -> Backend:
    __doc__ = _get_first_backend.__doc__
    if "fastlisaresponse_" not in backend:
        return _get_first_backend("fastlisaresponse_" + backend)
    else:
        return _get_first_backend(backend)


from .response import pyResponseTDI, ResponseWrapper


__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
    "pyResponseTDI",
    "ResponseWrapper",
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_file_manager",
    "has_backend",
]
