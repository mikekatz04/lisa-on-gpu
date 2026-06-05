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

# Phase 3L.7k (2026-06-04): the fastlisaresponse_<flavor> backend family
# has been retired. The LISA-response wraps it used to bundle now live
# directly on LAT's LISAToolsBackend; the GB- and SOBBH-specific wraps
# live on the gbgpu and bbhx backends respectively. Code that used to
# `fastlisaresponse.get_backend("cpu")` should now do one of:
#   * `lisatools.get_backend("cpu")` -- LAT response wraps + Orbits +
#     WDM/FD/Spline + TDITypeDict.
#   * `gbgpu.get_backend("cpu")` -- everything above PLUS GBTDIonTheFlyWrap +
#     GBComputationGroupWrap + GBGPUComputationWrap.
#   * `bbhx.get_backend("cpu")` -- everything LAT-side PLUS SOBBHTDIonTheFlyWrap +
#     SOBBHComputationGroupWrap + BBHxComputationWrap.
# This module deliberately no longer registers backends or provides
# `get_backend` shortcuts so import sites that still call them surface
# a clear `AttributeError` and migrate. See the deprecation notice
# above for the full migration table.


from .response import pyResponseTDI, ResponseWrapper


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
    "get_file_manager",
]
