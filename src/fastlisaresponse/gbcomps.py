"""Deprecation shim. Moved to ``gbgpu.gbcomps``.

Phase 3L.7i of the 2026 sprint reorg (2026-06-04) moved the GB and
SOBBH chunked-heterodyne Python frontends out of lisa-on-gpu and into
GBGPU as part of the lisa-on-gpu full-retirement work. This module
re-exports the class symbols from their new home and emits a
``DeprecationWarning`` so downstream code can migrate at its own pace.

Migration:

* ``from fastlisaresponse.gbcomps import GBWDMComputations`` ->
  ``from gbgpu.gbcomps import GBWDMComputations``
* ``from fastlisaresponse.gbcomps import SOBBHWDMComputations`` ->
  ``from gbgpu.gbcomps import SOBBHWDMComputations``
* ``from fastlisaresponse.gbcomps import GBFDComputations`` ->
  ``from gbgpu.gbcomps import GBFDComputations``

The shim will be removed when ``fastlisaresponse`` is fully retired
(see the lisa-on-gpu CLAUDE.md deprecation table).
"""

import warnings

warnings.warn(
    "fastlisaresponse.gbcomps has moved to gbgpu.gbcomps as of "
    "Phase 3L.7i (2026-06-04). Update your imports: "
    "`from gbgpu.gbcomps import GBWDMComputations, SOBBHWDMComputations, "
    "GBFDComputations`. This shim will be removed when fastlisaresponse "
    "is fully retired.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.gbcomps import (  # noqa: E402, F401
    GBFDComputations,
    GBWDMComputations,
    SOBBHWDMComputations,
)
