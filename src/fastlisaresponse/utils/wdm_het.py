"""Deprecation shim. Moved to ``gbgpu.wdm_het``.

Phase 3L.7i of the 2026 sprint reorg (2026-06-04) moved the
chunked-heterodyne host-side helpers out of lisa-on-gpu and into GBGPU
alongside ``gbcomps``. This module re-exports the helper symbols from
their new home and emits a ``DeprecationWarning`` so downstream code
can migrate at its own pace.

Migration:

* ``from fastlisaresponse.utils.wdm_het import compute_chunk_geometry`` ->
  ``from gbgpu.wdm_het import compute_chunk_geometry``

The shim will be removed when ``fastlisaresponse`` is fully retired
(see the lisa-on-gpu CLAUDE.md deprecation table).
"""

import warnings

warnings.warn(
    "fastlisaresponse.utils.wdm_het has moved to gbgpu.wdm_het as of "
    "Phase 3L.7i (2026-06-04). Update your imports to "
    "`from gbgpu.wdm_het import ...`. This shim will be removed when "
    "fastlisaresponse is fully retired.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.wdm_het import (  # noqa: E402, F401
    RECOMMENDED_TUKEY_ALPHA_HET_NARROW,
    RECOMMENDED_TUKEY_ALPHA_HET_WIDE,
    RECOMMENDED_TUKEY_ALPHA_TD,
    USE_RECOMMENDED_TUKEY,
    compute_chunk_geometry,
    compute_layer_groups,
    compute_swap_layer_groups,
    compute_wdm_window,
    recommended_tukey_alpha,
    resolve_tukey_alpha,
)
