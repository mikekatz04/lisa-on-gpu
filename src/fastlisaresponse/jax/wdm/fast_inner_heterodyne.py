"""Deprecation shim. Moved to ``gbgpu.jax.wdm.fast_inner_heterodyne``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.wdm.fast_inner_heterodyne has moved to "
    "gbgpu.jax.wdm.fast_inner_heterodyne; import from there instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.jax.wdm.fast_inner_heterodyne import *  # noqa: F401,F403,E402
from gbgpu.jax.wdm.fast_inner_heterodyne import (  # noqa: F401,E402
    ALPHA_AUTO,
    fast_wdm_inner_heterodyne_jax,
    gb_chunk_fd_to_wdm_jax,
)
