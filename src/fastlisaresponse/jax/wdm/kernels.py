"""Deprecation shim. Moved to ``gbgpu.jax.wdm.kernels``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.wdm.kernels has moved to gbgpu.jax.wdm.kernels; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.jax.wdm.kernels import *  # noqa: F401,F403,E402
from gbgpu.jax.wdm.kernels import (  # noqa: F401,E402
    gb_wdm_get_ll_jax,
    gb_wdm_fill_global_jax,
    gb_wdm_swap_ll_jax,
)
