"""Deprecation shim. Moved to ``gbgpu.jax.sources.ucb``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.sources.ucb has moved to gbgpu.jax.sources.ucb; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.jax.sources.ucb import *  # noqa: F401,F403,E402
from gbgpu.jax.sources.ucb import JaxUCBSource  # noqa: F401,E402
