"""Deprecation shim. Moved to ``bbhx.jax.sources.sobbh``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.sources.sobbh has moved to bbhx.jax.sources.sobbh; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from bbhx.jax.sources.sobbh import *  # noqa: F401,F403,E402
from bbhx.jax.sources.sobbh import JaxSOBBHSource  # noqa: F401,E402
