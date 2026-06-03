"""Deprecation shim. Moved to ``lisatools.jax.response.base``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.base has moved to lisatools.jax.response.base; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.response.base import *  # noqa: F401,F403,E402
from lisatools.jax.response.base import JaxAmpPhaseSource  # noqa: F401,E402
