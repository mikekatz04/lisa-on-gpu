"""Deprecation shim. Moved to ``lisatools.jax.response.tdi_config``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.tdi_config has moved to lisatools.jax.response.tdi_config; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.response.tdi_config import *  # noqa: F401,F403,E402
from lisatools.jax.response.tdi_config import TDIConfigWrapJAX  # noqa: F401,E402
