"""Deprecation shim. Moved to ``lisatools.jax.wdm.fast_inner``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.wdm.fast_inner has moved to "
    "lisatools.jax.wdm.fast_inner; import from there instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.wdm.fast_inner import *  # noqa: F401,F403,E402
from lisatools.jax.wdm.fast_inner import fast_wdm_inner_jax  # noqa: F401,E402
