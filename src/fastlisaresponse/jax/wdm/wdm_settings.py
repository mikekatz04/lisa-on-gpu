"""Deprecation shim. Moved to ``lisatools.jax.wdm.wdm_settings``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.wdm.wdm_settings has moved to "
    "lisatools.jax.wdm.wdm_settings; import from there instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.wdm.wdm_settings import *  # noqa: F401,F403,E402
from lisatools.jax.wdm.wdm_settings import WDMSettingsWrapJAX  # noqa: F401,E402
