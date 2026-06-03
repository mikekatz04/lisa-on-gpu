"""Deprecation shim. Moved to ``lisatools.jax.wdm.wdm_domain``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.wdm.wdm_domain has moved to "
    "lisatools.jax.wdm.wdm_domain; import from there instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.wdm.wdm_domain import *  # noqa: F401,F403,E402
from lisatools.jax.wdm.wdm_domain import WDMDomainWrapJAX  # noqa: F401,E402
