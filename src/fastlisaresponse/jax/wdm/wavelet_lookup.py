"""Deprecation shim. Moved to ``lisatools.jax.wdm.wavelet_lookup``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.wdm.wavelet_lookup has moved to "
    "lisatools.jax.wdm.wavelet_lookup; import from there instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.wdm.wavelet_lookup import *  # noqa: F401,F403,E402
from lisatools.jax.wdm.wavelet_lookup import WaveletLookupTableWrapJAX  # noqa: F401,E402
