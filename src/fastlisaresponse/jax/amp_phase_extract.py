"""Deprecation shim. Moved to ``lisatools.jax.response.amp_phase_extract``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.amp_phase_extract has moved to "
    "lisatools.jax.response.amp_phase_extract; import from there instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.response.amp_phase_extract import *  # noqa: F401,F403,E402
from lisatools.jax.response.amp_phase_extract import (  # noqa: F401,E402
    extract_amplitude_and_phase,
    extract_and_unwrap,
    unwrap_phase,
)
