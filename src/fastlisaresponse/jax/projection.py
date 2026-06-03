"""Deprecation shim. Moved to ``lisatools.jax.response.projection``."""
from __future__ import annotations

import warnings

warnings.warn(
    "fastlisaresponse.jax.projection has moved to lisatools.jax.response.projection; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.jax.response.projection import *  # noqa: F401,F403,E402
from lisatools.jax.response.projection import (  # noqa: F401,E402
    get_phase_ref,
    get_sky_vectors,
    get_tdi_Xf_single,
)
