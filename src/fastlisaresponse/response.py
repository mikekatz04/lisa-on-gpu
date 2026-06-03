"""Deprecation shim. Moved to ``lisatools.response.directresponse``.

Phase 3 of the 2026 sprint reorg moved the LISA-response Python
frontends into LISAanalysistools. This module re-exports everything
from its new home and emits a ``DeprecationWarning`` on first use.
Scheduled for removal one release cycle after Phase 3 ships.
"""

import warnings

warnings.warn(
    "fastlisaresponse.response has moved to lisatools.response.directresponse; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.response.directresponse import *  # noqa: F401,F403,E402
from lisatools.response.directresponse import (  # noqa: F401,E402
    ecliptic_to_icrs,
    pyResponseTDI,
    ResponseWrapper,
)
