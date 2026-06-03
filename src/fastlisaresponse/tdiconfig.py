"""Deprecation shim. Moved to ``lisatools.response.tdiconfig``.

Phase 3 of the 2026 sprint reorg moved the LISA-response Python
frontends into LISAanalysistools. This module re-exports ``TDIConfig``
from its new home and emits a ``DeprecationWarning`` on first use.
Scheduled for removal one release cycle after Phase 3 ships.
"""

import warnings

warnings.warn(
    "fastlisaresponse.tdiconfig has moved to lisatools.response.tdiconfig; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.response.tdiconfig import *  # noqa: F401,F403,E402
from lisatools.response.tdiconfig import TDIConfig  # noqa: F401,E402
