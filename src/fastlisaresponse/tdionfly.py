"""Deprecation shim. Moved to ``lisatools.response.tdionfly``.

Phase 3 of the 2026 sprint reorg moved the LISA-response Python
frontends into LISAanalysistools. This module re-exports everything
from its new home and emits a ``DeprecationWarning`` on first use.
Scheduled for removal one release cycle after Phase 3 ships.
"""

import warnings

warnings.warn(
    "fastlisaresponse.tdionfly has moved to lisatools.response.tdionfly; "
    "import from there instead. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from lisatools.response.tdionfly import *  # noqa: F401,F403,E402
