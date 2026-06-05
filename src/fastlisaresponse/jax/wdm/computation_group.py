"""Deprecation shim. Moved to ``gbgpu.jax.wdm.computation_group``.

Phase 3L.7j of the 2026 sprint reorg (2026-06-04) moved the JAX-side
chunked-het computation-group classes (``GBComputationGroupWrapJAX``,
``SOBBHComputationGroupWrapJAX``) out of lisa-on-gpu and into GBGPU
as part of the lisa-on-gpu full-retirement work. This module
re-exports the symbols from their new home and emits a
``DeprecationWarning``.

Migration:

* ``from fastlisaresponse.jax.wdm.computation_group import
  GBComputationGroupWrapJAX`` ->
  ``from gbgpu.jax.wdm.computation_group import GBComputationGroupWrapJAX``
* Same path for ``SOBBHComputationGroupWrapJAX``.
"""

import warnings

warnings.warn(
    "fastlisaresponse.jax.wdm.computation_group has moved to "
    "gbgpu.jax.wdm.computation_group as of Phase 3L.7j (2026-06-04). "
    "Update your imports. This shim will be removed when "
    "fastlisaresponse is fully retired.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.jax.wdm.computation_group import (  # noqa: E402, F401
    GBComputationGroupWrapJAX,
    SOBBHComputationGroupWrapJAX,
)
