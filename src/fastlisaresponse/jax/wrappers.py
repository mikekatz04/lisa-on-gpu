"""Deprecation shim. Moved to ``gbgpu.jax.wrappers``.

Phase 3L.7j of the 2026 sprint reorg (2026-06-04) moved the JAX
wrapper objects (``GBTDIonTheFlyWrapJAX``, ``SOBBHTDIonTheFlyWrapJAX``)
out of lisa-on-gpu and into GBGPU as part of the lisa-on-gpu
full-retirement work. This module re-exports the symbols from their
new home and emits a ``DeprecationWarning``.

Migration:

* ``from fastlisaresponse.jax.wrappers import GBTDIonTheFlyWrapJAX`` ->
  ``from gbgpu.jax.wrappers import GBTDIonTheFlyWrapJAX``
* ``from fastlisaresponse.jax.wrappers import SOBBHTDIonTheFlyWrapJAX`` ->
  ``from gbgpu.jax.wrappers import SOBBHTDIonTheFlyWrapJAX``
"""

import warnings

warnings.warn(
    "fastlisaresponse.jax.wrappers has moved to gbgpu.jax.wrappers as "
    "of Phase 3L.7j (2026-06-04). Update your imports. This shim will "
    "be removed when fastlisaresponse is fully retired.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.jax.wrappers import (  # noqa: E402, F401
    GBTDIonTheFlyWrapJAX,
    SOBBHTDIonTheFlyWrapJAX,
)
# OrbitsWrapJAX + TDIConfigWrapJAX were re-exported from this module
# pre-3L.7j alongside the wrapper classes; preserve that by sourcing
# them from their canonical LAT homes (Phase 3D).
from lisatools.jax.orbits import OrbitsWrapJAX  # noqa: E402, F401
from lisatools.jax.response.tdi_config import TDIConfigWrapJAX  # noqa: E402, F401
