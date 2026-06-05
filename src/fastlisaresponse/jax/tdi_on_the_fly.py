"""Deprecation shim. Moved to ``gbgpu.jax.tdi_on_the_fly``.

Phase 3L.7j of the 2026 sprint reorg (2026-06-04) moved the JAX
TDI-on-the-fly entry points (``run_wave_tdi``, ``gb_run_wave_tdi``,
``sobbh_run_wave_tdi``) out of lisa-on-gpu and into GBGPU as part of
the lisa-on-gpu full-retirement work. This module re-exports the
symbols from their new home and emits a ``DeprecationWarning``.

Migration:

* ``from fastlisaresponse.jax.tdi_on_the_fly import gb_run_wave_tdi`` ->
  ``from gbgpu.jax.tdi_on_the_fly import gb_run_wave_tdi``
* ``from fastlisaresponse.jax.tdi_on_the_fly import sobbh_run_wave_tdi`` ->
  ``from gbgpu.jax.tdi_on_the_fly import sobbh_run_wave_tdi``
"""

import warnings

warnings.warn(
    "fastlisaresponse.jax.tdi_on_the_fly has moved to "
    "gbgpu.jax.tdi_on_the_fly as of Phase 3L.7j (2026-06-04). "
    "Update your imports. This shim will be removed when "
    "fastlisaresponse is fully retired.",
    DeprecationWarning,
    stacklevel=2,
)

from gbgpu.jax.tdi_on_the_fly import (  # noqa: E402, F401
    gb_run_wave_tdi,
    run_wave_tdi,
    sobbh_run_wave_tdi,
)
