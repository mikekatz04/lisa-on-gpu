"""Duck-typed wrappers exposing the C++ ``*Wrap`` API surface in JAX.

The C++ backend exposes objects like
    backend.GBTDIonTheFlyWrap(orbits_wrap, tdi_config_wrap, T, t_ref)
        .run_wave_tdi_wrap(tdi_channels_arr, tdi_amp, tdi_phase,
                           phase_ref, params, t_arr, N, num_bin,
                           n_params, nchannels)
        .run_fd_wave_tdi_wrap(X_het, k_f0_out, f0_grid_out, params,
                              t_start, T, N_sparse, num_bin,
                              n_params, nchannels)

The JAX path needs the same API so the Python orchestration in
``tdionfly.py`` doesn't need a per-backend branch. We provide functional
return when called via the ``.run_wave_tdi`` (no ``_wrap`` suffix) entry
point so autograd works, and a thin in-place adapter at
``.run_wave_tdi_wrap(...)`` that mutates the user-supplied buffers.

JAX arrays are immutable; the in-place adapter does
    buf[...] = np.asarray(jnp_result)
so the caller sees the same array object updated -- preserving the C++
contract. For the autograd path users should call
``.run_wave_tdi(...)`` directly and consume its return values.

Each wrapper carries a reference to a default source class (``UCB``,
``SOBBH``) but the underlying functional kernel can accept any
:class:`JaxAmpPhaseSource`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import jax.numpy as jnp

from .base import JaxAmpPhaseSource
from .orbits import OrbitsWrapJAX
from .tdi_config import TDIConfigWrapJAX
from .tdi_on_the_fly import run_wave_tdi


def _copy_into(buf, value):
    """Copy a JAX result into a numpy-like buffer, preserving dtype.

    Used by the in-place ``run_wave_tdi_wrap`` adapter so the caller
    sees the same array object updated, matching the C++ contract.
    Tolerates either numpy or cupy ``buf`` (the JAX backend's xp is
    numpy; cupy buffers might be passed if mixing backends).
    """
    arr = np.asarray(value).astype(buf.dtype, copy=False).reshape(buf.shape)
    buf[...] = arr


class _BaseTDIonTheFlyWrapJAX:
    """Shared implementation for the GB/SOBBH JAX wrappers.

    Subclasses set ``_default_source_cls`` (subclass of
    :class:`JaxAmpPhaseSource`) and ``_n_params``.
    """

    _default_source_cls: type = None  # type: ignore[assignment]
    _n_params: int = 0

    def __init__(self, orbits_wrap, tdi_config_wrap, T: float, t_ref: float):
        if not isinstance(orbits_wrap, OrbitsWrapJAX):
            raise TypeError(
                "orbits_wrap must be an OrbitsWrapJAX; got "
                f"{type(orbits_wrap).__name__}"
            )
        if not isinstance(tdi_config_wrap, TDIConfigWrapJAX):
            raise TypeError(
                "tdi_config_wrap must be a TDIConfigWrapJAX; got "
                f"{type(tdi_config_wrap).__name__}"
            )
        self.orbits = orbits_wrap
        self.tdi_config = tdi_config_wrap
        self.T = float(T)
        self.t_ref = float(t_ref)
        # Default source -- callers can override via run_wave_tdi(source=...).
        self.source: JaxAmpPhaseSource = self._default_source_cls(t_ref=self.t_ref)

    # ----------------------------------------------------------------
    # Functional entry point (the autograd-friendly path).
    # ----------------------------------------------------------------
    def run_wave_tdi(self, params, t_arr,
                     source: Optional[JaxAmpPhaseSource] = None
                     ) -> "tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]":
        """Return ``(M, tdi_amp, tdi_phase, phase_ref)`` (JAX arrays).

        Shapes:
            M         : complex (num_bin, nchannels, N)
            tdi_amp   : real    (num_bin, nchannels, N)
            tdi_phase : real    (num_bin, nchannels, N)
            phase_ref : real    (num_bin, N)
        """
        if source is None:
            source = self.source
        return run_wave_tdi(params, t_arr, source, self.orbits, self.tdi_config)

    # ----------------------------------------------------------------
    # In-place adapter mirroring the C++ ``*WrapCPU::run_wave_tdi_wrap``.
    # ----------------------------------------------------------------
    def run_wave_tdi_wrap(self, tdi_channels_arr, tdi_amp, tdi_phase,
                          phase_ref, params, t_arr,
                          N, num_bin, n_params, nchannels,
                          source: Optional[JaxAmpPhaseSource] = None):
        """In-place buffer-fill variant of :meth:`run_wave_tdi`.

        Signature matches ``gb_run_wave_tdi_wrap`` /
        ``sobbh_run_wave_tdi_wrap`` in TDIonTheFly.hh exactly so this
        wrapper can be dropped into the existing
        :class:`GBTDIonTheFly.__call__` path.

        Buffer shapes (flat 1D, matching C++):
            tdi_channels_arr : complex, length ``N*nchannels*num_bin``
            tdi_amp          : float,   length ``N*nchannels*num_bin``
            tdi_phase        : float,   length ``N*nchannels*num_bin``
            phase_ref        : float,   length ``N*num_bin``
        """
        if n_params != self._n_params:
            raise ValueError(
                f"{type(self).__name__} expects n_params={self._n_params}, "
                f"got {n_params}."
            )
        # Re-shape the flat inputs into the (num_bin, n_params) /
        # (num_bin, N) shapes the functional kernel expects.
        params_2d = np.asarray(params).reshape(num_bin, n_params)
        t_2d = np.asarray(t_arr).reshape(num_bin, N)

        M, amps, phases, prefs = self.run_wave_tdi(params_2d, t_2d, source=source)

        # Copy back. C++ layout: index = ((bin * nchannels) + chan) * N + i.
        amps_flat = np.asarray(amps).reshape(num_bin * nchannels * N)
        phases_flat = np.asarray(phases).reshape(num_bin * nchannels * N)
        prefs_flat = np.asarray(prefs).reshape(num_bin * N)
        M_flat = np.asarray(M).reshape(num_bin * nchannels * N)

        _copy_into(tdi_amp, amps_flat)
        _copy_into(tdi_phase, phases_flat)
        _copy_into(phase_ref, prefs_flat)
        _copy_into(tdi_channels_arr, M_flat)

    # ----------------------------------------------------------------
    # FD path: NOT implemented yet -- the C++ kernel uses an in-place
    # shared-memory FFT that doesn't translate cleanly to a pure JAX
    # function. Will land in a follow-up slice once the TD path is
    # validated against C++.
    # ----------------------------------------------------------------
    def run_fd_wave_tdi_wrap(self, *args, **kwargs):
        raise NotImplementedError(
            "Heterodyne FD GB TDI is not yet implemented in the JAX backend. "
            "Use the C++ backend (force_backend='cpu' / 'cuda12x') for FD "
            "until the JAX FD kernel lands."
        )


class GBTDIonTheFlyWrapJAX(_BaseTDIonTheFlyWrapJAX):
    """JAX analog of ``GBTDIonTheFlyWrapCPU/GPU``."""

    @property
    def _default_source_cls(self):
        from .sources.ucb import JaxUCBSource
        return JaxUCBSource

    _n_params = 9


class SOBBHTDIonTheFlyWrapJAX(_BaseTDIonTheFlyWrapJAX):
    """JAX analog of ``SOBBHTDIonTheFlyWrapCPU/GPU``."""

    @property
    def _default_source_cls(self):
        from .sources.sobbh import JaxSOBBHSource
        return JaxSOBBHSource

    _n_params = 11
