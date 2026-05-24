"""Base class for JAX amp/phase source plugins.

Source classes provide the per-time-sample (amplitude, phase) functions
that the JAX TDI-on-the-fly projection composes with the LISA arm
geometry. The interface mirrors the C++ pattern in ``TDIonTheFly.hh``
where ``LISATDIonTheFly`` exposes virtual ``get_amp(t, params, bin_i)``
/ ``get_phase(t, params, bin_i)`` that source subclasses
(``GBTDIonTheFly``, ``SOBBHTDIonTheFly``) override.

JAX twist: the source object is *static* (configuration only). The
per-source parameters live in a flat 1D array (``params``) -- exactly the
``double *params`` the C++ kernel sees -- so that ``jax.vmap`` over the
binary axis is the natural way to drive a batch. The source class
declares where the sky-angle parameters live (``inc_index``, ...) so the
shared TDI projection can pull them out in a source-agnostic way.

Concrete plugins live in ``fastlisaresponse.jax.sources``:
    - ``JaxUCBSource``  -> matches GBTDIonTheFly::ucb_*
    - ``JaxSOBBHSource`` -> matches SOBBHTDIonTheFly::sobbh_*
"""
from __future__ import annotations

import abc
from typing import Tuple

import jax.numpy as jnp


class JaxAmpPhaseSource(abc.ABC):
    """Abstract amp/phase source for the JAX TDI-on-the-fly engine.

    Concrete subclasses populate the ``n_params``, ``param_names``, and
    sky-angle index attributes and implement :meth:`amplitude` and
    :meth:`phase`. The TDI projection consumes them via the static
    methods :meth:`get_amp`, :meth:`get_phase`, :meth:`get_sky_indices`.
    """

    #: Number of parameters per source (length of ``params`` array).
    n_params: int = 0
    #: Human-readable parameter names, length ``n_params``.
    param_names: Tuple[str, ...] = ()
    #: Indices into ``params`` for the four sky/polarization angles.
    #: Defaults match the GB layout (inc=5, psi=6, lam=7, beta=8).
    inc_index: int = 5
    psi_index: int = 6
    lam_index: int = 7
    beta_index: int = 8

    # ----------------------------------------------------------------
    # Per-sample physics. Subclasses MUST override.
    # ----------------------------------------------------------------
    @abc.abstractmethod
    def amplitude(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """Scalar amplitude :math:`A(t)` for a single source.

        ``t``: scalar or array of times (seconds).
        ``params``: 1D ``jnp.ndarray`` of length ``n_params``.

        Returns an array broadcasting with ``t``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def phase(self, t: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        """GW phase :math:`\\Phi(t)` (radians, unwrapped) for a single source.

        ``t``: scalar or array of times (seconds).
        ``params``: 1D ``jnp.ndarray`` of length ``n_params``.

        Returns an array broadcasting with ``t``.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------
    # Sky-angle accessors. Default GB layout; SOBBH overrides via attrs.
    # ----------------------------------------------------------------
    def get_sky_indices(self) -> Tuple[int, int, int, int]:
        """Return ``(inc_index, psi_index, lam_index, beta_index)``."""
        return (self.inc_index, self.psi_index, self.lam_index, self.beta_index)
