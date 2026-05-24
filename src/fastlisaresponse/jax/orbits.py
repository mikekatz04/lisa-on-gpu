"""JAX-native ``OrbitsWrap`` re-export.

The single source of truth for the JAX orbits wrapper lives in
:mod:`lisatools.jax.orbits` so all consumers (this package + any
future LISA waveform packages) share the same implementation.

This module re-exports it so existing imports inside
``fastlisaresponse.jax`` keep working.
"""
from __future__ import annotations

from lisatools.jax.orbits import OrbitsWrapJAX, NLINKS, NSC, _to_jnp

__all__ = ["OrbitsWrapJAX", "NLINKS", "NSC", "_to_jnp"]
