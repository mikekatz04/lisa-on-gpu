"""JAX port of ``new_extract_amplitude_and_phase`` + ``new_unwrap_phase``.

These two functions in ``TDIonTheFly.cu:4262-4341`` (extract) and
``TDIonTheFly.cu:4197-4258`` (unwrap) post-process the complex TDI
samples ``M[n]`` into the ``(tdi_amp[n], tdi_phase[n])`` pair the
downstream WDM / spline machinery consumes.

The algorithm:

1. Compute ``|M[n]|`` (real amplitudes).
2. Find local minima of ``|M|`` that look like sign-flips (the second-
   derivative test ``|dA2/dA1|<0.1`` or ``|dA3/dA1|<0.1``).
3. Per minimum, decide whether to flip ``M[n]`` or ``M[n+1]`` and store
   the cumulative jump count ``c[n]``.
4. Multiply ``As[n]`` by ``(-1)^c[n]`` and define
   ``Dphi[n] = -arg(M[n]) + c[n]*pi - rem(phiR[n], 2*pi)``.
5. Unwrap the resulting ``Dphi`` by adding multiples of ``2*pi`` so
   adjacent samples don't jump by more than ``pi``.

The JAX port keeps these steps but expresses them as fixed-shape array
operations so ``jax.jit`` can trace them. The local-min test uses
``jnp.where`` masks rather than the C++ branchy ``if``, and the
cumulative count is a ``cumsum``.
"""
from __future__ import annotations

import jax.numpy as jnp


# --------------------------------------------------------------------
# Extract amplitude / phase from complex M with sign-flip handling.
# --------------------------------------------------------------------
def extract_amplitude_and_phase(M: jnp.ndarray, phiR: jnp.ndarray
                                ) -> "tuple[jnp.ndarray, jnp.ndarray]":
    """Mirror of ``LISATDIonTheFly::new_extract_amplitude_and_phase``.

    Args:
        M: Complex array of shape ``(N,)`` -- single channel of the TDI
            output from ``get_tdi_Xf_single``.
        phiR: Real array of shape ``(N,)`` -- reference phase (``get_phase_ref``).

    Returns:
        ``(As, Dphi)``, both real arrays of shape ``(N,)``. ``As`` is the
        signed amplitude (with the sign flips folded in) and ``Dphi``
        is the un-unwrapped phase difference; call
        :func:`unwrap_phase` to get the production output.
    """
    As_raw = jnp.abs(M)
    N = As_raw.shape[0]

    # Local-min mask on the interior (i in [1, N-2]).
    # is_min[i] = As[i] < As[i-1] and As[i] < As[i+1] (i = 1..N-2)
    interior_prev = As_raw[:-2]
    interior_curr = As_raw[1:-1]
    interior_next = As_raw[2:]
    is_min = (interior_curr < interior_prev) & (interior_curr < interior_next)

    dA1 = interior_next + interior_prev - 2.0 * interior_curr
    dA2 = -interior_next + interior_prev - 2.0 * interior_curr
    dA3 = -interior_next + interior_prev + 2.0 * interior_curr

    # ``test1`` -> count++ at i+1; ``test2`` -> count++ at i.
    # Use ``where`` to dodge division-by-zero when dA1 == 0 at a smooth
    # interior point (in that case the inequality ``< 0.1`` is False).
    safe_dA1 = jnp.where(dA1 != 0.0, dA1, 1.0)
    test1 = is_min & (jnp.abs(dA2 / safe_dA1) < 0.1)
    test2 = is_min & ~test1 & (jnp.abs(dA3 / safe_dA1) < 0.1)

    # Per-sample bump array: shift test1 by +1 in the i-index space and
    # add test2 in place.
    count_increments = jnp.zeros(N, dtype=jnp.int32)
    # interior i -> global index i+1; test1 bumps i+2, test2 bumps i+1.
    count_increments = count_increments.at[2:].add(test1.astype(jnp.int32))
    count_increments = count_increments.at[1:-1].add(test2.astype(jnp.int32))

    count = jnp.cumsum(count_increments)

    flip = jnp.where(count % 2 == 0, 1.0, -1.0)
    pjump = count.astype(As_raw.dtype) * jnp.pi

    # As[Ns-1] = flip[Ns-2] * |M[Ns-1]| handled implicitly: flip is
    # constant beyond the last increment so flip[-1] == flip[-2].
    As = flip * As_raw

    v = jnp.remainder(phiR, 2.0 * jnp.pi)
    Dphi = -jnp.arctan2(M.imag, M.real) + pjump - v
    return As, Dphi


# --------------------------------------------------------------------
# Phase-unwrap (TDIonTheFly.cu:4197-4258, the ``new_unwrap_phase``).
# --------------------------------------------------------------------
def unwrap_phase(phase: jnp.ndarray) -> jnp.ndarray:
    """Mirror of ``LISATDIonTheFly::new_unwrap_phase``.

    Computes the per-sample correction
        d[i] = round((phase[i] - phase[i-1]) / (2 pi))
    and applies ``-2 pi * cumsum(d)`` so that adjacent samples differ by
    at most ``pi``. Equivalent to ``numpy.unwrap`` but starts at index 0
    (no rotation of the initial sample) -- which is what the C++ does
    so we don't drift relative to it.
    """
    diff = jnp.diff(phase)
    # round-to-nearest, ties-to-even -- matches C99 ``rint``/``round``
    # closely enough that the integer winding numbers agree on the
    # interior. The C++ uses an unconditional ``int(round(x))`` so we
    # mimic that with ``jnp.round`` + cast.
    correction = jnp.round(diff / (2.0 * jnp.pi)).astype(jnp.int32)
    cumulative = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(correction)]
    )
    return phase - 2.0 * jnp.pi * cumulative.astype(phase.dtype)


def extract_and_unwrap(M_channels: jnp.ndarray, phiR: jnp.ndarray
                       ) -> "tuple[jnp.ndarray, jnp.ndarray]":
    """Convenience: per-channel extract+unwrap on the full (C, N) TDI block.

    ``M_channels``: complex array of shape ``(nchannels, N)``.
    ``phiR``: real array of shape ``(N,)``.

    Returns ``(tdi_amp, tdi_phase)``, both shape ``(nchannels, N)``.
    """
    def per_chan(Mc):
        As, Dphi = extract_amplitude_and_phase(Mc, phiR)
        return As, unwrap_phase(Dphi)

    # Manual loop over channels (small fixed number, typically 3).
    amps = []
    phases = []
    for c in range(M_channels.shape[0]):
        a, p = per_chan(M_channels[c])
        amps.append(a)
        phases.append(p)
    return jnp.stack(amps, axis=0), jnp.stack(phases, axis=0)
