"""JAX port of the chunked-heterodyne WDM primitives.

Parallels the C++ implementation in
``../../cutils/TDIonthefly.cu`` (``fast_wdm_inner_heterodyne`` /
``gb_chunk_fd_to_wdm``). Validated algorithm reference lives in
``check_shortened_wdm.py`` (``Test K``: chunk-FD->chunk-WDM Python
reference matches lisatools to ~5e-16).

Conventions match the Python prototype in
``gb_heterodyne_fd.py:make_heterodyne_fd``:

    s(tau) = tdi_amp(tau) * exp(+i (tdi_phase + phase_ref - 2 pi f0_grid * tau))
    X_het  = 0.5 * dt_sparse * FFT(s)

placed onto the chunk's dense rfft grid at bins
``k_f0 + np.fft.fftfreq(N_sparse).astype(int)``.

The JAX functions here are pure-functional (no mutable buffers) so they
flow through ``jax.jit`` / ``jax.vmap`` / ``jax.grad`` cleanly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..base import JaxAmpPhaseSource
from ..orbits import OrbitsWrapJAX
from ..projection import get_phase_ref, get_tdi_Xf_single
from ..tdi_config import TDIConfigWrapJAX


# Recommended Tukey alphas mirror the C++ #defines in TDIonthefly.cu and
# the Python module ``check_shortened_wdm.py``. Use ``ALPHA_AUTO`` to
# trigger the auto-pick (HET_WIDE for N_sparse >= 512, HET_NARROW
# otherwise).
ALPHA_TD          = 0.02
ALPHA_HET_WIDE    = 0.01
ALPHA_HET_NARROW  = 0.05
ALPHA_AUTO        = -1.0


def _resolve_alpha(alpha: float, N_sparse: int) -> float:
    if alpha == ALPHA_AUTO:
        return ALPHA_HET_WIDE if N_sparse >= 512 else ALPHA_HET_NARROW
    return float(alpha)


def _tukey_window(N: int, alpha: float) -> jnp.ndarray:
    """Tukey window of length ``N``; alpha in [0, 1].

    Uses ``alpha * (N - 1) / 2`` as the taper-region length, matching
    ``scipy.signal.windows.tukey`` and the C++ kernel's
    ``fast_wdm_inner_heterodyne``. The previous ``alpha * N / 2`` form
    shifts ~0.1-0.3% of spectral leakage into adjacent bins relative
    to the lisatools reference -- the C++ side had the same bug; both
    are now fixed.
    """
    if alpha <= 0.0:
        return jnp.ones(N)
    n = jnp.arange(N)
    last = N - 1
    n_taper = 0.5 * alpha * (N - 1)
    # left taper (cosine ramp 0->1 over n_taper samples)
    xl = jnp.clip(n / n_taper, 0.0, 1.0)
    left = 0.5 * (1.0 + jnp.cos(jnp.pi * (xl - 1.0)))
    # right taper (mirror)
    xr = jnp.clip((last - n) / n_taper, 0.0, 1.0)
    right = 0.5 * (1.0 + jnp.cos(jnp.pi * (xr - 1.0)))
    return jnp.minimum(left, right)


def fast_wdm_inner_heterodyne_jax(
    chunk_t_start: float,
    T_chunk: float,
    N_sparse: int,
    N_chunk_td: int,
    params: jnp.ndarray,
    source: JaxAmpPhaseSource,
    orbits: OrbitsWrapJAX,
    tdi_config: TDIConfigWrapJAX,
    k_sky: jnp.ndarray, u_sky: jnp.ndarray, v_sky: jnp.ndarray,
    tukey_alpha: float = ALPHA_AUTO,
):
    """Build one chunk's dense rfft via the heterodyne path.

    Returns:
        chunk_fd: complex ``(nchannels, N_chunk_td // 2 + 1)`` array with
            ``N_sparse`` non-zero bins around ``k_f0``.
        k_f0: int (the chunk-df bin index of the snapped carrier).
    """
    f0 = params[1]
    dt_sparse = T_chunk / N_sparse
    t_sparse  = chunk_t_start + jnp.arange(N_sparse) * dt_sparse

    # Per-sample TDI + reference phase
    def _per_t(t):
        M = get_tdi_Xf_single(t, params, source, orbits, tdi_config,
                              k_sky, u_sky, v_sky)
        phi = get_phase_ref(t, params, source, orbits)
        return M, phi
    Ms, phis = jax.vmap(_per_t)(t_sparse)
    # Ms: (N_sparse, nchannels)  complex
    # phis: (N_sparse,) real
    # Project to (amp, phase_tdi) per channel:
    #   z_c(t) = A_c(t) * exp(-i(tdi_phase_c(t) + phase_ref(t)))  (lisatools)
    # so amp_c = |M_c|, tdi_phase_c = -arg(M_c) - phase_ref.
    tdi_amp   = jnp.abs(Ms)                                # (N_sparse, nch)
    tdi_phase = -jnp.angle(Ms) - phis[:, None]             # (N_sparse, nch)

    df_chunk = 1.0 / T_chunk
    k_f0     = jnp.round(f0 / df_chunk).astype(jnp.int32)
    f0_grid  = k_f0.astype(jnp.float64) * df_chunk

    tau     = t_sparse - chunk_t_start
    carrier = 2.0 * jnp.pi * f0_grid * tau                  # (N_sparse,)
    slow    = tdi_amp * jnp.exp(
        +1j * (tdi_phase + phis[:, None] - carrier[:, None])
    )                                                       # (N_sparse, nch)

    alpha_eff = _resolve_alpha(tukey_alpha, N_sparse)
    if alpha_eff > 0.0:
        slow = slow * _tukey_window(N_sparse, alpha_eff)[:, None]

    X_het = 0.5 * dt_sparse * jnp.fft.fft(slow, axis=0)     # (N_sparse, nch)

    n_rfft_chunk = N_chunk_td // 2 + 1
    m_idx = jnp.fft.fftfreq(N_sparse, d=1.0 / N_sparse).astype(jnp.int32)
    kbins = k_f0 + m_idx                                    # (N_sparse,)
    mask  = (kbins >= 0) & (kbins < n_rfft_chunk)

    # Functional placement: build the dense rfft array with .at[].set
    nch = slow.shape[-1]
    chunk_fd = jnp.zeros((nch, n_rfft_chunk), dtype=jnp.complex128)
    chunk_fd = chunk_fd.at[:, jnp.where(mask, kbins, 0)].set(
        jnp.where(mask[None, :], X_het.T, 0.0)
    )
    return chunk_fd, k_f0


def _phitilde_window(Nf: int, Nt_sub: int) -> jnp.ndarray:
    """Build the WDM analysis window (length Nt_sub) -- precompute on host.

    Computed via ``WDMSettings.setup_window`` semantics: dOmega_s = pi/Nf,
    A = dOmega_s/4, B = dOmega_s - 2A, phitilde(omega) is a regularised
    incomplete beta function. Because scipy's betainc isn't easily
    available in JAX we delegate the window precompute to the Python
    host (``gb_wdm_het.compute_wdm_window``) and accept it as input to
    :func:`gb_chunk_fd_to_wdm_jax`.
    """
    raise NotImplementedError(
        "Precompute the window on the host (e.g. with "
        "gb_wdm_het.compute_wdm_window) and pass it to "
        "gb_chunk_fd_to_wdm_jax."
    )


def gb_chunk_fd_to_wdm_jax(
    chunk_fd: jnp.ndarray,                # (nch, n_rfft_chunk) complex
    wdm_window: jnp.ndarray,              # (Nt_sub,) real, host-precomputed
    Nf: int,
    Nt_sub: int,
    data_dt: float,
) -> jnp.ndarray:
    """JAX port of the chunk FD -> chunk WDM transform.

    Mirrors :func:`check_shortened_wdm.gb_chunk_fd_to_wdm_python_reference`
    (which is bit-identical to lisatools' ``FDSignal.wdmtransform`` at
    Nt_sub-length chunk scale).

    Returns:
        w_mn: ``(nch, Nf, Nt_sub)`` real WDM coefficients for the chunk.
    """
    N_chunk_td = Nf * Nt_sub
    half_Nt_sub = Nt_sub // 2
    n_rfft_chunk = N_chunk_td // 2 + 1
    kappa = 2.0 * jnp.sqrt(jnp.pi * data_dt) / float(Nf)
    sqrt2 = jnp.sqrt(2.0)

    # Build per-layer FD slice indices once. For layer m, k_idx in [0,Nt_sub):
    #   k_global = m*half_Nt_sub + (k_idx - half_Nt_sub)
    # Hermitian wrap for k_global < 0 or > N_chunk_td/2.
    k_idx_arr = jnp.arange(Nt_sub)
    m_arr = jnp.arange(Nf + 1)
    k_global = (m_arr[:, None] * half_Nt_sub
                + (k_idx_arr[None, :] - half_Nt_sub))         # (Nf+1, Nt_sub)
    herm_lo = k_global < 0
    herm_hi = k_global > (N_chunk_td // 2)
    herm    = herm_lo | herm_hi
    k_safe  = jnp.where(herm_lo, -k_global,
              jnp.where(herm_hi, N_chunk_td - k_global, k_global))
    k_safe  = jnp.clip(k_safe, 0, n_rfft_chunk - 1)

    nch = chunk_fd.shape[0]

    def _layer(m_row_k_safe, herm_row):
        # gather + window
        fd_slice = chunk_fd[:, m_row_k_safe]                 # (nch, Nt_sub)
        fd_slice = jnp.where(herm_row[None, :], jnp.conj(fd_slice), fd_slice)
        fd_slice = (fd_slice / data_dt) * wdm_window[None, :]
        return jnp.fft.ifft(fd_slice, axis=-1)               # (nch, Nt_sub)

    after_ifft = jax.vmap(_layer)(k_safe, herm)              # (Nf+1, nch, Nt_sub)

    # Apply parity + Re/Im pick:
    #   (m+n) even: real_part = Re(after_ifft)
    #   (m+n) odd:  real_part = Im(after_ifft)
    #   sign = (-1)^((m+1)*n)
    #   boundary (m=0 or m=Nf) & (m+n) odd -> 0
    n_arr   = jnp.arange(Nt_sub)
    mn_par_even = (((m_arr[:, None] + n_arr[None, :]) & 1) == 0)
    boundary    = (m_arr[:, None] == 0) | (m_arr[:, None] == Nf)
    mask_keep   = ~(boundary & ~mn_par_even)
    sign        = jnp.where((((m_arr[:, None] + 1) * n_arr[None, :]) & 1) == 0,
                             1.0, -1.0)
    real_part = jnp.where(mn_par_even,
                           jnp.real(after_ifft.transpose(1, 0, 2)),
                           jnp.imag(after_ifft.transpose(1, 0, 2)))   # (nch, Nf+1, Nt_sub)
    val = kappa * sign[None, :, :] * real_part * mask_keep[None, :, :]

    # Folding: m=0 row holds DC (even n) and Nyquist (odd n).
    w_mn = jnp.zeros((nch, Nf, Nt_sub))
    # m in [1, Nf-1]: direct copy
    w_mn = w_mn.at[:, 1:Nf, :].set(val[:, 1:Nf, :])
    # m = 0, even n: w[0, n] = val[0, n] / sqrt(2)
    even_mask = (n_arr % 2 == 0)
    w_mn = w_mn.at[:, 0, :].set(jnp.where(even_mask, val[:, 0, :] / sqrt2,
                                          w_mn[:, 0, :]))
    # m = 0, odd n: w[0, n] = val[Nf, n-1] / sqrt(2)
    odd_mask = ~even_mask
    val_Nf_shifted = jnp.roll(val[:, Nf, :], 1, axis=-1)        # n-1
    w_mn = w_mn.at[:, 0, :].set(jnp.where(odd_mask, val_Nf_shifted / sqrt2,
                                          w_mn[:, 0, :]))
    return w_mn


__all__ = [
    "ALPHA_TD", "ALPHA_HET_WIDE", "ALPHA_HET_NARROW", "ALPHA_AUTO",
    "fast_wdm_inner_heterodyne_jax",
    "gb_chunk_fd_to_wdm_jax",
]
