"""Host-side helpers for the chunked-heterodyne WDM kernels.

These are the Python primitives that
:class:`fastlisaresponse.gbcomps.GBWDMComputations` (and any future
SOBBH analog) hands off to the C++ / JAX kernels. Keeping them inside
``fastlisaresponse`` lets ``gbcomps.py`` avoid any reach into the
repo-root scratch scripts.

Pieces here:

* :func:`compute_chunk_geometry` -- per-chunk ``starts / keep_lo /
  keep_hi / n_global_lo`` arrays the kernels consume. Handles the
  partial-slide tail when ``(Nt - Nt_sub) % step != 0``.
* :func:`compute_wdm_window` -- the Nt_sub-long phitilde sample the
  ``gb_chunk_fd_to_wdm`` kernel reads on the host.
* :func:`compute_layer_groups` / :func:`compute_swap_layer_groups` --
  cluster binaries by carrier WDM layer so each kernel launch reads
  data/PSD only over a narrow m-band per group.
* :func:`recommended_tukey_alpha` / :func:`resolve_tukey_alpha` --
  centralised Tukey alpha selection. ``USE_RECOMMENDED_TUKEY`` is the
  sentinel; ``recommended_tukey_alpha("heterodyne", N_sparse=...)``
  is the auto-pick (0.01 for N_sparse>=512, 0.05 otherwise).

Algorithm reference for the chunk geometry / layer-grouping lives in
``check_shortened_wdm.py`` and the design notes
``CHUNKED_HET_DESIGN_NOTES.md``.
"""
from __future__ import annotations

import numpy as np

from lisatools.domains import WDMSettings
from lisatools.utils.utility import get_array_module

# ----------------------------------------------------------------------
# Tukey alpha
# ----------------------------------------------------------------------

# Recommended alphas for the chunked-stitch variants. The TD path was
# validated at 0.02; the heterodyne path uses 0.01 at wide N_sparse
# (>=512) and 0.05 at narrow N_sparse (<512).
RECOMMENDED_TUKEY_ALPHA_TD = 0.02
RECOMMENDED_TUKEY_ALPHA_HET_WIDE = 0.01
RECOMMENDED_TUKEY_ALPHA_HET_NARROW = 0.05

# Sentinel: pass ``tukey_alpha=USE_RECOMMENDED_TUKEY`` to opt into the
# auto-pick. Float-typed so existing positional callers still work.
USE_RECOMMENDED_TUKEY = -1.0


def recommended_tukey_alpha(path, N_sparse=None):
    """Return the recommended Tukey alpha for a chunked-WDM workflow.

    Args:
        path: ``"td"`` for the TD-based chunked stitch or
            ``"heterodyne"`` / ``"het"`` / ``"fd"`` for the FD-direct
            heterodyne stitch.
        N_sparse: heterodyne FFT length. Wider for >= 512, narrower
            otherwise.
    """
    p = str(path).lower()
    if p in ("td", "td_stitched", "td-based"):
        return RECOMMENDED_TUKEY_ALPHA_TD
    if p in ("heterodyne", "het", "fd", "fd_heterodyne"):
        if N_sparse is None or N_sparse >= 512:
            return RECOMMENDED_TUKEY_ALPHA_HET_WIDE
        return RECOMMENDED_TUKEY_ALPHA_HET_NARROW
    raise ValueError(
        "path must be one of 'td', 'heterodyne', 'het', 'fd', 'fd_heterodyne'"
    )


def resolve_tukey_alpha(tukey_alpha, use_tukey, path, N_sparse=None):
    """Resolve the actual Tukey alpha to apply, per the ``use_tukey`` flag.

    Precedence: explicit numeric ``tukey_alpha`` (>= 0) is honoured;
    ``USE_RECOMMENDED_TUKEY`` triggers :func:`recommended_tukey_alpha`;
    ``use_tukey=False`` forces 0.0; otherwise the value is kept.
    """
    if tukey_alpha == USE_RECOMMENDED_TUKEY:
        if not use_tukey:
            return 0.0
        return recommended_tukey_alpha(path, N_sparse=N_sparse)
    if not use_tukey:
        return 0.0
    return float(tukey_alpha)


# ----------------------------------------------------------------------
# Chunk geometry + WDM window
# ----------------------------------------------------------------------

def compute_chunk_geometry(Nt, Nt_sub, n_pad):
    """Pre-compute per-chunk stitching geometry for the het kernels.

    Returns a dict of numpy arrays:

      ``starts``      (n_chunks,) int64   -- chunk start in WDM-pixel units
      ``keep_lo``     (n_chunks,) int32   -- chunk-local lo to keep
      ``keep_hi``     (n_chunks,) int32   -- chunk-local hi (exclusive)
      ``n_global_lo`` (n_chunks,) int32   -- global WDM pixel where the
                                             first kept sample lands

    Handles the partial-slide case (``(Nt - Nt_sub) % step != 0``) by
    appending one extra chunk at ``Nt - Nt_sub`` and trimming the prior
    ``keep_hi`` so the overlap region isn't double-counted.
    """
    step = int(Nt_sub) - 2 * int(n_pad)
    assert step > 0 and step % 2 == 0, (Nt_sub, n_pad, step)

    # Single-chunk case: Nt_sub >= Nt. One chunk covers the full WDM grid;
    # the chunk's TD span (Nt_sub samples) is longer than Nt but we only
    # keep [0, Nt) WDM pixels. The waveform is evaluated on the fly at any
    # t, so extending the chunk past t_obs_end is harmless.
    if int(Nt_sub) >= int(Nt):
        return dict(
            starts      = np.asarray([0],          dtype=np.int64),
            keep_lo     = np.asarray([0],          dtype=np.int32),
            keep_hi     = np.asarray([int(Nt)],    dtype=np.int32),
            n_global_lo = np.asarray([0],          dtype=np.int32),
        )

    n_full = (int(Nt) - int(Nt_sub)) // step + 1
    starts = [j * step for j in range(n_full)]
    last_full_end = starts[-1] + Nt_sub

    keep_lo = []
    keep_hi = []
    n_global_lo = []
    for j, n0 in enumerate(starts):
        klo = 0 if j == 0 else n_pad
        if j == n_full - 1 and last_full_end == Nt:
            khi = Nt_sub
        else:
            khi = Nt_sub - n_pad
        keep_lo.append(klo)
        keep_hi.append(khi)
        n_global_lo.append(n0 + klo)

    if last_full_end < Nt:
        n0_partial = Nt - Nt_sub
        new_lo_global = starts[-1] + (Nt_sub - n_pad)
        new_lo_local = new_lo_global - n0_partial
        starts.append(n0_partial)
        keep_lo.append(new_lo_local)
        keep_hi.append(Nt_sub)
        n_global_lo.append(new_lo_global)

    return dict(
        starts      = np.asarray(starts,      dtype=np.int64),
        keep_lo     = np.asarray(keep_lo,     dtype=np.int32),
        keep_hi     = np.asarray(keep_hi,     dtype=np.int32),
        n_global_lo = np.asarray(n_global_lo, dtype=np.int32),
    )


def compute_wdm_window(Nf, Nt_sub, dt, backend="cpu"):
    """Return the Nt_sub-long phitilde sampled at the chunk's grid.

    Equivalent to ``WDMSettings(Nf, Nt_sub, dt).window`` -- exposed
    here as a free function so host code doesn't need to construct a
    full settings object. The window is a real-valued host array; we
    always build it through the ``cpu`` backend (lisatools' WDMSettings
    in-place initializer doesn't support jax arrays).
    """
    del backend  # window is host-side; always CPU
    return np.asarray(
        WDMSettings(Nf=int(Nf), Nt=int(Nt_sub), dt=float(dt),
                    force_backend="cpu").window,
        dtype=float,
    )


# ----------------------------------------------------------------------
# Layer-grouping (narrow-band GB inner product)
# ----------------------------------------------------------------------

def compute_layer_groups(params_array, layer_df, f0_param_index=1,
                          group_band_layers=5, margin_layers=0,
                          data_index_all=None, noise_index_all=None):
    """Group binaries by (m-band, data_index) for get_ll dispatch.

    Each group's kernel iteration is restricted to the m-band
    ``[m_lo - margin_layers, m_hi + margin_layers)`` -- cuts WDM-data
    global-mem reads by ~``Nf / group_band_layers``.

    The default ``group_band_layers=5`` gives a 5-wide band centered on
    each group's carrier WDM layer (``[m_floor - 2, m_floor + 3)`` for
    ``group_band_layers=5``), matching the validated ``mm5`` narrow-band
    convention for GB inner products. ``group_band_layers=3`` (``m_floor
    ± 1``) is also enough when a Tukey (or other) window is applied
    to the *injection* data -- the window suppresses the m_floor-1
    spillover so 2-3 layers around the carrier capture essentially all
    of the in-band power.

    Returns the per-group arrays the kernels consume:
        ``binary_perm``, ``group_starts``, ``group_ends``,
        ``group_m_lo``, ``group_m_hi``, ``group_data_index``,
        ``n_groups``.
    """
    xp = get_array_module(params_array)
    params_array = xp.asarray(params_array)
    num_bin = params_array.shape[0]
    f0 = params_array[:, int(f0_param_index)]
    m_floor = xp.floor(f0 / float(layer_df)).astype(xp.int32)

    if data_index_all is None:
        data_index_all = xp.zeros(num_bin, dtype=xp.int32)
    else:
        data_index_all = xp.asarray(data_index_all, dtype=xp.int32)
        assert data_index_all.shape == (num_bin,)

    if noise_index_all is None:
        noise_index_all = xp.zeros(num_bin, dtype=xp.int32)
    else:
        noise_index_all = xp.asarray(noise_index_all, dtype=xp.int32)
        assert noise_index_all.shape == (num_bin,)
        if xp.any(noise_index_all != 0):
            assert xp.array_equal(noise_index_all, data_index_all), (
                "noise_index_all must equal data_index_all when noise_index "
                "is non-trivial; grouping path assumes data + PSD share a slab")
    if xp != np:
        # Manual lexsort equivalent to np.lexsort((m_floor, data_index_all)):
        # stable-sort by secondary key first, then stable-sort by primary.
        # Works under both numpy and cupy (cupy.lexsort requires a 2-D keys
        # array and rejects axis kwargs).
        _sec = xp.argsort(m_floor, kind='stable')
        order = _sec[xp.argsort(data_index_all[_sec], kind='stable')]
    else:
        order = xp.lexsort((m_floor, data_index_all))
    sorted_m = m_floor[order]
    sorted_data_idx = data_index_all[order]

    # Build the iterated m-band as a window of width ``group_band_layers``
    # *centred* on each group's carrier(s). Mirrors
    # :func:`compute_swap_layer_groups`'s ``pair_m_{lo,hi}_b`` convention so
    # a single-binary group still sees the full ``group_band_layers``-wide
    # window instead of just its carrier layer. ``half = group_band_layers
    # // 2`` puts more bins above the carrier than below (e.g. for
    # ``group_band_layers=5``, ``half=2`` -> band ``[m-2, m+3)``), matching
    # the WDM transform's slightly-asymmetric spectral spread.
    half = int(group_band_layers) // 2
    starts, ends, m_los, m_his, di = [], [], [], [], []
    i = 0
    while i < num_bin:
        m0 = sorted_m[i]
        d0 = sorted_data_idx[i]
        j = i
        while j < num_bin and sorted_data_idx[j] == d0 and (
                sorted_m[j] - m0) < group_band_layers:
            j += 1
        m_lo = int(m0 - half - margin_layers)
        m_hi = int(sorted_m[j - 1] + (group_band_layers - half) + margin_layers)
        starts.append(int(i))
        ends.append(int(j))
        m_los.append(m_lo)
        m_his.append(m_hi)
        di.append(int(d0))
        i = j

    return dict(
        binary_perm      = xp.asarray(order,  dtype=xp.int32),
        group_starts     = xp.asarray(starts, dtype=xp.int32),
        group_ends       = xp.asarray(ends,   dtype=xp.int32),
        group_m_lo       = xp.asarray(m_los,  dtype=xp.int32),
        group_m_hi       = xp.asarray(m_his,  dtype=xp.int32),
        group_data_index = xp.asarray(di,     dtype=xp.int32),
        n_groups         = len(starts),
    )


def compute_swap_layer_groups(params_add, params_remove, layer_df,
                               f0_param_index=1,
                               group_band_layers=5, margin_layers=0,
                               data_index_all=None, noise_index_all=None):
    """Layer-grouping for swap_ll's two-pass cache strategy.

    Returns the same fields as :func:`compute_layer_groups` (driven by
    the ADD template's carrier WDM layer) plus per-pair remove-template
    bands (``pair_m_lo_b``, ``pair_m_hi_b``) the kernel uses for its
    second pass over the leftover <d|rem> / <rem|rem> pixels.
    """
    xp = get_array_module(params_add)
    params_add    = xp.asarray(params_add)
    params_remove = xp.asarray(params_remove)
    num_bin = params_add.shape[0]
    assert params_remove.shape[0] == num_bin

    add_groups = compute_layer_groups(
        params_add, layer_df=layer_df, f0_param_index=f0_param_index,
        group_band_layers=group_band_layers, margin_layers=margin_layers,
        data_index_all=data_index_all, noise_index_all=noise_index_all)

    binary_perm = add_groups["binary_perm"]
    f0_b = params_remove[:, int(f0_param_index)]
    m_b  = xp.floor(f0_b / float(layer_df)).astype(xp.int32)
    half = group_band_layers // 2

    m_b_sorted     = m_b[binary_perm]
    pair_m_lo_b    = (m_b_sorted - half - margin_layers).astype(xp.int32)
    pair_m_hi_b    = (m_b_sorted + (group_band_layers - half)
                       + margin_layers).astype(xp.int32)

    add_groups["pair_m_lo_b"] = pair_m_lo_b
    add_groups["pair_m_hi_b"] = pair_m_hi_b
    return add_groups


__all__ = [
    "RECOMMENDED_TUKEY_ALPHA_TD",
    "RECOMMENDED_TUKEY_ALPHA_HET_WIDE",
    "RECOMMENDED_TUKEY_ALPHA_HET_NARROW",
    "USE_RECOMMENDED_TUKEY",
    "recommended_tukey_alpha",
    "resolve_tukey_alpha",
    "compute_chunk_geometry",
    "compute_wdm_window",
    "compute_layer_groups",
    "compute_swap_layer_groups",
]
