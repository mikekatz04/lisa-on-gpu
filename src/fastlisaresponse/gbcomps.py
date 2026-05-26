import numpy as np

from .utils.parallelbase import FastLISAResponseParallelModule
from .utils.wdm_het import (
    USE_RECOMMENDED_TUKEY,
    compute_chunk_geometry,
    compute_layer_groups,
    compute_swap_layer_groups,
    compute_wdm_window,
    resolve_tukey_alpha,
)
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import Orbits, EqualArmlengthOrbits
from lisatools.domains import WDMSettings
from copy import deepcopy
from .response import ecliptic_to_icrs


class GBWDMComputations(FastLISAResponseParallelModule):
    """Source-side WDM-domain GB likelihood (chunked-heterodyne path).

    Routes through the chunked-heterodyne kernel set
    ``gb_wdm_het_{fill_global, get_ll, swap_ll}`` on the backend (C++
    on CPU / CUDA, JAX on the ``jax`` backend). The constructor
    pre-computes the chunk geometry, the Nt_sub-long WDM window, and
    wraps orbits / TDI config so methods are a thin packing of
    per-call kernel arguments.

    Returns ``-0.5 * (d_d + h_h - 2 d_h)``. ``d_d`` defaults to 0; the
    return is then the source-only piece, and the caller adds
    ``-0.5 <d|d>`` for the full ``log p(d|h)``. There is **no**
    separate ``source_only`` flag.

    Subclassing for other narrow-band source classes (e.g. SOBBH) is
    supported through the routing constants below: override
    ``_WRAP_ATTR`` to pick a different ``*ComputationGroupWrap`` on the
    backend, ``_METHOD_PREFIX`` to pick the kernel-name family
    (``gb_wdm_het`` -> ``sobbh_wdm_het``), ``_NPARAMS`` to change the
    per-source parameter count, and ``_F0_PARAM_INDEX`` to point the
    layer-grouping logic at the carrier-frequency column of
    ``params``.
    """

    # Routes ``fill_global_wdm`` / ``get_ll_wdm`` / ``swap_ll_wdm``
    # through ``GBComputationGroupWrap.gb_wdm_het_*`` on the backend.
    # Subclasses (e.g. :class:`SOBBHWDMComputations`) override these.
    _WRAP_ATTR = "GBComputationGroupWrap"
    _METHOD_PREFIX = "gb_wdm_het"
    _NPARAMS = 9
    _F0_PARAM_INDEX = 1   # GBTDIonTheFly: params[1] = f0

    def __init__(self, wdm_settings, t_ref,
                 Nt_sub=256, n_pad=32, N_sparse=256,
                 tukey_alpha=USE_RECOMMENDED_TUKEY, use_tukey=True,
                 N_cp_sig=0, N_cp_orbit=0,
                 orbits=None, tdi_config=None, force_backend=None,
                 d_d=0.0, tdi_type="XYZ"):
        """Args:
            wdm_settings: :class:`lisatools.domains.WDMSettings` instance
                exposing the WDM grid (``Nf``, ``Nt``, ``data_dt``,
                ``Tobs``, ``t0``, ``layer_df``, ``layer_dt``) that the
                global template buffer is built on. ``fill_global_wdm`` /
                ``get_ll_wdm`` operate over this domain.
            t_ref: source-phase reference time in seconds. Forwarded to
                the kernel as ``t_ref_full``.
            Nt_sub: per-chunk WDM time pixels.
            n_pad: WDM pixels discarded at each chunk edge during the
                stitch.
            N_sparse: heterodyne FFT length per chunk; must be a power
                of two <= ``FAST_WDM_N_SPARSE_MAX`` in the C kernel.
            tukey_alpha, use_tukey: Tukey window selector. Default
                ``USE_RECOMMENDED_TUKEY`` auto-picks per
                ``recommended_tukey_alpha`` ("heterodyne" path).
            N_cp_sig: source-signal spline-cache density per chunk
                (0 = direct path; >0 = cache).
            N_cp_orbit: orbit spline-cache density per chunk
                (0 = global-mem lookups; >0 = cache).
            orbits, tdi_config, tdi_type: as on the parent classes.
            d_d: constant added to ``h_h - 2 d_h`` inside the returned
                likelihood (default 0 = source-only return).
            force_backend: chooses the dispatch backend at construction;
                all subsequent methods read ``self.backend`` /
                ``self.backend.xp``. Per the sprint-wide rule, no
                method on this class takes a backend kwarg.
        """
        super().__init__(force_backend=force_backend)

        if not isinstance(wdm_settings, WDMSettings):
            raise TypeError(
                "wdm_settings must be a lisatools.domains.WDMSettings "
                f"instance; got {type(wdm_settings).__name__}.")
        self.wdm_settings = wdm_settings

        # WDM grid + obs setup pulled from the WDMSettings instance.
        # Stored first so the orbits setter can configure on the full
        # obs span before kernel-arg packing.
        self.Nf       = int(wdm_settings.Nf)
        self.Nt       = int(wdm_settings.Nt)
        self.dt       = float(wdm_settings.data_dt)
        self.T        = float(wdm_settings.Tobs)
        self.t_ref    = float(t_ref)
        self.t_obs_start = float(wdm_settings.t0)
        self.Nt_sub   = int(Nt_sub)
        self.n_pad    = int(n_pad)
        self.N_sparse = int(N_sparse)
        self.tukey_alpha = float(tukey_alpha)
        self.use_tukey   = bool(use_tukey)
        self.N_cp_sig    = int(N_cp_sig)
        self.N_cp_orbit  = int(N_cp_orbit)

        # Derived quantities the kernel consumes directly. ``layer_df``
        # is read straight from the WDMSettings so the two definitions
        # cannot drift.
        self.T_chunk      = self.Nf * self.Nt_sub * self.dt
        self.layer_df     = float(wdm_settings.layer_df)
        self.n_rfft_chunk = self.Nf * self.Nt_sub // 2 + 1
        self.log2_N_sparse = int(np.log2(self.N_sparse))
        self.log2_Nt_sub   = int(np.log2(self.Nt_sub))
        assert 2 ** self.log2_N_sparse == self.N_sparse, (
            f"N_sparse={self.N_sparse} must be a power of two.")
        assert 2 ** self.log2_Nt_sub == self.Nt_sub, (
            f"Nt_sub={self.Nt_sub} must be a power of two.")

        # ``d_d`` constant the source-side likelihood adds to
        # ``h_h - 2 d_h``. Defaults to 0 -> source-only.
        self.d_d = float(d_d)

        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError(
                f"tdi_type must be one of 'XYZ', 'AET', 'AE'; got {tdi_type!r}.")
        self.tdi_type = tdi_type

        # Chunk geometry + WDM phitilde -- precomputed once.
        backend_short = self.backend.name.split("_")[-1]
        geom = compute_chunk_geometry(self.Nt, self.Nt_sub, self.n_pad)
        layer_dt = self.Nf * self.dt
        starts_f = np.asarray(geom["starts"], dtype=float)
        self.chunk_t_starts = (
            self.t_obs_start + starts_f * layer_dt
        ).copy()
        self.chunk_keep_lo  = np.asarray(geom["keep_lo"],     dtype=np.int32)
        self.chunk_keep_hi  = np.asarray(geom["keep_hi"],     dtype=np.int32)
        self.chunk_n_global_offset = np.asarray(geom["n_global_lo"],
                                                  dtype=np.int32)
        self.wdm_window = compute_wdm_window(self.Nf, self.Nt_sub, self.dt,
                                              backend=backend_short)
        self.n_chunks = int(starts_f.size)

        # Resolved Tukey alpha (one double passed to the kernel).
        self.resolved_tukey_alpha = float(resolve_tukey_alpha(
            self.tukey_alpha, self.use_tukey,
            path="heterodyne", N_sparse=self.N_sparse,
        ))

        # Orbits / TDI config setters wrap the C++ / JAX objects on the
        # chosen backend.
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.nchannels = self.tdi_config.nchannels

    @property
    def tdi_config(self) -> TDIConfig:
        return self._tdi_config

    @tdi_config.setter
    def tdi_config(self, tdi_config: TDIConfig):
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation", force_backend=self.backend)
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config, force_backend=self.backend)
        elif not isinstance(tdi_config, TDIConfig):
            raise ValueError("TDI Config needs to be a string or an instnace of TDIConfig.")
        self._tdi_config = tdi_config

        self.cpp_tdi_config = self.backend.TDIConfigWrap(*self._tdi_config.pytdiconfig_args)

    @property
    def xp(self) -> object:
        return self.backend.xp

    @property
    def orbits(self) -> object:
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        """Set response orbits and (re)build the backend orbits wrap."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        self._orbits = deepcopy(orbits)

        if not self._orbits.configured:
            # Configure on the full observation span at the kernel's dt
            # so pycppdetector_args populate for the C++ / JAX wrap.
            t_arr = np.arange(0.0, self.T + self.dt, self.dt) + self.t_obs_start
            try:
                self._orbits.configure(t_arr=t_arr, dt=self.dt,
                                        linear_interp_setup=True)
            except TypeError:
                self._orbits.configure(linear_interp_setup=True)
        try:
            self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)
        except:
            breakpoint()
    @classmethod
    def supported_backends(cls):
        # GPU_RECOMMENDED_WITH_JAX appends 'jax' to the CPU/GPU options
        # so this class can dispatch to the pure JAX backend in
        # fastlisaresponse.jax via force_backend='jax'.
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED_WITH_JAX()]

    # ------------------------------------------------------------------
    # Backend dispatch helpers
    # ------------------------------------------------------------------

    def _layer_groups(self, params_2d, group_band_layers, margin_layers,
                       data_index=None, noise_index=None):
        """Cluster binaries by carrier WDM layer for narrow-band dispatch."""
        return compute_layer_groups(
            self.xp.asarray(params_2d), layer_df=self.layer_df,
            f0_param_index=self._F0_PARAM_INDEX,
            group_band_layers=int(group_band_layers),
            margin_layers=int(margin_layers),
            data_index_all=data_index, noise_index_all=noise_index,
        )

    def _swap_layer_groups(self, params_add_2d, params_remove_2d,
                            group_band_layers, margin_layers,
                            data_index=None, noise_index=None):
        return compute_swap_layer_groups(
            np.asarray(params_add_2d), np.asarray(params_remove_2d),
            layer_df=self.layer_df,
            f0_param_index=self._F0_PARAM_INDEX,
            group_band_layers=int(group_band_layers),
            margin_layers=int(margin_layers),
            data_index_all=data_index, noise_index_all=noise_index,
        )

    def _comp_group(self):
        """The ``*ComputationGroupWrap`` instance for this source class."""
        return getattr(self.backend, self._WRAP_ATTR)()

    def _kernel(self, name):
        """Resolve ``self._METHOD_PREFIX + '_' + name`` on the comp group."""
        return getattr(self._comp_group(), f"{self._METHOD_PREFIX}_{name}")

    def _empty_groups(self, num_bin):
        """Zero-length grouping arrays for the un-grouped C-kernel path."""
        return dict(
            binary_perm  = np.zeros(num_bin, dtype=np.int32),
            group_starts = np.zeros(1,       dtype=np.int32),
            group_ends   = np.zeros(1,       dtype=np.int32),
            group_m_lo   = np.zeros(1,       dtype=np.int32),
            group_m_hi   = np.zeros(1,       dtype=np.int32),
            n_groups     = 0,
        )

    def _empty_swap_groups(self, num_bin):
        g = self._empty_groups(num_bin)
        g["pair_m_lo_b"] = np.zeros(num_bin, dtype=np.int32)
        g["pair_m_hi_b"] = np.zeros(num_bin, dtype=np.int32)
        return g

    def _prep_indices(self, num_bin, num_data, num_noise,
                       data_index, noise_index):
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)
        assert int(data_index.max()) < num_data
        assert int(noise_index.max()) < num_noise
        return data_index, noise_index

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None,
                   convert_to_ra_dec: bool = True,
                   grid_dim: int = 0,
                   use_layer_groups: bool = True,
                   group_band_layers: int = 5,
                   margin_layers: int = 0):
        """Per-binary chunked-heterodyne ``<d|h>`` / ``<h|h>``.

        Returns ``-0.5 * (d_d + h_h - 2 d_h)`` per binary. ``d_d``
        defaults to 0 (set in ``__init__``), making the return the
        source-only piece; pass ``d_d`` at construction time to fold
        the global ``<d|d>`` in.

        The raw per-binary inner products are stashed on
        ``self.d_h_out`` / ``self.h_h_out`` for callers that need them
        (e.g. analytic phase maximisation).

        With ``use_layer_groups=True`` (default) binaries are clustered
        by carrier WDM layer and each kernel launch reads
        ``data`` / ``invC`` only over a narrow ``group_band_layers``-wide
        m-band -- this is the canonical narrow-band GB inner product
        (mm5 / mm2). The full-Nf path (``use_layer_groups=False``)
        picks up spectral-tail contributions and is not the right
        physical model for narrow-band GBs.

        Args:
            params: ``(num_bin, nparams)`` array (1D auto-promoted via
                ``atleast_2d``). Last two columns are ``(lam, beta)``;
                with ``convert_to_ra_dec`` (default) they are converted
                to ICRS before dispatch.
            wdm_holder: ``AnalysisContainerArray`` providing
                ``linear_data_arr[0]`` (WDM data) and
                ``linear_psd_arr[0]`` (invC) -- both flat.
            data_index, noise_index: per-binary slab indices into
                ``wdm_holder``. Default = all zeros.
            grid_dim: CUDA launch grid size (use 0 for ``n_chunks``).
            use_layer_groups, group_band_layers, margin_layers: narrow-band
                grouping controls (see method docstring).
        """
        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin = params_tmp.shape[0]
        nparams = int(self._NPARAMS)

        if convert_to_ra_dec:
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        # JAX backend's wrap mutates host (numpy) buffers since jnp
        # arrays are immutable.
        if self.backend.name == "fastlisaresponse_jax":
            d_h_out = np.zeros(num_bin)
            h_h_out = np.zeros(num_bin)
        else:
            d_h_out = self.xp.zeros(num_bin)
            h_h_out = self.xp.zeros(num_bin)

        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_in = params_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._layer_groups(
                params_tmp, group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_groups(num_bin)
        breakpoint()
        self._kernel("get_ll")(
            d_h_out, h_h_out,
            self.cpp_orbits, self.cpp_tdi_config,
            params_in, data_index, noise_index,
            self.xp.asarray(self.chunk_t_starts),
            self.xp.asarray(self.chunk_keep_lo), self.xp.asarray(self.chunk_keep_hi),
            self.xp.asarray(self.chunk_n_global_offset),
            self.xp.asarray(self.wdm_window),
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nf), int(self.Nt), int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            self.xp.asarray(groups["binary_perm"],  dtype=np.int32),
            self.xp.asarray(groups["group_starts"], dtype=np.int32),
            self.xp.asarray(groups["group_ends"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_lo"],   dtype=np.int32),
            self.xp.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
        )

        self.d_h_out = d_h_out
        self.h_h_out = h_h_out
        return -0.5 * (self.d_d + h_h_out - 2.0 * d_h_out)

    def get_swap_ll_wdm(self, params_add, params_remove, wdm_holder,
                        data_index=None, noise_index=None,
                        convert_to_ra_dec: bool = True,
                        grid_dim: int = 0,
                        use_layer_groups: bool = True,
                        group_band_layers: int = 5,
                        margin_layers: int = 0):
        """Swap-proposal 5-way accumulator via chunked-heterodyne.

        Mirrors :meth:`get_ll_wdm` but evaluates the five inner products
        ``<d|h_add>``, ``<d|h_remove>``, ``<h_add|h_add>``,
        ``<h_remove|h_remove>``, ``<h_add|h_remove>`` per binary.

        Returns
        -------
        like_add, like_remove : ndarray
            ``-0.5 * (d_d + <hh> - 2 <dh>)`` for add and remove.
        d_h_add, d_h_remove, add_add, remove_remove, add_remove : ndarray
            Raw inner products (e.g. for cross-term Hastings ratios).
        """
        params_add_tmp = self.xp.asarray(self.xp.atleast_2d(params_add)).copy()
        params_remove_tmp = self.xp.asarray(self.xp.atleast_2d(params_remove)).copy()
        assert params_add_tmp.shape == params_remove_tmp.shape, (
            "params_add and params_remove must have the same shape; got "
            f"{params_add_tmp.shape} vs {params_remove_tmp.shape}"
        )
        num_bin = params_add_tmp.shape[0]
        nparams = int(self._NPARAMS)

        if convert_to_ra_dec:
            for p_tmp in (params_add_tmp, params_remove_tmp):
                lam = p_tmp[:, -2].copy()
                beta = p_tmp[:, -1].copy()
                lam, beta = ecliptic_to_icrs(lam, beta)
                p_tmp[:, -2] = lam
                p_tmp[:, -1] = beta

        if self.backend.name == "fastlisaresponse_jax":
            d_h_a = np.zeros(num_bin); d_h_r = np.zeros(num_bin)
            aa    = np.zeros(num_bin); rr    = np.zeros(num_bin)
            ar    = np.zeros(num_bin)
        else:
            d_h_a = self.xp.zeros(num_bin); d_h_r = self.xp.zeros(num_bin)
            aa    = self.xp.zeros(num_bin); rr    = self.xp.zeros(num_bin)
            ar    = self.xp.zeros(num_bin)

        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_add_in    = params_add_tmp.flatten().copy()
        params_remove_in = params_remove_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._swap_layer_groups(
                params_add_tmp, params_remove_tmp,
                group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_swap_groups(num_bin)

        self._kernel("swap_ll")(
            d_h_a, d_h_r, aa, rr, ar,
            self.cpp_orbits, self.cpp_tdi_config,
            params_add_in, params_remove_in,
            data_index, noise_index,
            self.chunk_t_starts,
            self.chunk_keep_lo, self.chunk_keep_hi,
            self.chunk_n_global_offset,
            self.wdm_window,
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nf), int(self.Nt), int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            np.asarray(groups["binary_perm"],  dtype=np.int32),
            np.asarray(groups["group_starts"], dtype=np.int32),
            np.asarray(groups["group_ends"],   dtype=np.int32),
            np.asarray(groups["group_m_lo"],   dtype=np.int32),
            np.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
            np.asarray(groups["pair_m_lo_b"],  dtype=np.int32),
            np.asarray(groups["pair_m_hi_b"],  dtype=np.int32),
        )

        self.d_h_add_out       = d_h_a
        self.d_h_remove_out    = d_h_r
        self.add_add_out       = aa
        self.remove_remove_out = rr
        self.add_remove_out    = ar

        like_add    = -0.5 * (self.d_d + aa - 2.0 * d_h_a)
        like_remove = -0.5 * (self.d_d + rr - 2.0 * d_h_r)
        return like_add, like_remove, d_h_a, d_h_r, aa, rr, ar

    # ------------------------------------------------------------------
    # Chain-rule gradients of get_ll_wdm / get_swap_ll_wdm
    #
    # Dispatch through ``self.backend.GBComputationGroupWrap()`` exactly
    # like the likelihood methods above. On the JAX backend the
    # gradient is built with ``jax.grad`` over the standalone JAX
    # chunked-het kernel. On the C++ backends the corresponding
    # ``gb_wdm_het_*_grad`` symbol is the canonical hook for a future
    # C++ implementation; until it lands the backend wrap will raise.
    # ------------------------------------------------------------------

    def get_ll_grad_wdm(self, params, wdm_holder,
                        data_index=None, noise_index=None,
                        convert_to_ra_dec: bool = True,
                        grid_dim: int = 0,
                        use_layer_groups: bool = True,
                        group_band_layers: int = 5,
                        margin_layers: int = 0):
        """Chain-rule gradient of :meth:`get_ll_wdm`.

        Dispatches to ``gb_wdm_het_get_ll_grad`` on the backend with
        the same C++-style argument list as :meth:`get_ll_wdm`. The JAX
        backend evaluates the gradient via autograd over the
        chunked-het kernel; the C++ backends raise until the matching
        kernel is implemented.

        Args:
            params: ``(num_bin, nparams)`` array.
            wdm_holder: ``AnalysisContainerArray``.
            data_index, noise_index, convert_to_ra_dec, grid_dim,
            use_layer_groups, group_band_layers, margin_layers: see
                :meth:`get_ll_wdm`.

        Returns:
            ``(num_bin, nparams)`` array, ``grad[i, k] = dL/dtheta_k``.
        """
        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin = params_tmp.shape[0]
        nparams = int(self._NPARAMS)
        assert params_tmp.shape[1] == nparams, (
            f"params has {params_tmp.shape[1]} columns, expected {nparams}")

        if convert_to_ra_dec:
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_in = params_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._layer_groups(
                params_tmp, group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_groups(num_bin)

        if self.backend.name == "fastlisaresponse_jax":
            grad_out = np.zeros(num_bin * nparams, dtype=np.float64)
        else:
            grad_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)

        self._kernel("get_ll_grad")(
            grad_out,
            self.cpp_orbits, self.cpp_tdi_config,
            params_in, data_index, noise_index,
            self.chunk_t_starts,
            self.chunk_keep_lo, self.chunk_keep_hi,
            self.chunk_n_global_offset,
            self.wdm_window,
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nf), int(self.Nt), int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            np.asarray(groups["binary_perm"],  dtype=np.int32),
            np.asarray(groups["group_starts"], dtype=np.int32),
            np.asarray(groups["group_ends"],   dtype=np.int32),
            np.asarray(groups["group_m_lo"],   dtype=np.int32),
            np.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
        )

        return self.xp.asarray(grad_out).reshape(num_bin, nparams)

    def get_swap_ll_grad_wdm(self, params_add, params_remove, wdm_holder,
                             data_index=None, noise_index=None,
                             convert_to_ra_dec: bool = True,
                             grid_dim: int = 0,
                             use_layer_groups: bool = True,
                             group_band_layers: int = 5,
                             margin_layers: int = 0):
        """Chain-rule gradient of :meth:`get_swap_ll_wdm`.

        Dispatches to ``gb_wdm_het_swap_ll_grad`` on the backend.
        Returns ``(grad_add, grad_remove)``, each ``(num_bin, nparams)``.
        """
        params_add_tmp = self.xp.asarray(self.xp.atleast_2d(params_add)).copy()
        params_remove_tmp = self.xp.asarray(self.xp.atleast_2d(params_remove)).copy()
        assert params_add_tmp.shape == params_remove_tmp.shape, (
            f"params_add {params_add_tmp.shape} != "
            f"params_remove {params_remove_tmp.shape}"
        )
        num_bin = params_add_tmp.shape[0]
        nparams = int(self._NPARAMS)
        assert params_add_tmp.shape[1] == nparams, (
            f"params has {params_add_tmp.shape[1]} columns, expected {nparams}")

        if convert_to_ra_dec:
            for p_tmp in (params_add_tmp, params_remove_tmp):
                lam = p_tmp[:, -2].copy()
                beta = p_tmp[:, -1].copy()
                lam, beta = ecliptic_to_icrs(lam, beta)
                p_tmp[:, -2] = lam
                p_tmp[:, -1] = beta

        num_data = num_noise = len(wdm_holder)
        data_index, noise_index = self._prep_indices(
            num_bin, num_data, num_noise, data_index, noise_index)

        params_add_in    = params_add_tmp.flatten().copy()
        params_remove_in = params_remove_tmp.flatten().copy()

        if use_layer_groups:
            groups = self._swap_layer_groups(
                params_add_tmp, params_remove_tmp,
                group_band_layers, margin_layers,
                data_index=data_index, noise_index=noise_index)
        else:
            groups = self._empty_swap_groups(num_bin)

        if self.backend.name == "fastlisaresponse_jax":
            grad_add_out    = np.zeros(num_bin * nparams, dtype=np.float64)
            grad_remove_out = np.zeros(num_bin * nparams, dtype=np.float64)
        else:
            grad_add_out    = self.xp.zeros(num_bin * nparams,
                                            dtype=self.xp.float64)
            grad_remove_out = self.xp.zeros(num_bin * nparams,
                                            dtype=self.xp.float64)

        self._kernel("swap_ll_grad")(
            grad_add_out, grad_remove_out,
            self.cpp_orbits, self.cpp_tdi_config,
            params_add_in, params_remove_in,
            data_index, noise_index,
            self.chunk_t_starts,
            self.chunk_keep_lo, self.chunk_keep_hi,
            self.chunk_n_global_offset,
            self.wdm_window,
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nf), int(self.Nt), int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
            np.asarray(groups["binary_perm"],  dtype=np.int32),
            np.asarray(groups["group_starts"], dtype=np.int32),
            np.asarray(groups["group_ends"],   dtype=np.int32),
            np.asarray(groups["group_m_lo"],   dtype=np.int32),
            np.asarray(groups["group_m_hi"],   dtype=np.int32),
            int(groups["n_groups"]),
            np.asarray(groups["pair_m_lo_b"],  dtype=np.int32),
            np.asarray(groups["pair_m_hi_b"],  dtype=np.int32),
        )

        grad_add    = self.xp.asarray(grad_add_out).reshape(num_bin, nparams)
        grad_remove = self.xp.asarray(grad_remove_out).reshape(num_bin, nparams)
        return grad_add, grad_remove

    def fill_global_wdm(self, params, templates,
                        convert_to_ra_dec: bool = True,
                        data_index=None, factors=None,
                        grid_dim: int = 0):
        """Scatter per-source chunked-heterodyne WDM templates into a global buffer.

        Argument order matches the other ``*_wdm`` methods on this
        class: ``params`` (1D or 2D ndarray; lists auto-promoted via
        ``atleast_2d`` internally) is the first positional argument,
        followed by the output ``templates`` buffer.

        ``templates`` may be supplied as ``(nchannels, Nf, Nt)``,
        ``(num_templates, nchannels, Nf, Nt)``, or already flattened to
        ``(num_templates * nchannels * Nf * Nt,)``. With
        ``force_backend='jax'`` it must be a *numpy* array (not jnp),
        since JAX arrays are immutable -- the kernel writes back into
        the numpy buffer to honour the C++ in-place contract.

        Args:
            params: ``(num_bin, nparams)`` array (1D auto-promoted).
            templates: output buffer (see shapes above). Pre-zero
                before calling -- the kernel accumulates.
            convert_to_ra_dec: ecliptic-to-ICRS conversion for the last
                two parameter columns.
            data_index: per-binary slab index into ``templates``.
                Default = all zeros (single shared template slab).
            factors: per-binary multiplicative factor at the
                accumulation step. Default ``+1``; pass ``-1`` to remove.
            grid_dim: CUDA launch grid size (use 0 to default to
                ``n_chunks``).
        """
        if self.backend.name == "fastlisaresponse_jax":
            assert isinstance(templates, np.ndarray), (
                "On the JAX backend, ``templates`` must be a numpy "
                "ndarray (not jnp.ndarray). The kernel writes back to "
                "it in place to match the C++ contract.")
        else:
            assert isinstance(templates, self.xp.ndarray)

        if templates.ndim == 1:
            per_template = self.nchannels * self.Nf * self.Nt
            num_templates = int(templates.shape[-1] // per_template)
            assert num_templates * per_template == templates.shape[-1], (
                f"templates flat size {templates.shape[-1]} not divisible "
                f"by nchannels*Nf*Nt = {per_template}")
        elif templates.ndim == 3:
            nch, _Nf, _Nt = templates.shape
            assert (nch, _Nf, _Nt) == (self.nchannels, self.Nf, self.Nt)
            num_templates = 1
        elif templates.ndim == 4:
            num_templates, nch, _Nf, _Nt = templates.shape
            assert (nch, _Nf, _Nt) == (self.nchannels, self.Nf, self.Nt)
        else:
            raise ValueError(
                "templates must be 3D (nchannels, Nf, Nt), 4D "
                "(num_templates, nchannels, Nf, Nt), or flat 1D.")

        params_tmp = self.xp.atleast_2d(self.xp.asarray(params)).copy()
        num_bin = params_tmp.shape[0]
        nparams = int(self._NPARAMS)
        assert params_tmp.shape[1] == nparams, (
            f"params has {params_tmp.shape[1]} columns, expected {nparams}")

        if convert_to_ra_dec:
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        params_in = params_tmp.flatten().copy()

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        assert int(data_index.max()) < num_templates

        # factor +1 (add) / -1 (remove). Mirrors
        # gbgpu.generate_global_template's factors API.
        if factors is None:
            factors = self.xp.ones(num_bin, dtype=self.xp.float64)
        else:
            factors = self.xp.ascontiguousarray(
                self.xp.asarray(factors, dtype=self.xp.float64))
            assert factors.shape == (num_bin,), (
                f"factors must have shape ({num_bin},), got {factors.shape}")

        self._kernel("fill_global")(
            templates,
            self.cpp_orbits, self.cpp_tdi_config,
            params_in, factors,
            self.chunk_t_starts,
            self.chunk_keep_lo, self.chunk_keep_hi,
            self.chunk_n_global_offset,
            self.wdm_window,
            self.n_chunks, int(num_bin), int(nparams),
            int(self.Nf), int(self.Nt), int(self.Nt_sub), int(self.log2_Nt_sub),
            int(self.N_sparse), int(self.log2_N_sparse),
            int(self.nchannels), int(self.n_rfft_chunk),
            float(self.T_chunk), float(self.dt),
            float(self.T), float(self.t_ref),
            float(self.resolved_tukey_alpha), int(grid_dim),
            int(self.N_cp_sig), int(self.N_cp_orbit),
        )


class SOBBHWDMComputations(GBWDMComputations):
    """Stellar-origin BBH analog of :class:`GBWDMComputations`.

    Same chunked-heterodyne pipeline as the GB version, only the
    routing constants differ:

    * Backend wrap class:
      ``SOBBHComputationGroupWrap`` instead of ``GBComputationGroupWrap``.
    * Kernel-name family:
      ``sobbh_wdm_het_{fill_global, get_ll, swap_ll, ...}``.
    * Per-source parameter count: ``11`` (vs 9 for GB).
    * Carrier-frequency column for layer-grouping: ``params[:, 5] = f_low``.

    Source intrinsic amp/phase are computed by the C++ side's
    ``SOBBHTDIonTheFly`` pointer (and by :class:`JaxSOBBHSource` on the
    JAX backend), so the only difference at the Python layer is which
    backend wrap class is invoked and how the parameter vector is
    interpreted -- everything else (chunk geometry, WDM window,
    layer-grouping, narrow-band dispatch, gradient hooks, fill_global)
    is inherited unchanged.

    Param order (matches ``SOBBHTDIonTheFly``):
        ``(m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta)``.
    """

    _WRAP_ATTR = "SOBBHComputationGroupWrap"
    _METHOD_PREFIX = "sobbh_wdm_het"
    _NPARAMS = 11
    _F0_PARAM_INDEX = 5   # SOBBHTDIonTheFly: params[5] = f_low


class GBFDComputations(FastLISAResponseParallelModule):
    """Frequency-domain heterodyne analog of :class:`GBWDMComputations`.

    Mirrors the WDM ``get_ll_wdm`` / ``get_swap_ll_wdm`` / ``fill_global``
    surface for the sparse FD heterodyne kernel.  All inner products are
    accumulated in C using the standard lisatools FD inner product
    ``(a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] * df``,
    so calling :func:`lisatools.diagnostic.inner_product` on the same data /
    invC / template gives an identical answer up to floating-point round-off.
    """

    def __init__(self, T, t_ref, t_start, N_sparse, df,
                 data_fd, invC,
                 orbits=None, tdi_config=None, force_backend=None,
                 d_d=0.0, tdi_type="XYZ", ind_min=None, ind_max=None):
        super().__init__(force_backend=force_backend)
        if N_sparse < 1 or (N_sparse & (N_sparse - 1)) != 0:
            raise ValueError("N_sparse must be a power of two.")
        if abs(float(t_start) - float(t_ref)) > 1e-9:
            raise ValueError("GBFDComputations requires t_start == t_ref so "
                             "the heterodyne phase factor is unity.")
        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError("tdi_type must be one of 'XYZ', 'AET', 'AE'.")

        self.T = float(T)
        self.t_ref = float(t_ref)
        self.t_start = float(t_start)
        self.N_sparse = int(N_sparse)
        self.df = float(df)
        self.d_d = float(d_d)
        self.tdi_type = tdi_type

        self.orbits = orbits
        self.tdi_config = tdi_config

        data_fd = self.xp.ascontiguousarray(data_fd)
        if data_fd.ndim != 3:
            raise ValueError(
                "data_fd must have shape (num_data, nchannels, n_rfft).")
        self.num_data, self.nchannels, self.n_rfft = data_fd.shape

        invC = self.xp.ascontiguousarray(invC, dtype=float)
        if tdi_type == "XYZ":
            if invC.ndim != 4 or invC.shape[1:] != (
                    self.nchannels, self.nchannels, self.n_rfft):
                raise ValueError(
                    f"For tdi_type=XYZ, invC must have shape "
                    f"(num_noise, {self.nchannels}, {self.nchannels}, "
                    f"{self.n_rfft}); got {invC.shape}.")
        else:
            if invC.ndim != 3 or invC.shape[1:] != (
                    self.nchannels, self.n_rfft):
                raise ValueError(
                    f"For tdi_type={tdi_type}, invC must have shape "
                    f"(num_noise, {self.nchannels}, {self.n_rfft}); "
                    f"got {invC.shape}.")
        self.num_noise = invC.shape[0]

        if ind_min is None: ind_min = 0
        if ind_max is None: ind_max = self.n_rfft - 1
        self.ind_min = int(ind_min)
        self.ind_max = int(ind_max)

        self._data_fd = data_fd
        self._invC    = invC

        self.cpp_fd = self.backend.FDDomainWrap(
            data_fd.reshape(-1),
            invC.reshape(-1),
            self.n_rfft, self.nchannels,
            self.num_data, self.num_noise,
            self.ind_min, self.ind_max, self.df,
        )

    @property
    def xp(self): return self.backend.xp

    @property
    def orbits(self): return self._orbits
    @orbits.setter
    def orbits(self, o):
        if o is None:
            o = EqualArmlengthOrbits()
        elif not isinstance(o, Orbits) and issubclass(o, Orbits):
            o = o()
        else:
            assert isinstance(o, Orbits)
        self._orbits = deepcopy(o)
        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)
        self.cpp_orbits = self.backend.OrbitsWrap(
            *self._orbits.pycppdetector_args)

    @property
    def tdi_config(self): return self._tdi_config
    @tdi_config.setter
    def tdi_config(self, tc):
        if tc is None:
            tc = TDIConfig("1st generation")
        elif isinstance(tc, str):
            tc = TDIConfig(tc)
        elif not isinstance(tc, TDIConfig):
            raise ValueError("tdi_config must be TDIConfig, str, or None.")
        self._tdi_config = tc
        self.cpp_tdi_config = self.backend.TDIConfigWrap(
            *self._tdi_config.pytdiconfig_args)

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _t for _t in cls.GPU_RECOMMENDED()]

    def _prep_params(self, params, convert_to_ra_dec):
        p = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        if convert_to_ra_dec:
            lam = p[:, -2].copy(); beta = p[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            p[:, -2] = lam; p[:, -1] = beta
        return p

    def get_ll_fd(self, params, data_index=None, noise_index=None,
                  convert_to_ra_dec: bool = True):
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin = p.shape[0]
        d_h_out = self.xp.zeros(num_bin)
        h_h_out = self.xp.zeros(num_bin)
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        self.backend.GBComputationGroupWrap().gb_fd_get_ll(
            d_h_out, h_h_out,
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            p.flatten().copy(),
            data_index, noise_index,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
        )
        self.d_h_out = d_h_out
        self.h_h_out = h_h_out
        return -0.5 * (self.d_d + h_h_out - 2.0 * d_h_out)

    def get_swap_ll_fd(self, params_add, params_remove,
                       data_index=None, noise_index=None,
                       convert_to_ra_dec: bool = True):
        pa = self._prep_params(params_add, convert_to_ra_dec)
        pr = self._prep_params(params_remove, convert_to_ra_dec)
        num_bin = pa.shape[0]
        assert pr.shape[0] == num_bin

        d_h_a = self.xp.zeros(num_bin); d_h_r = self.xp.zeros(num_bin)
        aa    = self.xp.zeros(num_bin); rr    = self.xp.zeros(num_bin)
        ar    = self.xp.zeros(num_bin)

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        self.backend.GBComputationGroupWrap().gb_fd_swap_ll(
            d_h_a, d_h_r, aa, rr, ar,
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            pa.flatten().copy(), pr.flatten().copy(),
            data_index, noise_index,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
        )
        like_add = -0.5 * (self.d_d + aa - 2.0 * d_h_a)
        like_rem = -0.5 * (self.d_d + rr - 2.0 * d_h_r)
        return like_add, like_rem, d_h_a, d_h_r, aa, rr, ar

    def fill_global(self, params, templates, data_index=None, factors=None,
                    convert_to_ra_dec: bool = True):
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin = p.shape[0]
        if templates.ndim != 3 or templates.shape[1:] != (
                self.nchannels, self.n_rfft):
            raise ValueError(
                f"templates must be (num_templates, {self.nchannels}, "
                f"{self.n_rfft}) complex; got {templates.shape}.")
        num_templates = templates.shape[0]
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if factors is None:
            factors = self.xp.ones(num_bin, dtype=float)
        else:
            factors = self.xp.asarray(factors, dtype=float)
        assert int(data_index.max()) < num_templates

        self.backend.GBComputationGroupWrap().gb_fd_fill_global(
            templates.reshape(-1),
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            p.flatten().copy(), data_index, factors,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
        )

    # ------------------------------------------------------------------
    # Chain-rule gradients of gb_fd_get_ll / gb_fd_swap_ll
    #
    # Mirrors GBWDMComputations.get_ll_grad_wdm / get_swap_ll_grad_wdm at the
    # API surface: same _DEFAULT_PARAM_EPS table, same _resolve_eps_and_scales
    # logic (param_eps + param_scales + param_eps_relative), same convention
    # that supplying param_scales returns the gradient in rescaled coordinates
    # eta = theta / Delta_theta.
    # ------------------------------------------------------------------
    _DEFAULT_PARAM_EPS = (
        1.0e-25,   # amp
        2.0e-14,   # f0    (Hz)
        1.0e-21,   # fdot  (Hz/s)
        1.0e-28,   # fddot (Hz/s^2)
        1.0e-6,    # phi0
        1.0e-6,    # iota
        1.0e-6,    # psi
        1.0e-6,    # lam (or RA after convert)
        1.0e-6,    # beta (or DEC after convert)
    )

    def _default_param_eps(self, nparams=9):
        eps = self.xp.asarray(self._DEFAULT_PARAM_EPS[:nparams],
                              dtype=self.xp.float64)
        if eps.shape[0] != nparams:
            extra = self.xp.full(nparams - eps.shape[0],
                                 eps[-1].item(), dtype=self.xp.float64)
            eps = self.xp.concatenate([eps, extra])
        return eps

    def _resolve_eps_and_scales(self, nparams,
                                param_eps, param_scales, param_eps_relative):
        if param_scales is not None:
            scales = self.xp.asarray(param_scales, dtype=self.xp.float64)
            assert scales.shape[0] == nparams, (
                f"param_scales length {scales.shape[0]} != nparams {nparams}"
            )
            if param_eps is None:
                eps_theta = scales * float(param_eps_relative)
            else:
                eps_theta = self.xp.asarray(param_eps, dtype=self.xp.float64)
                assert eps_theta.shape[0] == nparams
            return eps_theta, scales

        if param_eps is None:
            eps_theta = self._default_param_eps(nparams)
        else:
            eps_theta = self.xp.asarray(param_eps, dtype=self.xp.float64)
            assert eps_theta.shape[0] == nparams, (
                f"param_eps length {eps_theta.shape[0]} != nparams {nparams}"
            )
        return eps_theta, None

    def get_ll_grad_fd(self, params,
                       param_eps=None,
                       param_scales=None,
                       param_eps_relative=1.0e-6,
                       data_index=None, noise_index=None,
                       convert_to_ra_dec: bool = True):
        """Chain-rule gradient of :meth:`get_ll_fd`.

        See :meth:`GBWDMComputations.get_ll_grad_wdm` for the meaning of
        ``param_scales`` / ``param_eps_relative``: with ``param_scales``
        provided the returned gradient is in rescaled coordinates
        ``eta_k = theta_k / Delta_theta_k`` and the kernel uses a uniform FD
        step ``eps_theta_k = param_eps_relative * Delta_theta_k``.

        Returns
        -------
        grad : (num_bin, nparams) xp.ndarray
            ``dL/dtheta_k`` (default) or ``dL/d(eta_k)`` when
            ``param_scales`` is supplied.
        """
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin, nparams = p.shape

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        eps_theta, scales = self._resolve_eps_and_scales(
            nparams, param_eps, param_scales, param_eps_relative,
        )

        grad_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        self.backend.GBComputationGroupWrap().gb_fd_get_ll_grad(
            grad_out,
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            p.flatten().copy(),
            data_index, noise_index,
            eps_theta,
            num_bin, nparams, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
        )
        grad = grad_out.reshape(num_bin, nparams)
        if scales is not None:
            grad = grad * scales[None, :]
        return grad

    def get_swap_ll_grad_fd(self, params_add, params_remove,
                            param_eps_add=None, param_eps_remove=None,
                            param_scales_add=None, param_scales_remove=None,
                            param_eps_relative=1.0e-6,
                            data_index=None, noise_index=None,
                            convert_to_ra_dec: bool = True):
        """Chain-rule gradient of :meth:`get_swap_ll_fd`.

        Returns ``(grad_add, grad_remove)``, the per-binary derivatives of
        ``ll_diff = L(after swap) - L(before swap)`` with respect to
        ``theta_add`` and ``theta_remove`` respectively.  Rescaling
        semantics match :meth:`get_ll_grad_fd`.
        """
        pa = self._prep_params(params_add, convert_to_ra_dec)
        pr = self._prep_params(params_remove, convert_to_ra_dec)
        assert pa.shape == pr.shape, (
            f"params_add {pa.shape} != params_remove {pr.shape}"
        )
        num_bin, nparams = pa.shape

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        eps_theta_add, scales_add = self._resolve_eps_and_scales(
            nparams, param_eps_add, param_scales_add, param_eps_relative,
        )
        eps_theta_remove, scales_remove = self._resolve_eps_and_scales(
            nparams, param_eps_remove, param_scales_remove, param_eps_relative,
        )

        grad_add_out    = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        grad_remove_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        self.backend.GBComputationGroupWrap().gb_fd_swap_ll_grad(
            grad_add_out, grad_remove_out,
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            pa.flatten().copy(), pr.flatten().copy(),
            data_index, noise_index,
            eps_theta_add, eps_theta_remove,
            num_bin, nparams, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
        )
        grad_add = grad_add_out.reshape(num_bin, nparams)
        grad_remove = grad_remove_out.reshape(num_bin, nparams)
        if scales_add is not None:
            grad_add = grad_add * scales_add[None, :]
        if scales_remove is not None:
            grad_remove = grad_remove * scales_remove[None, :]
        return grad_add, grad_remove
