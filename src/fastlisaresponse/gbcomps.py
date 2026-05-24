import numpy as np

from .utils.parallelbase import FastLISAResponseParallelModule
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import Orbits, EqualArmlengthOrbits
from copy import deepcopy
from lisatools.domains import WDMLookupTable
from .response import ecliptic_to_icrs

# Seconds per Julian year, used to convert `coarse_pts_per_year` into the
# `coarse_dt` argument the spline-path C wraps consume.
_SECONDS_PER_YEAR = 365.25 * 86400.0

class GBWDMComputations(FastLISAResponseParallelModule):
    def __init__(self, wdm_lookup_table, T, t_ref, orbits=None, tdi_config=None, force_backend=None, d_d=0.0, tdi_type="XYZ"):

        super().__init__(force_backend=force_backend)
        # setup orbits
        self.orbits = orbits
         # setup TDI info
        self.tdi_config = tdi_config
        # setup WDM c class
        self.wdm_lookup_table = wdm_lookup_table
        self.T = T
        self.t_ref = t_ref
        self.d_d = d_d
        # Which kernel branch get_ll/get_swap_ll/fill_global drive:
        #   "XYZ" -> full 3x3 cross-channel inverse covariance per pixel.
        #   "AET" -> three orthogonal channels, diagonal noise per pixel.
        #   "AE"  -> two orthogonal channels (T dropped), diagonal noise.
        # The choice has to match the wdm_holder's noise buffer layout that
        # ``get_pixel_noise_value{,_cross_channel}`` consumes inside the kernel.
        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError(f"tdi_type must be one of 'XYZ', 'AET', 'AE'; got {tdi_type!r}.")
        self.tdi_type = tdi_type
        
    @property
    def tdi_config(self) -> TDIConfig:
        return self._tdi_config
    
    @tdi_config.setter
    def tdi_config(self, tdi_config: TDIConfig):
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation")
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config)
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
        """Set response orbits."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()
        
        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            # assumed default arguments if not initialized as input
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        self._orbits = deepcopy(orbits)

        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)

        self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)

    @property
    def wdm_lookup_table(self) -> object:
        return self._wdm_lookup_table

    @wdm_lookup_table.setter
    def wdm_lookup_table(self, wdm_lookup_table: WDMLookupTable) -> None:
        """Set wdm lookup table.

        Three table layouts are supported, picked by
        ``wdm_lookup_table.build_kind``:

          * ``'per_n'``        → shape ``(Nt, num_fdot, num_f)`` (legacy)
          * ``'n_ref_only'``   → shape ``(num_fdot, num_f)`` (Plan A, real)
          * ``'n_ref_complex'``→ same shape as ``'n_ref_only'`` but stored
                                as ONE complex table whose Re/Im equal the
                                real path's (cos, sin) — 2x faster to
                                build. Maps to ``LOOKUP_N_REF_ONLY`` on
                                the C side; we just split table_cx into
                                Re/Im before shipping it.

        The kind is forwarded to the C++ ``WaveletLookupTable`` as the
        ``kind`` int (matches the ``LookupKind`` enum in
        ``TDIonTheFly.hh`` — 0 = PER_N, 1 = N_REF_ONLY).
        """

        self._wdm_lookup_table = wdm_lookup_table

        Nt = wdm_lookup_table.settings.Nt
        num_fdot = wdm_lookup_table.fdot_steps
        num_f = wdm_lookup_table.f_steps
        build_kind = getattr(wdm_lookup_table, "build_kind", "per_n")
        if build_kind in ("n_ref_only", "n_ref_complex"):
            expected_shape = (num_fdot, num_f)
            kind_int = 1
        elif build_kind == "per_n":
            expected_shape = (Nt, num_fdot, num_f)
            kind_int = 0
        else:
            raise ValueError(
                f"Unknown WDMLookupTable.build_kind={build_kind!r}; "
                "expected 'per_n', 'n_ref_only', or 'n_ref_complex'."
            )

        if build_kind == "n_ref_complex":
            # Split the stored complex table into real cos / sin arrays for
            # the C kernel. ``Re(table_cx) == table_cos`` and
            # ``Im(table_cx) == table_sin`` of the real-path build (build
            # applies the same heroics in both paths), so the C side
            # consumes these identically.
            assert wdm_lookup_table.table_cx.shape == expected_shape, (
                f"table_cx shape {wdm_lookup_table.table_cx.shape} != "
                f"{expected_shape} for build_kind={build_kind!r}"
            )
            _cos_src = self.xp.real(self.xp.asarray(wdm_lookup_table.table_cx))
            _sin_src = self.xp.imag(self.xp.asarray(wdm_lookup_table.table_cx))
        else:
            assert wdm_lookup_table.table_cos.shape == expected_shape, (
                f"table_cos shape {wdm_lookup_table.table_cos.shape} != "
                f"{expected_shape} for build_kind={build_kind!r}"
            )
            assert wdm_lookup_table.table_sin.shape == expected_shape, (
                f"table_sin shape {wdm_lookup_table.table_sin.shape} != "
                f"{expected_shape} for build_kind={build_kind!r}"
            )
            _cos_src = self.xp.asarray(wdm_lookup_table.table_cos)
            _sin_src = self.xp.asarray(wdm_lookup_table.table_sin)

        # ``jax.numpy`` doesn't expose ``ascontiguousarray``; ``jnp.asarray``
        # already returns a contiguous immutable buffer. For the numpy /
        # cupy backends we keep the explicit contiguous coercion.
        if hasattr(self.xp, "ascontiguousarray"):
            self.c_nm_all = self.xp.ascontiguousarray(_cos_src)
            self.s_nm_all = self.xp.ascontiguousarray(_sin_src)
        else:
            self.c_nm_all = _cos_src
            self.s_nm_all = _sin_src

        delta_f = wdm_lookup_table.f_vals_norm[1] - wdm_lookup_table.f_vals_norm[0]
        try:
            delta_fdot = wdm_lookup_table.fdot_vals[1] - wdm_lookup_table.fdot_vals[0]
        except IndexError:
            # this happens when there is no fdot
            delta_fdot = 1.0

        self.cpp_wdm_lookup_table = self.backend.WaveletLookupTableWrap(
            self.c_nm_all,
            self.s_nm_all,
            wdm_lookup_table.f_steps,
            wdm_lookup_table.fdot_steps,
            delta_f,  # NOT .layer_df (that is the WDM basis info)
            delta_fdot,
            wdm_lookup_table.f_vals_norm.min().item(),
            wdm_lookup_table.fdot_vals.min().item(),
            wdm_lookup_table.settings.layer_df,
            wdm_lookup_table.settings.layer_dt,
            wdm_lookup_table.settings.Nf,  # calculates Nf_active inside
            wdm_lookup_table.settings.Nt,  # calculates Nt_active inside
            wdm_lookup_table.nchannels,
            wdm_lookup_table.settings.ind_min_t,
            wdm_lookup_table.settings.ind_max_t,
            wdm_lookup_table.settings.ind_min_f,
            wdm_lookup_table.settings.ind_max_f,
            int(wdm_lookup_table.m_ref),
            int(getattr(wdm_lookup_table, "n_ref", 0)),
            kind_int,
        )

    @classmethod
    def supported_backends(cls):
        # GPU_RECOMMENDED_WITH_JAX appends 'jax' to the CPU/GPU options
        # so this class can dispatch to the pure JAX backend in
        # fastlisaresponse.jax via force_backend='jax'.
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED_WITH_JAX()]

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None, convert_to_ra_dec: bool = True,
                   use_spline: bool = False, coarse_pts_per_year: int = 256):
        """Per-binary (d|h) and (h|h) accumulated as <h | h> := -2 (d|h) + (h|h) likelihood pieces.

        Set ``use_spline=True`` to dispatch to ``gb_wdm_spline_get_ll`` which
        replaces per-WDM-pixel fast_wdm_inner calls with cubic-spline
        interpolation of get_tdi outputs on a coarse uniform time grid of
        ``coarse_pts_per_year`` points per Julian year.
        """

        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin = params_tmp.shape[0]

        # The JAX backend's GBComputationGroupWrap can't mutate
        # immutable jnp arrays; allocate numpy host buffers in that
        # case and rebind ``d_h_out``/``h_h_out`` to jnp at the end.
        if self.backend.name == "fastlisaresponse_jax":
            self.d_h_out = np.zeros(num_bin)
            self.h_h_out = np.zeros(num_bin)
        else:
            self.d_h_out = self.xp.zeros(num_bin)
            self.h_h_out = self.xp.zeros(num_bin)

        if convert_to_ra_dec:
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        num_data = num_noise = len(wdm_holder)
        
        # TODO: move this part
        # TODO: need to check for num_data, num_noise
        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.settings.layer_df, 
            self.wdm_lookup_table.settings.layer_dt,
            self.wdm_lookup_table.settings.Nf, # calculates Nf_active inside
            self.wdm_lookup_table.settings.Nt, # calculates Nt_active inside
            self.tdi_config.nchannels,
            self.wdm_lookup_table.settings.ind_min_t,
            self.wdm_lookup_table.settings.ind_max_t,
            self.wdm_lookup_table.settings.ind_min_f,
            self.wdm_lookup_table.settings.ind_max_f,
            num_data, 
            num_noise
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif data_index.dtype == self.xp.int64:
            _data_index = data_index.copy().astype(self.xp.int32)
            del data_index
            data_index = _data_index
            
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif noise_index.dtype == self.xp.int64:
            _noise_index = noise_index.copy().astype(self.xp.int32)
            del noise_index
            noise_index = _noise_index

        assert noise_index.dtype == self.xp.int32
        
        assert data_index.max() < num_data
        assert noise_index.max() < num_noise
        nparams = 9

        params_in = params_tmp.flatten().copy()

        if use_spline:
            coarse_dt = _SECONDS_PER_YEAR / float(coarse_pts_per_year)
            self.backend.GBComputationGroupWrap().gb_wdm_spline_get_ll(
                self.d_h_out,
                self.h_h_out,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_in,
                data_index,
                noise_index,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                coarse_dt,
            )
        else:
            deriv_delta_t = 500.0  # seconds
            self.backend.GBComputationGroupWrap().gb_wdm_get_ll(
                self.d_h_out,
                self.h_h_out,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_in,
                data_index,
                noise_index,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                deriv_delta_t
            )

        like_out = -1. / 2. * (self.d_d + self.h_h_out - 2 * self.d_h_out)
        # TODO: phase maximize
        return like_out

    def get_swap_ll_wdm(self, params_add, params_remove, wdm_holder, data_index=None, noise_index=None, convert_to_ra_dec: bool = True,
                        use_spline: bool = False, coarse_pts_per_year: int = 256):
        """Swap-proposal likelihood pieces for an 'add' and a 'remove' template.

        Mirrors :meth:`get_ll_wdm` but evaluates the five inner products
        <d|h_add>, <d|h_remove>, <h_add|h_add>, <h_remove|h_remove>,
        <h_add|h_remove> for each binary in parallel. Used by RJMCMC swap moves
        where a single proposal replaces one source with another.

        Set ``use_spline=True`` to dispatch to ``gb_wdm_spline_swap_ll``.

        Returns
        -------
        like_add : xp.ndarray
            -0.5 * (d_d + <h_add|h_add> - 2 <d|h_add>)
        like_remove : xp.ndarray
            -0.5 * (d_d + <h_remove|h_remove> - 2 <d|h_remove>)
        d_h_add, d_h_remove, add_add, remove_remove, add_remove : xp.ndarray
            The raw inner products, useful for evaluating Hastings ratios that
            include the cross term <h_add|h_remove>.
        """
        params_add_tmp = self.xp.asarray(self.xp.atleast_2d(params_add)).copy()
        params_remove_tmp = self.xp.asarray(self.xp.atleast_2d(params_remove)).copy()
        assert params_add_tmp.shape == params_remove_tmp.shape, (
            "params_add and params_remove must have the same shape; got "
            f"{params_add_tmp.shape} vs {params_remove_tmp.shape}"
        )
        num_bin = params_add_tmp.shape[0]

        # See note in get_ll_wdm: the JAX-backend computation group
        # mutates host (numpy) buffers; jnp arrays would error out.
        if self.backend.name == "fastlisaresponse_jax":
            self.d_h_add_out = np.zeros(num_bin)
            self.d_h_remove_out = np.zeros(num_bin)
            self.add_add_out = np.zeros(num_bin)
            self.remove_remove_out = np.zeros(num_bin)
            self.add_remove_out = np.zeros(num_bin)
        else:
            self.d_h_add_out = self.xp.zeros(num_bin)
            self.d_h_remove_out = self.xp.zeros(num_bin)
            self.add_add_out = self.xp.zeros(num_bin)
            self.remove_remove_out = self.xp.zeros(num_bin)
            self.add_remove_out = self.xp.zeros(num_bin)

        if convert_to_ra_dec:
            for params_tmp in (params_add_tmp, params_remove_tmp):
                lam = params_tmp[:, -2].copy()
                beta = params_tmp[:, -1].copy()
                lam, beta = ecliptic_to_icrs(lam, beta)
                params_tmp[:, -2] = lam
                params_tmp[:, -1] = beta

        num_data = num_noise = len(wdm_holder)

        # TODO: move this part
        # TODO: need to check for num_data, num_noise
        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.settings.layer_df,
            self.wdm_lookup_table.settings.layer_dt,
            self.wdm_lookup_table.settings.Nf,  # calculates Nf_active inside
            self.wdm_lookup_table.settings.Nt,  # calculates Nt_active inside
            self.tdi_config.nchannels,
            self.wdm_lookup_table.settings.ind_min_t,
            self.wdm_lookup_table.settings.ind_max_t,
            self.wdm_lookup_table.settings.ind_min_f,
            self.wdm_lookup_table.settings.ind_max_f,
            num_data,
            num_noise,
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif data_index.dtype == self.xp.int64:
            _data_index = data_index.copy().astype(self.xp.int32)
            del data_index
            data_index = _data_index

        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif noise_index.dtype == self.xp.int64:
            _noise_index = noise_index.copy().astype(self.xp.int32)
            del noise_index
            noise_index = _noise_index

        assert noise_index.dtype == self.xp.int32

        assert data_index.max() < num_data
        assert noise_index.max() < num_noise
        nparams = 9

        params_add_in = params_add_tmp.flatten().copy()
        params_remove_in = params_remove_tmp.flatten().copy()

        if use_spline:
            coarse_dt = _SECONDS_PER_YEAR / float(coarse_pts_per_year)
            self.backend.GBComputationGroupWrap().gb_wdm_spline_swap_ll(
                self.d_h_add_out,
                self.d_h_remove_out,
                self.add_add_out,
                self.remove_remove_out,
                self.add_remove_out,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_add_in,
                params_remove_in,
                data_index,
                noise_index,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                coarse_dt,
            )
        else:
            deriv_delta_t = 500.0  # seconds
            self.backend.GBComputationGroupWrap().gb_wdm_swap_ll(
                self.d_h_add_out,
                self.d_h_remove_out,
                self.add_add_out,
                self.remove_remove_out,
                self.add_remove_out,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_add_in,
                params_remove_in,
                data_index,
                noise_index,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                deriv_delta_t,
            )

        like_add = -1. / 2. * (self.d_d + self.add_add_out - 2 * self.d_h_add_out)
        like_remove = -1. / 2. * (self.d_d + self.remove_remove_out - 2 * self.d_h_remove_out)
        # TODO: phase maximize
        return (
            like_add,
            like_remove,
            self.d_h_add_out,
            self.d_h_remove_out,
            self.add_add_out,
            self.remove_remove_out,
            self.add_remove_out,
        )

    # ------------------------------------------------------------------
    #  Chain-rule gradients of get_ll_wdm / get_swap_ll_wdm
    #
    #  These mirror the corresponding likelihood methods above and call the
    #  C++/CUDA gradient kernels in TDIonTheFly.cu, which compute, per binary
    #  and per parameter k,
    #
    #      dL/dtheta_k = 4 * sum_{m,n,c} (w_data - w_h)_{m n c}
    #                                  * (dw_h/dtheta_k)_{m n c} * N^{-1}_{m n c}
    #
    #  via per-pixel central differences on top of the existing fast_wdm_inner
    #  -> get_wdm_in_channel_over_layers pipeline.  The numerical step size is
    #  controlled by ``param_eps`` (one value per parameter); pass <= 0 to
    #  freeze a parameter.
    # ------------------------------------------------------------------

    # default central-difference step sizes for the 9 GB parameters.
    #
    # The optimal step size for central FD on a function with effective
    # angular frequency omega is roughly  eps_opt ~ (machine_eps)^{1/3} / omega
    # which balances truncation O(eps^2 omega^2) against round-off
    # O(machine_eps / (eps * omega)).  For GB parameters the relevant omega
    # comes from the phase term  2 pi f0 t  evaluated at t ~ T_obs:
    #
    #    omega_f0  ~ 2 pi T_obs           ~ 2e8 rad/Hz at T_obs = 1 yr
    #    omega_fdot ~ pi T_obs^2          ~ 3e15 rad / (Hz/s)
    #    omega_fddot ~ (pi/3) T_obs^3     ~ 3e22 rad / (Hz/s^2)
    #
    # so an eps_relative ~ (1e-16)^{1/3} ~ 5e-6 in the phase derivative
    # translates to absolute steps eps_k ~ 5e-6 / omega_k :
    #
    #    eps_f0    ~ 2e-14
    #    eps_fdot  ~ 1e-21
    #    eps_fddot ~ 1e-28
    #
    # For the angle parameters (phi0, iota, psi, lam, beta) the effective
    # omega is O(1), so eps ~ 1e-6 is appropriate.  For amp the dependence
    # is at most quadratic (in h_h) so truncation is essentially zero and
    # eps just needs to keep round-off down.  We use a *relative* amp step
    # (1e-3 of amp at evaluation time) handled by the caller -- the absolute
    # default below assumes amp ~ 1e-22.
    _DEFAULT_PARAM_EPS = (
        1.0e-25,   # amp                       (absolute; ~ 1e-3 * amp)
        2.0e-14,   # f0    (Hz)                ~ (eps_rel / 2*pi*T_obs)
        1.0e-21,   # fdot  (Hz/s)
        1.0e-28,   # fddot (Hz/s^2)
        1.0e-6,    # phi0
        1.0e-6,    # iota
        1.0e-6,    # psi
        1.0e-6,    # lam (or RA after convert)
        1.0e-6,    # beta (or DEC after convert)
    )

    def _default_param_eps(self, nparams=9):
        eps = self.xp.asarray(self._DEFAULT_PARAM_EPS[:nparams], dtype=self.xp.float64)
        if eps.shape[0] != nparams:
            # extend with last value if the user supplies extra params (e.g. third-body)
            extra = self.xp.full(nparams - eps.shape[0], eps[-1].item(), dtype=self.xp.float64)
            eps = self.xp.concatenate([eps, extra])
        return eps

    def _resolve_eps_and_scales(self, nparams, param_eps, param_scales, param_eps_relative):
        """Compute eps_theta to pass to the C kernel and the scale vector for
        post-multiplication of the returned gradient.

        Convention
        ----------
        With ``param_scales = Delta_theta = (theta_max - theta_min)`` (or any
        natural per-parameter width) and ``param_eps_relative = eps_rel`` we
        work in the rescaled coordinate ``eta_k = theta_k / Delta_theta_k``:

            eps_theta_k = eps_rel * Delta_theta_k   (FD step the kernel uses)
            grad_eta_k  = Delta_theta_k * grad_theta_k    (returned gradient)

        With ``param_scales is None`` the routine falls back to the raw
        ``param_eps`` argument (or ``_DEFAULT_PARAM_EPS`` if that is also
        None), and the returned gradient is ``dL/dtheta`` -- the legacy
        behavior, unchanged.
        """
        if param_scales is not None:
            scales = self.xp.asarray(param_scales, dtype=self.xp.float64)
            assert scales.shape[0] == nparams, (
                f"param_scales length {scales.shape[0]} != nparams {nparams}"
            )
            if param_eps is None:
                eps_theta = scales * float(param_eps_relative)
            else:
                # caller wants a specific eps in original units; still scale the
                # *output* gradient back to eta space at the end.
                eps_theta = self.xp.asarray(param_eps, dtype=self.xp.float64)
                assert eps_theta.shape[0] == nparams
            return eps_theta, scales

        # no scaling: legacy behavior
        if param_eps is None:
            eps_theta = self._default_param_eps(nparams)
        else:
            eps_theta = self.xp.asarray(param_eps, dtype=self.xp.float64)
            assert eps_theta.shape[0] == nparams, (
                f"param_eps length {eps_theta.shape[0]} != nparams {nparams}"
            )
        return eps_theta, None

    def get_ll_grad_wdm(self, params, wdm_holder,
                        param_eps=None,
                        param_scales=None,
                        param_eps_relative=1.0e-6,
                        data_index=None, noise_index=None,
                        convert_to_ra_dec: bool = True,
                        use_spline: bool = False, coarse_pts_per_year: int = 256):
        """Chain-rule gradient of :meth:`get_ll_wdm`.

        Parameters
        ----------
        params : array, (num_bin, nparams)
            Galactic-binary parameters per binary.
        wdm_holder : AnalysisContainerArray
        param_eps : array, (nparams,), optional
            Per-parameter central-difference step *in original units*.  Use
            this for fine control over the kernel FD step.  If both
            ``param_eps`` and ``param_scales`` are supplied, ``param_eps`` wins
            for the FD step; ``param_scales`` still controls the output
            gradient normalisation.  Default: ``_DEFAULT_PARAM_EPS`` (when
            ``param_scales`` is also None).
        param_scales : array, (nparams,), optional
            Per-parameter natural width ``Delta_theta_k`` (e.g.
            ``theta_max - theta_min``).  When supplied:

              * the C-kernel FD step is set uniformly in the rescaled
                coordinate eta_k = theta_k / Delta_theta_k via
                eps_theta_k = param_eps_relative * Delta_theta_k;
              * the returned gradient is in rescaled space,
                ``dL/d(eta_k) = Delta_theta_k * dL/d(theta_k)``.

            This is the recommended path for samplers / Newton-CG / Fisher
            mass-matrix work: the 9 gradient components become comparable in
            magnitude, the FD step is a single number, and per-parameter
            relative precision becomes meaningful.
        param_eps_relative : float, default 1e-6
            Uniform FD step in eta space; ignored when ``param_scales`` is
            None.

        Returns
        -------
        grad : (num_bin, nparams) xp.ndarray
            ``grad[i, k] = dL/dtheta_k`` (default) or ``dL/d(eta_k)`` when
            ``param_scales`` is supplied.
        """
        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin, nparams = params_tmp.shape

        if convert_to_ra_dec:
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        num_data = num_noise = len(wdm_holder)

        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.settings.layer_df,
            self.wdm_lookup_table.settings.layer_dt,
            self.wdm_lookup_table.settings.Nf,
            self.wdm_lookup_table.settings.Nt,
            self.tdi_config.nchannels,
            self.wdm_lookup_table.settings.ind_min_t,
            self.wdm_lookup_table.settings.ind_max_t,
            self.wdm_lookup_table.settings.ind_min_f,
            self.wdm_lookup_table.settings.ind_max_f,
            num_data,
            num_noise,
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif data_index.dtype == self.xp.int64:
            data_index = data_index.astype(self.xp.int32)

        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif noise_index.dtype == self.xp.int64:
            noise_index = noise_index.astype(self.xp.int32)

        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32
        assert data_index.max() < num_data
        assert noise_index.max() < num_noise

        eps_theta, scales = self._resolve_eps_and_scales(
            nparams, param_eps, param_scales, param_eps_relative,
        )

        grad_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        params_in = params_tmp.flatten().copy()

        if use_spline:
            coarse_dt = _SECONDS_PER_YEAR / float(coarse_pts_per_year)
            self.backend.GBComputationGroupWrap().gb_wdm_spline_get_ll_grad(
                grad_out,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_in,
                data_index,
                noise_index,
                eps_theta,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                coarse_dt,
            )
        else:
            deriv_delta_t = 500.0
            self.backend.GBComputationGroupWrap().gb_wdm_get_ll_grad(
                grad_out,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_in,
                data_index,
                noise_index,
                eps_theta,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                deriv_delta_t,
            )
        grad = grad_out.reshape(num_bin, nparams)
        if scales is not None:
            # convert dL/dtheta -> dL/d(eta) = Delta_theta * dL/dtheta
            grad = grad * scales[None, :]
        return grad

    def get_swap_ll_grad_wdm(self, params_add, params_remove, wdm_holder,
                             param_eps_add=None, param_eps_remove=None,
                             param_scales_add=None, param_scales_remove=None,
                             param_eps_relative=1.0e-6,
                             data_index=None, noise_index=None,
                             convert_to_ra_dec: bool = True):
        """Chain-rule gradient of :meth:`get_swap_ll_wdm`.

        See :meth:`get_ll_grad_wdm` for the meaning of ``param_scales`` and
        ``param_eps_relative``.  The swap variant accepts independent
        ``param_scales_add`` / ``param_scales_remove`` so the add and remove
        sides can be scaled by their own natural widths.

        Returns
        -------
        grad_add, grad_remove : (num_bin, nparams) each
            Partial derivatives of  ll_diff = L(after swap) - L(before swap)
            with respect to ``theta_add`` and ``theta_remove`` respectively.
            Returned in rescaled (eta) coordinates when the corresponding
            ``param_scales_{add,remove}`` is provided.
        """
        params_add_tmp = self.xp.asarray(self.xp.atleast_2d(params_add)).copy()
        params_remove_tmp = self.xp.asarray(self.xp.atleast_2d(params_remove)).copy()
        assert params_add_tmp.shape == params_remove_tmp.shape, (
            f"params_add {params_add_tmp.shape} != params_remove {params_remove_tmp.shape}"
        )
        num_bin, nparams = params_add_tmp.shape

        if convert_to_ra_dec:
            for params_tmp in (params_add_tmp, params_remove_tmp):
                lam = params_tmp[:, -2].copy()
                beta = params_tmp[:, -1].copy()
                lam, beta = ecliptic_to_icrs(lam, beta)
                params_tmp[:, -2] = lam
                params_tmp[:, -1] = beta

        num_data = num_noise = len(wdm_holder)

        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.settings.layer_df,
            self.wdm_lookup_table.settings.layer_dt,
            self.wdm_lookup_table.settings.Nf,
            self.wdm_lookup_table.settings.Nt,
            self.tdi_config.nchannels,
            self.wdm_lookup_table.settings.ind_min_t,
            self.wdm_lookup_table.settings.ind_max_t,
            self.wdm_lookup_table.settings.ind_min_f,
            self.wdm_lookup_table.settings.ind_max_f,
            num_data,
            num_noise,
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif data_index.dtype == self.xp.int64:
            data_index = data_index.astype(self.xp.int32)

        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif noise_index.dtype == self.xp.int64:
            noise_index = noise_index.astype(self.xp.int32)

        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32
        assert data_index.max() < num_data
        assert noise_index.max() < num_noise

        eps_theta_add, scales_add = self._resolve_eps_and_scales(
            nparams, param_eps_add, param_scales_add, param_eps_relative,
        )
        eps_theta_remove, scales_remove = self._resolve_eps_and_scales(
            nparams, param_eps_remove, param_scales_remove, param_eps_relative,
        )

        grad_add_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        grad_remove_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        params_add_in = params_add_tmp.flatten().copy()
        params_remove_in = params_remove_tmp.flatten().copy()

        deriv_delta_t = 500.0
        self.backend.GBComputationGroupWrap().gb_wdm_swap_ll_grad(
            grad_add_out,
            grad_remove_out,
            self.cpp_orbits,
            self.cpp_tdi_config,
            self.cpp_wdm_lookup_table,
            self.cpp_wdm,
            params_add_in,
            params_remove_in,
            data_index,
            noise_index,
            eps_theta_add,
            eps_theta_remove,
            num_bin,
            nparams,
            self.T,
            self.t_ref,
            self.backend.TDITypeDict[self.tdi_type],
            deriv_delta_t,
        )
        grad_add = grad_add_out.reshape(num_bin, nparams)
        grad_remove = grad_remove_out.reshape(num_bin, nparams)
        if scales_add is not None:
            grad_add = grad_add * scales_add[None, :]
        if scales_remove is not None:
            grad_remove = grad_remove * scales_remove[None, :]
        return grad_add, grad_remove

    def fill_global_wdm(self, templates, params, wdm_holder, convert_to_ra_dec: bool = True, data_index=None, factors=None,
                        use_spline: bool = False, coarse_pts_per_year: int = 256):
        """Scatter per-source WDM contributions into a global template buffer.

        Set ``use_spline=True`` to dispatch to ``gb_wdm_spline_fill_global``,
        which replaces fast_wdm_inner with cubic-spline interpolation of the
        get_tdi outputs on a coarse uniform time grid of
        ``coarse_pts_per_year`` points per Julian year.

        With ``force_backend='jax'`` the ``templates`` buffer must be a
        *numpy* array (not jnp). JAX arrays are immutable; the JAX
        kernel internally uses a functional ``segment_sum`` and writes
        the result back into the numpy buffer via standard host-side
        assignment. The caller's ``templates`` reference is mutated in
        place, matching the C++ contract.
        """
        if self.backend.name == "fastlisaresponse_jax":
            # Accept numpy on the JAX path -- jnp arrays are immutable
            # so the in-place buffer contract would silently break.
            assert isinstance(templates, np.ndarray), (
                "On the JAX backend, ``templates`` must be a numpy "
                "ndarray (not jnp.ndarray). The kernel mutates it "
                "in place to match the C++ contract."
            )
        else:
            assert isinstance(templates, self.xp.ndarray)

        if templates.ndim == 1:
            num_templates = int(templates.shape[-1] / (self.wdm_lookup_table.nchannels * self.wdm_lookup_table.settings.Nf_active * self.wdm_lookup_table.settings.Nt_active))
            assert num_templates * self.wdm_lookup_table.nchannels * self.wdm_lookup_table.settings.Nf_active * self.wdm_lookup_table.settings.Nt_active == templates.shape[-1]
            nchannels = self.wdm_lookup_table.nchannels
            _Nf_active = self.wdm_lookup_table.settings.Nf_active
            _Nt_active = self.wdm_lookup_table.settings.Nt_active

        elif templates.ndim == 2:
            raise ValueError("Template must be 3D (nchannels, Nf_active, Nt_active), 4D (num_templates, nchannels, Nf_active, Nt_active), or flattended to 1D.")
        elif templates.ndim == 3:
            num_templates = 1
            nchannels, _Nf_active, _Nt_active = templates.shape

        elif templates.ndim == 4:
            num_templates, nchannels, _Nf_active, _Nt_active = templates.shape
            
        assert (
            nchannels == self.wdm_lookup_table.nchannels
            and _Nf_active == self.wdm_lookup_table.Nf_active
            and _Nt_active == self.wdm_lookup_table.Nt_active
        )
        # templates = templates.flatten()
       
        params_tmp = self.xp.atleast_2d(self.xp.asarray(params)).copy()
        
        if convert_to_ra_dec:
            lam = params_tmp[:, -2].copy()
            beta = params_tmp[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            params_tmp[:, -2] = lam
            params_tmp[:, -1] = beta

        num_bin = params_tmp.shape[0]
        params_in = params_tmp.flatten().copy()

        # TODO: move this part
        # TODO: need to check for num_data, num_noise
        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.settings.layer_df, 
            self.wdm_lookup_table.settings.layer_dt,
            self.wdm_lookup_table.settings.Nf, # calculates Nf_active inside
            self.wdm_lookup_table.settings.Nt, # calculates Nt_active inside
            self.tdi_config.nchannels,
            self.wdm_lookup_table.settings.ind_min_t,
            self.wdm_lookup_table.settings.ind_max_t,
            self.wdm_lookup_table.settings.ind_min_f,
            self.wdm_lookup_table.settings.ind_max_f,
            num_templates, # data not needed here
            num_templates  # noise not needed here
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        elif data_index.dtype == self.xp.int64:
            _data_index = data_index.copy().astype(self.xp.int32)
            del data_index
            data_index = _data_index

        # Per-source multiplicative factor applied at the accumulation step
        # (template[m,n] += factor * w_mn). Default +1 (add); pass -1 to
        # remove a source. Mirrors gbgpu.generate_global_template's factors.
        if factors is None:
            factors = self.xp.ones(num_bin, dtype=self.xp.float64)
        else:
            factors = self.xp.ascontiguousarray(self.xp.asarray(factors, dtype=self.xp.float64))
            assert factors.shape == (num_bin,), (
                f"factors must have shape ({num_bin},), got {factors.shape}"
            )

        assert data_index.max() < num_templates
        nparams = 9

        if use_spline:
            coarse_dt = _SECONDS_PER_YEAR / float(coarse_pts_per_year)
            self.backend.GBComputationGroupWrap().gb_wdm_spline_fill_global(
                templates,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_in,
                data_index,
                factors,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                coarse_dt,
            )
        else:
            deriv_delta_t = 500.0  # seconds
            self.backend.GBComputationGroupWrap().gb_wdm_fill_global(
                templates,
                self.cpp_orbits,
                self.cpp_tdi_config,
                self.cpp_wdm_lookup_table,
                self.cpp_wdm,
                params_in,
                data_index,
                factors,
                num_bin,
                nparams,
                self.T,
                self.t_ref,
                self.backend.TDITypeDict[self.tdi_type],
                deriv_delta_t
            )


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
