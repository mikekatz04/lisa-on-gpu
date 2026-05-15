from .utils.parallelbase import FastLISAResponseParallelModule
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import Orbits, EqualArmlengthOrbits
from copy import deepcopy
from lisatools.domains import WDMLookupTable
from .response import ecliptic_to_icrs

class GBWDMComputations(FastLISAResponseParallelModule):
    def __init__(self, wdm_lookup_table, T, t_ref, orbits=None, tdi_config=None, force_backend=None, d_d=0.0):
        
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
        """Set wdm lookup table."""

        self._wdm_lookup_table = wdm_lookup_table
        # Tables are (Nt, num_fdot, num_f); the C lookup indexes by layer_n,
        # so the underlying buffer must be contiguous in this layout.
        Nt = wdm_lookup_table.settings.Nt
        num_fdot = wdm_lookup_table.fdot_steps
        num_f = wdm_lookup_table.f_steps
        expected_shape = (Nt, num_fdot, num_f)
        assert wdm_lookup_table.table_cos.shape == expected_shape, (
            f"table_cos shape {wdm_lookup_table.table_cos.shape} != {expected_shape}"
        )
        assert wdm_lookup_table.table_sin.shape == expected_shape, (
            f"table_sin shape {wdm_lookup_table.table_sin.shape} != {expected_shape}"
        )
        self.c_nm_all = self.xp.ascontiguousarray(self.xp.asarray(wdm_lookup_table.table_cos))
        self.s_nm_all = self.xp.ascontiguousarray(self.xp.asarray(wdm_lookup_table.table_sin))
        
        delta_f = wdm_lookup_table.f_vals_norm[1] - wdm_lookup_table.f_vals_norm[0]
        try:
            delta_fdot = wdm_lookup_table.fdot_vals[1] - wdm_lookup_table.fdot_vals[0]
        except IndexError:
            # this happens when there is no fdot
            delta_fdot = 1.0

        is_m_ref_n_ref_even = False

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
        )

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None, convert_to_ra_dec: bool = True):
        
        params_tmp = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        num_bin = params_tmp.shape[0]
        
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
            self.backend.TDITypeDict["XYZ"],
            deriv_delta_t
        )

        like_out = -1. / 2. * (self.d_d + self.h_h_out - 2 * self.d_h_out)
        # TODO: phase maximize
        return like_out

    def fill_global_wdm(self, templates, params, wdm_holder, convert_to_ra_dec: bool = True, data_index=None):
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
            
        assert data_index.max() < num_templates
        nparams = 9

        deriv_delta_t = 500.0  # seconds

        self.backend.GBComputationGroupWrap().gb_wdm_fill_global(
            templates, 
            self.cpp_orbits,
            self.cpp_tdi_config, 
            self.cpp_wdm_lookup_table, 
            self.cpp_wdm, 
            params_in, 
            data_index, 
            num_bin,
            nparams, 
            self.T,
            self.t_ref,
            self.backend.TDITypeDict["XYZ"],
            deriv_delta_t
        )