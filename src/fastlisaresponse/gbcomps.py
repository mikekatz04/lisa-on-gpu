from .utils.parallelbase import FastLISAResponseParallelModule
from fastlisaresponse.tdiconfig import TDIConfig
from lisatools.detector import Orbits, EqualArmlengthOrbits
from copy import deepcopy
from lisatools.domains import WDMLookupTable


class GBWDMComputations(FastLISAResponseParallelModule):
    def __init__(self, wdm_lookup_table, T, orbits=None, tdi_config=None, force_backend=None, d_d=0.0):
        
        super().__init__(force_backend=force_backend)
        # setup orbits
        self.orbits = orbits
         # setup TDI info
        self.tdi_config = tdi_config
        # setup WDM c class
        self.wdm_lookup_table = wdm_lookup_table
        self.T = T
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
        self.c_nm_all = self.xp.asarray(wdm_lookup_table.table.real.copy())
        self.s_nm_all = self.xp.asarray(wdm_lookup_table.table.imag.copy())
        self.cpp_wdm_lookup_table = self.backend.WaveletLookupTableWrap(
            self.c_nm_all, 
            self.s_nm_all, 
            wdm_lookup_table.f_steps, 
            wdm_lookup_table.fdot_steps, 
            wdm_lookup_table.deltaf,  # NOT .df (that is the WDM basis info) 
            wdm_lookup_table.d_fdot, 
            wdm_lookup_table.min_f_scaled,
            wdm_lookup_table.min_fdot,
            wdm_lookup_table.df,
            wdm_lookup_table.dt,
            wdm_lookup_table.NF,
            wdm_lookup_table.NT,
            wdm_lookup_table.num_channel
        )

    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None):
        params_tmp = self.xp.atleast_2d(self.xp.asarray(params))
        num_bin = params_tmp.shape[0]
        params_in = params_tmp.flatten().copy()

        self.d_h_out = self.xp.zeros(num_bin)
        self.h_h_out = self.xp.zeros(num_bin)

        # TODO: move this part
        # TODO: need to check for num_data, num_noise
        num_data = num_noise = len(wdm_holder)
        self.cpp_wdm = self.backend.WDMDomainWrap(
            wdm_holder.linear_data_arr[0],
            wdm_holder.linear_psd_arr[0],
            self.wdm_lookup_table.df, 
            self.wdm_lookup_table.dt,
            self.wdm_lookup_table.NF, 
            self.wdm_lookup_table.NT,
            self.tdi_config.nchannels, 
            num_data, 
            num_noise
        )

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert data_index.dtype == self.xp.int32
            
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            assert noise_index.dtype == self.xp.int32
            
        nparams = 9

        breakpoint()
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
            self.backend.TDITypeDict["XYZ"]
        )

        like_out = -1. / 2. * (self.d_d + self.h_h_out - 2 * self.d_h_out)
        # TODO: phase maximize

        return like_out
