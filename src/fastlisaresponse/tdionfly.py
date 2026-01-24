from __future__ import annotations
import numpy as np
from typing import Optional, List
import warnings
from typing import Tuple
from copy import deepcopy
from gpubackendtools import wrapper
        
import time
import h5py

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as cp

from scipy.interpolate import CubicSpline as CubicSpline_scipy
from gpubackendtools.interpolate import CubicSplineInterpolant

from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.utils.utility import AET
from gpubackendtools import wrapper
            
from .utils.parallelbase import FastLISAResponseParallelModule
from .tdiconfig import TDIConfig

# TODO: need to update constants setup
YRSID_SI = 31558149.763545603


def get_factorial(n):
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact


from math import factorial

factorials = np.array([factorial(i) for i in range(30)])

C_inv = 3.3356409519815204e-09


class CubicSpline:
    """Alias to cubic spline cython class."""
    pass

class TDIonTheFly(FastLISAResponseParallelModule):
    """Class container for LISA TDI on the fly.

    This class is also GPU-accelerated, which is very helpful for Bayesian inference
    methods.

    Args:
        tdi_config (str or list, optional): TDI setup. Currently, the stock options are
            :code:`'1st generation'` and :code:`'2nd generation'`. Or the user can provide
            a list of tdi_combinations of the form
            :code:`{"link": 12, "links_for_delay": [21, 13, 31], "sign": 1, "type": "delay"}`.
            :code:`'link'` (`int`) the link index (12, 21, 13, 31, 23, 32) for the projection (:math:`y_{ij}`).
            :code:`'links_for_delay'` (`list`) are the link indexes as a list used for delays
            applied to the link projections.
            ``'sign'`` is the sign in front of the contribution to the TDI observable. It takes the value of `+1` or `-1`.
            ``type`` is either ``"delay"`` or ``"advance"``. It is optional and defaults to ``"delay"``.
            (default: ``"1st generation"``)
        orbits (:class:`Orbits`, optional): Orbits class from LISA Analysis Tools. Works with LISA Orbits 
            outputs: ``lisa-simulation.pages.in2p3.fr/orbits/``.
            (default: :class:`EqualArmlengthOrbits`)
        tdi_chan (str, optional): Which TDI channel combination to return. Choices are :code:`'XYZ'`,
            :code:`AET`, or :code:`AE`. (default: :code:`'XYZ'`)
        force_backend (str, optional): If given, run this class on the requested backend. 
            Options are ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``. (default: ``None``)
        
    """

    def __init__(
        self,
        sampling_frequency,
        num_sub,
        n_params=4,
        tdi_config: Optional[TDIConfig] = None,
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        tdi_chan="XYZ",
        force_backend=None,
    ):

        # setup all quantities
        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency
        self.n_params = n_params
        self.num_sub = num_sub

        # setup TDI information
        self.tdi_chan = tdi_chan
        super().__init__(force_backend=force_backend)

        # setup orbits
        self.orbits = orbits
        self.tdi_config = tdi_config
        # setup TDI info
        
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
        
        elif issubclass(orbits, Orbits) and not isinstance(orbits, Orbits):
            # assumed default arguments if not initialized as input
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        self._orbits = deepcopy(orbits)

        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)

        self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)
    
    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(self, inc, psi, lam, beta, return_spline: bool =False) -> TDIOutput:
        
        params = self.xp.asarray([inc, psi, lam, beta]).T.flatten().copy()

        assert len(params) == 4 * self.num_sub

        tdi_channels_arr = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=complex)
        tdi_amp = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        tdi_phase = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        phase_ref = self.xp.zeros((self.N * self.num_sub), dtype=float)
        assert int(np.prod(self.t_arr.shape)) == self.N * self.num_sub

        self.wave_gen.run_wave_tdi_wrap(
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phase_ref,
            params, self.t_arr.flatten().copy(),
            self.N, self.num_sub, self.n_params, self.tdi_config.nchannels
        )
        
        breakpoint()
        reshape_shape = (self.num_sub, self.tdi_config.nchannels, self.N)
        return self.from_tdi_output(TDIOutput(
            self.t_arr, 
            tdi_amp.reshape(reshape_shape), 
            tdi_phase.reshape(reshape_shape), 
            phase_ref.reshape(self.t_arr.shape)
        ), fill_splines=return_spline)
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        return tdi_output


CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2
CUBIC_SPLINE_GENERAL_SPACING = 3

class TDTDIonTheFly(TDIonTheFly):
    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSpline,
        phase: np.ndarray | CubicSpline_scipy | CubicSpline,
        *args, 
        t_input: Optional[np.ndarray] = None, 
        **kwargs
    ): 
        super().__init__(*args, **kwargs)

        self.phase_input = phase
        self.amp_input = amp

        if isinstance(amp, np.ndarray) or isinstance(amp, cp.ndarray):
            if isinstance(amp, np.ndarray):
                assert isinstance(phase, np.ndarray) and isinstance(t, np.ndarray)
                assert t_input is not None and isinstance(t_input, np.ndarray)
            else:
                assert isinstance(phase, cp.ndarray) and isinstance(t, cp.ndarray)
                assert t_input is not None and isinstance(t_input, cp.ndarray)
            
            self.spline_length = len(phase)

            t_input = self.xp.atleast_2d(self.xp.asarray(t_input))
            
            if t_input.shape[0] == 1:
                t_input = self.xp.repeat(t_input, amp.shape[0], axis=0)

            amp = self.xp.atleast_2d(self.xp.asarray(amp))
            phase = self.xp.atleast_2d(self.xp.asarray(phase))

            # TODO: improve when gbt is fixed up
            amp = CubicSplineInterpolant(t_input.copy(), amp, force_backend=self.backend.name.split("_")[-1])
            phase = CubicSplineInterpolant(t_input.copy(), phase, force_backend=self.backend.name.split("_")[-1])

        elif isinstance(amp, CubicSpline_scipy):
            raise NotImplementedError
            assert isinstance(phase, CubicSpline_scipy)

            self.spline_length = phase.c.shape[-1] + 1

            phase_y = phase.c[3, :].copy()
            phase_c1 = phase.c[2, :].copy()
            phase_c2 = phase.c[1, :].copy()
            phase_c3 = phase.c[0, :].copy()

            amp_y = amp.c[3, :].copy()
            amp_c1 = amp.c[2, :].copy()
            amp_c2 = amp.c[1, :].copy()
            amp_c3 = amp.c[0, :].copy()

            breakpoint()
            # x = amp

            # convert to pointers
            targs, twkargs = wrapper(t, phase_y, phase_c1, phase_c2, phase_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            phase = self.backend.pyCubicSplineWrap(_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)

        elif isinstance(amp, CubicSplineInterpolant):
            assert isinstance(phase, CubicSplineInterpolant)

        else:
            raise ValueError("# TODO: fix this.")
        
        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))

        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
        self.amp = amp
        self.phase = phase

        # self.wave_gen = self.backend.pyTDSplineTDIWaveform()
        # self.wave_gen.add_orbit_information(*self.orbits.pycppdetector_args)
        # self.wave_gen.add_tdi_config(*self.tdi_config.pytdiconfig_args)
        # self.wave_gen.add_amp_spline(*self.amp.cpp_class_args)
        # self.wave_gen.add_phase_spline(*self.phase.cpp_class_args)
        
        # import time
        # time.sleep(1.0)
    @property
    def wave_gen(self) -> callable:
        return self._wave_gen
    
    @wave_gen.setter
    def wave_gen(self, wave_gen):
        self._wave_gen = wave_gen
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        assert self.xp.allclose(tdi_output.x, self.t_arr)
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines
        )
    

class TDIOutput(FastLISAResponseParallelModule):
    def __init__(self, x, tdi_amp, tdi_phase, phase_ref, fill_splines=True, **kwargs):
        
        self.fill_splines = fill_splines
        if self.fill_splines:
            self._splines = {}

        self.x = x
        super().__init__(**kwargs)

        # need to be after for proper setter
        self.tdi_amp, self.tdi_phase = tdi_amp, tdi_phase
        self.phase_ref = phase_ref
       
        
    def _get_spl(self, key: str) -> CubicSpline:
        assert self.fill_splines
        return self._splines[key]
    
    @property
    def phase_ref_spl(self) -> CubicSpline:
        return self._get_spl("phase_ref")
    
    def build_spline(self, x, y, **kwargs) -> CubicSplineInterpolant:
        if x.ndim == 2 and y.ndim == 3:
            x_in =  self.xp.repeat(x[:, None, :], y.shape[1], axis=1)
        else:
            x_in = x.copy()
        return CubicSplineInterpolant(x_in, y, **kwargs, force_backend=self.backend.name.split("_")[-1])
    
    @property
    def num_bin(self) -> int:
        if self.tdi_amp.ndim == 3:
            return self.tdi_amp.shape[0]
        elif self.tdi_amp.ndim == 2:
            return 1

    @classmethod
    def supported_backends(cls) -> list:
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    @property
    def X(self) -> np.ndarray:
        return self.Xamp * self.xp.exp(-1j * (self.Xphase + self.phase_ref))
    @property
    def Y(self) -> np.ndarray:
        return self.Yamp * self.xp.exp(-1j * (self.Xphase + self.phase_ref))
    @property
    def Z(self) -> np.ndarray:
        return self.Zamp * self.xp.exp(-1j * (self.Xphase + self.phase_ref))
    @property
    def Xamp(self) -> np.ndarray:
        return self.tdi_amp[:, 0]
    @property
    def Yamp(self) -> np.ndarray:
        return self.tdi_amp[:, 1]
    @property
    def Zamp(self) -> np.ndarray:
        return self.tdi_amp[:, 2]
    @property
    def Xphase(self) -> np.ndarray:
        return self.tdi_phase[:, 0]
    @property
    def Yphase(self) -> np.ndarray:
        return self.tdi_phase[:, 1]
    @property
    def Zphase(self) -> np.ndarray:
        return self.tdi_phase[:, 2]
    
    @property
    def tdi_amp(self) -> np.ndarray:
        return self._tdi_amp
    
    @tdi_amp.setter
    def tdi_amp(self, tdi_amp: np.ndarray):
        if self.fill_splines:
            self._splines["tdi_amp"] = self.build_spline(self.x, tdi_amp)

        self._tdi_amp = tdi_amp

    @property
    def tdi_phase(self) -> np.ndarray:
        return self._tdi_phase
    
    @tdi_phase.setter
    def tdi_phase(self, tdi_phase: np.ndarray):
        if self.fill_splines:
            self._splines["tdi_phase"] = self.build_spline(self.x, tdi_phase)

        self._tdi_phase = tdi_phase

    @property
    def phase_ref(self) -> np.ndarray:
        return self._phase_ref
    
    @phase_ref.setter
    def phase_ref(self, phase_ref: np.ndarray):
        if self.fill_splines:
            self._splines["phase_ref"] = self.build_spline(self.x, phase_ref)

        self._phase_ref = phase_ref
    
    @property
    def tdi_amp_spl(self):
        return self._get_spl("tdi_amp")
    
    @property
    def tdi_phase_spl(self):
        return self._get_spl("tdi_phase")
    
    @property
    def Aamp(self):
        raise NotImplementedError
    @property
    def Aphase(self):
        raise NotImplementedError
    @property
    def Eamp(self):
        raise NotImplementedError
    @property
    def Ephase(self):
        raise NotImplementedError
    @property
    def Tamp(self):
        raise NotImplementedError
    @property
    def Tphase(self):
        raise NotImplementedError
    
    def eval_spline_vals(self, x_new: np.ndarray, **kwargs) -> np.ndarray:

        if x_new.ndim == 1:
            t_amp_phase = self.xp.tile(x_new, (self.num_bin, 3, 1))
            t_phase_ref = self.xp.tile(x_new, (self.num_bin, 1))
        elif x_new.ndim == 2:
            t_amp_phase = self.xp.repeat(x_new[:, None, :], 3, axis=1)
            t_phase_ref = x_new

        tdi_amp_new = self.tdi_amp_spl(t_amp_phase, **kwargs)
        tdi_phase_new = self.tdi_phase_spl(t_amp_phase, **kwargs)
        phase_ref_new = self.phase_ref_spl(t_phase_ref, **kwargs)
        return (tdi_amp_new, tdi_phase_new, phase_ref_new)
    
    def eval_tdi(self, x_new: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
class TDTDIOutput(TDIOutput):
    @classmethod
    def from_tdi_output(cls, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> TDTDIOutput:
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines
        )

    def eval_tdi(self, t_new: np.ndarray, **kwargs) -> np.ndarray:
        tdi_amp_new, tdi_phase_new, phase_ref_new = self.eval_spline_vals(t_new, **kwargs)
        tdi_output = self.xp.real(tdi_amp_new * self.xp.exp(-1j * (tdi_phase_new + phase_ref_new)))
        return tdi_output
    
    @property
    def t_arr(self) -> np.ndarray:
        return self.x
        
    
class FDTDIOutput(TDIOutput):
    def eval_tdi(self, f_new: np.ndarray, **kwargs) -> np.ndarray:
        tdi_amp_new, tdi_phase_new, phase_ref_new = self.eval_spline_vals(f_new, **kwargs)
        tdi_output = tdi_amp_new * self.xp.exp(-1j * (tdi_phase_new + phase_ref_new[:, None, :]))
        return tdi_output
    
    @property
    def f_arr(self) -> np.ndarray:
        return self.x


# TODO: make it log spaced in frequency?

class FDTDIonTheFly(TDIonTheFly):
    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        freq: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        phase_ref: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        *args, 
        t_input: Optional[np.ndarray] = None, 
        spline_type: int = CUBIC_SPLINE_GENERAL_SPACING,
        force_backend: str = None,
        **kwargs
    ): 
        super().__init__(*args, force_backend=force_backend, **kwargs)

        self.freq_input = freq
        self.amp_input = amp
        self.phase_ref = phase_ref
        
        if isinstance(amp, np.ndarray) or isinstance(amp, cp.ndarray):
            if isinstance(amp, np.ndarray):
                assert isinstance(freq, np.ndarray) and isinstance(t, np.ndarray)
                assert t_input is not None and isinstance(t_input, np.ndarray)
            else:
                assert isinstance(freq, cp.ndarray) and isinstance(t, cp.ndarray)
                assert t_input is not None and isinstance(t_input, cp.ndarray)
            
            self.spline_length = len(freq)

            t_input = self.xp.atleast_2d(self.xp.asarray(t_input))
            
            if t_input.shape[0] == 1:
                t_input = self.xp.repeat(t_input, amp.shape[0], axis=0)

            amp = self.xp.atleast_2d(self.xp.asarray(amp))
            freq = self.xp.atleast_2d(self.xp.asarray(freq))

            # TODO: improve when gbt is fixed up
            amp = CubicSplineInterpolant(t_input.copy(), amp, force_backend=self.backend.name.split("_")[-1])
            freq = CubicSplineInterpolant(t_input.copy(), freq, force_backend=self.backend.name.split("_")[-1])
            

        elif isinstance(amp, CubicSpline_scipy):
            assert isinstance(freq, CubicSpline_scipy)
            raise NotImplementedError
            self.spline_length = freq.c.shape[-1] + 1

            freq_y = freq.c[3, :].copy()
            freq_c1 = freq.c[2, :].copy()
            freq_c2 = freq.c[1, :].copy()
            freq_c3 = freq.c[0, :].copy()

            amp_y = amp.c[3, :].copy()
            amp_c1 = amp.c[2, :].copy()
            amp_c2 = amp.c[1, :].copy()
            amp_c3 = amp.c[0, :].copy()

            breakpoint()
            # x = amp

            # convert to pointers
            targs, twkargs = wrapper(t, freq_y, freq_c1, freq_c2, freq_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            freq = self.backend.pyCubicSplineWrap(_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)

        elif isinstance(amp, CubicSplineInterpolant):
            assert isinstance(freq, CubicSplineInterpolant)
            # f = freq.y, t = freq.x

        else:
            raise ValueError("# TODO: fix this.")
        
        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))
        
        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
        self.amp = amp
        self.freq = freq

    @property
    def wave_gen(self) -> callable:
        self.cpp_amp = self.backend.CubicSplineWrap(*self.amp.cpp_class_args)
        self.cpp_freq = self.backend.CubicSplineWrap(*self.freq.cpp_class_args)
        self._wave_gen = self.backend.FDSplineTDIWaveformWrap(self.cpp_orbits, self.cpp_tdi_config, self.cpp_amp, self.cpp_freq)
        return self._wave_gen
    
    @property
    def spline_type(self) -> int:
        return self._spline_type
    
    @spline_type.setter
    def spline_type(self, spline_type: int):
        assert isinstance(spline_type, int)
        assert spline_type in [CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING]
        self._spline_type = spline_type
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        # TODO: remove the freq spline?
        return FDTDIOutput(
            self.freq(tdi_output.x), tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines
        )
    
