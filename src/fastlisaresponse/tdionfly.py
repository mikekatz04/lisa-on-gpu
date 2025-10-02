import numpy as np
from typing import Optional, List
import warnings
from typing import Tuple
from copy import deepcopy

import time
import h5py

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as cp

from scipy.interpolate import CubicSpline as CubicSpline_scipy

from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.utils.utility import AET
from gpubackendtools import wrapper
            
from .utils.parallelbase import FastLISAResponseParallelModule


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
        tdi (str or list, optional): TDI setup. Currently, the stock options are
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
        tdi="1st generation",
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        tdi_chan="XYZ",
        force_backend=None,
    ):

        # setup all quantities
        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency
        
        # setup TDI information
        self.tdi = tdi
        self.tdi_chan = tdi_chan

        super().__init__(force_backend=force_backend)

        # setup orbits
        self.response_orbits = orbits

        # setup TDI info
        self._init_TDI_delays()

    @property
    def xp(self) -> object:
        return self.backend.xp
    
    @property
    def orbits(self) -> object:
        return self.response_orbits

    @property
    def response_orbits(self) -> Orbits:
        """Response function orbits."""
        return self._response_orbits

    @response_orbits.setter
    def response_orbits(self, orbits: Orbits) -> None:
        """Set response orbits."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()
        
        elif issubclass(orbits, Orbits) and not isinstance(orbits, Orbits):
            # assumed default arguments if not initialized as input
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        self._response_orbits = deepcopy(orbits)

        if not self._response_orbits.configured:
            self._response_orbits.configure(linear_interp_setup=True)

    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def _init_TDI_delays(self):
        """Initialize TDI specific information"""

        # setup the actual TDI combination
        if self.tdi in ["1st generation", "2nd generation"]:
            # tdi 1.0
            tdi_combinations = [
                {"link": 13, "links_for_delay": [], "sign": +1},
                {"link": 31, "links_for_delay": [13], "sign": +1},
                {"link": 12, "links_for_delay": [13, 31], "sign": +1},
                {"link": 21, "links_for_delay": [13, 31, 12], "sign": +1},
                {"link": 12, "links_for_delay": [], "sign": -1},
                {"link": 21, "links_for_delay": [12], "sign": -1},
                {"link": 13, "links_for_delay": [12, 21], "sign": -1},
                {"link": 31, "links_for_delay": [12, 21, 13], "sign": -1},
            ]

            if self.tdi == "2nd generation":
                # tdi 2.0 is tdi 1.0 + additional terms
                tdi_combinations += [
                    {"link": 12, "links_for_delay": [13, 31, 12, 21], "sign": +1},
                    {"link": 21, "links_for_delay": [13, 31, 12, 21, 12], "sign": +1},
                    {
                        "link": 13,
                        "links_for_delay": [13, 31, 12, 21, 12, 21],
                        "sign": +1,
                    },
                    {
                        "link": 31,
                        "links_for_delay": [13, 31, 12, 21, 12, 21, 13],
                        "sign": +1,
                    },
                    {"link": 13, "links_for_delay": [12, 21, 13, 31], "sign": -1},
                    {"link": 31, "links_for_delay": [12, 21, 13, 31, 13], "sign": -1},
                    {
                        "link": 12,
                        "links_for_delay": [12, 21, 13, 31, 13, 31],
                        "sign": -1,
                    },
                    {
                        "link": 21,
                        "links_for_delay": [12, 21, 13, 31, 13, 31, 12],
                        "sign": -1,
                    },
                ]

        elif isinstance(self.tdi, list):
            tdi_combinations = self.tdi

        else:
            raise ValueError(
                "tdi kwarg should be '1st generation', '2nd generation', or a list with a specific tdi combination."
            )
        self.tdi_combinations = tdi_combinations

    @property
    def tdi_combinations(self) -> List:
        """TDI Combination setup"""
        return self._tdi_combinations

    @tdi_combinations.setter
    def tdi_combinations(self, tdi_combinations: List) -> None:
        """Set TDI combinations and fill out setup."""
        tdi_base_links = []
        tdi_link_combinations = []
        tdi_signs = []
        tdi_operation_index = []
        channels = []

        tdi_index = 0
        for permutation_number in range(3):
            for tmp in tdi_combinations:
                tdi_base_links.append(
                    self._cyclic_permutation(tmp["link"], permutation_number)
                )
                tdi_signs.append(tmp["sign"])
                channels.append(permutation_number)
                if len(tmp["links_for_delay"]) == 0:
                    tdi_link_combinations.append(-11)
                    tdi_operation_index.append(tdi_index)

                else:
                    for link_delay in tmp["links_for_delay"]:

                        tdi_link_combinations.append(
                            self._cyclic_permutation(link_delay, permutation_number)
                        )
                        tdi_operation_index.append(tdi_index)

                tdi_index += 1

        self.tdi_operation_index = self.xp.asarray(tdi_operation_index).astype(
            self.xp.int32
        )
        self.tdi_base_links = self.xp.asarray(tdi_base_links).astype(self.xp.int32)
        self.tdi_link_combinations = self.xp.asarray(tdi_link_combinations).astype(
            self.xp.int32
        )
        self.tdi_signs = self.xp.asarray(tdi_signs).astype(self.xp.int32)
        self.channels = self.xp.asarray(channels).astype(self.xp.int32)
        assert len(self.tdi_link_combinations) == len(self.tdi_operation_index)

        assert (
            len(self.tdi_base_links)
            == len(np.unique(self.tdi_operation_index))
            == len(self.tdi_signs)
            == len(self.channels)
        )

    def _cyclic_permutation(self, link, permutation):
        """permute indexes by cyclic permutation"""
        link_str = str(link)

        out = ""
        for i in range(2):
            sc = int(link_str[i])
            temp = sc + permutation
            if temp > 3:
                temp = temp % 3
            out += str(temp)

        return int(out)
    
    @property
    def t_tdi_eval(self) -> np.ndarray:
        """Get tdi values to avoid edge issues.
        
        # TODO: make this better and check this.

        # This should be avoidable in FD?
        
        """
        return self.t_arr[1:-1]
    
    def __call__(self, inc, psi, lam, beta, return_spline: bool =False):
        
        params = np.array([inc, psi, lam, beta])

        buffer = self.wave_gen.get_buffer_size(self.N)
        _size_of_double = 8
        num_points = int(buffer / _size_of_double)

        buffer = np.zeros(num_points)
        Xamp = np.zeros(self.N)
        Xphase = np.zeros(self.N)
        Yamp = np.zeros(self.N)
        Yphase = np.zeros(self.N)
        Zamp = np.zeros(self.N)
        Zphase = np.zeros(self.N)
        
        self.wave_gen.run_wave_tdi(
            buffer, buffer.shape[0] * _size_of_double,
            Xamp, Xphase,
            Yamp, Yphase,
            Zamp, Zphase,
            params, self.t_tdi_eval,
            self.N
        )

        return TDTDIOutput(self.t_arr, Xamp, Xphase, Yamp, Yphase, Zamp, Zphase, fill_splines=return_spline)


CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2
CUBIC_SPLINE_GENERAL_SPACING = 3

class TDTDIonTheFly(TDIonTheFly):
    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSpline,
        phase: np.ndarray | CubicSpline_scipy | CubicSpline,
        *args, **kwargs
    ): 
        super().__init__(*args, **kwargs)

        self.t_arr = t
        self.phase_input = phase
        self.amp_input = amp

        self.dt = t[1] - t[0]
        self.N = t.shape[0]

        # TODO: spacing?
        if not np.allclose(np.diff(t), np.full_like(t[:-1], self.dt)):
            raise NotImplementedError("Currently t needs to be evenly spaced.")
        
        if isinstance(amp, np.ndarray):
            assert isinstance(phase, np.ndarray) and isinstance(t, np.ndarray)
            self.spline_length = len(phase)

        elif isinstance(amp, cp.ndarray):
            assert isinstance(phase, cp.ndarray) and isinstance(t, cp.ndarray)
            self.spline_length = len(phase)

        elif isinstance(amp, CubicSpline_scipy):
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

            # convert to pointers
            targs, twkargs = wrapper(t, phase_y, phase_c1, phase_c2, phase_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            phase = self.backend.pyCubicSplineWrap(_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, self.dt, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.dt, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)

        elif isinstance(amp, CubicSpline):
            assert isinstance(phase, CubicSpline)
            raise NotImplementedError

        else:
            raise ValueError("# TODO: fix this.")
        
        self.amp = amp
        self.phase = phase

        self.wave_gen = self.backend.pyTDSplineTDIWaveform(self.orbits.ptr, self.amp.ptr, self.phase.ptr)


class TDTDIOutput(FastLISAResponseParallelModule):
    def __init__(self, t, Xamp, Xphase, Yamp, Yphase, Zamp, Zphase, fill_splines=True, **kwargs):
        self.Xamp, self.Xphase, self.Yamp, self.Yphase, self.Zamp, self.Zphase = Xamp, Xphase, Yamp, Yphase, Zamp, Zphase
        self.t = t
        super().__init__(**kwargs)
        self.fill_splines = fill_splines
        if fill_splines:
            self._splines = {}
            for name in ["Xamp", "Xphase", "Yamp", "Yphase", "Zamp", "Zphase"]:
                self._splines[name] = self.build_spline(self.t, getattr(self, name))
            
    def _get_spl(self, key: str) -> CubicSpline:
        assert self.fill_splines
        return self._splines[key]
    
    @property
    def Xamp_spl(self) -> CubicSpline:
        return self._get_spl("Xamp")
    @property
    def Xphase_spl(self) -> CubicSpline:
        return self._get_spl("Xphase")
    
    @property
    def Yamp_spl(self) -> CubicSpline:
        return self._get_spl("Yamp")
    @property
    def Yphase_spl(self) -> CubicSpline:
        return self._get_spl("Yphase")
    
    @property
    def Zamp_spl(self) -> CubicSpline:
        return self._get_spl("Zamp")
    @property
    def Zphase_spl(self) -> CubicSpline:
        return self._get_spl("Zphase")
    
    def build_spline(self, x, y) -> CubicSpline:
        _spl_scipy = CubicSpline_scipy(x, y)
        c1 = _spl_scipy.c[2, :]
        c2 = _spl_scipy.c[1, :]
        c3 = _spl_scipy.c[0, :]
        targs, twkargs = wrapper(x, y, c1, c2, c3)
        (_x, _y, _c1, _c2, _c3) = targs

        dx_all = np.diff(x)
        # TODO: fix this input setup
        dx = dx_all[0]
        _spl = self.backend.pyCubicSplineWrap(_x, _y, _c1, _c2, _c3, dx, y.shape[0], CUBIC_SPLINE_LINEAR_SPACING)
        return _spl
    
    @classmethod
    def supported_backends(cls) -> list:
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

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


# TODO: make it log spaced in frequency?

class FDTDIonTheFly(TDIonTheFly):
    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSpline,
        freq: np.ndarray | CubicSpline_scipy | CubicSpline,
        *args, 
        spline_type: int = CUBIC_SPLINE_GENERAL_SPACING,
        **kwargs
    ): 
        super().__init__(*args, **kwargs)

        self.t_arr = t
        self.freq_input = freq
        self.amp_input = amp
        self.spline_type = spline_type

        self.dt = t[1] - t[0]
        self.N = t.shape[0]

        # TODO: spacing?
        if isinstance(amp, np.ndarray):
            assert isinstance(freq, np.ndarray) and isinstance(t, np.ndarray)
            self.spline_length = len(freq)

        elif isinstance(amp, cp.ndarray):
            assert isinstance(freq, cp.ndarray) and isinstance(t, cp.ndarray)
            self.spline_length = len(freq)

        elif isinstance(amp, CubicSpline_scipy):
            assert isinstance(freq, CubicSpline_scipy)
            self.spline_length = freq.c.shape[-1] + 1
            freq_y = freq.c[3, :].copy()
            freq_c1 = freq.c[2, :].copy()
            freq_c2 = freq.c[1, :].copy()
            freq_c3 = freq.c[0, :].copy()

            amp_y = amp.c[3, :].copy()
            amp_c1 = freq.c[2, :].copy()
            amp_c2 = freq.c[1, :].copy()
            amp_c3 = freq.c[0, :].copy()

            # convert to pointers
            targs, twkargs = wrapper(t, freq_y, freq_c1, freq_c2, freq_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            freq = self.backend.pyCubicSplineWrap(_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, self.dt, self.spline_length, self.spline_type)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.dt, self.spline_length, self.spline_type)

        elif isinstance(amp, CubicSpline):
            assert isinstance(freq, CubicSpline)
            self.spline_length = freq.length
            
        else:
            raise ValueError("# TODO: fix this.")
        
        self.amp = amp
        self.freq = freq

        self.wave_gen = self.backend.pyFDSplineTDIWaveform(self.orbits.ptr, self.amp.ptr, self.freq.ptr)

    @property
    def spline_type(self) -> int:
        return self._spline_type
    
    @spline_type.setter
    def spline_type(self, spline_type: int):
        assert isinstance(spline_type, int)
        assert spline_type in [CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING]
        self._spline_type = spline_type