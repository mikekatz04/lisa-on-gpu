import unittest
import numpy as np
import warnings
import os

path_to_file = os.path.dirname(__file__)

import fastlisaresponse_backend_cpu.tdionthefly

from lisatools.detector import EqualArmlengthOrbits

try:
    import cupy as cp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    pass

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

YRSID_SI = 31558149.763545603


from fastlisaresponse.utils.parallelbase import FastLISAResponseParallelModule
from fastlisaresponse_backend_cpu import tdionthefly
from scipy.interpolate import CubicSpline as CubicSpline_scipy

CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2


class TDIonTheFlyTest(unittest.TestCase):
    def test_gb_tdi(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        
        T = 2.0 * YRSID_SI # years
        t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

        sampling_frequency = 0.1
        dt = 1 / sampling_frequency

        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        # define GB parameters
        A = 1.084702251e-22
        f = 2.35962078e-3
        fdot = 1.47197271e-17
        fddot = 0.0
        iota = 1.11820901
        phi0 = 4.91128699
        psi = 2.3290324

        beta = 0.9805742971871619
        lam = 5.22979888

        gb_tdi_on_fly = tdionthefly.pyGBTDIonTheFly(orbits.ptr, T)

        N = 1024

        t_arr = np.linspace(0.0, T, 1024, endpoint=False)

        buffer = gb_tdi_on_fly.get_buffer_size(N)
        _size_of_double = 8
        num_points = int(buffer / _size_of_double)

        buffer = np.zeros(num_points)
        Xamp = np.zeros(N)
        Xphase = np.zeros(N)
        Yamp = np.zeros(N)
        Yphase = np.zeros(N)
        Zamp = np.zeros(N)
        Zphase = np.zeros(N)

        params = np.zeros(9)

        params[0] = A
        params[1] = f
        params[2] = fdot
        params[3] = fddot
        params[4] = phi0
        params[5] = iota
        params[6] = psi
        params[7] = lam
        params[8] = beta

        gb_tdi_on_fly.run_wave_tdi(
            buffer, buffer.shape[0] * _size_of_double,
            Xamp, Xphase,
            Yamp, Yphase,
            Zamp, Zphase,
            params, t_arr,
            N
        )

    def test_td_spline_tdi(self):

        _Tobs = YRSID_SI
        dt = 10000.0
        N = int(_Tobs / dt)
        t_arr = np.arange(N) * dt
        
        phi_of_t = 2 * np.pi * 1e-3 * t_arr
        amp_of_t = np.ones_like(t_arr)
        
        phase_scipy_spl = CubicSpline_scipy(t_arr, phi_of_t)
        amp_scipy_spl = CubicSpline_scipy(t_arr, amp_of_t)

        from fastlisaresponse.tdionfly import TDTDIonTheFly

        sampling_frequency = 0.1
        td_spline_tdi = TDTDIonTheFly(t_arr, amp_scipy_spl, phase_scipy_spl, sampling_frequency)
        
        inc = 0.2
        psi = 0.8
        lam = 4.0923421
        beta = -0.234091341

        output_info_td = td_spline_tdi(inc, psi, lam, beta)  
            
    def test_fd_spline_tdi(self):

        _Tobs = YRSID_SI
        dt = 10000.0
        N = int(_Tobs / dt)
        t_arr = np.arange(N) * dt
        
        f_of_t = 1e-3 + 1e-12 * t_arr

        freq_scipy_spl = CubicSpline_scipy(t_arr, f_of_t)

        freq_c1 = freq_scipy_spl.c[2, :].copy()
        freq_c2 = freq_scipy_spl.c[1, :].copy()
        freq_c3 = freq_scipy_spl.c[0, :].copy()

        from gpubackendtools import wrapper
        (_t_arr, _phi_of_t, _freq_c1, _freq_c2, _freq_c3), twkargs = wrapper(t_arr, f_of_t, freq_c1, freq_c2, freq_c3)
        freq_spl = tdionthefly.pyCubicSplineWrap(_t_arr, _phi_of_t, _freq_c1, _freq_c2, _freq_c3, dt, N, CUBIC_SPLINE_LINEAR_SPACING)
        
        amp_of_t = np.ones_like(t_arr)
        amp_scipy_spl = CubicSpline_scipy(t_arr, amp_of_t)

        amp_c1 = amp_scipy_spl.c[2, :].copy()
        amp_c2 = amp_scipy_spl.c[1, :].copy()
        amp_c3 = amp_scipy_spl.c[0, :].copy()

        (_t_arr, _amp_of_t, _amp_c1, _amp_c2, _amp_c3), twkargs = wrapper(t_arr, amp_of_t, amp_c1, amp_c2, amp_c3)
        amp_spl = tdionthefly.pyCubicSplineWrap(_t_arr, _amp_of_t, _amp_c1, _amp_c2, _amp_c3, dt, N, CUBIC_SPLINE_LINEAR_SPACING)

        orbits = EqualArmlengthOrbits()
        orbits.configure(linear_interp_setup=True)
        (_orbits, _amp_spl, _freq_spl), twkargs = wrapper(orbits, amp_spl, freq_spl)

        fd_spline_tdi = tdionthefly.pyFDSplineTDIWaveform(_orbits, _amp_spl, _freq_spl)
        
        inc = 0.2
        psi = 0.8
        lam = 4.0923421
        beta = -0.234091341
        params = np.array([inc, psi, lam, beta])

        buffer = fd_spline_tdi.get_buffer_size(N)
        _size_of_double = 8
        num_points = int(buffer / _size_of_double)

        buffer = np.zeros(num_points)
        Xamp = np.zeros(N)
        Xphase = np.zeros(N)
        Yamp = np.zeros(N)
        Yphase = np.zeros(N)
        Zamp = np.zeros(N)
        Zphase = np.zeros(N)

        fd_spline_tdi.run_wave_tdi(
            buffer, buffer.shape[0] * _size_of_double,
            Xamp, Xphase,
            Yamp, Yphase,
            Zamp, Zphase,
            params, t_arr,
            N
        )

