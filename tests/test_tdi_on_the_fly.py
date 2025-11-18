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
    # def test_gb_tdi(self):
    #     force_backend = "cpu" if not gpu_available else "gpu"
        
    #     T = 2.0 * YRSID_SI # years
    #     t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

    #     sampling_frequency = 0.1
    #     dt = 1 / sampling_frequency

    #     orbits = EqualArmlengthOrbits(force_backend=force_backend)
    #     orbits.configure(linear_interp_setup=True)

    #     # define GB parameters
    #     f = 6.000000000000e-03
    #     costh = 2.000000000000e-01
    #     beta = np.pi / 2 - np.arccos(costh)
    #     phi = 1.000000000000e+00
    #     lam = phi
    #     A = 1.510465911233e-21
    #     cosi = -3.000000000000e-01
    #     iota = np.arccos(cosi)
    #     psi = 8.000000000000e-01
    #     phi0 = 1.200000000000e+00
    #     fdot = 1.590872976087e-15

    #     gb_tdi_on_fly = tdionthefly.pyGBTDIonTheFly(orbits.ptr, T)

    #     t_arr = np.linspace(0.0, T, 4096, endpoint=False)
    #     t_tdi_in = t_arr[1:-1]
    #     # t_tdi_in = np.array([0.000000000000e+00, 1.588751515152e+07])
        
    #     N = len(t_tdi_in)
    #     buffer = gb_tdi_on_fly.get_buffer_size(N)
    #     _size_of_double = 8
    #     num_points = int(buffer / _size_of_double)
    #     num_bin = 3

    #     buffer = np.zeros(num_points)
    #     X = np.zeros((num_bin, N), dtype=complex)  # TODO: flatten?
    #     Y = np.zeros((num_bin, N), dtype=complex)
    #     Z = np.zeros((num_bin, N), dtype=complex)
        
    #     params = np.zeros((num_bin, 9))
    #     fddot = 0.0
    #     params[:, 0] = A
    #     params[:, 1] = f
    #     params[:, 2] = fdot
    #     params[:, 3] = fddot
    #     params[:, 4] = phi0
    #     params[:, 5] = iota
    #     params[:, 6] = psi
    #     params[:, 7] = lam
    #     params[:, 8] = beta

    #     _params = params.flatten().copy()

    #     n_params = 9
    #     num_sub = num_bin
        
    #     _t_tdi_in = np.tile(t_tdi_in, (num_bin, 1)).flatten().copy()
    #     t_tdi_tmp = np.tile(t_tdi_in, (num_bin, 1))

    #     phase_ref = np.zeros_like(X, dtype=float)
    #     time_sc = np.zeros_like(phase_ref)



    #     gb_tdi_on_fly.run_wave_tdi(
    #         X, Y, Z, phase_ref,
    #         _params, _t_tdi_in,
    #         N, num_sub, n_params
    #     )
    #     breakpoint()

    def test_td_spline_tdi(self):

        _Tobs = YRSID_SI
        dt = 10000.0
        N = int(_Tobs / dt)
        t_arr = np.arange(N) * dt
        t_tdi_in = t_arr[1:-1]
        N = len(t_tdi_in)
        num_bin = 3
        phi_of_t = np.tile(2 * np.pi * 1e-3 * t_arr, (num_bin, 1))
        amp_of_t = np.tile(np.ones_like(t_arr), (num_bin, 1))
        
        from fastlisaresponse.tdionfly import TDTDIonTheFly
        
        sampling_frequency = 0.1

        td_spline_tdi = TDTDIonTheFly(t_tdi_in, amp_of_t, phi_of_t, sampling_frequency, num_bin, t_input=t_arr)
        
        inc = np.full(num_bin, 0.2)
        psi = np.full(num_bin, 0.8)
        lam = np.full(num_bin, 4.0923421)
        beta = np.full(num_bin, -0.234091341)

        # TODO: inclination appearing here? what about for emris (or precession)
        output_info_td = td_spline_tdi(inc, psi, lam, beta, return_spline=True)  
            
    def test_fd_spline_tdi(self):

        _Tobs = YRSID_SI
        dt = 10000.0
        N = int(_Tobs / dt)
        t_arr = np.arange(N) * dt
        t_tdi_in = t_arr[1:-1]
        N = len(t_tdi_in)
        f_of_t = 1e-3 + 1e-12 * t_arr

        num_bin = 5
        
        f_of_t = np.tile(1e-3 + 1e-12 * t_arr, (num_bin, 1))
        amp_of_t = np.tile(np.ones_like(t_arr), (num_bin, 1))

        # TODO: how does amp spline play in for FD

        from fastlisaresponse.tdionfly import FDTDIonTheFly

        phase_ref = (2 * np.pi * f_of_t * t_arr[None, :]).flatten().copy()

        sampling_frequency = 0.1
        fd_spline_tdi = FDTDIonTheFly(t_tdi_in, amp_of_t, f_of_t, phase_ref, sampling_frequency, num_bin, t_input=t_arr)
        
        inc = np.full(num_bin, 0.2)
        psi = np.full(num_bin, 0.8)
        lam = np.full(num_bin, 4.0923421)
        beta = np.full(num_bin, -0.234091341)

        output_info_fd = fd_spline_tdi(inc, psi, lam, beta, return_spline=True) 