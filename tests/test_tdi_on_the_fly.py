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

class TDIonTheFlyTest(unittest.TestCase):
    def test_tdi(self):
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

        gb_tdi_on_fly = fastlisaresponse_backend_cpu.tdionthefly.pyGBTDIonTheFly(orbits.ptr, T)

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
        
        # TODO: GET FREQUENCIES !!!!!
        breakpoint()
