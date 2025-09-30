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
from scipy.interpolate import CubicSpline as CubicSpline_scipy

CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2

class CubicSplineTest(unittest.TestCase):
    def test_gb_tdi(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        
        N = 1000
        x_in = np.linspace(0.0, 1.0, N)
        dx = x_in[1] - x_in[0]
        y_in = x_in ** 3 + x_in ** 2 + x_in ** 1 + x_in
        
        spl = CubicSpline_scipy(x_in, y_in)

        c1 = spl.c[2, :]
        c2 = spl.c[1, :]
        c3 = spl.c[0, :]

        x_new = np.random.uniform(x_in[0], x_in[-1], size=10000)
        scipy_check = spl(x_new)

        from gpubackendtools import wrapper
        (_x_in, _y_in, _c1, _c2, _c3), twkargs = wrapper(x_in, y_in, c1, c2, c3)
        our_spl = fastlisaresponse_backend_cpu.tdionthefly.pyCubicSplineWrap(_x_in, _y_in, _c1, _c2, _c3, dx.item(), N, CUBIC_SPLINE_LINEAR_SPACING)
        
        our_check = np.zeros_like(scipy_check)
        our_spl.eval(our_check, x_new, len(x_new))

        self.assertTrue(np.allclose(our_check, scipy_check))
