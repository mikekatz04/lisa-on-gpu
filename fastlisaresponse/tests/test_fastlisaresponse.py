import unittest
import numpy as np
import warnings
import os

path_to_file = os.path.dirname(__file__)

from fastlisaresponse import ResponseWrapper, ResponseWrapperDetectorFrame
from fastlisaresponse.utils import get_overlap

try:
    import cupy as xp

    gpu_available = True

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp

    warnings.warn(
        "CuPy is not installed or a gpu is not available. If trying to run on a gpu, please install CuPy."
    )
    gpu_available = False

YRSID_SI = 31558149.763545603


class GBWave:
    def __init__(self, use_gpu=False):

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0, t=None):

        # get the t array
        if t is None:
            t = self.xp.arange(0.0, T * YRSID_SI, dt)
        elif not isinstance(t, self.xp.ndarray):
            raise ValueError("If providing t, must be xp.ndarray.")
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot ** 2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t ** 2 + 1.0 / 6.0 * fddot * t ** 3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc


class ResponseTest(unittest.TestCase):
    def run_test(self, tdi_gen, use_gpu, delays_first=False):

        gb = GBWave(use_gpu=use_gpu)

        T = 2.0  # years
        t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

        sampling_frequency = 0.1
        dt = 1 / sampling_frequency

        # order of the langrangian interpolation
        order = 25

        orbit_file_esa = path_to_file + "/../../orbit_files/esa-trailing-orbits.h5"

        orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

        index_lambda = 6
        index_beta = 7

        tdi_kwargs_esa = dict(
            orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
        )

        if delays_first:
            gb_lisa_esa = ResponseWrapperDetectorFrame(
                gb,
                T,
                dt,
                index_lambda,
                index_beta,
                t0=t0,
                flip_hx=False,  # set to True if waveform is h+ - ihx
                use_gpu=use_gpu,
                remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
                is_ecliptic_latitude=True,  # False if using polar angle (theta)
                remove_garbage=True,  # removes the beginning of the signal that has bad information
                **tdi_kwargs_esa,
            )
        else:
            gb_lisa_esa = ResponseWrapper(
                gb,
                T,
                dt,
                index_lambda,
                index_beta,
                t0=t0,
                flip_hx=False,  # set to True if waveform is h+ - ihx
                use_gpu=use_gpu,
                remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
                is_ecliptic_latitude=True,  # False if using polar angle (theta)
                remove_garbage=True,  # removes the beginning of the signal that has bad information
                **tdi_kwargs_esa,
            )

        # define GB parameters
        A = 1.084702251e-22
        f = 2.35962078e-3
        fdot = 1.47197271e-17
        iota = 1.11820901
        phi0 = 4.91128699
        psi = 2.3290324

        beta = 0.9805742971871619
        lam = 5.22979888

        chans = gb_lisa_esa(A, f, fdot, iota, phi0, psi, lam, beta)

        return chans

    def test_tdi_1st_generation(self):

        waveform_cpu = self.run_test("1st generation", False)

        self.assertTrue(np.all(np.isnan(waveform_cpu) == False))

        if gpu_available:
            waveform_gpu = self.run_test("1st generation", True)
            mm = 1.0 - get_overlap(
                xp.asarray(waveform_cpu),
                xp.asarray(waveform_gpu),
                use_gpu=gpu_available,
            )
            self.assertLess(mm, 1e-10)

    def test_tdi_2nd_generation(self):

        waveform_cpu = self.run_test("2nd generation", False)

        self.assertTrue(np.all(np.isnan(waveform_cpu) == False))

        if gpu_available:
            waveform_gpu = self.run_test("2nd generation", True)
            mm = 1.0 - get_overlap(
                xp.asarray(waveform_cpu), waveform_gpu, use_gpu=gpu_available
            )
            self.assertLess(mm, 1e-10)

    def test_delays_first_tdi_2nd_generation(self):

        waveform_cpu1 = self.run_test("2nd generation", False, delays_first=True)
        self.assertTrue(np.all(np.isnan(waveform_cpu1) == False))

        if gpu_available:
            waveform_gpu1 = self.run_test("2nd generation", True, delays_first=True)
            self.assertTrue(np.all(np.isnan(waveform_gpu1) == False))
            mm = 1.0 - get_overlap(
                xp.asarray(waveform_cpu1), waveform_gpu1, use_gpu=gpu_available
            )
            self.assertLess(mm, 1e-10)
