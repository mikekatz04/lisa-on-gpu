import unittest
import numpy as np
import warnings
import os

path_to_file = os.path.dirname(__file__)

import fastlisaresponse_backend_cpu.tdionthefly
from fastlisaresponse.tdionfly import TDTDIOutput
        
from lisatools.detector import EqualArmlengthOrbits

from fastlisaresponse.tdiconfig import TDIConfig
from fastlisaresponse import ResponseWrapper

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


from fastlisaresponse.utils.parallelbase import FastLISAResponseParallelModule
class GBWave(FastLISAResponseParallelModule):

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0, t_vals = None):

        
        # get the t array
        if t_vals is None:
            t = self.xp.arange(0.0, T * YRSID_SI, dt)
        else:
            t = t_vals

        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 0.0  # 11.0 / 3.0 * fdot**2 / f

        amp = A * ( 1.0 + 2.0/3.0*fdot/f*t )

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = (
            2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t**2 + 1.0 / 6.0 * fddot * t**3)
            - phi0
        )

        hSp = -self.xp.cos(phase) * amp * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * amp * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc


class TDIonTheFlyTest(unittest.TestCase):
    def test_gb_tdi(self):
        force_backend = "cpu" if not gpu_available else "gpu"
        
        T = 2.0 * YRSID_SI # years
        t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

        sampling_frequency = 0.1
        dt = 1 / sampling_frequency

        orbits = EqualArmlengthOrbits(force_backend=force_backend)
        orbits.configure(linear_interp_setup=True)

        tdi_combinations = "1st generation"  # [
        #     {"link": 13, "links_for_delay": [], "sign": +1},
        #     {"link": 31, "links_for_delay": [13], "sign": +1},
        #     {"link": 12, "links_for_delay": [13, 31], "sign": +1},
        #     {"link": 21, "links_for_delay": [13, 31, 12], "sign": +1},
        #     {"link": 12, "links_for_delay": [], "sign": -1},
        #     {"link": 21, "links_for_delay": [12], "sign": -1},
        #     {"link": 13, "links_for_delay": [12, 21], "sign": -1},
        #     {"link": 31, "links_for_delay": [12, 21, 13], "sign": -1},
        # ]
        tdi_config = TDIConfig(tdi_combinations)  # "1st generation")
        
        # define GB parameters
        f = 6.000000000000e-03
        costh = 2.000000000000e-01
        beta = np.pi / 2 - np.arccos(costh)
        phi = 1.000000000000e+00
        lam = phi
        A = 1.510465911233e-21
        cosi = -3.000000000000e-01
        iota = np.arccos(cosi)
        psi = 8.000000000000e-01
        phi0 = 1.200000000000e+00
        fdot = 1.590872976087e-15
        
        import fastlisaresponse
        _backend = fastlisaresponse.get_backend(force_backend)

        cpp_orbits = _backend.OrbitsWrap(*orbits.pycppdetector_args)
        cpp_tdi_config = _backend.TDIConfigWrap(*tdi_config.pytdiconfig_args)

        gb_tdi_on_fly = _backend.GBTDIonTheFlyWrap(cpp_orbits, cpp_tdi_config, T)

        t_arr = np.linspace(0.0, T, 4096, endpoint=False)
        t_tdi_in = t_arr[1:-1]
        t_tdi_in[100] = 1.556340000000e+06
        # t_tdi_in = np.array([0.000000000000e+00, 1.588751515152e+07])
        
        N = len(t_tdi_in)
        buffer = gb_tdi_on_fly.get_buffer_size(N)
        _size_of_double = 8
        num_points = int(buffer / _size_of_double)
        num_bin = 3

        buffer = np.zeros(num_points)
        
        params = np.zeros((num_bin, 9))
        fddot = 0.0
        params[:, 0] = A
        params[:, 1] = f
        params[:, 2] = fdot
        params[:, 3] = fddot
        params[:, 4] = phi0
        params[:, 5] = iota
        params[:, 6] = psi
        params[:, 7] = lam
        params[:, 8] = beta

        _params = params.flatten().copy()

        n_params = 9
        num_sub = num_bin
        
        _t_tdi_in = np.tile(t_tdi_in, (num_bin, 1)).flatten().copy()
        t_tdi_tmp = np.tile(t_tdi_in, (num_bin, 1))
        
        # time_sc = np.zeros_like(phase_ref)
        # gb_tdi_on_fly.run_wave_tdi(
        #     X, Y, Z, phase_ref,
        #     _params, _t_tdi_in,
        #     N, num_sub, n_params
        # )

        assert len(_params) == n_params * num_sub

        tdi_channels_arr = np.zeros((N * tdi_config.nchannels * num_sub), dtype=complex)
        tdi_amp = np.zeros((N * tdi_config.nchannels * num_sub), dtype=float)
        tdi_phase = np.zeros((N * tdi_config.nchannels * num_sub), dtype=float)
        phase_ref = np.zeros((N * num_sub), dtype=float)
        assert int(np.prod(t_tdi_tmp.shape)) == N * num_sub

        buffer_length = gb_tdi_on_fly.get_buffer_size(N)
        # bool is 1 byte
        buffer = np.zeros(buffer_length, dtype=bool)

        gb_tdi_on_fly.run_wave_tdi_wrap(
            buffer, buffer_length,
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phase_ref,
            _params, _t_tdi_in,
            N, num_sub, n_params, tdi_config.nchannels
        )
        
        tdi_out_fly = TDTDIOutput(
            t_tdi_tmp, 
            tdi_amp.reshape(num_sub, tdi_config.nchannels, N), 
            tdi_phase.reshape(num_sub, tdi_config.nchannels, N), 
            phase_ref.reshape(num_sub, N), 
            fill_splines=True
        )

        force_backend = "cpu"  #  if not use_gpu else "gpu"
        gb = GBWave(force_backend=force_backend)

        T = 2.0  # years
        t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)

        sampling_frequency = 0.1
        dt = 1 / sampling_frequency

        # order of the langrangian interpolation
        order = 25

        # orbit_file_esa = path_to_file + "/../../orbit_files/esa-trailing-orbits.h5"

        # orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

        index_lambda = 6
        index_beta = 7

        tdi_kwargs_esa = dict(
            orbits=orbits,
            order=order,
            tdi=tdi_config,
            tdi_chan="XYZ",
        )

        gb_lisa_esa = ResponseWrapper(
            gb,
            T,
            dt,
            index_lambda,
            index_beta,
            t0=t0,
            flip_hx=False,  # set to True if waveform is h+ - ihx
            force_backend=force_backend,
            remove_sky_coords=True,  # True if the waveform generator does not take sky coordinates
            is_ecliptic_latitude=True,  # False if using polar angle (theta)
            remove_garbage=False,  # removes the beginning of the signal that has bad information
            **tdi_kwargs_esa,
        )

        # define GB parameters
        chans = gb_lisa_esa(A, f, fdot, iota, phi0, psi, lam, beta)

        t_new = np.tile(np.arange(chans[0].shape[-1]) * dt, (num_bin, 1))
        
        keep = (np.sum(
            (t_new > tdi_out_fly.t_arr.min(axis=-1).max()[None, None])
            & (t_new < tdi_out_fly.t_arr.max(axis=-1).min()[None, None])
        , axis=0) > 0)

        chans_fly = np.zeros((num_bin, tdi_config.nchannels, t_new.shape[-1]))
        chans_fly[:, :, keep] = tdi_out_fly.eval_tdi(t_new[:, keep])
        
        import matplotlib.pyplot as plt
        
        # fig, ax = plt.subplots(3, 1)
        # ax[0].plot(t_new[0], chans_fly[0, 0], color="C0")
        # ax[1].plot(t_new[0], chans_fly[0, 1], color="C0")
        # ax[2].plot(t_new[0], chans_fly[0, 2], color="C0")

        # ax[0].plot(t_new[0], chans[0], ls='--', color="C1")
        # ax[1].plot(t_new[0], chans[1], ls='--', color="C1")
        # ax[2].plot(t_new[0], chans[2], ls='--', color="C1")

        # plt.show()
        chans_fft = np.fft.rfft(np.asarray(chans)[:, keep], axis=-1)
        chans_fly_fft = np.fft.rfft(chans_fly[0][:, keep], axis=-1)

        overlap = np.sum(chans_fly_fft.conj() * chans_fft) / np.sqrt(np.sum(chans_fft.conj() * chans_fft) * np.sum(chans_fly_fft.conj() * chans_fly_fft))
        breakpoint()

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

        dt_new = 5.0
        t_new = np.tile(np.arange(
            output_info_td.t_arr[0].min().item(),
            output_info_td.t_arr[0].max().item(),
            dt_new,
        ), (num_bin, 1))

        tdi_out = output_info_td.eval_tdi(t_new)

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

        df_new = 1 / YRSID_SI

        f_new = np.tile(np.arange(
            np.sort(f_of_t, axis=-1)[0, 1].item(),
            np.sort(f_of_t, axis=-1)[0, -2].item(),
            df_new,
        ), (num_bin, 1))
        tdi_out = output_info_fd.eval_tdi(f_new)


