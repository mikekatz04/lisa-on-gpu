import numpy as np

try:
    import cupy as xp

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu = False

from response import pyResponseTDI
from few.utils.constants import *

from ldc.waveform.waveform import HpHc
from ldc.lisa.orbits import Orbits


class GBLike(pyResponseTDI):
    def __init__(self, sampling_frequency, Tobs, tdi_kwargs, use_gpu=False):
        pyResponseTDI.__init__(self, sampling_frequency, use_gpu=use_gpu, **tdi_kwargs)

        if use_gpu:
            self.xp = xp
        else:
            self.xp = np
        self.n = int(Tobs * YRSID_SI / self.dt)
        self.Tobs = self.n * self.dt

        # add the buffer
        self.t_buffer = self.total_buffer * self.dt
        t = self.xp.arange(0, self.n + 2 * self.total_buffer) * self.dt

        self.t_in = t - self.t_buffer

    def _get_h(self, A, f, fdot, iota, phi0, psi):
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        # phi0 is phi(t = 0, which is shifted due to t_buffer)
        phase = 2 * np.pi * (f * self.t_in + 1.0 / 2.0 * fdot * self.t_in ** 2) - phi0

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc

    def __call__(self, A, f, fdot, iota, phi0, psi, lam, beta):

        h = self._get_h(A, f, fdot, iota, phi0, psi)

        self.get_projections(h, lam, beta)
        tdi_out = self.get_tdi_delays()

        return list(tdi_out)


if __name__ == "__main__":
    from astropy import units as un
    import doctest
    from ldc.waveform.waveform import HpHc
    from ldc.lisa.orbits import Orbits
    from ldc.lisa.projection import ProjectedStrain

    use_gpu = gpu

    num_pts_in = int(3e6)

    sampling_frequency = 0.1
    dt = 1 / sampling_frequency
    T = (num_pts_in * dt) / YRSID_SI

    order = 25

    orbit_file = "keplerian-orbits.h5"

    config = dict(
        {
            "nominal_arm_length": 2.5e9 * un.m,
            "initial_rotation": 0 * un.rad,
            "initial_position": 0 * un.rad,
            "orbit_type": "analytic",
        }
    )

    pGB = dict(
        {
            "Amplitude": 1.07345e-22,
            "EclipticLatitude": 0.312414 * un.rad,
            "EclipticLongitude": -2.75291 * un.rad,
            "Frequency": 0.00135962 * un.Hz,
            "FrequencyDerivative": 8.94581279e-19 * un.Unit("Hz2"),
            "Inclination": 0.523599 * un.rad,
            "InitialPhase": 3.0581565 * un.rad,
            "Polarization": 3.5621656 * un.rad,
        }
    )

    GB = HpHc.type("my-galactic-binary", "GB", "TD_fdot")
    GB.set_param(pGB)
    """
    orbits = Orbits.type(config)

    proj = ProjectedStrain(orbits)
    """
    tdi_kwargs = dict(
        orbit_kwargs=dict(orbits_file=orbit_file),
        order=order,
        tdi="1st generation",
        tdi_chan="XYZ",
    )
    import time

    gb = GBLike(sampling_frequency, T, tdi_kwargs, use_gpu=use_gpu)

    A = GB.source_parameters["Amplitude"]
    f = GB.source_parameters["Frequency"]
    fdot = GB.source_parameters["FrequencyDerivative"]
    iota = GB.source_parameters["Inclination"]
    phi0 = GB.source_parameters["InitialPhase"]
    psi = GB.source_parameters["Polarization"]

    beta = GB.source_parameters["EclipticLatitude"]
    lam = GB.source_parameters["EclipticLongitude"]

    num = 1
    chans = gb(A, f, fdot, iota, phi0, psi, lam, beta)
    st = time.perf_counter()
    for i in range(num):
        chans = gb(A, f, fdot, iota, phi0, psi, lam, beta)

    et = time.perf_counter()

    X1, Y1, Z1 = chans
    print("num delays:", num_pts_in, (et - st) / num)
    breakpoint()
    proj = ProjectedStrain(orbits)

    t_data = np.load("t_data.npy")

    sampling_frequency = 0.1
    dt = 1 / sampling_frequency

    yArm = proj.arm_response(t_data[0], t_data[-1], dt, [GB])

    X = proj.compute_tdi_x(np.arange(t_data[0], t_data[-1], dt))[:num_pts_in]
    Y = proj.compute_tdi_y(np.arange(t_data[0], t_data[-1], dt))[:num_pts_in]
    Z = proj.compute_tdi_z(np.arange(t_data[0], t_data[-1], dt))[:num_pts_in]
    try:
        X1, Y1, Z1 = X1.get(), Y1.get(), Z1.get()

    except AttributeError:
        pass

    mismatch = [
        1.0 - np.dot(K, K1) / np.sqrt(np.dot(K, K) * np.dot(K1, K1))
        for K, K1 in zip([X, Y, Z], [X1, Y1, Z1])
    ]
    print(mismatch)
    breakpoint()
