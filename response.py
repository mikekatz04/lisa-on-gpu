import numpy as np

try:
    import cupy as xp
    from pyresponse import get_response_wrap, get_tdi_delays_wrap

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu = False

import time
import h5py

from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *

def get_factorial(n):
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact

from few.waveform import FastSchwarzschildEccentricFlux

class pyResponseTDI(object):
    def __init__(self, sampling_frequency, orbits_file="orbits.h5", num_interp_points=100, order=25, num_factorials=100):

        self.sampling_frequency = sampling_frequency
        self.dt = 1/sampling_frequency

        self.order = order
        self.buffer_integer = self.order * 2

        self.num_factorials = num_factorials
        factorials = np.asarray([float(get_factorial(n)) for n in range(num_factorials)])
        self.factorials_in = xp.asarray(factorials)

        self.num_interp_points = num_interp_points
        self._init_link_indices()
        self._init_orbit_information(orbits_file)

    def _init_link_indices(self):
        self.nlinks = 6

        link_space_craft_0 = np.zeros((self.nlinks,), dtype=int)
        link_space_craft_1 = np.zeros((self.nlinks,), dtype=int)
        link_space_craft_0[0] = 0
        link_space_craft_1[0] = 1
        link_space_craft_0[1] = 1
        link_space_craft_1[1] = 0

        link_space_craft_0[2] = 0
        link_space_craft_1[2] = 2
        link_space_craft_0[3] = 2
        link_space_craft_1[3] = 0

        link_space_craft_0[4] = 1
        link_space_craft_1[4] = 2
        link_space_craft_0[5] = 2
        link_space_craft_1[5] = 1

        self.link_space_craft_0_in = xp.asarray(link_space_craft_0).astype(xp.int32)
        self.link_space_craft_1_in = xp.asarray(link_space_craft_1).astype(xp.int32)

    def _init_orbit_information(self, orbits_file):
        out = {}
        with h5py.File(orbits_file, "r") as f:
            for key in f:
                out[key] = f[key][:]

        self.num_spacecraft_inputs = length = len(out['n12_1'])
        L_vals = []
        n_in = []

        self.t_vals = out['n12_1'].T[0].astype(xp.float64)

        slice_int = int(length / self.num_interp_points)

        inds = np.arange(0, length, slice_int)

        self.init_len = len(inds)
        self.ninterps = 33

        if inds[-1] != length - 1:
            inds = np.append(inds, length -1)
            self.init_len += 1

        self.t_vals = self.t_vals[inds] * 1000.0
        interp_in = xp.zeros((self.ninterps, self.init_len))

        for i in range(3):  # number of spacecraft

            interp_in[i * 3 + 0] = xp.asarray(out["x" + str(i + 1) + "_1"].T[1][inds]).astype(xp.float64)
            interp_in[i * 3 + 1] = xp.asarray(out["x" + str(i + 1) + "_2"].T[1][inds]).astype(xp.float64)
            interp_in[i * 3 + 2] = xp.asarray(out["x" + str(i + 1) + "_3"].T[1][inds]).astype(xp.float64)

        for i in range(self.nlinks):
            start_ind = 3 * 3
            link_0 = self.link_space_craft_0_in[i]
            link_1 = self.link_space_craft_1_in[i]

            x_val = xp.asarray(out["n" + str(link_0 + 1) + str(link_1 + 1) + "_1"].T[1][inds]).astype(xp.float64)
            y_val = xp.asarray(out["n" + str(link_0 + 1) + str(link_1 + 1) + "_2"].T[1][inds]).astype(xp.float64)
            z_val = xp.asarray(out["n" + str(link_0 + 1) + str(link_1 + 1) + "_3"].T[1][inds]).astype(xp.float64)

            norm = (x_val ** 2 + y_val ** 2 + z_val ** 2) ** (1 / 2)

            n_ij_x = x_val / norm
            n_ij_y = y_val / norm
            n_ij_z = z_val / norm

            interp_in[start_ind + i * 3 + 0] = n_ij_x
            interp_in[start_ind + i * 3 + 1] = n_ij_y
            interp_in[start_ind + i * 3 + 2] = n_ij_z

            start_ind = 3 * 3 + 6 * 3

            interp_in[start_ind + i] = xp.asarray(out["L" + str(link_0 + 1) + str(link_1 + 1)].T[1][inds]).astype(xp.float64)

        L_start_ind = 3 * 3 + 6 * 3
        self.spline = CubicSplineInterpolant(xp.asarray(self.t_vals), interp_in, use_gpu=gpu)
        self.L_only_spline = CubicSplineInterpolant(xp.asarray(self.t_vals), interp_in[L_start_ind:], use_gpu=gpu)

    @property
    def y_gw(self):
        return self.y_gw_flat.reshape(self.nlinks, -1)

    def get_projections(self, num_delays, input_in, beta, lam, input_start_time):

        k = np.zeros(3, dtype=np.float)
        u = np.zeros(3, dtype=np.float)
        v = np.zeros(3, dtype=np.float)

        cosbeta = np.cos(beta)
        sinbeta = np.sin(beta)

        coslam = np.cos(lam)
        sinlam = np.sin(lam)

        u[0] = -sinbeta * coslam
        u[1] = sinbeta * sinlam
        u[2] = cosbeta
        v[0] = sinlam
        v[1] = -coslam
        v[2] = 0.0
        k[0] = -cosbeta * coslam
        k[1] = -cosbeta * sinlam
        k[2] = -cosbeta

        y_gw = xp.zeros((self.nlinks * num_delays,), dtype=xp.float)
        k_in = xp.asarray(k)
        u_in = xp.asarray(u)
        v_in = xp.asarray(v)

        input_in = xp.asarray(input_in)
        num_pts_in = len(input_in)

        st = time.perf_counter()
        num = 200

        for i in range(num):

            get_response_wrap(
                y_gw,
                k_in,
                u_in,
                v_in,
                self.dt,
                num_delays,
                self.link_space_craft_0_in,
                self.link_space_craft_1_in,
                input_in,
                num_pts_in,
                self.order,
                self.sampling_frequency,
                self.buffer_integer,
                self.factorials_in,
                self.num_factorials,
                input_start_time,
                self.spline.interp_array,
                len(self.t_vals),
                self.t_vals,
            )
        et = time.perf_counter()
        print("Num delays:", num_delays, (et - st) / num)

        self.y_gw_flat = y_gw
        self.y_gw_length = num_delays

    @property
    def delayed_links(self):
        return self.delayed_links_flat.reshape(self.num_units, -1)

    @property
    def tdi_delay_info(self):
        temp = self.delayed_links
        return [dict(unit=i, link=self.link_inds[i], factor=self.delay_factor[i], delayed_link=temp[i]) for i in range(self.num_units)]

    def get_tdi_delays(self, num_delays, link_inds, delay_factor, input_start_time):

        self.num_delays_tdi = num_delays
        self.num_units = len(link_inds)
        self.delayed_links_flat = xp.zeros(self.num_units * self.num_delays_tdi, dtype=xp.float64)
        self.link_inds = xp.asarray(link_inds).astype(xp.int32)
        self.delay_factor = xp.asarray(delay_factor).astype(xp.int32)

        get_tdi_delays_wrap(self.delayed_links_flat, self.y_gw_flat, self.y_gw_length,
                            self.num_delays_tdi, self.dt, self.link_inds, self.delay_factor, self.num_units,
                            self.order, self.sampling_frequency, self.buffer_integer, self.factorials_in,
                            self.num_factorials, input_start_time,
                            self.L_only_spline.interp_array, len(self.t_vals), self.t_vals)


use_gpu = gpu

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
)

input_start_time = -10000.0

num_pts_in = int(4e7)
A = 1e-22

M = 1e6
mu = 1e1
p0 = 12.0
e0 = 0.4
theta = np.pi/3  # polar viewing angle
phi = np.pi/4  # azimuthal viewing angle

sampling_frequency = 1.0
dt = 1/sampling_frequency
T = (num_pts_in * dt) / YRSID_SI

beta = 0.4
lam = 1.3

input_in = A * few(M, mu, p0, e0, theta, phi, dt=dt, T=T)

response = pyResponseTDI(sampling_frequency, orbits_file="orbits.h5", order=16, num_factorials=100)

link_inds = np.concatenate([np.arange(6), np.arange(6)]).astype(dtype=np.int32)
delay_factor = np.ones_like(link_inds)
delay_factor[6:] = 2
link_inds = delay_factor.copy()

if gpu:

    response.get_projections(int(1e5), input_in, beta, lam, input_start_time)
    response.get_tdi_delays(int(1e4), link_inds, delay_factor, input_start_time)
        # print(i)

breakpoint()
