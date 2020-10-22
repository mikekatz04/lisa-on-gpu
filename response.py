import numpy as np

try:
    import cupy as xp
    from pyresponse import get_response_wrap

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu = False

import time
import h5py


def get_factorial(n):
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact

from few.waveform import FastSchwarzschildEccentricFlux

class pyResponseTDI(object):
    def __init__(self, sampling_frequency, orbits_file="orbits.h5", order=25, num_factorials=100):

        self.sampling_frequency = sampling_frequency
        self.dt = 1/sampling_frequency

        self.order = order
        self.buffer_integer = self.order * 2

        self.num_factorials = num_factorials
        factorials = np.asarray([float(get_factorial(n)) for n in range(num_factorials)])
        self.factorials_in = xp.asarray(factorials)

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

        L_vals = []
        n_in = []
        for i in range(self.nlinks):
            link_0 = self.link_space_craft_0_in[i]
            link_1 = self.link_space_craft_1_in[i]

            L_vals.append(out["L" + str(link_0 + 1) + str(link_1 + 1)].T[1])

            x_val = out["n" + str(link_0 + 1) + str(link_1 + 1) + "_1"].T[1]
            y_val = out["n" + str(link_0 + 1) + str(link_1 + 1) + "_2"].T[1]
            z_val = out["n" + str(link_0 + 1) + str(link_1 + 1) + "_3"].T[1]

            norm = (x_val ** 2 + y_val ** 2 + z_val ** 2) ** (1 / 2)

            n_ij_x = x_val / norm
            n_ij_y = y_val / norm
            n_ij_z = z_val / norm

            n_in.append(np.concatenate([n_ij_x, n_ij_y, n_ij_z]))

        self.L_vals = np.concatenate(L_vals)
        self.n_in = np.concatenate(n_in)

        x = []
        for i in range(3):  # number of spacecraft
            x_i_x = out["x" + str(i + 1) + "_1"].T[1]
            x_i_y = out["x" + str(i + 1) + "_2"].T[1]
            x_i_z = out["x" + str(i + 1) + "_3"].T[1]

            x.append(np.concatenate([x_i_x, x_i_y, x_i_z]))

        self.num_spacecraft_inputs = len(out["x1_1"])
        self.x = np.concatenate(x)

    def __call__(self, input_in, beta, lam, input_start_time, rep=32):

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

        num_delays = rep * self.num_spacecraft_inputs

        y_gw = xp.zeros((self.nlinks * num_delays,), dtype=xp.float)
        k_in = xp.asarray(k)
        u_in = xp.asarray(u)
        v_in = xp.asarray(v)

        x = xp.concatenate(
            [xp.asarray(self.x).reshape(3, 3, self.num_spacecraft_inputs) for i in range(rep)], axis=-1
        ).flatten()

        n_in = xp.concatenate(
            [xp.asarray(self.n_in).reshape(self.nlinks, 3, self.num_spacecraft_inputs) for i in range(rep)], axis=-1
        ).flatten()

        L_vals = xp.concatenate(
            [xp.asarray(self.L_vals).reshape(self.nlinks, self.num_spacecraft_inputs) for i in range(rep)]
        ).flatten()

        input_in = xp.asarray(input_in)
        num_pts_in = len(input_in)

        st = time.perf_counter()
        num = 25
        for i in range(num):
            get_response_wrap(
                y_gw,
                k_in,
                u_in,
                v_in,
                dt,
                x,
                n_in,
                num_delays,
                self.link_space_craft_0_in,
                self.link_space_craft_1_in,
                L_vals,
                input_in,
                num_pts_in,
                self.order,
                self.sampling_frequency,
                self.buffer_integer,
                self.factorials_in,
                self.num_factorials,
                input_start_time,
            )
        et = time.perf_counter()
        print("Num delays:", self.num_spacecraft_inputs * rep, (et - st) / num)

        return y_gw.reshape(self.nlinks, -1)


use_gpu = True

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

num_pts_in = int(1e7)
A = 1e-22

M = 1e6
mu = 1e1
p0 = 12.0
e0 = 0.4
theta = np.pi/3  # polar viewing angle
phi = np.pi/4  # azimuthal viewing angle

sampling_frequency = 1.0
dt = 1/sampling_frequency
T = num_pts_in * dt

beta = 0.4
lam = 1.3

input_in = A * few(M, mu, p0, e0, theta, phi, dt=dt, T=T)

response = pyResponseTDI(sampling_frequency, orbits_file="orbits.h5", order=25, num_factorials=100)
rep = 1
if gpu:

    y_gw = response(input_in, beta, lam, input_start_time, rep=rep)
        # print(i)

breakpoint()
