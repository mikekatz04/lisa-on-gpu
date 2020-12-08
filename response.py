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

from scipy.interpolate import CubicSpline

from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.utils.constants import *


def get_factorial(n):
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact


from few.waveform import FastSchwarzschildEccentricFlux

from math import factorial

factorials = np.array([factorial(i) for i in range(30)])

C_inv = 3.3356409519815204e-09


class VariableFractionDelay:
    def __init__(self, sampling_frequency, order=25):

        self.sampling_frequency = sampling_frequency

        self.order = order

        if (self.order % 2) != 1:
            raise ValueError("order must be odd.")

        self.point_count = self.order + 1
        self.half_point_count = int(self.point_count / 2)

        self.h = self.half_point_count

        self.min_delay = self.half_point_count / self.sampling_frequency

        self.max_delay = 100000.0
        self.max_integer_delay = int(np.floor(self.max_delay * self.sampling_frequency))

    def update_coefficients(self, fraction):

        e = fraction
        h = self.half_point_count
        A = 1.0

        for i in range(1, h):
            A *= (i + e) * (i + 1 - e)

        denominator = factorials[h - 1] * factorials[h]
        A /= denominator

        self.A = A
        self.B = 1.0 - e
        self.C = e
        self.D = e * (1.0 - e)

        E = [0.0]
        F = [0.0]
        G = [0.0]

        for j in range(1, h):
            first_term = factorials[h - 1] / factorials[h - 1 - j]
            second_term = factorials[h] / factorials[h + j]
            value = first_term * second_term
            if (j % 2) != 0:
                value = -value

            E.append(value)
            F.append(j + e)
            G.append(j + (1 - e))

        self.E = np.asarray(E)
        self.F = np.asarray(F)
        self.G = np.asarray(G)

    def var_frac_delay(self, integer_delay, fraction, input):
        # st()
        # clipped_delay = np.clip(
        #    delay - input_start_time, self.min_delay, self.max_integer_delay
        # )
        # integer_delay = int(np.ceil(clipped_delay * sampling_frequency) - 1)
        # fraction = 1.0 + integer_delay - clipped_delay * sampling_frequency
        self.update_coefficients(fraction)

        d = integer_delay
        h = self.half_point_count
        sumit = 0.0

        for j in range(1, h):
            sumit += self.E[j] * (
                input[d + 1 + j] / self.F[j] + input[d - j] / self.G[j]
            )

        result = self.A * (self.B * input[d + 1] + self.C * input[d] + self.D * sumit)

        return result

    def __call__(self, input_in, i, dt, x0, x1, n, L, lam, beta, input_start_time):

        t = i * dt
        k = np.zeros(3, dtype=np.float)
        u = np.zeros(3, dtype=np.float)
        v = np.zeros(3, dtype=np.float)

        cosbeta = np.cos(beta)
        sinbeta = np.sin(beta)

        coslam = np.cos(lam)
        sinlam = np.sin(lam)

        u[0] = -sinbeta * coslam
        u[1] = -sinbeta * sinlam
        u[2] = cosbeta
        v[0] = sinlam
        v[1] = -coslam
        v[2] = 0.0
        k[0] = -cosbeta * coslam
        k[1] = -cosbeta * sinlam
        k[2] = -sinbeta
        u_dot_n = np.dot(u, n)
        v_dot_n = np.dot(v, n)

        xi_p = (u_dot_n * u_dot_n) - (v_dot_n * v_dot_n)
        xi_c = 2.0 * u_dot_n * v_dot_n

        k_dot_n = np.dot(k, n)
        k_dot_x0 = np.dot(k, x0)
        k_dot_x1 = np.dot(k, x1)

        delay0 = t - L - k_dot_x0 * C_inv
        delay1 = t - k_dot_x1 * C_inv

        clipped_delay0 = delay0 - input_start_time
        integer_delay0 = int(np.ceil(clipped_delay0 * sampling_frequency)) - 1
        fraction0 = 1.0 + integer_delay0 - clipped_delay0 * sampling_frequency

        clipped_delay1 = delay1 - input_start_time
        integer_delay1 = int(np.ceil(clipped_delay1 * sampling_frequency)) - 1
        fraction1 = 1.0 + integer_delay1 - clipped_delay1 * sampling_frequency

        temp = self.var_frac_delay(integer_delay0, fraction0, input_in)
        hp_del0, hc_del0 = temp.real, temp.imag

        temp = self.var_frac_delay(integer_delay1, fraction1, input_in)
        hp_del1, hc_del1 = temp.real, temp.imag

        pre_factor = 1.0 / (2 * (1.0 - k_dot_n))
        large_factor = (hp_del0 - hp_del1) * xi_p + (hc_del0 - hc_del1) * xi_c

        return pre_factor * large_factor


class pyResponseTDI(object):
    def __init__(
        self,
        sampling_frequency,
        orbits_file="orbits.h5",
        num_interp_points=100,
        order=25,
        num_factorials=100,
        tdi="1st generation",
    ):

        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency

        self.order = order
        self.buffer_integer = self.order * 2 + 1
        self.half_order = int((order + 1) / 2)

        self.tdi = tdi

        self.num_interp_points = num_interp_points
        self._fill_A_E()
        self._init_link_indices()
        self._init_orbit_information(orbits_file)
        self._init_TDI_delays()

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

    def _fill_A_E(self):

        factorials = np.asarray([float(get_factorial(n)) for n in range(40)])

        self.num_A = 1001
        self.deps = 1.0 / (self.num_A - 1)

        eps = np.arange(self.num_A) * self.deps

        h = self.half_order

        denominator = factorials[h - 1] * factorials[h]

        A_in = np.zeros_like(eps)
        for j, eps_i in enumerate(eps):
            A = 1.0
            for i in range(1, h):
                A *= (i + eps_i) * (i + 1 - eps_i)

            A /= denominator
            A_in[j] = A

        self.A_in = xp.asarray(A_in)

        E_in = xp.zeros((self.half_order,))

        for j in range(1, self.half_order):
            first_term = factorials[h - 1] / factorials[h - 1 - j]
            second_term = factorials[h] / factorials[h + j]
            value = first_term * second_term
            value = value * (-1.0) ** j
            E_in[j - 1] = value

        self.E_in = xp.asarray(E_in)

    def _init_orbit_information(self, orbits_file):
        out = {}
        with h5py.File(orbits_file, "r") as f:
            for key in f:
                out[key] = f[key][:]

        t_in = out["t"]
        length_in = len(t_in)

        x_in = []
        for i in range(3):
            for let in ["x", "y", "z"]:
                x_in.append(out["sc_" + str(i + 1)][let])

        n_in = []
        L_in = []

        for link_i in range(self.nlinks):
            sc0 = self.link_space_craft_0_in[link_i] + 1
            sc1 = self.link_space_craft_1_in[link_i] + 1

            x_val = out["sc_" + str(sc0)]["x"] - out["sc_" + str(sc1)]["x"]
            y_val = out["sc_" + str(sc0)]["y"] - out["sc_" + str(sc1)]["y"]
            z_val = out["sc_" + str(sc0)]["z"] - out["sc_" + str(sc1)]["z"]

            norm = np.sqrt(x_val ** 2 + y_val ** 2 + z_val ** 2)

            n_in.append(x_val / norm)
            n_in.append(y_val / norm)
            n_in.append(z_val / norm)

            L_in.append(out["l_" + str(sc0) + str(sc1)]["delay"])

        t_new = np.arange(0.0, t_in[-1] + self.dt, self.dt)

        for i in range(9):
            x_in[i] = CubicSpline(t_in, x_in[i])(t_new)

        for i in range(self.nlinks * 3):
            n_in[i] = CubicSpline(t_in, n_in[i])(t_new)

        for i in range(self.nlinks):
            L_in[i] = CubicSpline(t_in, L_in[i])(t_new)

        self.x_in = xp.asarray(np.concatenate(x_in))
        self.n_in = xp.asarray(np.concatenate(n_in))

        self.L_in_for_TDI = L_in
        self.L_in = xp.asarray(np.concatenate(L_in))

        self.num_orbit_inputs = len(t_new)
        self.t_data = t_new
        self.final_t = t_new[-1]

    def get_xnL(self, i, link_i):
        sc0 = self.link_space_craft_0_in[link_i]
        sc1 = self.link_space_craft_1_in[laink_i]
        x0 = np.zeros(3)
        x1 = np.zeros(3)
        n = np.zeros(3)
        for coord in range(3):
            ind0 = (sc0 * 3 + coord) * self.num_orbit_inputs + i
            ind1 = (sc1 * 3 + coord) * self.num_orbit_inputs + i
            ind_n = (link_i * 3 + coord) * self.num_orbit_inputs + i

            x0[coord] = self.x_in[ind0]
            x1[coord] = self.x_in[ind1]
            n[coord] = self.n_in[ind_n]

        L_ind = link_i * self.num_orbit_inputs + i

        L = self.L_in[L_ind]

        return x0, x1, n, L

    def _init_TDI_delays(self):
        self.num_delays_tdi = int(3.1e6)

        link_dict = {12: 0, 21: 1, 13: 2, 31: 3, 23: 4, 32: 5}
        if self.tdi == "1st generation":
            tdi_combinations = [
                {"link": 13, "links_for_delay": [], "sign": 1},
                {"link": 31, "links_for_delay": [21], "sign": 1},
                {"link": 12, "links_for_delay": [21, 23], "sign": 1},
                {"link": 21, "links_for_delay": [21, 23, 31], "sign": 1},
                {"link": 12, "links_for_delay": [], "sign": -1},
                {"link": 21, "links_for_delay": [31], "sign": -1},
                {"link": 13, "links_for_delay": [31, 32], "sign": -1},
                {"link": 31, "links_for_delay": [31, 32, 21], "sign": -1},
            ]

        elif self.tdi == "2nd generation":
            tdi_combinations = [
                {"link": 13, "links_for_delay": [], "sign": 1},
                {"link": 31, "links_for_delay": [21], "sign": 1},
                {"link": 12, "links_for_delay": [21, 23], "sign": 1},
                {"link": 21, "links_for_delay": [21, 23, 31], "sign": 1},
                {"link": 12, "links_for_delay": [21, 23, 31, 32], "sign": 1},
                {"link": 21, "links_for_delay": [21, 23, 31, 32, 31], "sign": 1},
                {"link": 13, "links_for_delay": [21, 23, 31, 32, 31, 32], "sign": 1},
                {
                    "link": 31,
                    "links_for_delay": [21, 23, 31, 32, 31, 32, 21],
                    "sign": 1,
                },
                {"link": 12, "links_for_delay": [], "sign": -1},
                {"link": 21, "links_for_delay": [31], "sign": -1},
                {"link": 13, "links_for_delay": [31, 32], "sign": -1},
                {"link": 31, "links_for_delay": [31, 32, 21], "sign": -1},
                {"link": 13, "links_for_delay": [31, 32, 21, 23], "sign": -1},
                {"link": 31, "links_for_delay": [31, 32, 21, 23, 21], "sign": -1},
                {"link": 12, "links_for_delay": [31, 32, 21, 23, 21, 23], "sign": -1},
                {
                    "link": 21,
                    "links_for_delay": [31, 32, 21, 23, 21, 23, 31],
                    "sign": -1,
                },
            ]

        elif isinstance(self.tdi, list):
            tdi_combinations = self.tdi

        else:
            raise ValueError(
                "tdi kwarg should be '1st generation', '2nd generation', or a list with a specific tdi combination."
            )

        channels_no_delays = [[], [], []]

        num_delay_comps = 0
        for i, tdi in enumerate(tdi_combinations):
            if len(tdi["links_for_delay"]) == 0:
                for j in range(3):
                    ind = link_dict[self._cyclic_permutation(tdi["link"], j)]
                    channels_no_delays[j].append([ind, tdi["sign"]])

            else:
                num_delay_comps += 1

        self.channels_no_delays = np.asarray(channels_no_delays)

        self.num_tdi_combinations = len(tdi_combinations)
        self.num_tdi_delay_comps = num_delay_comps

        delays = np.zeros((3, self.num_tdi_delay_comps, self.num_delays_tdi))

        delays[:] = self.t_data[: self.num_delays_tdi]
        link_inds = np.zeros((3, self.num_tdi_delay_comps), dtype=np.int32)

        # cyclic permuatations for X, Y, Z
        for j in range(3):
            i = 0
            for tdi in tdi_combinations:

                if len(tdi["links_for_delay"]) == 0:
                    continue

                link = self._cyclic_permutation(tdi["link"], j)
                link_inds[j][i] = link_dict[link]

                for link in tdi["links_for_delay"]:
                    link = self._cyclic_permutation(link, j)
                    link_index = link_dict[link]
                    delays[j][i] -= self.L_in_for_TDI[link_index][: self.num_delays_tdi]
                i += 1

        self.num_units = self.num_tdi_delay_comps
        self.num_channels = 3
        self.link_inds = xp.asarray(link_inds.flatten()).astype(xp.int32)
        self.tdi_delays = xp.asarray(delays.flatten())

    def _cyclic_permutation(self, link, permutation):

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
    def y_gw(self):
        return self.y_gw_flat.reshape(self.nlinks, -1)

    def get_projections(self, num_delays, input_in, beta, lam, input_start_time):

        k = np.zeros(3, dtype=np.float)
        u = np.zeros(3, dtype=np.float)
        v = np.zeros(3, dtype=np.float)

        assert num_delays <= self.num_orbit_inputs
        assert num_delays * self.dt < self.final_t

        cosbeta = np.cos(beta)
        sinbeta = np.sin(beta)

        coslam = np.cos(lam)
        sinlam = np.sin(lam)

        u[0] = -sinbeta * coslam
        u[1] = -sinbeta * sinlam
        u[2] = cosbeta
        v[0] = sinlam
        v[1] = -coslam
        v[2] = 0.0
        k[0] = -cosbeta * coslam
        k[1] = -cosbeta * sinlam
        k[2] = -sinbeta

        y_gw = xp.zeros((self.nlinks * num_delays,), dtype=xp.float)
        k_in = xp.asarray(k)
        u_in = xp.asarray(u)
        v_in = xp.asarray(v)

        input_in = xp.asarray(input_in)
        num_pts_in = len(input_in)

        st = time.perf_counter()
        num = 1
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
                self.A_in,
                self.deps,
                len(self.A_in),
                self.E_in,
                input_start_time,
                self.x_in,
                self.n_in,
                self.L_in,
                self.num_orbit_inputs,
            )
        et = time.perf_counter()
        print("Num delays:", num_delays, (et - st) / num)

        self.y_gw_flat = y_gw
        self.y_gw_length = num_delays

    @property
    def XYZ(self):
        return self.delayed_links_flat.reshape(3, -1)

    def get_tdi_delays(self):
        assert self.y_gw_length >= self.num_delays_tdi

        input_start_time = -3000.00
        self.delayed_links_flat = xp.zeros((3, self.num_delays_tdi), dtype=xp.float64)

        for j in range(3):
            for link_ind, sign in self.channels_no_delays[j]:
                # TODO: check boundaries
                self.delayed_links_flat[j] += (
                    sign * self.y_gw[link_ind, : self.num_delays_tdi]
                )

        self.delayed_links_flat = self.delayed_links_flat.flatten()
        st = time.perf_counter()
        num = 100
        for i in range(num):
            get_tdi_delays_wrap(
                self.delayed_links_flat,
                self.y_gw_flat,
                self.y_gw_length,
                self.tdi_delays,
                self.num_delays_tdi,
                self.dt,
                self.link_inds,
                self.num_units,
                self.num_channels,
                self.order,
                self.sampling_frequency,
                self.buffer_integer,
                self.A_in,
                self.deps,
                len(self.A_in),
                self.E_in,
                input_start_time,
            )

        et = time.perf_counter()
        print(
            "Num delays:",
            self.num_delays_tdi,
            "num total units:",
            self.num_units * self.num_channels,
            "per unit:",
            (et - st) / (num * self.num_units * self.num_channels),
            "all units:",
            (et - st) / (num),
        )


use_gpu = gpu

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu,  # GPU is available in this class
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

num_pts_in = int(4e6)

M = 1e6
mu = 1e1
p0 = 12.0
e0 = 0.1
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 1.0

sampling_frequency = 0.2
dt = 1 / sampling_frequency
T = (num_pts_in * dt) / YRSID_SI

beta = 0.4
lam = 1.3

input_in = few(M, mu, p0, e0, theta, phi, dist, dt=dt, T=T)

order = 25
response = pyResponseTDI(
    sampling_frequency,
    orbits_file="orbits.h5",
    order=order,
    num_factorials=100,
    tdi="1st generation",
)


link_inds = np.concatenate([np.arange(6), np.arange(6)]).astype(dtype=np.int32)
delay_factor = np.ones_like(link_inds)
delay_factor[6:] = 2
link_inds = delay_factor.copy()

if gpu is False:
    frac_delay_check = VariableFractionDelay(sampling_frequency, order=order)
    out = []
    for link_i in range(6):
        out.append([])
        for i in range(1000):
            x0, x1, n, L = response.get_xnL(i, link_i)
            tt1 = frac_delay_check(
                input_in, i, dt, x0, x1, n, L, lam, beta, input_start_time
            )
            out[link_i].append(tt1)

    out = np.asarray(out)
    import matplotlib.pyplot as plt

    check = np.load("test_resp.npy")

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)

    ax = [ax]
    for link_i, ax_i in zip(range(6), ax):
        ax_i.plot(out[link_i])
        ax_i.plot(check[link_i][:1000], ls="--")
    plt.show()
    breakpoint()

if gpu:
    response.get_projections(int(3.3e6), input_in, beta, lam, input_start_time)
    response.get_tdi_delays()
    # print(i)

breakpoint()
