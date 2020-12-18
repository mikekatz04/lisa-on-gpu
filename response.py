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

from few.utils.constants import *
import matplotlib.pyplot as plt


def get_factorial(n):
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact


from math import factorial

factorials = np.array([factorial(i) for i in range(30)])

C_inv = 3.3356409519815204e-09


def AET(X, Y, Z):
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


class pyResponseTDI(object):
    def __init__(
        self,
        sampling_frequency,
        orbits_file="orbits.h5",
        order=25,
        tdi="1st generation",
        max_t_orbits=3.15576e7,
        tdi_chan="XYZ",
    ):

        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency

        self.order = order
        self.buffer_integer = self.order * 2 + 1
        self.half_order = int((order + 1) / 2)

        self.tdi = tdi
        self.tdi_chan = tdi_chan

        self._fill_A_E()
        self._init_link_indices()
        self._init_orbit_information(orbits_file, max_t_orbits=max_t_orbits)
        self._init_TDI_delays()

        self.total_buffer = self.tdi_buffer + self.projection_buffer

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

    def _init_orbit_information(self, orbits_file, max_t_orbits=3.15576e7):

        out = {}
        with h5py.File(orbits_file, "r") as f:
            for key in f:
                out[key] = f[key][:]

        t_in = out["t"]
        t_in = t_in - t_in[0]
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

            L_in.append(out["l_" + str(sc0) + str(sc1)]["tt"])

        t_max = t_in[-1] if t_in[-1] < max_t_orbits else max_t_orbits
        t_new = np.arange(0.0, t_max + self.dt, self.dt)

        for i in range(9):
            x_in[i] = CubicSpline(t_in, x_in[i])(t_new)

        for i in range(self.nlinks * 3):
            n_in[i] = CubicSpline(t_in, n_in[i])(t_new)

        for i in range(self.nlinks):
            L_in[i] = CubicSpline(t_in, L_in[i])(t_new)

        # get max buffer for projections
        self.projection_buffer = (
            int(np.max(x_in) * C_inv + np.max(np.abs(L_in))) + 4 * self.order
        )

        self.x_in = xp.asarray(np.concatenate(x_in))
        self.n_in = xp.asarray(np.concatenate(n_in))

        self.L_in_for_TDI = L_in
        self.L_in = xp.asarray(np.concatenate(L_in))

        self.num_orbit_inputs = len(t_new)
        self.t_data = t_new
        self.final_t = t_new[-1]

    def get_xnL(self, i, link_i):
        sc0 = self.link_space_craft_0_in[link_i]
        sc1 = self.link_space_craft_1_in[link_i]
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

        delays = np.zeros((3, self.num_tdi_delay_comps, self.num_orbit_inputs))

        delays[:] = self.t_data
        link_inds = np.zeros((3, self.num_tdi_delay_comps), dtype=np.int32)

        # cyclic permuatations for X, Y, Z
        for j in range(3):
            i = 0
            for tdi in tdi_combinations:

                if len(tdi["links_for_delay"]) == 0:
                    continue

                link = self._cyclic_permutation(tdi["link"], j)
                link_inds[j][i] = link_dict[link]

                temp_delay = 0.0
                for link in tdi["links_for_delay"]:
                    link = self._cyclic_permutation(link, j)
                    link_index = link_dict[link]
                    delays[j][i] -= self.L_in_for_TDI[link_index]

                i += 1

        self.max_delay = np.max(np.abs(self.t_data - delays[:]))

        # get necessary buffer for TDI
        check_tdi_buffer = (
            int(self.max_delay * self.sampling_frequency) + 4 * self.order
        )
        self.tdi_buffer = 200

        assert check_tdi_buffer < self.tdi_buffer

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

    def get_projections(self, input_in, lam, beta):

        k = np.zeros(3, dtype=np.float)
        u = np.zeros(3, dtype=np.float)
        v = np.zeros(3, dtype=np.float)

        self.num_total_points = len(input_in)
        num_delays_proj = self.num_total_points - 2 * self.projection_buffer

        assert num_delays_proj <= self.num_orbit_inputs
        assert num_delays_proj * self.dt < self.final_t

        cosbeta = np.cos(beta)
        sinbeta = np.sin(beta)

        coslam = np.cos(lam)
        sinlam = np.sin(lam)

        v[0] = -sinbeta * coslam
        v[1] = -sinbeta * sinlam
        v[2] = cosbeta
        u[0] = sinlam
        u[1] = -coslam
        u[2] = 0.0
        k[0] = -cosbeta * coslam
        k[1] = -cosbeta * sinlam
        k[2] = -sinbeta

        y_gw = xp.zeros((self.nlinks * num_delays_proj,), dtype=xp.float)
        k_in = xp.asarray(k)
        u_in = xp.asarray(u)
        v_in = xp.asarray(v)

        input_in = xp.asarray(input_in)

        get_response_wrap(
            y_gw,
            k_in,
            u_in,
            v_in,
            self.dt,
            num_delays_proj,
            self.link_space_craft_0_in,
            self.link_space_craft_1_in,
            input_in,
            self.num_total_points,
            self.order,
            self.sampling_frequency,
            self.buffer_integer,
            self.A_in,
            self.deps,
            len(self.A_in),
            self.E_in,
            self.projection_buffer,
            self.x_in,
            self.n_in,
            self.L_in,
            self.num_orbit_inputs,
        )

        self.y_gw_flat = y_gw
        self.y_gw_length = num_delays_proj

    @property
    def XYZ(self):
        return self.delayed_links_flat.reshape(3, -1)

    def get_tdi_delays(self):

        self.num_delays_tdi = self.num_total_points - 2 * self.total_buffer
        assert self.y_gw_length >= self.num_delays_tdi

        self.delayed_links_flat = xp.zeros((3, self.num_delays_tdi), dtype=xp.float64)

        for j in range(3):
            for link_ind, sign in self.channels_no_delays[j]:
                # TODO: check boundaries
                self.delayed_links_flat[j] += (
                    sign
                    * self.y_gw[
                        link_ind,
                        self.tdi_buffer : self.num_delays_tdi + self.tdi_buffer,
                    ]
                )

        self.delayed_links_flat = self.delayed_links_flat.flatten()

        get_tdi_delays_wrap(
            self.delayed_links_flat,
            self.y_gw_flat,
            self.y_gw_length,
            self.num_orbit_inputs,
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
            self.projection_buffer,
            self.total_buffer,
        )

        if self.tdi_chan == "XYZ":
            return self.XYZ

        elif self.tdi_chan == "AET" or self.tdi_chan == "AE":
            X, Y, Z = self.XYZ
            A, E, T = AET(X, Y, Z)
            if self.tdi_chan == "AET":
                return A, E, T

            else:
                return A, E

        else:
            raise ValueError("tdi_chan must be 'XYZ', 'AET' or 'AE'.")
