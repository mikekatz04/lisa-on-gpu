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


num_factorials = 100
factorials = np.asarray([float(get_factorial(n)) for n in range(num_factorials)])

dt = 1.0
Re = 1.496e11
L = 2.5e9

Phi0 = 0.0
Omega0 = 1 / (365.25 * 24.0 * 3600.0)

sc0_delta = np.array([L / 2, -L / (2.0 * np.sqrt(3.0))])
sc1_delta = np.array([-L / 2, -L / (2.0 * np.sqrt(3.0))])
sc2_delta = np.array([0.0, L / (np.sqrt(3.0))])

nlinks = 6

link_space_craft_0 = np.zeros((6,), dtype=int)
link_space_craft_1 = np.zeros((6,), dtype=int)
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


beta = 0.5
lam = 1.0

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


sampling_frequency = 1.0
dt = 1.0 / sampling_frequency
input_start_time = -10000.0

order = 25
buffer_integer = order * 2

num_pts_in = int(1e7)
input_in = np.zeros(num_pts_in, dtype=np.complex128)
A = 1e-22


input_in = A * (
    np.sin((np.arange(num_pts_in) * dt + input_start_time) / 1e2)
    + 1j * np.cos((np.arange(num_pts_in) * dt + input_start_time) / 1e2)
)


out = {}
with h5py.File("orbits.h5", "r") as f:
    for key in f:
        out[key] = f[key][:]

L_vals = []
n_in = []
for i in range(nlinks):
    link_0 = link_space_craft_0[i]
    link_1 = link_space_craft_1[i]

    L_vals.append(out["L" + str(link_0 + 1) + str(link_1 + 1)].T[1])

    x_val = out["n" + str(link_0 + 1) + str(link_1 + 1) + "_1"].T[1]
    y_val = out["n" + str(link_0 + 1) + str(link_1 + 1) + "_2"].T[1]
    z_val = out["n" + str(link_0 + 1) + str(link_1 + 1) + "_3"].T[1]

    norm = (x_val ** 2 + y_val ** 2 + z_val ** 2) ** (1 / 2)

    n_ij_x = x_val / norm
    n_ij_y = y_val / norm
    n_ij_z = z_val / norm

    n_in.append(np.concatenate([n_ij_x, n_ij_y, n_ij_z]))

L_vals = np.concatenate(L_vals)
n_in = np.concatenate(n_in)

x = []
for i in range(3):  # number of spacecraft
    x_i_x = out["x" + str(i + 1) + "_1"].T[1]
    x_i_y = out["x" + str(i + 1) + "_2"].T[1]
    x_i_z = out["x" + str(i + 1) + "_3"].T[1]

    x.append(np.concatenate([x_i_x, x_i_y, x_i_z]))

x = np.concatenate(x)

num_delays = len(out["x1_1"])

rep = 32
num_delays_orig = num_delays
num_delays = rep * num_delays


y_gw = xp.zeros((nlinks * num_delays,), dtype=xp.float)
k_in = xp.asarray(k)
u_in = xp.asarray(u)
v_in = xp.asarray(v)

x = xp.concatenate(
    [xp.asarray(x).reshape(3, 3, num_delays_orig) for i in range(rep)], axis=-1
).flatten()

n_in = xp.concatenate(
    [xp.asarray(n_in).reshape(nlinks, 3, num_delays_orig) for i in range(rep)], axis=-1
).flatten()

link_space_craft_0_in = xp.asarray(link_space_craft_0)
link_space_craft_1_in = xp.asarray(link_space_craft_1)

L_vals = xp.concatenate(
    [xp.asarray(L_vals).reshape(nlinks, num_delays_orig) for i in range(rep)]
).flatten()
input_in = xp.asarray(input_in)
factorials_in = xp.asarray(factorials)


if gpu:
    st = time.perf_counter()
    num = 100
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
            link_space_craft_0_in,
            link_space_craft_1_in,
            L_vals,
            input_in,
            num_pts_in,
            order,
            sampling_frequency,
            buffer_integer,
            factorials_in,
            num_factorials,
            input_start_time,
        )
        # print(i)
    et = time.perf_counter()
    print((et - st) / num)

breakpoint()
