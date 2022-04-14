import numpy as np
import warnings 

try:
    import cupy as xp
    from pyresponse import get_response_wrap as get_response_wrap_gpu
    from pyresponse import get_tdi_delays_wrap as get_tdi_delays_wrap_gpu

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu = False

from pyresponse_cpu import get_response_wrap as get_response_wrap_cpu
from pyresponse_cpu import get_tdi_delays_wrap as get_tdi_delays_wrap_cpu
import time
import h5py

from scipy.interpolate import CubicSpline

YRSID_SI = 31558149.763545603
AU = 1.49597870660e11


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


def pointer_adjust(func):
    def func_wrapper(*args, **kwargs):
        targs = []
        for arg in args:
            if gpu:
                if isinstance(arg, xp.ndarray):
                    targs.append(arg.data.mem.ptr)
                    continue

            if isinstance(arg, np.ndarray):
                targs.append(arg.__array_interface__["data"][0])
                continue

            try:
                targs.append(arg.ptr)
                continue
            except AttributeError:
                targs.append(arg)

        return func(*targs, **kwargs)

    return func_wrapper


class pyResponseTDI(object):
    """Class container for fast LISA response function generation.

    The class computes the generic time-domain response function for LISA.
    It takes LISA constellation orbital information as input and properly determines
    the response for these orbits numerically. This includes both the projection
    of the gravitational waves onto the LISA constellation arms and combinations \
    of projections into TDI observables. The methods and maths used can be found
    in # TODO: add url for paper.

    This class is also GPU-accelerated, which is very helpful for Bayesian inference
    methods.

    Args:
        sampling_frequency (double): The sampling rate in Hz.
        num_pts (int): Number of points to produce for the final output template.
        orbit_kwargs (dict): Dictionary containing orbital information. The kwargs and defaults
            are: :code:`orbit_module=None, order=0, max_t_orbits=3.15576e7, orbit_file=None`.
            :code:`orbit_module` is an orbit module from the LDC package. :code:`max_t_orbits` is
            the maximum time desired for the orbital information. `orbit_file` is
            an h5 file of the form used `here <https://gitlab.in2p3.fr/lisa-simulation/orbits>`_.
            :code:`order` is the order of interpolation used in the orbit modules.
        order (int, optional): Order of Lagrangian interpolation technique. Lower orders
            will be faster. The user must make sure the order is sufficient for the
            waveform being used. (default: 25)
        tdi (str or list, optional): TDI setup. Currently, the stock options are
            :code:`'1st generation'` and :code:`'2nd generation'`. Or the user can provide
            a list of tdi_combinations of the form
            :code:`{"link": 12, "links_for_delay": [21, 13, 31], "sign": 1, "type": "delay"}`.
            :code:`'link'` (`int`) the link index (12, 21, 13, 31, 23, 32) for the projection (:math:`y_{ij}`).
            :code:`'links_for_delay'` (`list`) are the link indexes as a list used for delays
            applied to the link projections.
            ``'sign'`` is the sign in front of the contribution to the TDI observable. It takes the value of `+1` or `-1`.
            ``type`` is either ``"delay"`` or ``"advance"``. It is optional and defaults to ``"delay"``.
            (default: ``"1st generation"``)
        tdi_orbit_kwargs (dict, optional): Same as :code:`orbit_kwargs`, but specifically for the TDI
            portion of the response computation. This allows the user to use two different orbits
            for the projections and TDI. For example, this can be used to examine the efficacy of
            frequency domain TDI codes that can handle generic orbits for the projections, but
            assume equal armlength orbits to reduce and simplify the expression for TDI
            computations. (default: :code:`None`, this means the orbits for the projections
            and TDI will be the same and will be built from :code:`orbit_kwargs`)
        tdi_chan (str, optional): Which TDI channel combination to return. Choices are :code:`'XYZ'`,
            :code:`AET`, or :code:`AE`. (default: :code:`'XYZ'`)
        use_gpu (bool, optional): If True, run code on the GPU. (default: :code:`False`)

    Attributes:
        A_in (xp.ndarray): Array containing y values for linear spline of A
            during Lagrangian interpolation.
        buffer_integer (int): Self-determined buffer necesary for the given
            value for :code:`order`.
        channels_no_delays (2D np.ndarray): Carrier of link index and sign information
            for arms that do not get delayed during TDI computation.
        deps (double): The spacing between Epsilon values in the interpolant
            for the A quantity in Lagrangian interpolation. Hard coded to
            1/(:code:`num_A` - 1).
        dt (double): Inverse of the sampling_frequency.
        E_in (xp.ndarray): Array containing y values for linear spline of E
            during Lagrangian interpolation.
        half_order (int): Half of :code:`order` adjusted to be :code:`int`.
        link_inds (xp.ndarray): Link indexes for delays in TDI.
        link_space_craft_0_in (xp.ndarray): Link indexes for receiver on each
            arm of the LISA constellation.
        link_space_craft_1_in (xp.ndarray): Link indexes for emitter on each
            arm of the LISA constellation.
        nlinks (int): The number of links in the constellation. Typically 6.
        num_A (int): Number of points to use for A spline values used in the Lagrangian
            interpolation. This is hard coded to 1001.
        num_channels (int): 3.
        num_pts (int): Number of points to produce for the final output template.
        num_tdi_combinations (int): Number of independent arm computations.
        num_tdi_delay_comps (int): Number of independent arm computations that require delays.
        orbits_store (dict): Contains orbital information for the projection and TDI
            steps.
        order (int): Order of Lagrangian interpolation technique.
        response_gen (func): Projection generator function.
        sampling_frequency (double): The sampling rate in Hz.
        tdi (str or list): TDI setup.
        tdi_buffer (int): The buffer necessary for all information needed at early times
            for the TDI computation. This is set to 200.
        tdi_chan (str): Which TDI channel combination to return.
        tdi_delays (xp.ndarray): TDI delays.
        tdi_gen (func): TDI generating function.
        tdi_signs (xp.ndarray): Signs applied to the addition of a delayed link. (+1 or -1)
        use_gpu (bool): If True, run on GPU.
        xp (obj): Either Numpy or Cupy.

    """

    def __init__(
        self,
        sampling_frequency,
        num_pts,
        orbit_kwargs,
        order=25,
        tdi="1st generation",
        tdi_orbit_kwargs={},
        tdi_chan="XYZ",
        use_gpu=False,
    ):

        # setup all quantities
        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency

        self.num_pts = num_pts

        # Lagrangian interpolation setup
        self.order = order
        self.buffer_integer = self.order * 2 + 1
        self.half_order = int((order + 1) / 2)

        # setup TDI information
        self.tdi = tdi
        self.tdi_chan = tdi_chan

        # setup functions for GPU or CPU
        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
            self.response_gen = get_response_wrap_gpu
            self.tdi_gen = get_tdi_delays_wrap_gpu

        else:
            self.xp = np
            self.response_gen = get_response_wrap_cpu
            self.tdi_gen = get_tdi_delays_wrap_cpu

        # prepare the interpolation of A and E in the Lagrangian interpolation
        self._fill_A_E()

        orbit_kwargs["order"] = order
        # setup orbits
        self.orbits_store = {}

        self.orbits_store["projection"] = self._init_orbit_information(**orbit_kwargs)

        # if tdi_orbit_kwargs are given, fill TDI specific orbit info
        if tdi_orbit_kwargs == {}:
            self.orbits_store["tdi"] = self.orbits_store["projection"]
        else:
            tdi_orbit_kwargs["order"] = order
            self.orbits_store["tdi"] = self._init_orbit_information(**tdi_orbit_kwargs)

        # setup TDI info
        self._init_TDI_delays()

    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """

    def _fill_A_E(self):
        """Set up A and E terms inside the Lagrangian interpolant"""

        factorials = np.asarray([float(get_factorial(n)) for n in range(40)])

        # base quantities for linear interpolant over A
        self.num_A = 1001
        self.deps = 1.0 / (self.num_A - 1)

        eps = np.arange(self.num_A) * self.deps

        h = self.half_order

        denominator = factorials[h - 1] * factorials[h]

        # prepare A
        A_in = np.zeros_like(eps)
        for j, eps_i in enumerate(eps):
            A = 1.0
            for i in range(1, h):
                A *= (i + eps_i) * (i + 1 - eps_i)

            A /= denominator
            A_in[j] = A

        self.A_in = self.xp.asarray(A_in)

        # prepare E
        E_in = self.xp.zeros((self.half_order,))

        for j in range(1, self.half_order):
            first_term = factorials[h - 1] / factorials[h - 1 - j]
            second_term = factorials[h] / factorials[h + j]
            value = first_term * second_term
            value = value * (-1.0) ** j
            E_in[j - 1] = value

        self.E_in = self.xp.asarray(E_in)

    def _init_orbit_information(
        self, orbit_module=None, max_t_orbits=None, orbit_file=None, order=25,
    ):
        """Initialize orbital information"""

        if orbit_module is None:
            if orbit_file is None:
                raise ValueError("Must provide either orbit file or orbit module.")

            self.nlinks = 6

            # link order: 21, 12, 31, 13, 32, 23
            # setup spacecraft links indexes
            link_space_craft_0 = np.zeros((self.nlinks,), dtype=int)
            link_space_craft_1 = np.zeros((self.nlinks,), dtype=int)
            link_space_craft_0[0] = 1
            link_space_craft_1[0] = 0
            link_space_craft_0[1] = 0
            link_space_craft_1[1] = 1

            link_space_craft_0[2] = 2
            link_space_craft_1[2] = 0
            link_space_craft_0[3] = 0
            link_space_craft_1[3] = 2

            link_space_craft_0[4] = 2
            link_space_craft_1[4] = 1
            link_space_craft_0[5] = 1
            link_space_craft_1[5] = 2

            self.link_space_craft_0_in = self.xp.asarray(link_space_craft_0).astype(
                self.xp.int32
            )
            self.link_space_craft_1_in = self.xp.asarray(link_space_craft_1).astype(
                self.xp.int32
            )

            # get info from the file
            out = {}
            with h5py.File(orbit_file, "r") as f:
                for key in f["tcb"]:
                    out[key] = f["tcb"][key][:]

            # get t and normalize so first point is at t=0
            t_in = out["t"]
            t_in = t_in - t_in[0]
            length_in = len(t_in)

            # get x and L information
            x_in = []
            for i in range(3):
                for let in ["x", "y", "z"]:
                    x_in.append(out["sc_" + str(i + 1)][let])

            L_in = []

            for link_i in range(self.nlinks):
                sc0 = self.link_space_craft_0_in[link_i] + 1
                sc1 = self.link_space_craft_1_in[link_i] + 1

                x_val = out["sc_" + str(sc0)]["x"] - out["sc_" + str(sc1)]["x"]
                y_val = out["sc_" + str(sc0)]["y"] - out["sc_" + str(sc1)]["y"]
                z_val = out["sc_" + str(sc0)]["z"] - out["sc_" + str(sc1)]["z"]

                norm = np.sqrt(x_val ** 2 + y_val ** 2 + z_val ** 2)

                L_in.append(out["l_" + str(sc0) + str(sc1)]["tt"])

            # constrain maximum time used for orbits
            if max_t_orbits is not None and max_t_orbits < t_in[-1]:
                t_max = max_t_orbits
            else:
                t_max = t_in[-1]

            if t_max < self.num_pts * self.dt:
                raise ValueError(
                    "End time for projection is greater than end time for orbital information."
                )

            # new time array
            t_new = np.arange(self.num_pts) * self.dt

            # evaluate splines on everything
            for i in range(self.nlinks):
                L_in[i] = CubicSpline(t_in, L_in[i])(t_new)

            x_in_receiver = [None for _ in range(2 * len(x_in))]
            x_in_emitter = [None for _ in range(2 * len(x_in))]
            for link_i in range(self.nlinks):
                sc0 = self.link_space_craft_0_in[link_i].item()  # receiver
                sc1 = self.link_space_craft_1_in[link_i].item()  # emitter

                for j in range(3):
                    x_in_receiver[link_i * 3 + j] = CubicSpline(
                        t_in, x_in[sc0 * 3 + j]
                    )(t_new)
                    x_in_emitter[link_i * 3 + j] = CubicSpline(t_in, x_in[sc1 * 3 + j])(
                        t_new - L_in[link_i]
                    )

        else:
            raise NotImplementedError
            # perform computations from LDC orbit class
            t_new = np.arange(self.num_pts) * self.dt
            self.nlinks = orbit_module.number_of_arms
            self.link_space_craft_0_in = self.xp.zeros(self.nlinks, dtype=self.xp.int32)
            self.link_space_craft_1_in = self.xp.zeros(self.nlinks, dtype=self.xp.int32)

            L_in = []
            for i in range(orbit_module.number_of_arms):
                emitter, receiver = orbit_module.get_pairs()[i]

                self.link_space_craft_0_in[i] = emitter - 1
                self.link_space_craft_1_in[i] = receiver - 1

                L_in.append(
                    orbit_module.compute_travel_time(
                        emitter, receiver, t_new, order=order
                    )
                )

            x_in = []
            # alphas = orbit_module.compute_alpha(t_new)
            for i in range(1, orbit_module.number_of_spacecraft + 1):
                temp = orbit_module.compute_position(i, t_new)
                for j in range(3):
                    x_in.append(temp[j])

        # get max buffer for projections
        projection_buffer = int(1.05 * AU * C_inv + np.max(np.abs(L_in))) + 4 * order
        x_in_receiver = self.xp.asarray(np.concatenate(x_in_receiver))
        x_in_emitter = self.xp.asarray(np.concatenate(x_in_emitter))
        L_in_for_TDI = L_in
        L_in = self.xp.asarray(np.concatenate(L_in))
        num_orbit_inputs = len(t_new)
        t_data_cpu = t_new
        t_data = self.xp.asarray(t_new)
        final_t = t_new[-1]

        # read out all info in a dictionary
        orbits_store = dict(
            projection_buffer=projection_buffer,
            x_in_receiver=x_in_receiver,
            x_in_emitter=x_in_emitter,
            L_in_for_TDI=L_in_for_TDI,
            L_in=L_in,
            num_orbit_inputs=num_orbit_inputs,
            t_data_cpu=t_data_cpu,
            t_data=t_data,
            final_t=final_t,
            orbit_file=orbit_file,
            orbit_module=orbit_module,
            max_t_orbits=max_t_orbits,
        )
        return orbits_store

    def _init_TDI_delays(self):
        """Initialize TDI specific information"""

        # sets attributes in class related to orbits_store
        for key, item in self.orbits_store["tdi"].items():
            setattr(self, key, item)

        link_dict = {}
        for link_i in range(self.nlinks):
            sc0 = self.link_space_craft_0_in[link_i]
            sc1 = self.link_space_craft_1_in[link_i]
            link_dict[int(str(sc0 + 1) + str(sc1 + 1))] = link_i

        # setup the actual TDI combination
        if self.tdi in ["1st generation", "2nd generation"]:
            # tdi 1.0
            tdi_combinations = [
                {"link": 13, "links_for_delay": [], "sign": +1},
                {"link": 31, "links_for_delay": [13], "sign": +1},
                {"link": 12, "links_for_delay": [13, 31], "sign": +1},
                {"link": 21, "links_for_delay": [13, 31, 12], "sign": +1},
                {"link": 12, "links_for_delay": [], "sign": -1},
                {"link": 21, "links_for_delay": [12], "sign": -1},
                {"link": 13, "links_for_delay": [12, 21], "sign": -1},
                {"link": 31, "links_for_delay": [12, 21, 13], "sign": -1},
            ]

            if self.tdi == "2nd generation":
                # tdi 2.0 is tdi 1.0 + additional terms
                tdi_combinations += [
                    {"link": 12, "links_for_delay": [13, 31, 12, 21], "sign": +1},
                    {"link": 21, "links_for_delay": [13, 31, 12, 21, 12], "sign": +1},
                    {
                        "link": 13,
                        "links_for_delay": [13, 31, 12, 21, 12, 21],
                        "sign": +1,
                    },
                    {
                        "link": 31,
                        "links_for_delay": [13, 31, 12, 21, 12, 21, 13],
                        "sign": +1,
                    },
                    {"link": 13, "links_for_delay": [12, 21, 13, 31], "sign": -1},
                    {"link": 31, "links_for_delay": [12, 21, 13, 31, 13], "sign": -1},
                    {
                        "link": 12,
                        "links_for_delay": [12, 21, 13, 31, 13, 31],
                        "sign": -1,
                    },
                    {
                        "link": 21,
                        "links_for_delay": [12, 21, 13, 31, 13, 31, 12],
                        "sign": -1,
                    },
                ]

        elif isinstance(self.tdi, list):
            tdi_combinations = self.tdi

        else:
            raise ValueError(
                "tdi kwarg should be '1st generation', '2nd generation', or a list with a specific tdi combination."
            )

        # setup computation of channels that are not delayed in order to save time
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

        delays = np.zeros((3, self.num_tdi_delay_comps, self.num_pts))

        delays[:] = self.t_data_cpu

        link_inds = np.zeros((3, self.num_tdi_delay_comps), dtype=np.int32)

        signs = []
        # cyclic permuatations for X, Y, Z
        # computing all of the delays a priori
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

                    # handles advancements
                    if "type" in tdi and tdi["type"] == "advance":
                        delays[j][i] += self.L_in_for_TDI[link_index]
                    else:
                        delays[j][i] -= self.L_in_for_TDI[link_index]

                if j == 0:
                    signs.append(tdi["sign"])
                i += 1

        try:
            t_arr = self.t_data.get()
        except AttributeError:
            t_arr = self.t_data

        # find the maximum delayed applied to the combinations
        self.max_delay = np.max(np.abs(t_arr - delays[:]))

        # get necessary buffer for TDI
        self.tdi_buffer = (
            int(self.max_delay * self.sampling_frequency) + 4 * self.order
        )

        # prepare final info needed for TDI
        self.num_channels = 3
        self.link_inds = self.xp.asarray(link_inds.flatten()).astype(self.xp.int32)
        self.tdi_delays = self.xp.asarray(delays.flatten())
        self.tdi_signs = self.xp.asarray(signs, dtype=np.int32)

    @property
    def tdi_buffer(self):
        return self._tdi_buffer

    @tdi_buffer.setter
    def tdi_buffer(self, tdi_buffer):
        if tdi_buffer < int(self.max_delay * self.sampling_frequency) + 4 * self.order:
            warnings.warn("Inputing tdi_buffer that is lower than the default determined value: int(self.max_delay * self.sampling_frequency) + 4 * self.order. Proceed with caution.")
        self._tdi_buffer = tdi_buffer

    @property
    def projection_buffer(self):
        return self._projection_buffer

    @projection_buffer.setter
    def projection_buffer(self, projection_buffer):
        if projection_buffer < int(AU * C_inv) + 4 * self.order:
            warnings.warn("Inputing projection_buffer that is lower than the advised value: int(AU * C_inv) + 4 * self.order. Proceed with caution.")
        self._projection_buffer = projection_buffer

    def _cyclic_permutation(self, link, permutation):
        """permute indexes by cyclic permutation"""
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
        """Projections along the arms"""
        return self.y_gw_flat.reshape(self.nlinks, -1)

    def get_projections(self, input_in, lam, beta, t0=10000.0, projections_start_ind=None, projections_cut_ind=None, remove_projection_buffer=False):
        """Compute projections of GW signal on to LISA constellation

        Args:
            input_in (xp.ndarray): Input complex time-domain signal. It should be of the form:
                :math:`h_+ + ih_x`. If using the GPU for the response, this should be a CuPy array.
            lam (double): Ecliptic Longitude in radians.
            beta (double): Ecliptic Latitude in radians.
            t0 (double, optional): Time at which to the waveform. Because of the delays
                and interpolation towards earlier times, the beginning of the waveform
                is garbage. ``t0`` tells the waveform generator where to start the waveform
                compraed to ``t=0``.

        Raises:
            ValueError: If ``t0`` is not large enough.


        """

        # get break points
        for key, item in self.orbits_store["projection"].items():
            if key == "projection_buffer" and remove_projection_buffer:
                setattr(self, "projection_buffer", -1e10)
                warnings.warn("Not using default projection_buffer. Proceed with caution.")
                continue

            setattr(self, key, item)

        self.tdi_start_ind = int(t0 / self.dt)

        base_projections_start_ind = self.tdi_start_ind - 2 * self.tdi_buffer

        if projections_start_ind is None:
            projections_start_ind = base_projections_start_ind

        if projections_cut_ind is None:
            # set to default which is just - projections_start_ind effectively
            projections_cut_ind = base_projections_start_ind

        if projections_cut_ind < self.projection_buffer:
            raise ValueError(
                "Need to increase t0. The initial buffer is not large enough."
            )

        if projections_start_ind < self.projection_buffer:
            raise ValueError(
                "Need to increase t0. The initial buffer is not large enough."
            )

        # determine sky vectors
        k = np.zeros(3, dtype=np.float)
        u = np.zeros(3, dtype=np.float)
        v = np.zeros(3, dtype=np.float)

        assert len(input_in) >= self.num_pts
        self.num_total_points = len(input_in)

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

        y_gw = self.xp.zeros((self.nlinks * self.num_pts,), dtype=self.xp.float)
        k_in = self.xp.asarray(k)
        u_in = self.xp.asarray(u)
        v_in = self.xp.asarray(v)

        input_in = self.xp.asarray(input_in)

        self.projections_start_ind = projections_start_ind
        self.projections_cut_ind = projections_cut_ind

        self.response_gen(
            y_gw,
            self.t_data,
            k_in,
            u_in,
            v_in,
            self.dt,
            self.num_pts,
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
            projections_start_ind,
            self.x_in_receiver,
            self.x_in_emitter,
            self.L_in,
            self.num_orbit_inputs,
            projections_cut_ind
        )

        self.y_gw_flat = y_gw
        self.y_gw_length = self.num_pts

    @property
    def XYZ(self):
        """Return links as an array"""
        return self.delayed_links_flat.reshape(3, -1)

    def get_tdi_delays(self, y_gw=None, tdi_cut_ind=None):
        """Get TDI combinations from projections.

        This functions generates the TDI combinations from the projections
        computed with ``get_projections``. It can return XYZ, AET, or AE depending
        on what was input for ``tdi_chan`` into ``__init__``.

        Args:
            y_gw (xp.ndarray, optional): Projections along each link. Must be
                a 2D ``numpy`` or ``cupy`` array with shape: ``(nlinks, num_pts)``.
                The links must be entered in the proper order in the code:
                21, 12, 31, 13, 32, 23. (Default: None)

        Returns:
            tuple: (X,Y,Z) or (A,E,T) or (A,E)

        Raises:
            ValueError: If ``tdi_chan`` is not one of the options.


        """
        for key, item in self.orbits_store["tdi"].items():
            setattr(self, key, item)

        self.delayed_links_flat = self.xp.zeros(
            (3, self.num_pts), dtype=self.xp.float64
        )

        # y_gw entered directly
        if y_gw is not None:
            assert y_gw.shape == (len(self.link_space_craft_0_in), self.num_pts)
            self.y_gw_flat = y_gw.flatten().copy()
            self.y_gw_length = self.num_pts

        elif self.y_gw_flat is None:
            raise ValueError(
                "Need to either enter projection array or have this code determine projections."
            )

        for j in range(3):
            for link_ind, sign in self.channels_no_delays[j]:
                self.delayed_links_flat[j] += sign * self.y_gw[link_ind]

        self.delayed_links_flat = self.delayed_links_flat.flatten()

        if tdi_cut_ind is None:
            # set to default which is just - tdi_start_ind effectively
            tdi_cut_ind = self.tdi_start_ind

        if tdi_cut_ind < self.tdi_buffer:
            raise ValueError(
                "Need to increase t0. The initial buffer is not large enough."
            )

        self.tdi_cut_ind = tdi_cut_ind
        self.tdi_gen(
            self.delayed_links_flat,
            self.y_gw_flat,
            self.y_gw_length,
            self.num_orbit_inputs,
            self.tdi_delays,
            self.num_pts,
            self.dt,
            self.link_inds,
            self.tdi_signs,
            self.num_tdi_delay_comps,
            self.num_channels,
            self.order,
            self.sampling_frequency,
            self.buffer_integer,
            self.A_in,
            self.deps,
            len(self.A_in),
            self.E_in,
            self.tdi_start_ind,
            tdi_cut_ind,
        )

        if self.tdi_chan == "XYZ":
            X, Y, Z = self.XYZ
            return X, Y, Z

        elif self.tdi_chan == "AET" or self.tdi_chan == "AE":
            X, Y, Z = self.XYZ
            A, E, T = AET(X, Y, Z)
            if self.tdi_chan == "AET":
                return A, E, T

            else:
                return A, E

        else:
            raise ValueError("tdi_chan must be 'XYZ', 'AET' or 'AE'.")


class ResponseWrapper:
    """Wrapper to produce LISA TDI from TD waveforms

    This class takes a waveform generator that produces :math:`h_+ \pm ih_x`.
    (:code:`flip_hx` is used if the waveform produces :math:`h_+ - ih_x`).
    It takes the complex waveform in the SSB frame and produces the TDI channels
    according to settings chosen for :class:`pyResponseTDI`.

    The waveform generator must have :code:`kwargs` with :code:`T` for the observation
    time in years and :code:`dt` for the time step in seconds.

    Args:
        waveform_gen (obj): Function or class (with a :code:`__call__` function) that takes parameters and produces
            :math:`h_+ \pm h_x`.
        Tobs (double): Observation time in years.
        dt (double): Time between time samples in seconds. The inverse of the sampling frequency.
        index_lambda (int): The user will input parameters. The code will read these in
            with the :code:`*args` formalism producing a list. :code:`index_lambda`
            tells the class the index of the ecliptic longitude within this list of
            parameters.
        index_beta (int): The user will input parameters. The code will read these in
            with the :code:`*args` formalism producing a list. :code:`index_beta`
            tells the class the index of the ecliptic latitude (or ecliptic polar angle)
            within this list of parameters.
        t0 (double, optional): Start of returned waveform in seconds leaving ample time for garbage at
            the beginning of the waveform. It also removed the same amount from the end. (Default: 10000.0)
        flip_hx (bool, optional): If True, :code:`waveform_gen` produces :math:`h_+ - ih_x`.
            :class:`pyResponseTDI` takes :math:`h_+ + ih_x`, so this setting will
            multiply the cross polarization term out of the waveform generator by -1.
            (Default: :code:`False`)
        remove_sky_coords (bool, optional): If True, remove the sky coordinates from
            the :code:`*args` list. This should be set to True if the waveform
            generator does not take in the sky information. (Default: :code:`False`)
        is_ecliptic_latitude (bool, optional): If True, the latitudinal sky
            coordinate is the ecliptic latitude. If False, thes latitudinal sky
            coordinate is the polar angle. In this case, the code will
            convert it with :math:`\beta=\pi / 2 - \Theta`. (Default: :code:`True`)
        use_gpu (bool, optional): If True, use GPU. (Default: :code:`False`)
        remove_garbage (bool, optional): If True, it removes everything before ``t0``
            and after the end time - ``t0``. (Default: ``True``)
        **kwargs (dict, optional): Keyword arguments passed to :class:`pyResponseTDI`.

    """

    def __init__(
        self,
        waveform_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=10000.0,
        flip_hx=False,
        remove_sky_coords=False,
        is_ecliptic_latitude=True,
        use_gpu=False,
        remove_garbage=True,
        **kwargs,
    ):

        # store all necessary information
        self.waveform_gen = waveform_gen
        self.index_lambda = index_lambda
        self.index_beta = index_beta
        self.dt = dt
        self.t0 = t0
        self.sampling_frequency = 1.0 / dt
        self.n = int(Tobs * YRSID_SI / dt)
        self.Tobs = self.n * dt
        self.is_ecliptic_latitude = is_ecliptic_latitude
        self.remove_sky_coords = remove_sky_coords
        self.flip_hx = flip_hx
        self.remove_garbage = remove_garbage

        # initialize response function class
        self.response_model = pyResponseTDI(
            self.sampling_frequency, self.n, use_gpu=use_gpu, **kwargs
        )

        self.use_gpu = use_gpu
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np

        self.Tobs = (self.n * self.response_model.dt) / YRSID_SI

    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """

    def __call__(self, *args, projections_start_ind=None, projections_cut_ind=None, remove_projection_buffer=False, tdi_cut_ind=None, **kwargs):
        """Run the waveform and response generation

        Args:
            *args (list): Arguments to the waveform generator. This must include
                the sky coordinates.
            **kwargs (dict): kwargs necessary for the waveform generator.

        Return:
            list: TDI Channels.

        """

        args = list(args)

        # get sky coords
        beta = args[self.index_beta]
        lam = args[self.index_lambda]

        # remove them from the list if waveform generator does not take them
        if self.remove_sky_coords:
            args.pop(self.index_beta)
            args.pop(self.index_lambda)

        # transform polar angle
        if not self.is_ecliptic_latitude:
            beta = np.pi / 2.0 - beta

        # add the new Tobs and dt info to the waveform generator kwargs
        kwargs["T"] = self.Tobs
        kwargs["dt"] = self.dt

        # get the waveform
        h = self.waveform_gen(*args, **kwargs)

        if self.flip_hx:
            h = h.real - 1j * h.imag

        self.response_model.get_projections(h, lam, beta, t0=self.t0, projections_start_ind=projections_start_ind, projections_cut_ind=projections_cut_ind, remove_projection_buffer=remove_projection_buffer)
        tdi_out = self.response_model.get_tdi_delays(tdi_cut_ind=tdi_cut_ind)

        out = list(tdi_out)
        if self.remove_garbage:
            for i in range(len(out)):
                out[i] = out[i][
                    self.response_model.tdi_start_ind : -self.response_model.tdi_start_ind
                ]

        return out
