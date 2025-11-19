from multiprocessing.sharedctypes import Value
import numpy as np
from typing import Optional, List
import warnings
from typing import Tuple
from copy import deepcopy

import time
import h5py

from scipy.interpolate import CubicSpline

from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.utils.utility import AET

from .utils.parallelbase import FastLISAResponseParallelModule


# TODO: need to update constants setup
YRSID_SI = 31558149.763545603


def get_factorial(n):
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact


from math import factorial

factorials = np.array([factorial(i) for i in range(30)])

C_inv = 3.3356409519815204e-09


class pyResponseTDI(FastLISAResponseParallelModule):
    """Class container for fast LISA response function generation.

    The class computes the generic time-domain response function for LISA.
    It takes LISA constellation orbital information as input and properly determines
    the response for these orbits numerically. This includes both the projection
    of the gravitational waves onto the LISA constellation arms and combinations \
    of projections into TDI observables. The methods and maths used can be found
    [here](https://arxiv.org/abs/2204.06633).

    This class is also GPU-accelerated, which is very helpful for Bayesian inference
    methods.

    Args:
        sampling_frequency (double): The sampling rate in Hz.
        num_pts (int): Number of points to produce for the final output template.
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
        orbits (:class:`Orbits`, optional): Orbits class from LISA Analysis Tools. Works with LISA Orbits 
            outputs: ``lisa-simulation.pages.in2p3.fr/orbits/``.
            (default: :class:`EqualArmlengthOrbits`)
        tdi_chan (str, optional): Which TDI channel combination to return. Choices are :code:`'XYZ'`,
            :code:`AET`, or :code:`AE`. (default: :code:`'XYZ'`)
        tdi_orbits (:class:`Orbits`, optional): Set if different orbits from projection.
            Orbits class from LISA Analysis Tools. Works with LISA Orbits 
            outputs: ``lisa-simulation.pages.in2p3.fr/orbits/``.
            (default: :class:`EqualArmlengthOrbits`)
        force_backend (str, optional): If given, run this class on the requested backend. 
            Options are ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``. (default: ``None``)

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
        order (int): Order of Lagrangian interpolation technique.
        sampling_frequency (double): The sampling rate in Hz.
        tdi (str or list): TDI setup.
        tdi_buffer (int): The buffer necessary for all information needed at early times
            for the TDI computation. This is set to 200.
        xp (obj): Either Numpy or Cupy.

    """

    def __init__(
        self,
        sampling_frequency,
        num_pts,
        order=25,
        tdi="1st generation",
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        tdi_orbits: Optional[Orbits] = None,
        tdi_chan="XYZ",
        force_backend=None,
    ):

        # setup all quantities
        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency
        self.tdi_buffer = 200

        self.num_pts = num_pts

        # Lagrangian interpolation setup
        self.order = order
        self.buffer_integer = self.order * 2 + 1
        self.half_order = int((order + 1) / 2)

        # setup TDI information
        self.tdi = tdi
        self.tdi_chan = tdi_chan

        super().__init__(force_backend=force_backend)

        # prepare the interpolation of A and E in the Lagrangian interpolation
        self._fill_A_E()

        # setup orbits
        self.response_orbits = orbits

        if tdi_orbits is None:
            tdi_orbits = self.response_orbits

        self.tdi_orbits = tdi_orbits

        if self.num_pts * self.dt > self.response_orbits.t_base.max():
            warnings.warn(
                "Input number of points is longer in time than available orbital information. Trimming to fit orbital information."
            )
            self.num_pts = int(self.response_orbits.t_base.max() / self.dt)

        # setup spacecraft links indexes

        # setup TDI info
        self._init_TDI_delays()

    @property
    def response_gen(self) -> callable:
        """CPU/GPU function for generating the projections."""
        return self.backend.get_response_wrap

    @property
    def tdi_gen(self) -> callable:
        """CPU/GPU function for generating tdi."""
        return self.backend.get_tdi_delays_wrap

    @property
    def xp(self) -> object:
        return self.backend.xp

    @property
    def response_orbits(self) -> Orbits:
        """Response function orbits."""
        return self._response_orbits

    @response_orbits.setter
    def response_orbits(self, orbits: Orbits) -> None:
        """Set response orbits."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        assert isinstance(orbits, Orbits)

        self._response_orbits = deepcopy(orbits)

        if not self._response_orbits.configured:
            self._response_orbits.configure(linear_interp_setup=True)

    @property
    def tdi_orbits(self) -> Orbits:
        """TDI function orbits."""
        return self._tdi_orbits

    @tdi_orbits.setter
    def tdi_orbits(self, orbits: Orbits) -> None:
        """Set TDI orbits."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        assert isinstance(orbits, Orbits)
        assert orbits.backend.name.split("_")[-1] == self.backend.name.split("_")[-1]

        self._tdi_orbits = deepcopy(orbits)

        if not self._tdi_orbits.configured:
            self._tdi_orbits.configure(linear_interp_setup=True)

    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

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

    def _init_TDI_delays(self):
        """Initialize TDI specific information"""

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
        self.tdi_combinations = tdi_combinations

    @property
    def tdi_combinations(self) -> List:
        """TDI Combination setup"""
        return self._tdi_combinations

    @tdi_combinations.setter
    def tdi_combinations(self, tdi_combinations: List) -> None:
        """Set TDI combinations and fill out setup."""
        tdi_base_links = []
        tdi_link_combinations = []
        tdi_signs = []
        tdi_operation_index = []
        channels = []

        tdi_index = 0
        for permutation_number in range(3):
            for tmp in tdi_combinations:
                tdi_base_links.append(
                    self._cyclic_permutation(tmp["link"], permutation_number)
                )
                tdi_signs.append(float(tmp["sign"]))
                channels.append(permutation_number)
                if len(tmp["links_for_delay"]) == 0:
                    tdi_link_combinations.append(-11)
                    tdi_operation_index.append(tdi_index)

                else:
                    for link_delay in tmp["links_for_delay"]:

                        tdi_link_combinations.append(
                            self._cyclic_permutation(link_delay, permutation_number)
                        )
                        tdi_operation_index.append(tdi_index)

                tdi_index += 1

        self.tdi_operation_index = self.xp.asarray(tdi_operation_index).astype(
            self.xp.int32
        )
        self.tdi_base_links = self.xp.asarray(tdi_base_links).astype(self.xp.int32)
        self.tdi_link_combinations = self.xp.asarray(tdi_link_combinations).astype(
            self.xp.int32
        )
        self.tdi_signs = self.xp.asarray(tdi_signs).astype(self.xp.float64)
        self.channels = self.xp.asarray(channels).astype(self.xp.int32)
        assert len(self.tdi_link_combinations) == len(self.tdi_operation_index)

        assert (
            len(self.tdi_base_links)
            == len(np.unique(self.tdi_operation_index))
            == len(self.tdi_signs)
            == len(self.channels)
        )

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

    def _data_time_check(
        self, t_data: np.ndarray, input_in: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # remove input data that goes beyond orbital information
        if t_data.max() > self.response_orbits.t.max():
            warnings.warn(
                "Input waveform is longer than available orbital information. Trimming to fit orbital information."
            )

            max_ind = np.where(t_data <= self.response_orbits.t.max())[0][-1]

            t_data = t_data[:max_ind]
            input_in = input_in[:max_ind]
        return (t_data, input_in)

    def get_projections(self, input_in, lam, beta, t0=10000.0):
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
        self.tdi_start_ind = int(t0 / self.dt)
        # get necessary buffer for TDI
        self.check_tdi_buffer = int(100.0 * self.sampling_frequency) + 4 * self.order

        from copy import deepcopy

        tmp_orbits = deepcopy(self.response_orbits.x_base)
        self.projection_buffer = (
            int(
                (
                    np.sum(
                        tmp_orbits.copy() * tmp_orbits.copy(),
                        axis=-1,
                    )
                    ** (1 / 2)
                ).max()
                * C_inv
            )
            + 4 * self.order
        )
        self.projections_start_ind = self.tdi_start_ind - 2 * self.check_tdi_buffer

        if self.projections_start_ind < self.projection_buffer:
            raise ValueError(
                "Need to increase t0. The initial buffer is not large enough."
            )

        # determine sky vectors
        k = np.zeros(3, dtype=np.float64)
        u = np.zeros(3, dtype=np.float64)
        v = np.zeros(3, dtype=np.float64)

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

        self.nlinks = 6
        k_in = self.xp.asarray(k)
        u_in = self.xp.asarray(u)
        v_in = self.xp.asarray(v)

        input_in = self.xp.asarray(input_in)

        t_data = self.xp.arange(len(input_in)) * self.dt

        t_data, input_in = self._data_time_check(t_data, input_in)

        assert len(input_in) >= self.num_pts
        y_gw = self.xp.zeros((self.nlinks * self.num_pts,), dtype=self.xp.float64)

        self.response_gen(
            y_gw,
            t_data,
            k_in,
            u_in,
            v_in,
            self.dt,
            len(input_in),
            input_in,
            len(input_in),
            self.order,
            self.sampling_frequency,
            self.buffer_integer,
            self.A_in,
            self.deps,
            len(self.A_in),
            self.E_in,
            self.projections_start_ind,
            self.response_orbits,
        )

        self.y_gw_flat = y_gw
        self.y_gw_length = self.num_pts

    @property
    def XYZ(self):
        """Return links as an array"""
        return self.delayed_links_flat.reshape(3, -1)

    def get_tdi_delays(self, y_gw=None):
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

        self.delayed_links_flat = self.delayed_links_flat.flatten()

        t_data = self.xp.arange(self.y_gw_length) * self.dt

        num_units = int(self.tdi_operation_index.max() + 1)

        assert np.all(
            (np.diff(self.tdi_operation_index) == 0)
            | (np.diff(self.tdi_operation_index) == 1)
        )

        _, unit_starts, unit_lengths = np.unique(
            self.tdi_operation_index,
            return_index=True,
            return_counts=True,
        )

        unit_starts = unit_starts.astype(np.int32)
        unit_lengths = unit_lengths.astype(np.int32)

        self.tdi_gen(
            self.delayed_links_flat,
            self.y_gw_flat,
            self.y_gw_length,
            self.num_pts,
            t_data,
            unit_starts,
            unit_lengths,
            self.tdi_base_links,
            self.tdi_link_combinations,
            self.tdi_signs,
            self.channels,
            num_units,
            3,  # num channels
            self.order,
            self.sampling_frequency,
            self.buffer_integer,
            self.A_in,
            self.deps,
            len(self.A_in),
            self.E_in,
            self.tdi_start_ind,
            self.tdi_orbits,
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


class ResponseWrapper(FastLISAResponseParallelModule):
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
        force_backend (str, optional): If given, run this class on the requested backend. 
            Options are ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``. (default: ``None``)
        remove_garbage (bool or str, optional): If True, it removes everything before ``t0``
            and after the end time - ``t0``. If ``str``, it must be ``"zero"``. If ``"zero"``,
            it will not remove the points, but set them to zero. This is ideal for PE. (Default: ``True``)
        n_overide (int, optional): If not ``None``, this will override the determination of
            the number of points, ``n``, from ``int(T/dt)`` to the ``n_overide``. This is used
            if there is an issue matching points between the waveform generator and the response
            model.
        orbits (:class:`Orbits`, optional): Orbits class from LISA Analysis Tools. Works with LISA Orbits
            outputs: ``lisa-simulation.pages.in2p3.fr/orbits/``.
            (default: :class:`EqualArmlengthOrbits`)
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
        force_backend=None,
        remove_garbage=True,
        n_overide=None,
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        **kwargs,
    ):

        # store all necessary information
        self.waveform_gen = waveform_gen
        self.index_lambda = index_lambda
        self.index_beta = index_beta
        self.dt = dt
        self.t0 = t0
        self.sampling_frequency = 1.0 / dt
        super().__init__(force_backend=force_backend)

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        assert isinstance(orbits, Orbits)

        if Tobs * YRSID_SI > orbits.t_base.max():
            warnings.warn(
                f"Tobs is larger than available orbital information time array. Reducing Tobs to {orbits.t_base.max()}"
            )
            Tobs = orbits.t_base.max() / YRSID_SI

        if n_overide is not None:
            if not isinstance(n_overide, int):
                raise ValueError("n_overide must be an integer if not None.")
            self.n = n_overide

        else:
            self.n = int(Tobs * YRSID_SI / dt)

        self.Tobs = self.n * dt
        self.is_ecliptic_latitude = is_ecliptic_latitude
        self.remove_sky_coords = remove_sky_coords
        self.flip_hx = flip_hx
        self.remove_garbage = remove_garbage

        # initialize response function class
        self.response_model = pyResponseTDI(
            self.sampling_frequency, self.n, orbits=orbits, force_backend=force_backend, **kwargs
        )

        self.Tobs = (self.n * self.response_model.dt) / YRSID_SI

    @property
    def xp(self) -> object:
        return self.backend.xp

    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls):
        return ["fastlisaresponse_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(self, *args, **kwargs):
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

        self.response_model.get_projections(h, lam, beta, t0=self.t0)
        tdi_out = self.response_model.get_tdi_delays()

        out = list(tdi_out)
        if self.remove_garbage is True:  # bool
            for i in range(len(out)):
                out[i] = out[i][
                    self.response_model.tdi_start_ind : -self.response_model.tdi_start_ind
                ]

        elif isinstance(self.remove_garbage, str):  # bool
            if self.remove_garbage != "zero":
                raise ValueError("remove_garbage must be True, False, or 'zero'.")
            for i in range(len(out)):
                out[i][: self.response_model.tdi_start_ind] = 0.0
                out[i][-self.response_model.tdi_start_ind :] = 0.0

        return out
