import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from gpubackendtools import wrapper


cdef extern from "Detector.hpp":
    cdef cppclass VecWrap "Vec":
        double x
        double y
        double z
        VecWrap(double x_, double y_, double z_) except+

    cdef cppclass OrbitsWrap "Orbits":
        OrbitsWrap(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_) except+
        int get_window(double t) except+
        void get_normal_unit_vec_ptr(VecWrap *vec, double t, int link)
        int get_link_ind(int link) except+
        int get_sc_ind(int sc) except+
        double get_light_travel_time(double t, int link) except+
        VecWrap get_pos_ptr(VecWrap* out, double t, int sc) except+
        void get_light_travel_time_arr(double *ltt, double *t, int *link, int num) except+
        void dealloc();
        void get_pos_arr(double *pos_x, double *pos_y, double *pos_z, double *t, int *sc, int num) except+
        void get_normal_unit_vec_arr(double *normal_unit_vec_x, double *normal_unit_vec_y, double *normal_unit_vec_z, double *t, int *link, int num) except+
        int get_sc_r_from_arr(int i) except+
        int get_sc_e_from_arr(int i) except+
        int get_link_from_arr(int i) except+
        

cdef class pycppDetector:
    cdef OrbitsWrap *g
    cdef double dt
    cdef int N
    cdef size_t n_arr
    cdef size_t L_arr
    cdef size_t x_arr
    cdef size_t links
    cdef size_t sc_r
    cdef size_t sc_e
    cdef double armlength

    def __cinit__(self, 
        *args, 
        **kwargs
    ):
        (
            dt,
            N, 
            n_arr,
            L_arr, 
            x_arr,
            links,
            sc_r, 
            sc_e,
            armlength
        ), tkwargs = wrapper(*args, **kwargs)

        self.dt = dt
        self.N = N 
        self.n_arr = n_arr
        self.L_arr = L_arr 
        self.x_arr = x_arr
        self.links = links
        self.sc_r = sc_r
        self.sc_e = sc_e
        self.armlength = armlength

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        cdef size_t links_in = links
        cdef size_t sc_r_in = sc_r
        cdef size_t sc_e_in = sc_e
        
        self.g = new OrbitsWrap(
            dt,
            N,
            <double*> n_arr_in,
            <double*> L_arr_in, 
            <double*> x_arr_in, 
            <int*> links_in, 
            <int*> sc_r_in, 
            <int*> sc_e_in,
            armlength
        )

    def __dealloc__(self):
        self.g.dealloc()
        if self.g:
            del self.g

    def __reduce__(self):
        return (rebuild, (self.dt, self.N, self.n_arr, self.L_arr, self.x_arr, self.links, self.sc_r, self.sc_e, self.armlength))

    def get_window(self, t: float) -> int:
        return self.g.get_window(t)

    def get_normal_unit_vec_single(self, t: float, link: int) -> np.ndarray:
        cdef VecWrap *out = new VecWrap(0.0, 0.0, 0.0)
        self.g.get_normal_unit_vec_ptr(out, t, link)

        return np.array([out.x, out.y, out.z])

    def get_normal_unit_vec_arr(self, t: np.ndarray, link: int) -> np.ndarray:
        output = np.zeros((len(t), 3), dtype=float)
        assert t.ndim == 1
        for i in range(len(t)):
            output[i] = self.get_normal_unit_vec_single(t[i], link)

        return output

    def get_normal_unit_vec(self, t: np.ndarray | float, link: int) -> np.ndarray:

        if isinstance(t, float):
            return self.get_normal_unit_vec_single(t, link)
        elif isinstance(t, np.ndarray):
            return self.get_normal_unit_vec_arr(t, link)

    def get_link_ind(self, link: int) -> int:
        return self.g.get_link_ind(link)

    def get_sc_ind(self, sc: int) -> int:
        return self.g.get_sc_ind(sc)

    def get_light_travel_time_single(self, t: float, link: int) -> float:
        return self.g.get_light_travel_time(t, link)

    def get_light_travel_time_arr(self, t: np.ndarray, link: int) -> np.ndarray:
        output = np.zeros((len(t),), dtype=float)
        assert t.ndim == 1
        for i in range(len(t)):
            output[i] = self.get_light_travel_time_single(t[i], link)

        return output

    def get_light_travel_time(self, t: np.ndarray | float, link: int) -> np.ndarray:

        if isinstance(t, float):
            print("t", t)
            return self.get_light_travel_time_single(t, link)
        elif isinstance(t, np.ndarray):
            return self.get_light_travel_time_arr(t, link)
    
    def get_pos_single(self, t: float, sc: int) -> np.ndarray:
  
        cdef VecWrap *out = new VecWrap(0.0, 0.0, 0.0)
        self.g.get_pos_ptr(out, t, sc)

        return np.array([out.x, out.y, out.z])

    def get_pos_arr(self, t: np.ndarray, sc: int) -> np.ndarray:
        output = np.zeros((len(t), 3), dtype=float)
        assert t.ndim == 1
        for i in range(len(t)):
            output[i] = self.get_pos_single(t[i], sc)

        return output

    def get_pos(self, t: np.ndarray | float, sc: int) -> np.ndarray:

        if isinstance(t, float):
            return self.get_pos_single(t, sc)
        elif isinstance(t, np.ndarray):
            return self.get_pos_arr(t, sc)

    def get_pos_arr_wrap(self, *args, **kwargs):
        (pos_x, pos_y, pos_z, t, sc, num), tkwargs = wrapper(*args, **kwargs)

        cdef size_t pos_x_in = pos_x
        cdef size_t pos_y_in = pos_y
        cdef size_t pos_z_in = pos_z
        cdef size_t t_in = t
        cdef size_t sc_in = sc

        self.g.get_pos_arr(
            <double *>pos_x_in,
            <double *>pos_y_in,
            <double *>pos_z_in,
            <double *>t_in,
            <int *>sc_in,
            num
        )
    
    def get_normal_unit_vec_arr_wrap(self, *args, **kwargs):
        (normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z, t, sc, num), tkwargs = wrapper(*args, **kwargs)

        cdef size_t normal_unit_vec_x_in = normal_unit_vec_x
        cdef size_t normal_unit_vec_y_in = normal_unit_vec_y
        cdef size_t normal_unit_vec_z_in = normal_unit_vec_z
        cdef size_t t_in = t
        cdef size_t sc_in = sc

        self.g.get_normal_unit_vec_arr(
            <double *>normal_unit_vec_x_in,
            <double *>normal_unit_vec_y_in,
            <double *>normal_unit_vec_z_in,
            <double *>t_in,
            <int *>sc_in,
            num
        )

    def get_light_travel_time_arr_wrap(self, *args, **kwargs):
        (ltt, t, link, num), tkwargs = wrapper(*args, **kwargs)

        cdef size_t ltt_in = ltt
        cdef size_t t_in = t
        cdef size_t link_in = link

        self.g.get_light_travel_time_arr(
            <double *>ltt_in,
            <double *>t_in,
            <int *>link_in,
            num
        )

    def get_sc_r_from_arr(self, i: int) -> int:
        return self.g.get_sc_r_from_arr(i)

    def get_sc_e_from_arr(self, i: int) -> int:
        return self.g.get_sc_e_from_arr(i)

    def get_link_from_arr(self, i: int) -> int:
        return self.g.get_link_from_arr(i)
        

    @property
    def ptr(self) -> long:
        return <uintptr_t>self.g
    
        
def rebuild(dt,
    N, 
    n_arr,
    L_arr, 
    x_arr,
    links,
    sc_r, 
    sc_e,
    armlength
):
    c = pycppDetector(
        dt,
        N, 
        n_arr,
        L_arr, 
        x_arr,
        links,
        sc_r, 
        sc_e,
        armlength
    )
    return c

    
import numpy as np
cimport numpy as np

from gpubackendtools import wrapper

assert sizeof(int) == sizeof(np.int32_t)


cdef extern from "LISAResponse.hh":
    ctypedef void* cmplx 'cmplx'
    void get_response(double* y_gw, double* t_data, double* k_in, double* u_in, double* v_in, double dt,
                  int num_delays,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer, double* A_in, double deps, int num_A, double* E_in,
                  int projections_start_ind,
                  OrbitsWrap *orbits);

    void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *unit_starts, int *unit_lengths, int *tdi_base_link, int *tdi_link_combinations, double *tdi_signs, int *channels, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind, OrbitsWrap *orbits);


def get_response_wrap(*args, **kwargs):

    (y_gw, t_data, k_in, u_in, v_in, dt,
              num_delays,
              input_in, num_inputs, order, sampling_frequency, buffer_integer,
              A_in, deps, num_A, E_in, projections_start_ind,
              orbits), tkwargs = wrapper(*args, **kwargs)

    cdef size_t y_gw_in = y_gw
    cdef size_t t_data_in = t_data
    cdef size_t k_in_in = k_in
    cdef size_t u_in_in = u_in
    cdef size_t v_in_in = v_in

    cdef size_t _orbits = orbits
    cdef OrbitsWrap *orbits_in = <OrbitsWrap *>_orbits
    cdef size_t input_in_in = input_in

    cdef size_t A_in_in = A_in
    cdef size_t E_in_in = E_in

    get_response(<double* >y_gw_in, <double*> t_data_in, <double* >k_in_in, <double* >u_in_in, <double* >v_in_in, dt,
                num_delays, 
                <cmplx *>input_in_in, num_inputs, order, sampling_frequency, buffer_integer,
                <double*> A_in_in, deps, num_A, <double*> E_in_in, projections_start_ind,
                orbits_in)


def get_tdi_delays_wrap(*args, **kwargs):

    (delayed_links, y_gw, num_inputs, num_delays, t_arr, unit_starts, unit_lengths, tdi_base_link, tdi_link_combinations, tdi_signs, channels, num_units, num_channels,
               order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind, orbits), tkwargs = wrapper(*args, **kwargs)

    cdef size_t delayed_links_in = delayed_links
    cdef size_t y_gw_in = y_gw
    cdef size_t A_in_in = A_in
    cdef size_t E_in_in = E_in
    cdef size_t t_arr_in = t_arr
    cdef size_t unit_starts_in = unit_starts
    cdef size_t unit_lengths_in = unit_lengths
    cdef size_t tdi_base_link_in = tdi_base_link
    cdef size_t tdi_link_combinations_in = tdi_link_combinations
    cdef size_t tdi_signs_in = tdi_signs
    cdef size_t channels_in = channels
    cdef size_t _orbits = orbits
    cdef OrbitsWrap *orbits_in = <OrbitsWrap *>_orbits
    
    get_tdi_delays(<double*> delayed_links_in, <double*> y_gw_in, num_inputs, num_delays, <double*> t_arr_in, <int *>unit_starts_in, <int *>unit_lengths_in, <int*> tdi_base_link_in, <int*> tdi_link_combinations_in, <double*> tdi_signs_in, <int*> channels_in, num_units, num_channels,
                   order, sampling_frequency, buffer_integer, <double*> A_in_in, deps, num_A, <double*> E_in_in, tdi_start_ind, orbits_in)
