import numpy as np
cimport numpy as np

from gpubackendtools import wrapper

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "LISAResponse.hh":
    ctypedef void* cmplx 'cmplx'

    cdef cppclass LISAResponseWrap "LISAResponse":
        void add_orbit_information(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_) except+
        void dealloc() except+
        void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *unit_starts, int *unit_lengths, int *tdi_base_link, int *tdi_link_combinations, double *tdi_signs, int *channels, int num_units, int num_channels,
                int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind) except+

        void get_response(double* y_gw, double* t_data, double* k_in, double* u_in, double* v_in, double dt,
                  int num_delays,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer, double* A_in, double deps, int num_A, double* E_in,
                  int projections_start_ind) except+

cdef class pyLISAResponseWrap:
    cdef LISAResponseWrap *g

    def __cinit__(self):
        
        self.g = new LISAResponseWrap()

    def add_orbit_information(self, *args, **kwargs):
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

        cdef size_t n_arr_in = n_arr
        cdef size_t L_arr_in = L_arr
        cdef size_t x_arr_in = x_arr
        cdef size_t links_in = links
        cdef size_t sc_r_in = sc_r
        cdef size_t sc_e_in = sc_e
        
        self.g.add_orbit_information(
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

    def get_response_wrap(self, *args, **kwargs):

        (y_gw, t_data, k_in, u_in, v_in, dt,
                num_delays,
                input_in, num_inputs, order, sampling_frequency, buffer_integer,
                A_in, deps, num_A, E_in, projections_start_ind), tkwargs = wrapper(*args, **kwargs)

        cdef size_t y_gw_in = y_gw
        cdef size_t t_data_in = t_data
        cdef size_t k_in_in = k_in
        cdef size_t u_in_in = u_in
        cdef size_t v_in_in = v_in

        cdef size_t input_in_in = input_in

        cdef size_t A_in_in = A_in
        cdef size_t E_in_in = E_in

        self.g.get_response(<double* >y_gw_in, <double*> t_data_in, <double* >k_in_in, <double* >u_in_in, <double* >v_in_in, dt,
                    num_delays, 
                    <cmplx *>input_in_in, num_inputs, order, sampling_frequency, buffer_integer,
                    <double*> A_in_in, deps, num_A, <double*> E_in_in, projections_start_ind)


    def get_tdi_delays_wrap(self, *args, **kwargs):

        (delayed_links, y_gw, num_inputs, num_delays, t_arr, unit_starts, unit_lengths, tdi_base_link, tdi_link_combinations, tdi_signs, channels, num_units, num_channels,
                order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind), tkwargs = wrapper(*args, **kwargs)

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
        
        self.g.get_tdi_delays(<double*> delayed_links_in, <double*> y_gw_in, num_inputs, num_delays, <double*> t_arr_in, <int *>unit_starts_in, <int *>unit_lengths_in, <int*> tdi_base_link_in, <int*> tdi_link_combinations_in, <double*> tdi_signs_in, <int*> channels_in, num_units, num_channels,
                    order, sampling_frequency, buffer_integer, <double*> A_in_in, deps, num_A, <double*> E_in_in, tdi_start_ind)
