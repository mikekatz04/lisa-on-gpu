import numpy as np
cimport numpy as np

from lisatools.utils.pointeradjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "Detector.hpp":
    ctypedef void* Orbits 'Orbits'

cdef extern from "LISAResponse.hh":
    ctypedef void* cmplx 'cmplx'
    void get_response(double* y_gw, double* t_data, double* k_in, double* u_in, double* v_in, double dt,
                  int num_delays,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer, double* A_in, double deps, int num_A, double* E_in,
                  int projections_start_ind,
                  Orbits *orbits);

    void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *tdi_base_link, int *tdi_link_combinations, int *tdi_signs, int *channels, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind, Orbits *orbits);

@pointer_adjust
def get_response_wrap(y_gw, t_data, k_in, u_in, v_in, dt,
              num_delays,
              input_in, num_inputs, order, sampling_frequency, buffer_integer,
              A_in, deps, num_A, E_in, projections_start_ind,
              orbits):

    cdef size_t y_gw_in = y_gw
    cdef size_t t_data_in = t_data
    cdef size_t k_in_in = k_in
    cdef size_t u_in_in = u_in
    cdef size_t v_in_in = v_in

    cdef size_t orbits_in = orbits
    cdef size_t input_in_in = input_in

    cdef size_t A_in_in = A_in
    cdef size_t E_in_in = E_in

    get_response(<double* >y_gw_in, <double*> t_data_in, <double* >k_in_in, <double* >u_in_in, <double* >v_in_in, dt,
                num_delays, 
                <cmplx *>input_in_in, num_inputs, order, sampling_frequency, buffer_integer,
                <double*> A_in_in, deps, num_A, <double*> E_in_in, projections_start_ind,
                <Orbits*> orbits_in)


@pointer_adjust
def get_tdi_delays_wrap(delayed_links, y_gw, num_inputs, num_delays, t_arr, tdi_base_link, tdi_link_combinations, tdi_signs, channels, num_units, num_channels,
               order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind, orbits):

    cdef size_t delayed_links_in = delayed_links
    cdef size_t y_gw_in = y_gw
    cdef size_t A_in_in = A_in
    cdef size_t E_in_in = E_in
    cdef size_t t_arr_in = t_arr
    cdef size_t tdi_base_link_in = tdi_base_link
    cdef size_t tdi_link_combinations_in = tdi_link_combinations
    cdef size_t tdi_signs_in = tdi_signs
    cdef size_t channels_in = channels
    cdef size_t orbits_in = orbits

    get_tdi_delays(<double*> delayed_links_in, <double*> y_gw_in, num_inputs, num_delays, <double*> t_arr_in, <int*> tdi_base_link_in, <int*> tdi_link_combinations_in, <int*> tdi_signs_in, <int*> channels_in, num_units, num_channels,
                   order, sampling_frequency, buffer_integer, <double*> A_in_in, deps, num_A, <double*> E_in_in, tdi_start_ind, <Orbits*> orbits_in)
