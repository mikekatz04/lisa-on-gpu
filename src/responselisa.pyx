import numpy as np
cimport numpy as np

from fastlisaresponse.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "LISAResponse.hh":
    ctypedef void* cmplx 'cmplx'
    void get_response(double* y_gw, double* t_data, double* k_in, double* u_in, double* v_in, double dt,
                  int num_delays, int *link_space_craft_0_in, int *link_space_craft_1_in,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer, double* A_in, double deps, int num_A, double* E_in,
                  int projections_start_ind,
                  double* x_in_receiver, double* x_in_emitter, double* L_in, int num_orbit_inputs, int projections_cut_ind);

    void get_tdi_delays(double* delayed_links, double* input_links, int num_inputs, int num_orbit_info, double* delays, int num_delays, double dt, int* link_inds_in, int* tdi_signs_in, int num_units, int num_channels,
                   int order, double sampling_frequency, int buffer_integer, double* A_in, double deps, int num_A, double* E_in, int tdi_start_ind, int tdi_cut_ind);
    void get_delays_wrap(double *delay0_out, double *delay1_out, double *xi_p_out, double *xi_c_out, double *k_dot_n_out, 
              double* t_data, double *k_in, double *u_in, double *v_in, double dt,
              int num_delays, int *link_space_craft_0_in, int *link_space_craft_1_in, int projections_start_ind,
              double* x_in_receiver, double* x_in_emitter, double* L_in, int num_orbit_inputs, int projections_cut_ind);


@pointer_adjust
def get_response_wrap(y_gw, t_data, k_in, u_in, v_in, dt,
              num_delays, link_space_craft_0_in, link_space_craft_1_in,
              input_in, num_inputs, order, sampling_frequency, buffer_integer,
              A_in, deps, num_A, E_in, projections_start_ind,
              x_in_receiver, x_in_emitter, L_in, num_orbit_inputs, projections_cut_ind):

    cdef size_t y_gw_in = y_gw
    cdef size_t t_data_in = t_data
    cdef size_t k_in_in = k_in
    cdef size_t u_in_in = u_in
    cdef size_t v_in_in = v_in

    cdef size_t x_in_receiver_in = x_in_receiver
    cdef size_t x_in_emitter_in = x_in_emitter
    cdef size_t L_in_in = L_in

    cdef size_t link_space_craft_0_in_in = link_space_craft_0_in
    cdef size_t link_space_craft_1_in_in = link_space_craft_1_in
    cdef size_t input_in_in = input_in

    cdef size_t A_in_in = A_in
    cdef size_t E_in_in = E_in

    get_response(<double* >y_gw_in, <double*> t_data_in, <double* >k_in_in, <double* >u_in_in, <double* >v_in_in, dt,
                num_delays, <int *>link_space_craft_0_in_in, <int *>link_space_craft_1_in_in,
                <cmplx *>input_in_in, num_inputs, order, sampling_frequency, buffer_integer,
                <double*> A_in_in, deps, num_A, <double*> E_in_in, projections_start_ind,
                <double* > x_in_receiver_in, <double* > x_in_emitter_in, <double* > L_in_in, num_orbit_inputs, projections_cut_ind)


@pointer_adjust
def get_tdi_delays_wrap(delayed_links, y_gw, num_inputs, num_orbit_info, delays, num_delays, dt, link_inds, tdi_signs, num_units, num_channels,
               order, sampling_frequency, buffer_integer, A_in, deps, num_A, E_in, tdi_start_ind, tdi_cut_ind):

    cdef size_t delayed_links_in = delayed_links
    cdef size_t y_gw_in = y_gw
    cdef size_t link_inds_in = link_inds
    cdef size_t tdi_signs_in = tdi_signs
    cdef size_t delays_in = delays
    cdef size_t A_in_in = A_in
    cdef size_t E_in_in = E_in

    get_tdi_delays(<double*> delayed_links_in, <double*> y_gw_in, num_inputs, num_orbit_info, <double*> delays_in, num_delays, dt, <int*> link_inds_in, <int*> tdi_signs_in, num_units, num_channels,
                   order, sampling_frequency, buffer_integer, <double*> A_in_in, deps, num_A, <double*> E_in_in, tdi_start_ind, tdi_cut_ind)

@pointer_adjust
def get_delays(delay0_out, delay1_out, xi_p_out, xi_c_out, k_dot_n_out, 
              t_data, k_in, u_in, v_in, dt,
              num_delays, link_space_craft_0_in, link_space_craft_1_in, projections_start_ind,
              x_in_receiver, x_in_emitter, L_in, num_orbit_inputs, projections_cut_ind):

    cdef size_t t_data_in = t_data
    cdef size_t k_in_in = k_in
    cdef size_t u_in_in = u_in
    cdef size_t v_in_in = v_in

    cdef size_t x_in_receiver_in = x_in_receiver
    cdef size_t x_in_emitter_in = x_in_emitter
    cdef size_t L_in_in = L_in

    cdef size_t link_space_craft_0_in_in = link_space_craft_0_in
    cdef size_t link_space_craft_1_in_in = link_space_craft_1_in

    cdef size_t delay0_out_in = delay0_out
    cdef size_t delay1_out_in = delay1_out
    cdef size_t xi_p_out_in = xi_p_out
    cdef size_t xi_c_out_in = xi_c_out
    cdef size_t k_dot_n_out_in = k_dot_n_out
    
    get_delays_wrap(<double *>delay0_out_in, <double *>delay1_out_in, <double *>xi_p_out_in, <double *>xi_c_out_in, <double *>k_dot_n_out_in, 
              <double *> t_data_in, <double *>k_in_in, <double *>u_in_in, <double *>v_in_in, dt,
             num_delays, <int *>link_space_craft_0_in_in, <int *>link_space_craft_1_in_in, projections_start_ind,
              <double *> x_in_receiver_in, <double *> x_in_emitter_in, <double *> L_in_in, num_orbit_inputs, projections_cut_ind)

