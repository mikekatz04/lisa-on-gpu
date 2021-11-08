#ifndef __LISA_RESPONSE__
#define __LISA_RESPONSE__

#include "cuda_complex.hpp"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

typedef gcmplx::complex<double> cmplx;


void get_response(double *y_gw, double* t_data, double *k_in, double *u_in, double *v_in, double dt,
              int num_delays, int *link_space_craft_0_in, int *link_space_craft_1_in,
              cmplx *input_in, int num_inputs, int order,
              double sampling_frequency, int buffer_integer, double* A_in, double deps, int num_A, double* E_in,
              int projections_start_ind,
              double* x_in_emitter, double* x_in_receiver, double* L_in, int num_orbit_inputs);

void get_tdi_delays(double* delayed_links, double* input_links, int num_inputs, int num_orbit_info, double* delays, int num_delays, double dt, int* link_inds_in, int* tdi_signs_in, int num_units, int num_channels,
               int order, double sampling_frequency, int buffer_integer, double* A_in, double deps, int num_A, double* E_in, int tdi_start_ind);

#endif // __LISA_RESPONSE__
