#ifndef __GEO_PROJ__
#define __GEO_PROJ__

#include "cuda_complex.hpp"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

typedef gcmplx::complex<double> cmplx;


void get_response(double *y_gw, double *k_in, double *u_in, double *v_in, double dt,
              int num_delays, int *link_space_craft_0_in, int *link_space_craft_1_in,
              cmplx *input_in, int num_inputs, int order,
              double sampling_frequency, int buffer_integer, double *factorials_in,
              int num_factorials, double input_start_time,
              double *interp_array, int init_len, double* h_t);

void get_tdi_delays(double* delayed_links, double *y_gw, int num_inputs, int num_delays, double dt, int* link_inds, int* delay_factor, int num_units,
              int order, double sampling_frequency, int buffer_integer, double *factorials_in, int num_factorials, double input_start_time,
              double *interp_array, int init_len, double* h_t);

#endif // __GEO_PROJ__
