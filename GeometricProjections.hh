#ifndef __GEO_PROJ__
#define __GEO_PROJ__

#include "cuda_complex.hpp"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

typedef gcmplx::complex<double> cmplx;


void get_response(double *y_gw, double *k_in, double *u_in, double *v_in, double dt, double *x, double *n_in,
              int num_delays, int *link_space_craft_0_in, int *link_space_craft_1_in,
              double *L_vals, cmplx *input_in, int num_inputs, int order, double sampling_frequency, int buffer_integer, double *factorials_in, int num_factorials, double input_start_time);


#endif // __GEO_PROJ__
