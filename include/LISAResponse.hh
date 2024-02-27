#ifndef __LISA_RESPONSE__
#define __LISA_RESPONSE__

#include "cuda_complex.hpp"
#include "Detector.hpp"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

typedef gcmplx::complex<double> cmplx;

void get_response(double *y_gw, double *t_data, double *k_in, double *u_in, double *v_in, double dt,
                  int num_delays,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in,
                  int projections_start_ind,
                  Orbits *orbits);

void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *tdi_base_link, int *tdi_link_combinations, int *tdi_signs_in, int *channels, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind, Orbits *orbits);

#endif // __LISA_RESPONSE__
