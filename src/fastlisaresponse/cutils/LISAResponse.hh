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

void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *unit_starts, int *unit_lengths, int *tdi_base_link, int *tdi_link_combinations, double *tdi_signs_in, int *channels, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind, Orbits *orbits);


class LISAResponse{
  public:
    Orbits *orbits;

    void add_orbit_information(double dt_, int N_, double *n_arr_, double *L_arr_, double *x_arr_, int *links_, int *sc_r_, int *sc_e_, double armlength_)
    {
        if (orbits != NULL)
        {
            delete orbits;
        }
        orbits = new Orbits(dt_, N_, n_arr_, L_arr_, x_arr_, links_, sc_r_, sc_e_, armlength_);
    };
    void dealloc(){delete orbits;};
    void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *unit_starts, int *unit_lengths, int *tdi_base_link, int *tdi_link_combinations, double *tdi_signs_in, int *channels, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind);
    void get_response(double *y_gw, double *t_data, double *k_in, double *u_in, double *v_in, double dt,
                  int num_delays,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in,
                  int projections_start_ind);
};
#endif // __LISA_RESPONSE__
