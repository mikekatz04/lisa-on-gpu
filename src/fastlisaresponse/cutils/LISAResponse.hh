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



class TDIConfig2{
  public:
    int *unit_starts;
    int *unit_lengths;
    int *tdi_base_link;
    int *tdi_link_combinations;
    double *tdi_signs_in;
    int *channels;
    int num_units;
    int num_channels;

    CUDA_CALLABLE_MEMBER 
    TDIConfig2(int *unit_starts_, int *unit_lengths_, int *tdi_base_link_, int *tdi_link_combinations_, double *tdi_signs_in_, int *channels_, int num_units_, int num_channels_)
    {
        unit_starts = unit_starts_;
        unit_lengths = unit_lengths_;
        tdi_base_link = tdi_base_link_;
        tdi_link_combinations = tdi_link_combinations_;
        tdi_signs_in = tdi_signs_in_;
        channels = channels_;
        num_units = num_units_;
        num_channels = num_channels_;
    };
    CUDA_CALLABLE_MEMBER 
    ~TDIConfig2(){};
    CUDA_CALLABLE_MEMBER 
    void dealloc(){};
};


class AddTDIConfig2{
  public:
    TDIConfig2 *tdi_config;

    void add_tdi_config(int *unit_starts_, int *unit_lengths_, int *tdi_base_link_, int *tdi_link_combinations_, double *tdi_signs_in_, int *channels_, int num_units_, int num_channels_){
        printf("new tdi config\n");
        if (tdi_config != NULL)
        {
            printf("delete tdi config\n");
            delete tdi_config;
        }
        tdi_config = new TDIConfig2(unit_starts_,  unit_lengths_,  tdi_base_link_,  tdi_link_combinations_,  tdi_signs_in_,  channels_,  num_units_,  num_channels_);
        printf("tdi config: %d\n", tdi_config->num_channels);
    };
    void dealloc(){
        printf("dealloc tdi config\n");
        if (tdi_config != NULL) 
            delete tdi_config;
    };
};


class LISAResponse: public AddOrbits, public AddTDIConfig2{
  public:
    Orbits *orbits;

    void get_tdi_delays(double *delayed_links, double *input_links, int num_inputs, int num_delays, double *t_arr, int *unit_starts, int *unit_lengths, int *tdi_base_link, int *tdi_link_combinations, double *tdi_signs_in, int *channels, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in, int tdi_start_ind);
    void get_response(double *y_gw, double *t_data, double *k_in, double *u_in, double *v_in, double dt,
                  int num_delays,
                  cmplx *input_in, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer, double *A_in, double deps, int num_A, double *E_in,
                  int projections_start_ind);
    void dealloc(){
        AddOrbits::dealloc();
        AddTDIConfig2::dealloc();
    }
};
#endif // __LISA_RESPONSE__
