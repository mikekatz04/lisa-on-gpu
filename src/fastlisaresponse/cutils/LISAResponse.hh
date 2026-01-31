#ifndef __LISA_RESPONSE__
#define __LISA_RESPONSE__

#include "cuda_complex.hpp"
#include "Detector.hpp"
#include "gbt_global.h"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS_RESPONSE 256
#define NLINKS 6

typedef gcmplx::complex<double> cmplx;


#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define LISAResponse LISAResponseGPU
#define TDIConfig TDIConfigGPU
#else
#define LISAResponse LISAResponseCPU
#define TDIConfig TDIConfigCPU
#endif


class TDIConfig{
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
    TDIConfig(int *unit_starts_, int *unit_lengths_, int *tdi_base_link_, int *tdi_link_combinations_, double *tdi_signs_in_, int *channels_, int num_units_, int num_channels_)
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
    ~TDIConfig(){};
};


class LISAResponse{
  public:
    Orbits *orbits;
    Orbits *orbits_gpu;
    TDIConfig *tdi_config;
    TDIConfig *tdi_config_gpu;
    LISAResponse(Orbits *orbits_, TDIConfig *tdi_config_){
      orbits = orbits_;
      tdi_config = tdi_config_;
    };
    ~LISAResponse(){};
    void get_tdi_delays(double *delayed_links_, double *input_links_, int num_inputs, int num_delays, double *t_arr_,
                    int order, double sampling_frequency, int buffer_integer, double *A_in_, double deps, int num_A, double *E_in_, int tdi_start_ind);
                    
    void get_response(double *y_gw_, double *t_data_, double *k_in_, double *u_in_, double *v_in_, double dt,
                  int num_delays,
                  cmplx* input_in_, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer,
                  double *A_in_, double deps, int num_A, double *E_in_, int projections_start_ind);
};
#endif // __LISA_RESPONSE__
