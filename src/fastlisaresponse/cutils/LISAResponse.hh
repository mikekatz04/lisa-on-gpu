#ifndef __LISA_RESPONSE__
#define __LISA_RESPONSE__

#include "cuda_complex.hpp"
#include "Detector.hpp"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

typedef gcmplx::complex<double> cmplx;



#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#define CUDA_KERNEL __global__
#define CUDA_SHARED __shared__
#define CUDA_SYNC_THREADS __syncthreads()
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL
#define CUDA_SHARED
#define CUDA_SYNC_THREADS
#endif

#if defined(__CUDACC__) || defined(__CUDA_COMPILATION__)
#define LISAResponse LISAResponseGPU
#define Orbits OrbitsGPU
#else
#define LISAResponse LISAResponseCPU
#define Orbits OrbitsCPU
#endif

#ifdef __CUDACC__
#define gpuErrchk(ans)                         \
    {                                          \
        gpuAssert2((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert2(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif


class LISAResponse{
  public:
    Orbits *orbits;
    LISAResponse(Orbits *orbits_){
      orbits = orbits_;
      // TODO: add GPU orbits now?
    };
    void get_tdi_delays(double *delayed_links_, double *input_links_, int num_inputs, int num_delays, double *t_arr_, int *unit_starts_, int *unit_lengths_, int *tdi_base_link_, int *tdi_link_combinations_, double *tdi_signs_in_, int *channels_, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, double *A_in_, double deps, int num_A, double *E_in_, int tdi_start_ind);
                    
    void get_response(double *y_gw_, double *t_data_, double *k_in_, double *u_in_, double *v_in_, double dt,
                  int num_delays,
                  cmplx* input_in_, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer,
                  double *A_in_, double deps, int num_A, double *E_in_, int projections_start_ind);
};
#endif // __LISA_RESPONSE__
