#ifndef __TDI_ON_THE_FLY_HH__
#define __TDI_ON_THE_FLY_HH__

#include "Detector.hpp"
#include "Interpolate.hh"

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

class LISATDIonTheFly{
    public:
        Orbits *orbits;

        CUDA_CALLABLE_MEMBER 
        LISATDIonTheFly(Orbits *orbits_);
        CUDA_CALLABLE_MEMBER 
        ~LISATDIonTheFly();
        CUDA_CALLABLE_MEMBER
        void LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k);
        CUDA_CALLABLE_MEMBER
        void get_tdi(void *buffer, int buffer_length, cmplx *X, cmplx *Y, cmplx *Z, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double* phi_ref, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, int bin_i);
        CUDA_CALLABLE_MEMBER
        virtual void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i);
        CUDA_CALLABLE_MEMBER
        void get_tdi_sub(cmplx *M, int n, int N, int a, int b, int c, double t, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
        CUDA_CALLABLE_MEMBER
        void get_tdi_n(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi, int bin_i);
        CUDA_CALLABLE_MEMBER
        void get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n);
        CUDA_CALLABLE_MEMBER
        void get_tdi_Xf(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, int bin_i);
        CUDA_CALLABLE_MEMBER
        void extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
        CUDA_CALLABLE_MEMBER
        void new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR);
        CUDA_CALLABLE_MEMBER
        int get_tdi_buffer_size(int N);
        CUDA_CALLABLE_MEMBER
        void unwrap_phase(int N, double *phase);
        CUDA_CALLABLE_MEMBER
        void new_unwrap_phase(double *ph_correct_buffer, int N, double *phase);
        CUDA_CALLABLE_MEMBER
        void new_extract_phase(cmplx *M, double *phiR, int N, double *t_arr);
        CUDA_CALLABLE_MEMBER
        virtual void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int bin_i, int m);
};

class GBTDIonTheFly : public LISATDIonTheFly {
    public:
        double T;

        CUDA_CALLABLE_MEMBER
        GBTDIonTheFly(Orbits *orbits_, double T_);
        CUDA_CALLABLE_MEMBER
        ~GBTDIonTheFly();
        CUDA_CALLABLE_MEMBER
        void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i) override;
        CUDA_CALLABLE_MEMBER
        double ucb_amplitude(double t, double *params);
        CUDA_CALLABLE_MEMBER
        double ucb_phase(double t, double *params);
        CUDA_CALLABLE_MEMBER
        int get_gb_buffer_size(int N);
        CUDA_CALLABLE_MEMBER
        void run_wave_tdi(void *buffer, int buffer_length, cmplx *X, cmplx *Y, cmplx *Z, 
            double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase,
            double *phi_ref, double *params, double *t_arr, int N, int num_sub, int n_params);
        CUDA_CALLABLE_MEMBER
        void dealloc();
        CUDA_CALLABLE_MEMBER
        void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int bin_i, int m);
};



class TDSplineTDIWaveform : public LISATDIonTheFly{
    public:
        CubicSpline *amp_spline;
        CubicSpline *phase_spline;
        int binary_index_storage;

    CUDA_CALLABLE_MEMBER
    TDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *phase_spline_);
    CUDA_CALLABLE_MEMBER
    ~TDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    CUDA_CALLABLE_MEMBER
    void run_wave_tdi(void *buffer, int buffer_length, cmplx *X, cmplx *Y, cmplx *Z, 
    double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase,
    double *phi_ref, double *params, double *t_arr, int N, int num_sub, int n_params);
    CUDA_CALLABLE_MEMBER
    int get_td_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_CALLABLE_MEMBER
    void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int spline_i, int m);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
    CUDA_CALLABLE_MEMBER
    void check_x();
};


class FDSplineTDIWaveform : public LISATDIonTheFly{
    public:
        CubicSpline *amp_spline;
        CubicSpline *freq_spline;
        double *phase_ref_store;

    CUDA_CALLABLE_MEMBER
    FDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *freq_spline_, double *phase_ref_);
    CUDA_CALLABLE_MEMBER
    ~FDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    CUDA_CALLABLE_MEMBER
    void run_wave_tdi(void *buffer, int buffer_length, cmplx *X, cmplx *Y, cmplx *Z, 
    double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase,
    double *phi_ref, double *params, double *t_arr, int N, int num_sub, int n_params);
    CUDA_CALLABLE_MEMBER
    int get_fd_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_CALLABLE_MEMBER
    void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int spline_i, int m);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
};

#endif // __TDI_ON_THE_FLY_HH__