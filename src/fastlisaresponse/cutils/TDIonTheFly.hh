#ifndef __TDI_ON_THE_FLY_HH__
#define __TDI_ON_THE_FLY_HH__

#include "Detector.hpp"

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
        void get_tdi(double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double t, int N, double costh, double phi, double cosi, double psi);
        CUDA_CALLABLE_MEMBER
        void LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k);
        CUDA_CALLABLE_MEMBER
        void get_tdi_amp_phase(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi, double *phi_ref);
        CUDA_CALLABLE_MEMBER
        void get_tdi_n(double *M, double *Mf, int n, int N, int a, int b, int c, double* tarray, double *amp_tdi_vals, double *phase_vals, struct CubicSpline *freq_spline, struct CubicSpline *phase_spline, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
        CUDA_CALLABLE_MEMBER
        virtual void get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N);
        CUDA_CALLABLE_MEMBER
        void get_tdi_sub(double *M, double *Mf, int n, int N, int a, int b, int c, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
        CUDA_CALLABLE_MEMBER
        void get_tdi_n(double *X, double *Xf, double *Y, double *Yf, double *Z, double *Zf, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi);
        CUDA_CALLABLE_MEMBER
        void get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n);
        CUDA_CALLABLE_MEMBER
        void get_tdi_Xf(double *X, double *Xf, double *Y, double *Yf, double *Z, double *Zf, double *params, double *t_arr, int N, double costh, double phi, double cosi, double psi);
        CUDA_CALLABLE_MEMBER
        void extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
        CUDA_CALLABLE_MEMBER
        int get_tdi_buffer_size(int N);
        CUDA_CALLABLE_MEMBER
        void unwrap_phase(int N, double *phase);
};

class GBTDIonTheFly : public LISATDIonTheFly {
    public:
        double T;

        CUDA_CALLABLE_MEMBER
        GBTDIonTheFly(Orbits *orbits_, double T_);
        CUDA_CALLABLE_MEMBER
        ~GBTDIonTheFly();
        CUDA_CALLABLE_MEMBER
        void get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N) override;
        CUDA_CALLABLE_MEMBER
        double ucb_amplitude(double t, double *params);
        CUDA_CALLABLE_MEMBER
        double ucb_phase(double t, double *params);
        CUDA_CALLABLE_MEMBER
        int get_gb_buffer_size(int N);
        CUDA_CALLABLE_MEMBER
        void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N);
        CUDA_CALLABLE_MEMBER
        void dealloc();
        CUDA_CALLABLE_MEMBER
        void get_phase_ref(double *t, double *phase, double *params, int N);
};

#endif // __TDI_ON_THE_FLY_HH__