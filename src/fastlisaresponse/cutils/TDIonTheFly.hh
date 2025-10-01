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

class CubicSplineSegment{
   public:
    double x0;
    double y0;
    double c1;
    double c2;
    double c3;
    int spline_type;

    CUDA_CALLABLE_MEMBER
    CubicSplineSegment(double x0_, double y0_, double c1_, double c2_, double c3_, int spline_type_)
    {
        x0 = x0_;
        y0 = y0_;
        c1 = c1_;
        c2 = c2_;
        c3 = c3_;
        spline_type = spline_type_;
    };
    CUDA_CALLABLE_MEMBER
    double eval(double x_new)
    {
        double dx = x_new - x0;
        double out = y0 + c1 * dx + c2 * dx * dx + c3 * dx * dx * dx;
        return out;
    };
    CUDA_CALLABLE_MEMBER
    double eval_single_derivative(double x_new)
    {
        double dx = x_new - x0;
        double out = c1+ c2 * dx + c3 * dx * dx;
        return out;
    };
    CUDA_CALLABLE_MEMBER
    double eval_double_derivative(double x_new)
    {
        double dx = x_new - x0;
        double out = c2 + c3 * dx;
        return out;
    };
    CUDA_CALLABLE_MEMBER
    double eval_triple_derivative(double x_new)
    {
        double out = c3;
        return out;
    };
};

class CubicSpline{
public:
    double *x0;
    double *y0;
    double *c1;
    double *c2;
    double *c3;
    double dx;
    int N;
    int spline_type;

    CUDA_CALLABLE_MEMBER
    CubicSpline(double *x0_, double *y0_, double *c1_, double *c2_, double *c3_, double dx_, int N_, int spline_type_)
    {
        x0 = x0_;
        y0 = y0_;
        c1 = c1_;
        c2 = c2_;
        c3 = c3_;
        dx = dx_;
        N = N_;
        spline_type = spline_type_;
    };
    CUDA_CALLABLE_MEMBER
    CUDA_CALLABLE_MEMBER
    ~CubicSpline(){};
    CUDA_CALLABLE_MEMBER
    int get_window(double x_new);
    CUDA_CALLABLE_MEMBER
    CubicSplineSegment get_cublic_spline_segment(double x_new);
    CUDA_CALLABLE_MEMBER
    double eval_single(double x_new);
    CUDA_CALLABLE_MEMBER
    void eval(double *y_new, double *x_new, int N);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
};

class TDSplineTDIWaveform : public LISATDIonTheFly{
    public:
        CubicSpline *amp_spline;
        CubicSpline *phase_spline;

    CUDA_CALLABLE_MEMBER
    TDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *phase_spline_);
    CUDA_CALLABLE_MEMBER
    ~TDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    void get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N);
    CUDA_CALLABLE_MEMBER
    void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N);
    CUDA_CALLABLE_MEMBER
    int get_td_spline_buffer_size(int N){return N * sizeof(double) + get_tdi_buffer_size(N);};
    CUDA_CALLABLE_MEMBER
    void get_phase_ref(double *t, double *phase, double *params, int N);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
};


class FDSplineTDIWaveform : public LISATDIonTheFly{
    public:
        CubicSpline *amp_spline;
        CubicSpline *freq_spline;

    CUDA_CALLABLE_MEMBER
    FDSplineTDIWaveform(Orbits *orbits_, CubicSpline *amp_spline_, CubicSpline *freq_spline_);
    CUDA_CALLABLE_MEMBER
    ~FDSplineTDIWaveform(){};
    CUDA_CALLABLE_MEMBER
    void get_amp_and_phase(double *t, double *amp, double *phase, double *params, int N);
    CUDA_CALLABLE_MEMBER
    void run_wave_tdi(void *buffer, int buffer_size, double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *params, double *t_arr, int N);
    CUDA_CALLABLE_MEMBER
    int get_td_spline_buffer_size(int N){return N * sizeof(double) + get_tdi_buffer_size(N);};
    CUDA_CALLABLE_MEMBER
    void get_phase_ref(double *t, double *phase, double *params, int N);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
};

#endif // __TDI_ON_THE_FLY_HH__