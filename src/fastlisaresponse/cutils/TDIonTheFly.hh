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
    CUDA_CALLABLE_MEMBER 
    void dealloc(){};
};

class LISATDIonTheFly{
    public:
        Orbits *orbits;
        TDIConfig *tdi_config;

        CUDA_CALLABLE_MEMBER 
        void run_wave_tdi(
            void *buffer, int buffer_length, cmplx *tdi_channels_arr, 
            double *tdi_amp, double *tdi_phase, double *phi_ref, 
            double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        );
        CUDA_CALLABLE_MEMBER 
        LISATDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_);
        CUDA_CALLABLE_MEMBER 
        ~LISATDIonTheFly();
        // CUDA_CALLABLE_MEMBER
        // void LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k);
        CUDA_CALLABLE_MEMBER
        void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
        // CUDA_CALLABLE_MEMBER
        // virtual void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i);
        // CUDA_CALLABLE_MEMBER
        // void get_tdi_sub(cmplx *M, int n, int N, int a, int b, int c, double t, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
        // CUDA_CALLABLE_MEMBER
        // void get_tdi_n(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi, int bin_i);
        // CUDA_CALLABLE_MEMBER
        // void get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n);
        CUDA_CALLABLE_MEMBER
        void get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i);
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
        // CUDA_CALLABLE_MEMBER
        // virtual void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int bin_i, int m);
        CUDA_CALLABLE_MEMBER
        virtual void get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i);
        CUDA_CALLABLE_MEMBER
        virtual int get_beta_index(); 
        CUDA_CALLABLE_MEMBER
        virtual int get_lam_index(); 
        CUDA_CALLABLE_MEMBER
        virtual int get_psi_index(); 
        CUDA_CALLABLE_MEMBER
        virtual int get_inc_index(); 
        CUDA_CALLABLE_MEMBER
        void get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params);
        CUDA_CALLABLE_MEMBER
        void xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n);
        CUDA_CALLABLE_MEMBER
        virtual double get_amp(double t, double *params, int bin_i);
        CUDA_CALLABLE_MEMBER
        virtual double get_phase(double t, double *params, int bin_i);
};

class GBTDIonTheFly : public LISATDIonTheFly {
    public:
        double T;

        CUDA_CALLABLE_MEMBER
        GBTDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, double T_);
        CUDA_CALLABLE_MEMBER
        ~GBTDIonTheFly();
        // CUDA_CALLABLE_MEMBER
        // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i) override;
        CUDA_CALLABLE_MEMBER
        double ucb_amplitude(double t, double *params);
        CUDA_CALLABLE_MEMBER
        double ucb_phase(double t, double *params);
        CUDA_CALLABLE_MEMBER
        int get_gb_buffer_size(int N);
        CUDA_CALLABLE_MEMBER
        // void run_wave_tdi(
        //     cmplx *tdi_channels_arr, 
        //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
        //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        // );
        CUDA_CALLABLE_MEMBER
        void dealloc();
        CUDA_CALLABLE_MEMBER
        void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int bin_i, int m);
        CUDA_CALLABLE_MEMBER
        int get_amplitude_index();
        CUDA_CALLABLE_MEMBER
        int get_f0_index();
        CUDA_CALLABLE_MEMBER
        int get_fdot0_index();
        CUDA_CALLABLE_MEMBER
        int get_fddot0_index();
        CUDA_CALLABLE_MEMBER
        int get_phi0_index();
        CUDA_CALLABLE_MEMBER
        int get_beta_index(); 
        CUDA_CALLABLE_MEMBER
        int get_lam_index(); 
        CUDA_CALLABLE_MEMBER
        int get_psi_index(); 
        CUDA_CALLABLE_MEMBER
        int get_inc_index();
        double get_amp(double t, double *params, int bin_i);
        double get_phase(double t, double *params, int bin_i);
};



class TDSplineTDIWaveform : public LISATDIonTheFly{
    public:
        CubicSpline *amp_spline;
        CubicSpline *phase_spline;
        int binary_index_storage;

    CUDA_CALLABLE_MEMBER
    TDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *phase_spline_);
    CUDA_CALLABLE_MEMBER
    ~TDSplineTDIWaveform(){};
    // CUDA_CALLABLE_MEMBER
    // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    CUDA_CALLABLE_MEMBER
    // void run_wave_tdi(
    //     cmplx *tdi_channels_arr, 
    //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
    //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
    // );
    CUDA_CALLABLE_MEMBER
    int get_td_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_CALLABLE_MEMBER
    void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int spline_i, int m);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
    CUDA_CALLABLE_MEMBER
    void check_x();
    CUDA_CALLABLE_MEMBER
    double get_amp(double t, double *params, int spline_i);
    CUDA_CALLABLE_MEMBER
    double get_phase(double t, double *params, int spline_i);
    CUDA_CALLABLE_MEMBER
    int get_beta_index(); 
    CUDA_CALLABLE_MEMBER
    int get_lam_index(); 
    CUDA_CALLABLE_MEMBER
    int get_psi_index(); 
    CUDA_CALLABLE_MEMBER
    int get_inc_index();
};


class FDSplineTDIWaveform : public LISATDIonTheFly{
    public:
        CubicSpline *amp_spline;
        CubicSpline *freq_spline;
        double *phase_ref_store;

    CUDA_CALLABLE_MEMBER
    FDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_, double *phase_ref_);
    CUDA_CALLABLE_MEMBER
    ~FDSplineTDIWaveform(){};
    // CUDA_CALLABLE_MEMBER
    // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    CUDA_CALLABLE_MEMBER
    // void run_wave_tdi(
    //     cmplx *tdi_channels_arr, 
    //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
    //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
    // );
    CUDA_CALLABLE_MEMBER
    int get_fd_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_CALLABLE_MEMBER
    void get_phase_ref(double t, double t_sc, double *phase, double *params, int N, int spline_i, int m);
    CUDA_CALLABLE_MEMBER
    void dealloc(){};
    CUDA_CALLABLE_MEMBER
    double get_amp(double t, double *params, int spline_i);
    CUDA_CALLABLE_MEMBER
    double get_phase(double t, double *params, int spline_i);
    CUDA_CALLABLE_MEMBER
    int get_beta_index(); 
    CUDA_CALLABLE_MEMBER
    int get_lam_index(); 
    CUDA_CALLABLE_MEMBER
    int get_psi_index(); 
    CUDA_CALLABLE_MEMBER
    int get_inc_index();
};

#endif // __TDI_ON_THE_FLY_HH__