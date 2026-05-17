#ifndef __TDI_ON_THE_FLY_HH__
#define __TDI_ON_THE_FLY_HH__

#include "Detector.hpp"
#include "Interpolate.hh"
#include "LISAResponse.hh"
#include "gbt_global.h"


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GBTDIonTheFly GBTDIonTheFlyGPU
#define FDSplineTDIWaveform FDSplineTDIWaveformGPU
#define TDSplineTDIWaveform TDSplineTDIWaveformGPU
#define WaveletLookupTable WaveletLookupTableGPU
#define WDMDomain WDMDomainGPU
#define FDDomain FDDomainGPU
#define GBComputationGroup GBComputationGroupGPU
#else
#define GBTDIonTheFly GBTDIonTheFlyCPU
#define FDSplineTDIWaveform FDSplineTDIWaveformCPU
#define TDSplineTDIWaveform TDSplineTDIWaveformCPU
#define WDMDomain WDMDomainCPU
#define FDDomain FDDomainCPU
#define GBComputationGroup GBComputationGroupCPU
#endif

#define TDI_XYZ 1
#define TDI_AET 2
#define TDI_AE 3

class LISATDIonTheFly{
    public:
        Orbits *orbits;
        TDIConfig *tdi_config;
        int inc_index;
        int psi_index;
        int lam_index;
        int beta_index;
        int N_store;

        CUDA_DEVICE 
        void run_wave_tdi(
            void *buffer, int buffer_length, cmplx *tdi_channels_arr, 
            double *tdi_amp, double *tdi_phase, double *phi_ref, 
            double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        );
        CUDA_CALLABLE_MEMBER 
        LISATDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, int inc_index_, int psi_index_, int lam_index_, int beta_index_)
        {
            orbits = orbits_;
            tdi_config = tdi_config_;
            inc_index = inc_index_;
            psi_index = psi_index_;
            lam_index = lam_index_;
            beta_index = beta_index_;
        };
        CUDA_CALLABLE_MEMBER 
        ~LISATDIonTheFly();
        CUDA_DEVICE
        void print_orbits_tdi();
        CUDA_DEVICE
        void fill_link_arrays(int *link_space_craft_rec, int *link_space_craft_em);
        // CUDA_DEVICE
        // void LISA_polarization_tensor(double costh, double phi, double *eplus, double *ecross, double *k);
        CUDA_DEVICE
        void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
        // CUDA_DEVICE
        // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i);
        // CUDA_DEVICE
        // void get_tdi_sub(cmplx *M, int n, int N, int a, int b, int c, double t, double* tarray, double *amp_tdi_vals, double *phase_tdi_vals, double Aplus, double Across, double cos2psi, double sin2psi, double *App, double *Apm, double *Acp, double *Acm, double *kr, double *Larm);
        // CUDA_DEVICE
        // void get_tdi_n(cmplx *X, cmplx *Y, cmplx *Z, double* phi_ref, double *params, double t, int m, int N, double costh, double phi, double cosi, double psi, int bin_i);
        // CUDA_DEVICE
        // void get_t_tdi(double *t_out, double *kr, double *Larm, double t, int a, int b, int c, int n);
        CUDA_DEVICE
        void get_tdi_Xf(cmplx *tdi_channels_arr, double *params, double *t_data, int N, int bin_i, int *link_space_craft_rec, int *link_space_craft_em, Vec k, Vec u, Vec v);
        CUDA_DEVICE
        void get_tdi_Xf_single(cmplx *tdi_channel, double t, double *params, Vec k, Vec u, Vec v, int *link_space_craft_rec, int *link_space_craft_em, int bin_i);
        CUDA_DEVICE
        void extract_amplitude_and_phase(double *flip, double *pjump, int Ns, double *As, double *Dphi, double *M, double *Mf, double *phiR);
        CUDA_DEVICE
        void new_extract_amplitude_and_phase(int *count, bool *fix_count, double *flip, double *pjump, int Ns, double *As, double *Dphi, cmplx *M, double *phiR);
        int get_tdi_buffer_size(int N);
        CUDA_DEVICE
        void unwrap_phase(int N, double *phase);
        CUDA_DEVICE
        void new_unwrap_phase(double *ph_correct_buffer, int N, double *phase);
        CUDA_DEVICE
        void new_extract_phase(cmplx *M, double *phiR, int N, double *t_arr);
        CUDA_DEVICE
        double get_phase_ref(double t, double *params, int bin_i);
        CUDA_DEVICE
        void get_hp_hc(double *hp, double *hc, double t, double *params, double phase_change, int bin_i);
        CUDA_DEVICE
        void get_sky_vectors(Vec *k, Vec *u, Vec *v, double *params);
        CUDA_DEVICE
        void xi_projections(double *xi_p, double *xi_c, Vec u, Vec v, Vec n);
        CUDA_DEVICE
        virtual double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        virtual double get_fdot(double t, double *params, int bin_i);
};

class GBTDIonTheFly : public LISATDIonTheFly{
    public:
        double T;
        double t_ref;
        int amplitude_index;
        int f0_index;
        int fdot0_index;
        int fddot0_index;
        int phi0_index;

        CUDA_CALLABLE_MEMBER
        GBTDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, double T_, double t_ref_) : LISATDIonTheFly(orbits_, tdi_config_, 5, 6, 7, 8)
        {
            T = T_;
            t_ref = t_ref_;
            amplitude_index = 0;
            f0_index = 1;
            fdot0_index = 2;
            fddot0_index = 3;
            phi0_index = 4;
        };
        CUDA_CALLABLE_MEMBER
        ~GBTDIonTheFly();
        // CUDA_DEVICE
        // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int bin_i) override;
        CUDA_DEVICE
        double ucb_amplitude(double t, double *params);
        CUDA_DEVICE
        double ucb_phase(double t, double *params);
        CUDA_DEVICE
        double ucb_fdot(double t, double *params);
        CUDA_DEVICE
        double ucb_f(double t, double *params);
        int get_gb_buffer_size(int N);
        // Total bytes of dynamic shared memory needed by the heterodyne FD
        // kernel: params + tdi_channels_arr (complex, also FFT scratch) +
        // tdi_amp + tdi_phase + phi_ref + get_tdi scratch.  N here is the
        // sparse FFT length (must be a power of two).
        int get_gb_fd_buffer_size(int N, int nchannels);
        // double get_phase_ref(double t, double *params, int bin_i);
        // CUDA_DEVICE
        // void run_wave_tdi(
        //     cmplx *tdi_channels_arr, 
        //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
        //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
        // );
        CUDA_DEVICE
        double get_amp(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_phase(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_f(double t, double *params, int bin_i);
        CUDA_DEVICE
        double get_fdot(double t, double *params, int bin_i);
};

void gb_run_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);

// Heterodyned frequency-domain GB TDI on the fly.
//
//   For each GB binary, builds the slow positive-frequency complex signal
//     s_c(tau) = A_c(tau) * exp(+i (phi_c(tau) + phi_ref(tau) - 2*pi*f0_grid*tau))
//   on a sparse, power-of-two-length time grid tau_n = n * dt_sparse, with
//   dt_sparse = Tobs / N_sparse and f0_grid = round(f0/df) * df,
//   df = 1/Tobs.  All three (or `nchannels`) channels live in shared memory
//   simultaneously so any XYZ-style cross-channel post-processing can happen
//   before the data hits global memory.  An in-place shared-memory radix-2
//   FFT then yields
//     X_het_c[m] = 0.5 * dt_sparse * FFT[s_c][m],
//   which equals the dense rfft of the real GB time series at dense bin
//   (k_f0 + m), bins outside +/- N_sparse/2 of f0_grid being zero.
//
// X_het output layout: (num_bin, nchannels, N_sparse) complex doubles,
//                       FFT-order (DC at index 0).
// k_f0_out: (num_bin,) integer dense rfft bin closest to f0.
// f0_grid_out: (num_bin,) double snapped carrier frequency [Hz].
//
// Requires N_sparse to be a power of two and t_ref == t_start; the caller
// should pass tau = t_local = absolute_t - t_start in t_arr_sparse (but the
// kernel just reads f0, Tobs, N_sparse and t_start to rebuild tau).
void gb_run_fd_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels);


class TDSplineTDIWaveform : public LISATDIonTheFly{
  public:
    // Orbits *orbits;
    // TDIConfig *tdi_config;
    
    CubicSpline *amp_spline;
    CubicSpline *phase_spline;
    int binary_index_storage;

    CUDA_CALLABLE_MEMBER
    TDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *phase_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3){
        amp_spline = amp_spline_;
        phase_spline = phase_spline_;
    };
    CUDA_CALLABLE_MEMBER
    ~TDSplineTDIWaveform(){};
    // CUDA_DEVICE
    // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    // void run_wave_tdi(
    //     cmplx *tdi_channels_arr, 
    //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
    //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
    // );
    int get_td_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    void check_x();
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);

};

void td_spline_run_wave_tdi_wrap(TDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);


class FDSplineTDIWaveform : public LISATDIonTheFly {
    public:
        CubicSpline *amp_spline;
        CubicSpline *freq_spline;
        // double *phase_ref_store;

    CUDA_CALLABLE_MEMBER
    FDSplineTDIWaveform(Orbits* orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_): LISATDIonTheFly(orbits_, tdi_config_, 0, 1, 2, 3)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
    };

    // CUDA_DEVICE
    // FDSplineTDIWaveform(Orbits *orbits_, TDIConfig *tdi_config_, CubicSpline *amp_spline_, CubicSpline *freq_spline_, double *phase_ref_);
    CUDA_CALLABLE_MEMBER
    ~FDSplineTDIWaveform(){};
    // CUDA_DEVICE
    // void get_amp_and_phase(double t_ssb, double *t, double *amp, double *phase, double *params, int N, int spline_i);
    // CUDA_DEVICE
    // void run_wave_tdi(
    //     cmplx *tdi_channels_arr, 
    //     double *Xamp, double *Xphase, double *Yamp, double *Yphase, double *Zamp, double *Zphase, double *phi_ref, 
    //     double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels
    // );
    int get_fd_spline_buffer_size(int N){return get_tdi_buffer_size(N);};
    CUDA_DEVICE
    double get_phase_ref(double t, double *params, int bin_i);
    CUDA_DEVICE
    double get_amp(double t, double *params, int spline_i);
    CUDA_DEVICE
    double get_phase(double t, double *params, int spline_i);
    CUDA_DEVICE
    void get_tdi(void *buffer, int buffer_length, cmplx *tdi_channels_arr, double *tdi_amp, double *tdi_phase, double* phi_ref, double *params, double *t_arr, int N, int bin_i, int nchannels);
    CUDA_DEVICE
    double get_amp_f(double t, double *params, int spline_i);
};

class WDMSettings{
  public:
    int Nt;
    int Nf;
    int num_channel;
    double layer_df;
    double layer_dt;
    int ind_min_t;
    int ind_max_t;
    int ind_min_f;
    int ind_max_f;
    int Nf_active;
    int Nt_active;

    // TODO: add to this?
    CUDA_CALLABLE_MEMBER
    WDMSettings(double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_){
        Nf = Nf_;
        Nt = Nt_;
        num_channel = num_channel_;
        layer_df = layer_df_;
        layer_dt = layer_dt_;
        ind_min_t = ind_min_t_;
        ind_max_t = ind_max_t_;
        ind_min_f = ind_min_f_;
        ind_max_f = ind_max_f_;
        Nf_active = ind_max_f - ind_min_f + 1; // inclusive
        Nt_active = ind_max_t - ind_min_t + 1; // inclusive
    };
};

class WDMDomain : public WDMSettings{
  public:
    
    double *wdm_data;
    double *wdm_noise;
    int num_data;
    int num_noise;

    CUDA_CALLABLE_MEMBER
    WDMDomain(double *wdm_data_, double *wdm_noise_, double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_, int num_data_, int num_noise_):
    WDMSettings(layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_)
    {
        wdm_data = wdm_data_;
        wdm_noise = wdm_noise_;
        num_data = num_data_;
        num_noise = num_noise_;
    };
    CUDA_DEVICE
    int get_pixel_index(int m, int n, int channel, int data_index);
    CUDA_DEVICE
    int get_pixel_index_noise(int m, int n, int channel, int noise_index);
    CUDA_DEVICE
    int get_pixel_index_noise_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index);
    CUDA_DEVICE
    double get_pixel_data_value(int m, int n, int channel,  int data_index);
    CUDA_DEVICE
    double get_pixel_noise_value(int m, int n, int channel, int noise_index);
    CUDA_DEVICE
    double get_pixel_noise_value_cross_channel(int m, int n, int channel_i, int channel_j, int noise_index);
    CUDA_DEVICE
    void get_inner_product_value(double *d_h, double *h_h, double wdm_template_nm, int m, int n, int channel, int data_index, int noise_index);
    CUDA_DEVICE
    void get_inner_product_value_cross_channel(double *d_h, double *h_h, double wdm_template_nm_i, double wdm_template_nm_j, int m, int n, int channel_i, int channel_j, int data_index, int noise_index);
    CUDA_DEVICE
    void add_ip_contrib(double *d_h_tmp, double *h_h_tmp, double *wdm_nm, int layer_m, int n, int data_index, int noise_index, int tdi_type);
    CUDA_DEVICE
    void add_ip_swap_contrib(double *d_h_add_acc, double *d_h_remove_acc, double *add_add_acc, double *remove_remove_acc, double *add_remove_acc, double *wdm_nm_add, double *wdm_nm_remove, int layer_m, int n, int data_index, int noise_index, int tdi_type);
    // Per-pixel chain-rule contribution:
    //   grad_acc_k += sum_{c,c'} (w_d - w_h)_c * (dw_h/dtheta_k)_{c'} * N^{-1}_{cc'} * 0.25
    // (XYZ cross-channel; the AET / AE branches use the diagonal noise).
    // The caller passes the *un-perturbed* central template w_mn[c] and the
    // central-FD parameter derivative dw_mn_dk[c] = (w_+ - w_-)/(2 eps_k);
    // this matches the analytic chain rule whenever the FD of w is
    // unbiased (polynomial-degree-2 dependence) and otherwise carries an
    // O(eps^2 d^3 w / dtheta_k^3) truncation that is small for the tuned
    // _DEFAULT_PARAM_EPS in GBWDMComputations.
    CUDA_DEVICE
    void add_grad_contrib(double *grad_acc_k, const double *w_mn, const double *dw_mn_dk,
                          int layer_m, int n, int data_index, int noise_index, int tdi_type);
    // Swap variant: accumulates +/- r_after * dw * N^{-1}, where
    //   r_after = w_d - w_add_center + w_rem_center.
    // `sign` selects between the add side (+1, dw = dw_add) and the remove
    // side (-1, dw = dw_rem); the helper is called once per parameter and
    // once per side.
    CUDA_DEVICE
    void add_swap_grad_contrib_one_side(double *grad_acc_k, double sign,
                                        const double *w_mn_add, const double *w_mn_rem,
                                        const double *dw_mn_dk,
                                        int layer_m, int n, int data_index, int noise_index, int tdi_type);
};


// ---------------------------------------------------------------------------
// FDDomain  -- mirror of WDMDomain for the heterodyne FD path.
//
// Holds the frequency-domain data array (complex, length n_rfft per channel,
// num_data instances) and an inverse-covariance array invC (3x3 cross-channel
// for tdi_type == TDI_XYZ; diagonal for TDI_AET / TDI_AE).  The convention is
// the lisatools rfft grid (df = 1/Tobs), with the active band specified by
// [ind_min, ind_max] inclusive.  Inner products are the standard lisatools
// formula  (a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] * df.
class FDDomain {
  public:
    cmplx  *fd_data;   // (num_data, num_channel, n_rfft) complex
    double *fd_invC;   // tdi_type=TDI_XYZ: (num_noise, num_channel, num_channel, n_rfft)
                       // tdi_type=TDI_AET/AE: (num_noise, num_channel, n_rfft)
    int    n_rfft;
    int    num_channel;
    int    num_data;
    int    num_noise;
    int    ind_min;    // inclusive
    int    ind_max;    // inclusive
    double df;
    double Tobs;       // = 1/df, kept for convenience

    CUDA_CALLABLE_MEMBER
    FDDomain(cmplx *fd_data_, double *fd_invC_, int n_rfft_,
             int num_channel_, int num_data_, int num_noise_,
             int ind_min_, int ind_max_, double df_)
    {
        fd_data     = fd_data_;
        fd_invC     = fd_invC_;
        n_rfft      = n_rfft_;
        num_channel = num_channel_;
        num_data    = num_data_;
        num_noise   = num_noise_;
        ind_min     = ind_min_;
        ind_max     = ind_max_;
        df          = df_;
        Tobs        = 1.0 / df_;
    };
    CUDA_DEVICE inline cmplx get_data(int k, int channel, int data_index) const
    {
        return fd_data[(size_t) data_index * num_channel * n_rfft
                       + (size_t) channel * n_rfft + k];
    }
    CUDA_DEVICE inline double get_invC_diag(int k, int channel, int noise_index) const
    {
        return fd_invC[(size_t) noise_index * num_channel * n_rfft
                       + (size_t) channel * n_rfft + k];
    }
    CUDA_DEVICE inline double get_invC_cross(int k, int c1, int c2, int noise_index) const
    {
        return fd_invC[(((size_t) noise_index * num_channel + c1)
                        * num_channel + c2) * n_rfft + k];
    }
    CUDA_DEVICE inline bool in_band(int k) const
    {
        return (k >= ind_min) && (k <= ind_max);
    }
};


class WaveletLookupTable : public WDMSettings{
  public:
    double *c_nm_all;
    double *s_nm_all;
    
    int num_f;
    int num_fdot;
    double df_interp;
    double dfdot_interp;
    double min_f_scaled;
    double min_fdot;

    CUDA_CALLABLE_MEMBER
    WaveletLookupTable(double *c_nm_all_, double *s_nm_all_, int num_f_, int num_fdot_, double df_interp_, double dfdot_interp_, double min_f_scaled_, double min_fdot_, 
        double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_): WDMSettings(layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_) {
        // n * Nf + m 
        c_nm_all = c_nm_all_;
        s_nm_all = s_nm_all_;
        num_f = num_f_;
        num_fdot = num_fdot_;
        df_interp = df_interp_;
        dfdot_interp = dfdot_interp_;
        min_f_scaled = min_f_scaled_;
        min_fdot = min_fdot_;
    };
    CUDA_DEVICE
    double get_wdm_in_channel_over_layers(cmplx tdi_channel_val, double f, double fdot, int m, int n);
    CUDA_DEVICE
    double linear_interp(double f_scaled, double fdot, double *z_vals, int layer_n);
    CUDA_DEVICE
    double get_w_mn_lookup(cmplx tdi_channel_val, double f, double fdot, int layer_m, int layer_n);
};

void fd_spline_run_wave_tdi_wrap(FDSplineTDIWaveform *tdi_on_fly, cmplx *tdi_channels_arr, 
    double *tdi_amp, double *tdi_phase, double *phi_ref, 
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels);

class GBComputationGroup{
  public:
    void gb_wdm_fill_global_wrap(double *template_fill, Orbits* orbits, TDIConfig *tdi_config, WaveletLookupTable* wdm_lookup, WDMDomain* wdm, double *params_all, int *data_index_all, double *factors_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
    void gb_wdm_get_ll_wrap(double *d_h_out, double *h_h_out, Orbits* orbits, TDIConfig *tdi_config, WaveletLookupTable* wdm_lookup, WDMDomain* wdm, double *params_all, int *data_index_all, int *noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
    void gb_wdm_swap_ll_wrap(double *d_h_add_out, double *d_h_remove_out, double *add_add_out, double *remove_remove_out, double *add_remove_out, Orbits* orbits, TDIConfig *tdi_config, WaveletLookupTable* wdm_lookup, WDMDomain* wdm, double *params_add_all, double *params_remove_all, int *data_index_all, int *noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);

    // Chain-rule parameter gradients of the two likelihood kernels.
    //
    //   grad_out             (num_bin, nparams)  -- dL/dtheta for get_ll
    //   grad_{add,remove}_out (num_bin, nparams) -- d(ll_diff)/d(theta_{add,remove}) for swap_ll
    //
    // The per-parameter central-difference step size is supplied via
    // ``param_eps`` (length nparams).  Passing eps <= 0 freezes that
    // parameter (gradient stays zero).
    void gb_wdm_get_ll_grad_wrap(double *grad_out, Orbits* orbits, TDIConfig *tdi_config, WaveletLookupTable* wdm_lookup, WDMDomain* wdm, double *params_all, int *data_index_all, int *noise_index_all, double *param_eps, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
    void gb_wdm_swap_ll_grad_wrap(double *grad_add_out, double *grad_remove_out, Orbits* orbits, TDIConfig *tdi_config, WaveletLookupTable* wdm_lookup, WDMDomain* wdm, double *params_add_all, double *params_remove_all, int *data_index_all, int *noise_index_all, double *param_eps_add, double *param_eps_remove, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);

    // Diagnostic: evaluate the per-pixel inputs (|M|, arg(M_mod), f, fdot, phase_ref)
    // that the fill_global / get_ll kernels feed into the WDM lookup, without doing
    // the lookup itself. Mirrors fast_wdm_inner: calls get_tdi_Xf_single + numerical
    // differentiation of phase_ref + tdi_phase across +/- deriv_delta_t.
    //
    // Layouts (all C-contiguous, 1-bin compatible if num_bin == 1):
    //   amp_out, phi_out, f_out, fdot_out: (num_bin, num_t, nchannels)
    //   phase_ref_out:                     (num_bin, num_t)
    void gb_wdm_eval_inputs_wrap(
        Orbits *orbits, TDIConfig *tdi_config,
        double *params_all, double *tn_arr,
        int num_bin, int nparams, int num_t, int nchannels,
        double T, double t_ref, double deriv_delta_t,
        double *amp_out, double *phi_out, double *f_out, double *fdot_out,
        double *phase_ref_out);

    // ------------------------------------------------------------------
    // FD analogs of the WDM methods above.
    //
    // gb_fd_fill_global_wrap: add (or subtract -- via factors_all[bin_i] in {+1,-1})
    //    each GB's heterodyne FD piece into a global rfft-grid template buffer
    //    of shape (num_data, num_channel, n_rfft) addressed by data_index_all.
    //
    // gb_fd_get_ll_wrap: compute (d|h) and (h|h) per binary using the
    //    standard lisatools FD inner product
    //         (a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] df
    //    with cross-channel invC for tdi_type=TDI_XYZ and diagonal invC for
    //    tdi_type=TDI_AET / TDI_AE.
    //
    // gb_fd_swap_ll_wrap: same accumulators as the WDM swap, restricted to the
    //    union of the add- and remove-source sparse supports.
    //
    // All three share the same per-source heterodyne FD pass that
    // GBFDTDIonTheFly already uses.  N_sparse must be a power of two.
    void gb_fd_fill_global_wrap(cmplx *template_fill,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, double *factors_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels);

    void gb_fd_get_ll_wrap(double *d_h_out, double *h_h_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, int *noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll_wrap(
        double *d_h_add_out, double *d_h_remove_out,
        double *add_add_out, double *remove_remove_out, double *add_remove_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    // Chain-rule parameter gradients of the two FD likelihood kernels.
    // Same convention as the WDM counterparts:
    //   grad_out             (num_bin, nparams)  -- dL/dtheta for get_ll
    //   grad_{add,remove}_out (num_bin, nparams) -- d(ll_diff)/d(theta_{add,remove})
    // Per-parameter central-FD step is supplied via ``param_eps_*`` arrays
    // (length nparams).  Passing eps_k <= 0 freezes parameter k (grad stays 0).
    // Both routines are CPU-fully-wired; the GPU branch prints a TODO and
    // returns (matching gb_fd_swap_ll_wrap's status).
    void gb_fd_get_ll_grad_wrap(double *grad_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, int *noise_index_all,
        double *param_eps,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll_grad_wrap(
        double *grad_add_out, double *grad_remove_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        double *param_eps_add, double *param_eps_remove,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);
};

#endif // __TDI_ON_THE_FLY_HH__