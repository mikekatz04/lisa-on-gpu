#ifndef __BINDING_TOF_HPP__
#define __BINDING_TOF_HPP__

#include "TDIonTheFly.hh"
#include "LISAResponse.hh"
#include "Detector.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding.hpp"
#include "gbt_binding.hpp"
#include "gbt_global.h"
#include "binding_flr.hpp"

namespace py = pybind11;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#define GBTDIonTheFlyWrap GBTDIonTheFlyWrapGPU
#define SOBBHTDIonTheFlyWrap SOBBHTDIonTheFlyWrapGPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapGPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapGPU
#define WaveletLookupTableWrap WaveletLookupTableWrapGPU
#define WDMDomainWrap WDMDomainWrapGPU
#define FDDomainWrap FDDomainWrapGPU
#define GBComputationGroupWrap GBComputationGroupWrapGPU
#define SOBBHComputationGroupWrap SOBBHComputationGroupWrapGPU
#else
#define GBTDIonTheFlyWrap GBTDIonTheFlyWrapCPU
#define SOBBHTDIonTheFlyWrap SOBBHTDIonTheFlyWrapCPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapCPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapCPU
#define WaveletLookupTableWrap WaveletLookupTableWrapCPU
#define WDMDomainWrap WDMDomainWrapCPU
#define FDDomainWrap FDDomainWrapCPU
#define GBComputationGroupWrap GBComputationGroupWrapCPU
#define SOBBHComputationGroupWrap SOBBHComputationGroupWrapCPU
#endif


class LISATDIonTheFlyWrap : public ReturnPointerBase {
  public:
    OrbitsWrap_responselisa *orbits;
    TDIConfigWrap *tdi_config;
    LISATDIonTheFlyWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_){
        orbits = orbits_;
        tdi_config = tdi_config_;
    };
};

class FDSplineTDIWaveformWrap : public LISATDIonTheFlyWrap {
  public:
    CubicSplineWrap_responselisa *amp_spline;
    CubicSplineWrap_responselisa *freq_spline;
    FDSplineTDIWaveform *waveform;
    FDSplineTDIWaveformWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_, CubicSplineWrap_responselisa *amp_spline_, CubicSplineWrap_responselisa *freq_spline_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
        waveform = new FDSplineTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, amp_spline_->spline, freq_spline_->spline);
    };
    ~FDSplineTDIWaveformWrap(){
        delete waveform;
    };

    void run_wave_tdi_wrap(
        array_type<std::complex<double>>tdi_channels_arr, 
        array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref, 
        array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
    );
    
    int get_buffer_size(int N){return waveform->get_fd_spline_buffer_size(N);};

};



class TDSplineTDIWaveformWrap : public LISATDIonTheFlyWrap {
  public:
    CubicSplineWrap_responselisa *amp_spline;
    CubicSplineWrap_responselisa *phase_spline;
    TDSplineTDIWaveform *waveform;
    TDSplineTDIWaveformWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_, CubicSplineWrap_responselisa *amp_spline_, CubicSplineWrap_responselisa *phase_spline_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        amp_spline = amp_spline_;
        phase_spline = phase_spline_;
        waveform = new TDSplineTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, amp_spline_->spline, phase_spline_->spline);
    };
    ~TDSplineTDIWaveformWrap(){
        delete waveform;
    };

    void run_wave_tdi_wrap(
        array_type<std::complex<double>>tdi_channels_arr, 
        array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref, 
        array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
    );
    
    int get_buffer_size(int N){return waveform->get_td_spline_buffer_size(N);};

};


class GBTDIonTheFlyWrap : public LISATDIonTheFlyWrap {
  public:
    GBTDIonTheFly *waveform;
    double T;
    double t_ref;

    GBTDIonTheFlyWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_, double T_, double t_ref_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        T = T_;
        t_ref = t_ref_;
        waveform = new GBTDIonTheFly(orbits_->orbits, tdi_config_->tdi_config, T_, t_ref_);
    };
    ~GBTDIonTheFlyWrap(){
        delete waveform;
    };

    void run_wave_tdi_wrap(
        array_type<std::complex<double>>tdi_channels_arr, 
        array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref, 
        array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
    );

    int get_buffer_size(int N){return waveform->get_gb_buffer_size(N);};

    // Heterodyne FD GB on a sparse time grid.  Builds the slow positive-freq
    // signal in shared memory, FFTs it, and returns (num_bin, nchannels,
    // N_sparse) complex doubles plus the per-source dense-bin index k_f0 and
    // snapped carrier frequency f0_grid.
    void run_fd_wave_tdi_wrap(
        array_type<std::complex<double>> X_het,
        array_type<int>    k_f0_out,
        array_type<double> f0_grid_out,
        array_type<double> params,
        double t_start, double Tobs,
        int N_sparse, int num_bin, int n_params, int nchannels);

    int get_fd_buffer_size(int N_sparse, int nchannels){
        return waveform->get_gb_fd_buffer_size(N_sparse, nchannels);
    }

};


// Pybind11 wrapper for SOBBHTDIonTheFly. Mirrors GBTDIonTheFlyWrap but
// exposes only the time-domain run_wave_tdi path; SOBBH currently has no
// WDM-lookup / heterodyne-FD counterparts.
class SOBBHTDIonTheFlyWrap : public LISATDIonTheFlyWrap {
  public:
    SOBBHTDIonTheFly *waveform;
    double T;
    double t_ref;

    SOBBHTDIonTheFlyWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_, double T_, double t_ref_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        T = T_;
        t_ref = t_ref_;
        waveform = new SOBBHTDIonTheFly(orbits_->orbits, tdi_config_->tdi_config, T_, t_ref_);
    };
    ~SOBBHTDIonTheFlyWrap(){
        delete waveform;
    };

    void run_wave_tdi_wrap(
        array_type<std::complex<double>>tdi_channels_arr,
        array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref,
        array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
    );

    int get_buffer_size(int N){return waveform->get_sobbh_buffer_size(N);};
};


class WaveletLookupTableWrap : public ReturnPointerBase {
  public:
    WaveletLookupTable *wdm_lookup;
    // array_type<double> c_nm_all;
    // array_type<double> s_nm_all;
    // int num_f;
    // int num_fdot;
    // double df;
    // double dfdot_interp;
    // double min_f;
    // double min_fdot;

    WaveletLookupTableWrap(array_type<double>c_nm_all_, array_type<double>s_nm_all_, int num_f_, int num_fdot_, double df_interp_, double dfdot_interp_, double min_f_, double min_fdot_, double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_, int m_ref_, int n_ref_, int kind_)
    {
        // PER_N tables are (Nt, num_fdot, num_f); N_REF_ONLY tables are
        // (num_fdot, num_f). Size-check the input arrays accordingly.
        size_t expected_len = (kind_ == LOOKUP_N_REF_ONLY)
            ? (size_t)num_fdot_ * (size_t)num_f_
            : (size_t)Nt_ * (size_t)num_fdot_ * (size_t)num_f_;
        wdm_lookup = new WaveletLookupTable(
            return_pointer_and_check_length(c_nm_all_, "c_nm_all", expected_len, 1),
            return_pointer_and_check_length(s_nm_all_, "s_nm_all", expected_len, 1),
            num_f_, num_fdot_, df_interp_, dfdot_interp_, min_f_, min_fdot_, layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_, m_ref_, n_ref_, kind_
        );
    };
    ~WaveletLookupTableWrap(){
        delete wdm_lookup;
    };

    // Diagnostic: directly evaluate w_mn for caller-supplied (amp, phi, f, fdot, m, n).
    // Mirrors the C kernel post-conj/shift: tdi_channel_val = amp * (cos(phi) + i*sin(phi)).
    // CPU-only — used for unit tests against the Python lookup.
    void get_w_mn_arr(
        array_type<double> out,
        array_type<double> amp_arr,
        array_type<double> phi_arr,
        array_type<double> f_arr,
        array_type<double> fdot_arr,
        array_type<int> m_arr,
        array_type<int> n_arr,
        int N);
};


class WDMDomainWrap : public ReturnPointerBase {
  public:
    WDMDomain *wdm;
    // array_type<double> c_nm_all;
    // array_type<double> s_nm_all;
    // int num_f;
    // int num_fdot;
    // double df;
    // double dfdot_interp;
    // double min_f;
    // double min_fdot;

    WDMDomainWrap(array_type<double>wdm_data_, array_type<double>wdm_noise_, double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_, int num_data_, int num_noise_)
    {
        // TODO: adjust noise length check to TDI setups
        int Nt_active = ind_max_t_ - ind_min_t_ + 1;
        int Nf_active = ind_max_f_ - ind_min_f_ + 1;
        wdm = new WDMDomain(
            return_pointer_and_check_length(wdm_data_, "wdm_data", Nt_active * Nf_active * num_channel_ * num_data_, 1),
            return_pointer(wdm_noise_, "wdm_noise"),  // return_pointer_and_check_length(wdm_noise_, "wdm_noise", Nt_ * Nf_ * num_channel_ * num_noise_, 1),
            layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_, num_data_, num_noise_
        );
    };
    ~WDMDomainWrap(){
        delete wdm;
    };

};


// FDDomainWrap: thin pybind11 holder for FDDomain, mirroring WDMDomainWrap.
// invC array layout depends on tdi_type:
//   tdi_type == TDI_XYZ   : (num_noise, num_channel, num_channel, n_rfft)
//   tdi_type == TDI_AET/AE: (num_noise, num_channel, n_rfft)
class FDDomainWrap : public ReturnPointerBase {
  public:
    FDDomain *fd;
    FDDomainWrap(
        array_type<std::complex<double>> fd_data_,
        array_type<double>               fd_invC_,
        int n_rfft_, int num_channel_, int num_data_, int num_noise_,
        int ind_min_, int ind_max_, double df_)
    {
        fd = new FDDomain(
            (cmplx*) return_pointer_and_check_length(
                fd_data_, "fd_data",
                n_rfft_ * num_channel_ * num_data_, 1),
            return_pointer(fd_invC_, "fd_invC"),
            n_rfft_, num_channel_, num_data_, num_noise_,
            ind_min_, ind_max_, df_);
    };
    ~FDDomainWrap(){ delete fd; };
};


class GBComputationGroupWrap: public GBComputationGroup, public ReturnPointerBase {
  public:
    void gb_wdm_fill_global(array_type<double>template_fill, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<double>factors_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
    void gb_wdm_get_ll(array_type<double>d_h_out, array_type<double>h_h_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
    void gb_wdm_swap_ll(array_type<double>d_h_add_out, array_type<double>d_h_remove_out, array_type<double>add_add_out, array_type<double>remove_remove_out, array_type<double>add_remove_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_add_all, array_type<double>params_remove_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);

    // Chain-rule gradients of the two likelihood kernels. ``param_eps`` is the
    // per-parameter central-difference step size (length nparams).
    void gb_wdm_get_ll_grad(array_type<double>grad_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, array_type<double>param_eps, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
    void gb_wdm_swap_ll_grad(array_type<double>grad_add_out, array_type<double>grad_remove_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_add_all, array_type<double>params_remove_all, array_type<int>data_index_all, array_type<int>noise_index_all, array_type<double>param_eps_add, array_type<double>param_eps_remove, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);

    // Diagnostic — see TDIonTheFly.hh for layout.
    void gb_wdm_eval_inputs(
        OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_all, array_type<double> tn_arr,
        int num_bin, int nparams, int num_t, int nchannels,
        double T, double t_ref, double deriv_delta_t,
        array_type<double> amp_out, array_type<double> phi_out,
        array_type<double> f_out, array_type<double> fdot_out,
        array_type<double> phase_ref_out);

    // ---- Spline-path mirrors --------------------------------------------
    // `coarse_dt` is the coarse-grid spacing (seconds) used to fit cubic
    // splines to the smooth get_tdi outputs. Smaller -> more accurate at the
    // cost of more get_tdi work. A typical Python-side choice is
    //     coarse_dt = (1 year in seconds) / coarse_pts_per_year
    // with `coarse_pts_per_year` defaulting to 256.
    void gb_wdm_spline_fill_global(array_type<double>template_fill, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<double>factors_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt);
    void gb_wdm_spline_get_ll(array_type<double>d_h_out, array_type<double>h_h_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt);
    void gb_wdm_spline_swap_ll(array_type<double>d_h_add_out, array_type<double>d_h_remove_out, array_type<double>add_add_out, array_type<double>remove_remove_out, array_type<double>add_remove_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_add_all, array_type<double>params_remove_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt);
    void gb_wdm_spline_get_ll_grad(array_type<double>grad_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, array_type<double>param_eps, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt);

    // Spline diagnostic — same output layout as gb_wdm_eval_inputs; builds a
    // single WDM_SPLINE_L-point spline window starting at t_window_start with
    // spacing coarse_dt and evaluates at every tn in tn_arr.
    void gb_wdm_spline_eval_inputs(
        OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_all, array_type<double> tn_arr,
        int num_bin, int nparams, int num_t, int nchannels,
        double T, double t_ref,
        double t_window_start, double coarse_dt,
        array_type<double> amp_out, array_type<double> phi_out,
        array_type<double> f_out, array_type<double> fdot_out,
        array_type<double> phase_ref_out);

    // ---- FD analogs ---------------------------------------------------
    void gb_fd_fill_global(
        array_type<std::complex<double>> template_fill,
        OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_all, array_type<int> data_index_all,
        array_type<double> factors_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels);

    void gb_fd_get_ll(
        array_type<double> d_h_out, array_type<double> h_h_out,
        OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll(
        array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
        array_type<double> add_add_out, array_type<double> remove_remove_out,
        array_type<double> add_remove_out,
        OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    // Chain-rule parameter gradients of gb_fd_get_ll / gb_fd_swap_ll.
    // param_eps[k] is the per-parameter central-FD step (length nparams);
    // pass eps_k <= 0 to freeze parameter k.
    void gb_fd_get_ll_grad(
        array_type<double> grad_out,
        OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> param_eps,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll_grad(
        array_type<double> grad_add_out, array_type<double> grad_remove_out,
        OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> param_eps_add, array_type<double> param_eps_remove,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    // ---- Chunked-heterodyne path (no lookup table) ----------------------
    // Geometry arrays (chunk_t_starts, chunk_keep_lo, chunk_keep_hi,
    // chunk_n_global_offset, wdm_window) are precomputed on the Python side
    // by ``gb_wdm_het.compute_chunk_geometry`` / ``compute_wdm_window``.
    // ``grid_dim`` is the CUDA launch grid size (chosen via
    // ``chunked_het_grid_dim``); pass anything > 0 on CPU (ignored).
    void gb_wdm_het_fill_global(
        array_type<double> template_fill,
        OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_all, array_type<double> factors_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        int n_chunks, int num_bin, int nparams,
        int Nf, int Nt, int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit);

    void gb_wdm_het_get_ll(
        array_type<double> d_h_out, array_type<double> h_h_out,
        OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nf, int Nt, int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups);

    void gb_wdm_het_swap_ll(
        array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
        array_type<double> add_add_out, array_type<double> remove_remove_out,
        array_type<double> add_remove_out,
        OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nf, int Nt, int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
        array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b);
};


// Parallel SOBBH API. Same chunked-het methods, sobbh_ prefix; routes
// to the templated kernel with SourceT = SOBBHTDIonTheFly via
// SOBBHComputationGroup. Pybind exposure mirrors GBComputationGroupWrap.
class SOBBHComputationGroupWrap: public SOBBHComputationGroup, public ReturnPointerBase {
  public:
    void sobbh_wdm_het_fill_global(
        array_type<double> template_fill,
        OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_all, array_type<double> factors_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        int n_chunks, int num_bin, int nparams,
        int Nf, int Nt, int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit);

    void sobbh_wdm_het_get_ll(
        array_type<double> d_h_out, array_type<double> h_h_out,
        OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nf, int Nt, int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups);

    void sobbh_wdm_het_swap_ll(
        array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
        array_type<double> add_add_out, array_type<double> remove_remove_out,
        array_type<double> add_remove_out,
        OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nf, int Nt, int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
        array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b);
};

#endif // __BINDING_TOF_HPP__
