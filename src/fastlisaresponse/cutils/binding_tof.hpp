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
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapGPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapGPU
#define WaveletLookupTableWrap WaveletLookupTableWrapGPU
#define WDMDomainWrap WDMDomainWrapGPU
#define GBComputationGroupWrap GBComputationGroupWrapGPU
#else
#define GBTDIonTheFlyWrap GBTDIonTheFlyWrapCPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapCPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapCPU
#define WaveletLookupTableWrap WaveletLookupTableWrapCPU
#define WDMDomainWrap WDMDomainWrapCPU
#define GBComputationGroupWrap GBComputationGroupWrapCPU
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

    WaveletLookupTableWrap(array_type<double>c_nm_all_, array_type<double>s_nm_all_, int num_f_, int num_fdot_, double df_interp_, double dfdot_interp_, double min_f_, double min_fdot_, double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_)
    {
        
        wdm_lookup = new WaveletLookupTable(
            return_pointer_and_check_length(c_nm_all_, "c_nm_all", num_f_ * num_fdot_, 1),
            return_pointer_and_check_length(s_nm_all_, "s_nm_all", num_f_ * num_fdot_, 1),
            num_f_, num_fdot_, df_interp_, dfdot_interp_, min_f_, min_fdot_, layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_
        );
    };
    ~WaveletLookupTableWrap(){
        delete wdm_lookup;
    };

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

class GBComputationGroupWrap: public GBComputationGroup, public ReturnPointerBase {
  public:
    void gb_wdm_fill_global(array_type<double>template_fill, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
    void gb_wdm_get_ll(array_type<double>d_h_out, array_type<double>h_h_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t);
};

#endif // __BINDING_TOF_HPP__