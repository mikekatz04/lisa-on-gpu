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
#else
#define GBTDIonTheFlyWrap GBTDIonTheFlyWrapCPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapCPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapCPU
#define WaveletLookupTableWrap WaveletLookupTableWrapCPU
#define WDMDomainWrap WDMDomainWrapCPU
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

    GBTDIonTheFlyWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_, double T_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        T = T_;
        waveform = new GBTDIonTheFly(orbits_->orbits, tdi_config_->tdi_config, T_);
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
    // double dfdot;
    // double min_f;
    // double min_fdot;

    WaveletLookupTableWrap(array_type<double>c_nm_all_, array_type<double>s_nm_all_, int num_f_, int num_fdot_, double df_, double dfdot_, double min_f_, double min_fdot_)
    {
        
        wdm_lookup = new WaveletLookupTable(
            return_pointer_and_check_length(c_nm_all_, "c_nm_all", num_f_ * num_fdot_, 1),
            return_pointer_and_check_length(s_nm_all_, "s_nm_all", num_f_ * num_fdot_, 1),
            num_f_, num_fdot_, df_, dfdot_, min_f_, min_fdot_
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
    // double dfdot;
    // double min_f;
    // double min_fdot;

    WDMDomainWrap(array_type<double>wdm_data_, array_type<double>wdm_noise_, double df_, double dt_, int num_m_, int num_n_, int num_channel_, int num_data_, int num_noise_)
    {
        wdm = new WDMDomain(
            return_pointer_and_check_length(wdm_data_, "wdm_data", num_n_ * num_m_ * num_channel_ * num_data_, 1),
            return_pointer_and_check_length(wdm_noise_, "wdm_noise", num_n_ * num_m_ * num_channel_ * num_noise_, 1),
            df_, dt_, num_m_, num_n_, num_channel_, num_data_, num_noise_
        );
    };
    ~WDMDomainWrap(){
        delete wdm;
    };

};

#endif // __BINDING_TOF_HPP__