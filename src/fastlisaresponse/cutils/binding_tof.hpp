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
#else
#define GBTDIonTheFlyWrap GBTDIonTheFlyWrapCPU
#define FDSplineTDIWaveformWrap FDSplineTDIWaveformWrapCPU
#define TDSplineTDIWaveformWrap TDSplineTDIWaveformWrapCPU
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
    CubicSplineWrap_responselisa *freq_spline;
    TDSplineTDIWaveform *waveform;
    TDSplineTDIWaveformWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_, CubicSplineWrap_responselisa *amp_spline_, CubicSplineWrap_responselisa *freq_spline_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        amp_spline = amp_spline_;
        freq_spline = freq_spline_;
        waveform = new TDSplineTDIWaveform(orbits_->orbits, tdi_config_->tdi_config, amp_spline_->spline, freq_spline_->spline);
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

#endif // __BINDING_TOF_HPP__