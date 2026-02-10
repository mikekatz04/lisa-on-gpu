#ifndef __BINDING_FLR_HPP__
#define __BINDING_FLR_HPP__

#include "LISAResponse.hh"
#include "Detector.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding.hpp"
#include "gbt_binding.hpp"
#include "Interpolate.hh"

namespace py = pybind11;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
template<typename T>
using array_type = cai::cuda_array_t<T>;
#define LISAResponseWrap LISAResponseWrapGPU
#else
template<typename T>
using array_type = py::array_t<T>;
#define LISAResponseWrap LISAResponseWrapCPU
#endif

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define OrbitsWrap_responselisa OrbitsWrapGPU_responselisa
#define CubicSplineWrap_responselisa CubicSplineWrapGPU_responselisa
#define TDIConfigWrap TDIConfigWrapGPU
#else
#define OrbitsWrap_responselisa OrbitsWrapCPU_responselisa
#define TDIConfigWrap TDIConfigWrapCPU
#endif


class ReturnPointerBase {
  public:
    Orbits *orbits;
    OrbitsWrap_responselisa(double dt_, int N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    // OrbitsWrap_responselisa(double sc_t0_, double sc_dt_, int sc_N_, double ltt_t0_, double ltt_dt_, int ltt_N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {

        // double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", sc_N_, 6 * 3);
        // double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", ltt_N_, 6);
        // double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", sc_N_, 3 * 3);

        double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", N_, 6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", N_, 6);
        double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", N_, 3 * 3);

        int *_sc_r = return_pointer_and_check_length(sc_r_, "sc_r", 6, 1);
        int *_sc_e = return_pointer_and_check_length(sc_e_, "sc_e", 6, 1);
        int *_links = return_pointer_and_check_length(links_, "links", 6, 1);

        orbits = new Orbits(dt_, N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
        // orbits = new Orbits(sc_t0_, sc_dt_, sc_N_, ltt_t0_, ltt_dt_, ltt_N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
    };
    ~OrbitsWrap_responselisa(){
        delete orbits;
    };
    template<typename T>
    static T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
        
#else
        py::buffer_info buf1 = input1.request();

        if (buf1.size != N * multiplier)
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(N * multiplier) + ". It's length is " + std::to_string(buf1.size) + ".";
            throw std::invalid_argument(err_out);
        }
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

    template<typename T>
    static T* return_pointer(array_type<T> input1, std::string name)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
#else
        py::buffer_info buf1 = input1.request();
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

    static cmplx* return_pointer_cmplx(array_type<std::complex<double>> input1, std::string name)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        cmplx *ptr1 = (cmplx *)(input1.get_compatible_typed_pointer());
#else
        py::buffer_info buf1 = input1.request();
        cmplx* ptr1 = static_cast<cmplx *>(buf1.ptr);
#endif
        return ptr1;
    };

};

class CubicSplineWrap_responselisa : public ReturnPointerBase {
  public:
    CubicSpline *spline;
    CubicSplineWrap_responselisa(array_type<double> x0_, array_type<double> y0_, array_type<double> c1_, array_type<double> c2_, array_type<double> c3_, int ninterps_, int length_, int spline_type_)
    {

        double *_x0 = return_pointer_and_check_length(x0_, "x0", length_, ninterps_);
        double *_y0 = return_pointer_and_check_length(y0_, "y0", length_, ninterps_);
        double *_c1 = return_pointer_and_check_length(c1_, "c1", length_, ninterps_);
        double *_c2 = return_pointer_and_check_length(c2_, "c2", length_, ninterps_);
        double *_c3 = return_pointer_and_check_length(c3_, "c3", length_, ninterps_);

        spline = new CubicSpline(_x0, _y0, _c1, _c2, _c3, ninterps_, length_, spline_type_);
    };
    ~CubicSplineWrap_responselisa(){
        delete spline;
    };
    void eval_wrap(array_type<double>y_new, array_type<double>x_new, array_type<int>spline_index, int N);

};



class OrbitsWrap_responselisa : public ReturnPointerBase{
  public:
    Orbits *orbits;
    OrbitsWrap_responselisa(double dt_, int N_, array_type<double> n_arr_, array_type<double> ltt_arr_, array_type<double> x_arr_, array_type<int> links_, array_type<int> sc_r_, array_type<int> sc_e_, double armlength_)
    {

        double *_n_arr = return_pointer_and_check_length(n_arr_, "n_arr", N_, 6 * 3);
        double *_ltt_arr = return_pointer_and_check_length(ltt_arr_, "ltt_arr", N_, 6);
        double *_x_arr = return_pointer_and_check_length(x_arr_, "x_arr", N_, 3 * 3);

        int *_sc_r = return_pointer_and_check_length(sc_r_, "sc_r", 6, 1);
        int *_sc_e = return_pointer_and_check_length(sc_e_, "sc_e", 6, 1);
        int *_links = return_pointer_and_check_length(links_, "links", 6, 1);

        orbits = new Orbits(dt_, N_, _n_arr, _ltt_arr, _x_arr, _links,  _sc_r, _sc_e, armlength_);
    };
    ~OrbitsWrap_responselisa(){
        delete orbits;
    };
};


class TDIConfigWrap : public ReturnPointerBase{
  public:
    TDIConfig *tdi_config;
    TDIConfigWrap(array_type<int>unit_starts_, array_type<int>unit_lengths_, array_type<int>tdi_base_link_, array_type<int>tdi_link_combinations_, array_type<double>tdi_signs_in_, array_type<int>channels_, int num_units_, int num_channels_)
    {
        // TODO: add check for length of all units
        int *_unit_starts = return_pointer_and_check_length(unit_starts_, "unit_starts", num_units_, 1);
        int *_unit_lengths = return_pointer_and_check_length(unit_lengths_, "unit_lengths", num_units_, 1);

        int *_tdi_base_link = return_pointer(tdi_base_link_, "tdi_base_link");
        int *_tdi_link_combinations = return_pointer(tdi_link_combinations_, "tdi_link_combinations");
        double *_tdi_signs_in = return_pointer(tdi_signs_in_, "tdi_signs_in");
        int *_channels = return_pointer(channels_, "channels");
        tdi_config = new TDIConfig(_unit_starts, _unit_lengths, _tdi_base_link, _tdi_link_combinations, _tdi_signs_in, _channels,  num_units_, num_channels_);
    };
    ~TDIConfigWrap(){
        delete tdi_config;
    };
};

class LISAResponseWrap : public ReturnPointerBase {
  public:
    LISAResponse *response;
    OrbitsWrap_responselisa *orbits;
    TDIConfigWrap *tdi_config;
    LISAResponseWrap(OrbitsWrap_responselisa *orbits_, TDIConfigWrap *tdi_config_)
    {
        orbits = orbits_;
        tdi_config = tdi_config_;
        response = new LISAResponse(orbits_->orbits, tdi_config_->tdi_config);
    };
    ~LISAResponseWrap(){
        delete response;
    };

    void get_tdi_delays_wrap(array_type<double> delayed_links_, array_type<double> input_links_, int num_inputs, int num_delays, array_type<double> t_arr_,
                    int order, double sampling_frequency, int buffer_integer, array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int tdi_start_ind);
                    
    void get_response_wrap(array_type<double> y_gw_, array_type<double> t_data_, array_type<double> k_in_, array_type<double> u_in_, array_type<double> v_in_, double dt,
                  int num_delays,
                  array_type<std::complex<double>> input_in_, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer,
                  array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int projections_start_ind);
    
};

#endif // __BINDING_FLR_HPP__