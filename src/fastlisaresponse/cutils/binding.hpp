#ifndef __BINDING_HPP__
#define __BINDING_HPP__

#include "LISAResponse.hh"
#include "Detector.hpp"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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

class LISAResponseWrap {
  public:
    LISAResponse *response;
    Orbits *orbits;
    LISAResponseWrap(Orbits * orbits)
    {
        response = new LISAResponse(orbits);
    };
    ~LISAResponseWrap(){
        delete response;
    };

    void get_tdi_delays_wrap(array_type<double> delayed_links_, array_type<double> input_links_, int num_inputs, int num_delays, array_type<double> t_arr_, array_type<int> unit_starts_, array_type<int> unit_lengths_, array_type<int> tdi_base_link_, array_type<int> tdi_link_combinations_, array_type<double> tdi_signs_in_, array_type<int> channels_, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int tdi_start_ind);
                    
    void get_response_wrap(array_type<double> y_gw_, array_type<double> t_data_, array_type<double> k_in_, array_type<double> u_in_, array_type<double> v_in_, double dt,
                  int num_delays,
                  array_type<std::complex<double>> input_in_, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer,
                  array_type<double> A_in_, double deps, int num_A, array_type<double> E_in_, int projections_start_ind);
    
    template<typename T>
    T* return_pointer_and_check_length(array_type<T> input1, std::string name, int N, int multiplier)
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
    T* return_pointer(array_type<T> input1, std::string name)
    {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
        T *ptr1 = static_cast<T *>(input1.get_compatible_typed_pointer());
#else
        py::buffer_info buf1 = input1.request();
        T* ptr1 = static_cast<T *>(buf1.ptr);
#endif
        return ptr1;
    };

    cmplx* return_pointer_cmplx(array_type<std::complex<double>> input1, std::string name)
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

#endif // __BINDING_HPP__