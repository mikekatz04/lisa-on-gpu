#ifndef __LISA_RESPONSE__
#define __LISA_RESPONSE__

#include "cuda_complex.hpp"
#include "Detector.hpp"

#define C_inv 3.3356409519815204e-09
#define NUM_THREADS 256

typedef gcmplx::complex<double> cmplx;

class LISAResponse{
  public:
    Orbits *orbits;
    LISAResponse(Orbits *orbits_){
      orbits = orbits_;
      // TODO: add GPU orbits now?
    };
    template<typename T>
    T* return_pointer_and_check_length(py::array_t<T> input1, std::string name, int N, int multiplier)
    {
        py::buffer_info buf1 = input1.request();

        if (buf1.size != N * multiplier)
        {
            std::string err_out = name + ": input arrays have the incorrect length. Should be " + std::to_string(N * multiplier) + ". It's length is " + std::to_string(buf1.size) + ".";
            throw std::invalid_argument(err_out);
        }
        
        T *ptr1 = static_cast<T *>(buf1.ptr);
        return ptr1;
    };
    void get_tdi_delays(py::array_t<double> delayed_links_, py::array_t<double> input_links_, int num_inputs, int num_delays, py::array_t<double> t_arr_, py::array_t<int> unit_starts_, py::array_t<int> unit_lengths_, py::array_t<int> tdi_base_link_, py::array_t<int> tdi_link_combinations_, py::array_t<double> tdi_signs_in_, py::array_t<int> channels_, int num_units, int num_channels,
                    int order, double sampling_frequency, int buffer_integer, py::array_t<double> A_in_, double deps, int num_A, py::array_t<double> E_in_, int tdi_start_ind);
                    
    void get_response(py::array_t<double> y_gw_, py::array_t<double> t_data_, py::array_t<double> k_in_, py::array_t<double> u_in_, py::array_t<double> v_in_, double dt,
                  int num_delays,
                  py::array_t<std::complex<double>> input_in_, int num_inputs, int order,
                  double sampling_frequency, int buffer_integer,
                  py::array_t<double> A_in_, double deps, int num_A, py::array_t<double> E_in_, int projections_start_ind);
};
#endif // __LISA_RESPONSE__
