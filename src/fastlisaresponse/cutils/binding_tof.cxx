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
#include "binding_tof.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;

void GBTDIonTheFlyWrap::run_wave_tdi_wrap(
    array_type<std::complex<double>>tdi_channels_arr, 
    array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref, 
    array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
)
{
    gb_run_wave_tdi_wrap(
        waveform,
        (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels), // TODO: add length check
        return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
        return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
        return_pointer_and_check_length(params, "params", n_params, num_bin),
        return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
        N, num_bin, n_params, nchannels
    );
}


void TDSplineTDIWaveformWrap::run_wave_tdi_wrap(
    array_type<std::complex<double>>tdi_channels_arr, 
    array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref, 
    array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
)
{
    td_spline_run_wave_tdi_wrap(
        waveform,
        (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels), // TODO: add length check
        return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
        return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
        return_pointer_and_check_length(params, "params", n_params, num_bin),
        return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
        N, num_bin, n_params, nchannels
    );
}

void FDSplineTDIWaveformWrap::run_wave_tdi_wrap(
    array_type<std::complex<double>>tdi_channels_arr, 
    array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref, 
    array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
)
{
    fd_spline_run_wave_tdi_wrap(
        waveform,
        (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels), // TODO: add length check
        return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
        return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
        return_pointer_and_check_length(params, "params", n_params, num_bin),
        return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
        N, num_bin, n_params, nchannels
    );
}

void gb_wdm_get_ll(array_type<double>d_h_out, array_type<double>h_h_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, int tdi_type)
{
    gb_wdm_get_ll_wrap(
        ReturnPointerBase::return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1), 
        ReturnPointerBase::return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1), 
        orbits_wrap->orbits, 
        tdi_config_wrap->tdi_config, 
        wdm_lookup_wrap->wdm_lookup, 
        wdm_wrap->wdm, 
        ReturnPointerBase::return_pointer_and_check_length(params_all, "params_all", nparams, num_bin), 
        ReturnPointerBase::return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1), 
        ReturnPointerBase::return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1), 
        num_bin, nparams, T, tdi_type);
}

std::string get_module_path_tdionthefly() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    py::gil_scoped_acquire acquire;

    // Import the module by its name
    // Note: The module name here ("tdionthefly") must match the name used in PYBIND11_MODULE
    py::object module = py::module::import("tdionthefly");

    // Access the __file__ attribute and cast it to a C++ string
    try {
        std::string path = module.attr("__file__").cast<std::string>();
        return path;
    } catch (const py::error_already_set& e) {
        // Handle the error if __file__ attribute is missing (e.g., if module is a namespace package)
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}


// PYBIND11_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void tdionthefly_part(py::module &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<FDSplineTDIWaveformWrap>(m, "FDSplineTDIWaveformWrapGPU")
#else
    py::class_<FDSplineTDIWaveformWrap>(m, "FDSplineTDIWaveformWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<OrbitsWrap_responselisa *, TDIConfigWrap *, CubicSplineWrap_responselisa *, CubicSplineWrap_responselisa *>(), 
         py::arg("orbits"), py::arg("tdi_config"), py::arg("amp_spline"), py::arg("freq_spline"))
    // Bind member functions
    .def("run_wave_tdi_wrap", &FDSplineTDIWaveformWrap::run_wave_tdi_wrap, "Preform TDI combinations.")
    .def("get_buffer_size", &FDSplineTDIWaveformWrap::get_buffer_size, "Get needed buffer size.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &FDSplineTDIWaveformWrap::orbits)
    .def_readwrite("tdi_config", &FDSplineTDIWaveformWrap::tdi_config)
    .def_readwrite("amp_spline", &FDSplineTDIWaveformWrap::amp_spline)
    .def_readwrite("freq_spline", &FDSplineTDIWaveformWrap::freq_spline)
    
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<FDSplineTDIWaveform>(m, "FDSplineTDIWaveformGPU")
#else
    py::class_<FDSplineTDIWaveform>(m, "FDSplineTDIWaveformCPU")
#endif

    // Bind the constructor
    .def(py::init<Orbits *, TDIConfig*, CubicSpline*, CubicSpline*>(), 
         py::arg("orbits"), py::arg("tdi_config"), py::arg("amp_spline"), py::arg("freqs_spline"))
    ;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<TDSplineTDIWaveformWrap>(m, "TDSplineTDIWaveformWrapGPU")
#else
    py::class_<TDSplineTDIWaveformWrap>(m, "TDSplineTDIWaveformWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<OrbitsWrap_responselisa *, TDIConfigWrap *, CubicSplineWrap_responselisa *, CubicSplineWrap_responselisa *>(), 
         py::arg("orbits"), py::arg("tdi_config"), py::arg("amp_spline"), py::arg("phase_spline"))
    // Bind member functions
    .def("run_wave_tdi_wrap", &TDSplineTDIWaveformWrap::run_wave_tdi_wrap, "Preform TDI combinations.")
    .def("get_buffer_size", &TDSplineTDIWaveformWrap::get_buffer_size, "Get needed buffer size.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &TDSplineTDIWaveformWrap::orbits)
    .def_readwrite("tdi_config", &TDSplineTDIWaveformWrap::tdi_config)
    .def_readwrite("amp_spline", &TDSplineTDIWaveformWrap::amp_spline)
    .def_readwrite("phase_spline", &TDSplineTDIWaveformWrap::phase_spline)
    
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<TDSplineTDIWaveform>(m, "TDSplineTDIWaveformGPU")
#else
    py::class_<TDSplineTDIWaveform>(m, "TDSplineTDIWaveformCPU")
#endif

    // Bind the constructor
    .def(py::init<Orbits *, TDIConfig*, CubicSpline*, CubicSpline*>(), 
         py::arg("orbits"), py::arg("tdi_config"), py::arg("amp_spline"), py::arg("phase_spline"))
    ;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<GBTDIonTheFlyWrap>(m, "GBTDIonTheFlyWrapGPU")
#else
    py::class_<GBTDIonTheFlyWrap>(m, "GBTDIonTheFlyWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<OrbitsWrap_responselisa *, TDIConfigWrap *, double>(), 
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"))
    // Bind member functions
    .def("run_wave_tdi_wrap", &GBTDIonTheFlyWrap::run_wave_tdi_wrap, "Preform TDI combinations.")
    .def("get_buffer_size", &GBTDIonTheFlyWrap::get_buffer_size, "Get needed buffer size.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &GBTDIonTheFlyWrap::orbits)
    .def_readwrite("tdi_config", &GBTDIonTheFlyWrap::tdi_config)
    
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<GBTDIonTheFly>(m, "GBTDIonTheFlyGPU")
#else
    py::class_<GBTDIonTheFly>(m, "GBTDIonTheFlyCPU")
#endif

    // Bind the constructor
    .def(py::init<Orbits *, TDIConfig*, double>(), 
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WaveletLookupTableWrap>(m, "WaveletLookupTableWrapGPU")
#else
    py::class_<WaveletLookupTableWrap>(m, "WaveletLookupTableWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<array_type<double>,array_type<double>,int,int,double,double,double,double>(), 
         py::arg("c_nm_all"), py::arg("s_nm_all"), py::arg("num_f"), py::arg("num_fdot"), py::arg("df"), py::arg("dfdot"), py::arg("min_f"), py::arg("min_fdot"))
    // Bind member functions
    
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("wdm_lookup", &WaveletLookupTableWrap::wdm_lookup)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WaveletLookupTable>(m, "WaveletLookupTableGPU")
#else
    py::class_<WaveletLookupTable>(m, "WaveletLookupTableCPU")
#endif
    // Bind the constructor
    .def(py::init<double*,double*,int,int,double,double,double,double>(), 
         py::arg("c_nm_all"), py::arg("s_nm_all"), py::arg("num_f"), py::arg("num_fdot"), py::arg("df"), py::arg("dfdot"), py::arg("min_f"), py::arg("min_fdot"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WDMDomainWrap>(m, "WDMDomainWrapGPU")
#else
    py::class_<WDMDomainWrap>(m, "WDMDomainWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<array_type<double>,array_type<double>, double, double, int, int, int, int, int>(), 
         py::arg("wdm_data"), py::arg("wdm_noise"), py::arg("df"), py::arg("dt"), py::arg("num_m"), py::arg("num_n"), py::arg("num_channel"), py::arg("num_data"), py::arg("num_noise"))
    // Bind member functions
    
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("wdm", &WDMDomainWrap::wdm)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WDMDomain>(m, "WDMDomainGPU")
#else
    py::class_<WDMDomain>(m, "WDMDomainCPU")
#endif

    // Bind the constructor
    .def(py::init<double*,double*, double, double, int, int, int, int, int>(), 
         py::arg("wdm_data"), py::arg("wdm_noise"), py::arg("df"), py::arg("dt"), py::arg("num_m"), py::arg("num_n"), py::arg("num_channel"), py::arg("num_data"), py::arg("num_noise"))
    ;

    m.def("gb_wdm_get_ll", &gb_wdm_get_ll, "wdm get likelihood.");

}



PYBIND11_MODULE(tdionthefly, m) {
     m.doc() = "TDI on the Fly."; // Optional module docstring

    // Call initialization functions from other files
    tdionthefly_part(m);
    
    m.def("get_module_path_cpp", &get_module_path_tdionthefly, "Returns the file path of the module");

    // Optionally, get the path during module initialization and store it
    // This can cause an AttributeError if not handled carefully, as m.attr("__file__")
    // might not be fully set during the initial call if the module is loaded in
    // a specific way (e.g., via pythonw or as a namespace package).
    try {
        std::string path_at_init = m.attr("__file__").cast<std::string>();
        // std::cout << "Module loaded from: " << path_at_init << std::endl;
        m.attr("module_dir") = py::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (py::error_already_set &e) {
         // Handle potential error here, e.g., by logging or setting a default value
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore(); // Restore exception state for proper Python handling
        PyErr_Clear();
    }
}

