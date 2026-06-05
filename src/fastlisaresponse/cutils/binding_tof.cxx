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
// Phase 3J: lisa-on-gpu is a CONSUMER of LAT-owned shared wrappers (OrbitsWrap,
// LISAResponseWrap, TDIConfigWrap, CubicSplineWrap_responselisa). It must NOT
// register them with pybind11 -- that would re-introduce the past duplicate-
// registration pain (LAT's pycppdetector + lisa-on-gpu's tdionthefly both
// trying to claim the same C++ type, pybind11 throwing
// "type already registered" at import). The static_assert below makes that
// violation a compile-time error if someone ever adds the registration here.
#include "lisatools_header_abi.hpp"
static_assert(!LISATOOLS_IS_WRAPPER_OWNER,
    "Single-registrant rule: lisa-on-gpu (binding_tof.cxx) must NOT register "
    "OrbitsWrap / LISAResponseWrap / TDIConfigWrap / CubicSplineWrap_responselisa. "
    "Those are owned by LISAanalysistools (pycppdetector). "
    "See plan section 'OrbitsWrap-symbol-unification' and "
    "memory project_phase3_efg_shipped for context.");

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;

// GBTDIonTheFlyWrap::run_*_wrap method bodies moved to GBGPU at
// Phase 3L.7g (2026-06-04). See GBGPU/src/gbgpu/cutils/binding_gbgpu.cxx.


void SOBBHTDIonTheFlyWrap::run_wave_tdi_wrap(
    array_type<std::complex<double>>tdi_channels_arr,
    array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref,
    array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
)
{
    sobbh_run_wave_tdi_wrap(
        waveform,
        (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
        return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
        return_pointer_and_check_length(params, "params", n_params, num_bin),
        return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
        N, num_bin, n_params, nchannels
    );
}


// TDSplineTDIWaveformWrap::run_wave_tdi_wrap + FDSplineTDIWaveformWrap::run_wave_tdi_wrap
// moved to LAT at Phase 3L.6 (2026-06-03). Bodies are now inline in
// lisatools/cutils/binding_lat_spline_tdi.hpp.


// WaveletLookupTableWrap + GBComputationGroupWrap::gb_wdm_spline_*

// disabled #if 0 historical bodies removed at Phase 3L.7g (lookup-table

// path retired earlier; spline path methods carved out with the wrapper).


// GBComputationGroupWrap::gb_fd_* method bodies moved to GBGPU at

// Phase 3L.7g (2026-06-04). See GBGPU/src/gbgpu/cutils/binding_gbgpu.cxx.


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


// GBComputationGroupWrap::gb_wdm_het_* + gb_signal_het_* method bodies

// moved to GBGPU at Phase 3L.7g (2026-06-04). See

// GBGPU/src/gbgpu/cutils/binding_gbgpu.cxx. SOBBH equivalents below stay.



// ---- SOBBH-flavored pybind shims -------------------------------------------
void SOBBHComputationGroupWrap::sobbh_wdm_het_fill_global(
    array_type<double> template_fill,
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    WDMSettingsWrap *wdm_settings_wrap,
    array_type<double> params_all, array_type<double> factors_all,
    array_type<double> chunk_t_starts,
    array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
    array_type<int> chunk_n_global_offset,
    array_type<double> wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit)
{
    const int Nf = wdm_settings_wrap->wdm_settings->Nf;
    const int Nt = wdm_settings_wrap->wdm_settings->Nt;
    sobbh_wdm_het_fill_global_wrap(
        return_pointer_and_check_length(template_fill, "template_fill",
                                        (size_t) nchannels * Nf * Nt, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit);
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_get_ll(
    array_type<double> d_h_out, array_type<double> h_h_out,
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    WDMSettingsWrap *wdm_settings_wrap,
    array_type<double> params_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    array_type<double> chunk_t_starts,
    array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
    array_type<int> chunk_n_global_offset,
    array_type<double> wdm_window,
    array_type<double> data_d, array_type<double> invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
    array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    sobbh_wdm_het_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // Active-band layout (see het kernel docs):
        //   data_d : (nchannels, Nf_active, Nt_active)
        //   invC   : (nchannels, [nchannels], Nf_active, Nt_active)
        //            cross-channel Hermitian for TDI_XYZ (extra nch dim),
        //            diagonal-only for TDI_AET / TDI_AE.
        return_pointer_and_check_length(
            data_d, "data_d",
            (size_t) nchannels * Nf_active * Nt_active, 1),
        return_pointer_and_check_length(
            invC, "invC",
            ((tdi_type == TDI_XYZ)
                 ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                 : (size_t) nchannels * Nf_active * Nt_active),
            1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        return_pointer_and_check_length(binary_perm,  "binary_perm",  num_bin, 1),
        return_pointer_and_check_length(group_starts, "group_starts", gn, 1),
        return_pointer_and_check_length(group_ends,   "group_ends",   gn, 1),
        return_pointer_and_check_length(group_m_lo,   "group_m_lo",   gn, 1),
        return_pointer_and_check_length(group_m_hi,   "group_m_hi",   gn, 1),
        n_groups);
}

void SOBBHComputationGroupWrap::sobbh_wdm_het_swap_ll(
    array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
    array_type<double> add_add_out, array_type<double> remove_remove_out,
    array_type<double> add_remove_out,
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    WDMSettingsWrap *wdm_settings_wrap,
    array_type<double> params_add_all, array_type<double> params_remove_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    array_type<double> chunk_t_starts,
    array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
    array_type<int> chunk_n_global_offset,
    array_type<double> wdm_window,
    array_type<double> data_d, array_type<double> invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
    array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
    array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    sobbh_wdm_het_swap_ll_wrap(
        return_pointer_and_check_length(d_h_add_out,       "d_h_add_out",       num_bin, 1),
        return_pointer_and_check_length(d_h_remove_out,    "d_h_remove_out",    num_bin, 1),
        return_pointer_and_check_length(add_add_out,       "add_add_out",       num_bin, 1),
        return_pointer_and_check_length(remove_remove_out, "remove_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_remove_out,    "add_remove_out",    num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_add_all,    "params_add_all",    nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin, 1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts,    "chunk_t_starts",    n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo,     "chunk_keep_lo",     n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi,     "chunk_keep_hi",     n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // Active-band layout (see het kernel docs):
        //   data_d : (nchannels, Nf_active, Nt_active)
        //   invC   : (nchannels, [nchannels], Nf_active, Nt_active)
        //            cross-channel Hermitian for TDI_XYZ (extra nch dim),
        //            diagonal-only for TDI_AET / TDI_AE.
        return_pointer_and_check_length(
            data_d, "data_d",
            (size_t) nchannels * Nf_active * Nt_active, 1),
        return_pointer_and_check_length(
            invC, "invC",
            ((tdi_type == TDI_XYZ)
                 ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                 : (size_t) nchannels * Nf_active * Nt_active),
            1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        return_pointer_and_check_length(binary_perm,  "binary_perm",  num_bin, 1),
        return_pointer_and_check_length(group_starts, "group_starts", gn, 1),
        return_pointer_and_check_length(group_ends,   "group_ends",   gn, 1),
        return_pointer_and_check_length(group_m_lo,   "group_m_lo",   gn, 1),
        return_pointer_and_check_length(group_m_hi,   "group_m_hi",   gn, 1),
        n_groups,
        return_pointer_and_check_length(pair_m_lo_b, "pair_m_lo_b", num_bin, 1),
        return_pointer_and_check_length(pair_m_hi_b, "pair_m_hi_b", num_bin, 1));
}


void SOBBHComputationGroupWrap::sobbh_wdm_het_get_fstat_ll(
    array_type<double> N_arr_re_out, array_type<double> N_arr_im_out,
    array_type<double> M_mat_re_out, array_type<double> M_mat_im_out,
    OrbitsWrap_responselisa *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    WDMSettingsWrap *wdm_settings_wrap,
    array_type<double> params_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    array_type<double> chunk_t_starts,
    array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
    array_type<int> chunk_n_global_offset,
    array_type<double> wdm_window,
    array_type<double> data_d, array_type<double> invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int m_band_half_width)
{
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    sobbh_wdm_het_get_fstat_ll_wrap(
        return_pointer_and_check_length(N_arr_re_out, "N_arr_re_out", num_bin, 4),
        return_pointer_and_check_length(N_arr_im_out, "N_arr_im_out", num_bin, 4),
        return_pointer_and_check_length(M_mat_re_out, "M_mat_re_out", num_bin, 10),
        return_pointer_and_check_length(M_mat_im_out, "M_mat_im_out", num_bin, 10),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        return_pointer_and_check_length(
            data_d, "data_d",
            (size_t) nchannels * Nf_active * Nt_active, 1),
        return_pointer_and_check_length(
            invC, "invC",
            ((tdi_type == TDI_XYZ)
                 ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                 : (size_t) nchannels * Nf_active * Nt_active),
            1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type,
        tukey_alpha, grid_dim, m_band_half_width);
}


// PYBIND11_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void tdionthefly_part(py::module &m) {

    m.attr("TDI_XYZ") = TDI_XYZ;
    m.attr("TDI_AET") = TDI_AET;
    m.attr("TDI_AE") = TDI_AE;
    
    // Phase 3L.6 (2026-06-03): FDSplineTDIWaveform[Wrap] + TDSplineTDIWaveform[Wrap]
    // pybind11 registrations moved to LAT's binding_flr.cxx (registered in
    // pycppdetector via response_part(m)). The
    // static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...) at the top of this TU
    // guards against any future re-registration here.


// py::class_<GBTDIonTheFlyWrap> + py::class_<GBTDIonTheFly> registrations

// moved to GBGPU's cgbgpu module at Phase 3L.7g (2026-06-04).


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<SOBBHTDIonTheFlyWrap>(m, "SOBBHTDIonTheFlyWrapGPU")
#else
    py::class_<SOBBHTDIonTheFlyWrap>(m, "SOBBHTDIonTheFlyWrapCPU")
#endif

    .def(py::init<OrbitsWrap_responselisa *, TDIConfigWrap *, double, double>(),
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"), py::arg("t_ref"))
    .def("run_wave_tdi_wrap", &SOBBHTDIonTheFlyWrap::run_wave_tdi_wrap, "Run SOBBH TDI on the fly.")
    .def("get_buffer_size", &SOBBHTDIonTheFlyWrap::get_buffer_size, "Get needed buffer size.")
    .def_readwrite("orbits", &SOBBHTDIonTheFlyWrap::orbits)
    .def_readwrite("tdi_config", &SOBBHTDIonTheFlyWrap::tdi_config)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<SOBBHTDIonTheFly>(m, "SOBBHTDIonTheFlyGPU")
#else
    py::class_<SOBBHTDIonTheFly>(m, "SOBBHTDIonTheFlyCPU")
#endif

    .def(py::init<Orbits *, TDIConfig*, double, double>(),
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"), py::arg("t_ref"))
    ;

    // === WaveletLookupTable + WaveletLookupTableWrap pybind11 registrations
    // disabled at Phase 3L (2026-06-02) -- lookup-table path retiring ===

    // Phase 3L (2026-06-02): WDMSettingsWrap pybind11 registration moved
    // to LAT's binding_flr.cxx (registered in pycppdetector via
    // response_part(m)).

    // Phase 3L (2026-06-02): WDMDomainWrap + WDMDomain pybind11 registrations
    // moved to LAT's binding_flr.cxx (registered in pycppdetector via
    // response_part(m)). The static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)
    // at the top of this TU guards against any future re-registration here.

    // Phase 3L (2026-06-02): FDDomain + FDDomainWrap pybind11 registrations
    // moved to LAT's binding_flr.cxx (registered in pycppdetector via
    // response_part(m)). The static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)
    // at the top of this TU guards against any future re-registration here.

// py::class_<GBComputationGroupWrap> registration moved to GBGPU's

// cgbgpu module at Phase 3L.7g (2026-06-04).

    #if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<SOBBHComputationGroupWrap>(m, "SOBBHComputationGroupWrapGPU")
#else
    py::class_<SOBBHComputationGroupWrap>(m, "SOBBHComputationGroupWrapCPU")
#endif
    .def(py::init<>())
    .def("sobbh_wdm_het_fill_global", &SOBBHComputationGroupWrap::sobbh_wdm_het_fill_global,
         "SOBBH chunked-heterodyne fill_global. Same as gb_wdm_het_fill_global "
         "but with SOBBHTDIonTheFly as the source class. 11-parameter source "
         "vector (see SOBBHTDIonTheFly).")
    .def("sobbh_wdm_het_get_ll", &SOBBHComputationGroupWrap::sobbh_wdm_het_get_ll,
         "SOBBH chunked-heterodyne get_ll.")
    .def("sobbh_wdm_het_swap_ll", &SOBBHComputationGroupWrap::sobbh_wdm_het_swap_ll,
         "SOBBH chunked-heterodyne swap_ll.")
    .def("sobbh_wdm_het_get_fstat_ll", &SOBBHComputationGroupWrap::sobbh_wdm_het_get_fstat_ll,
         "SOBBH chunked-heterodyne F-stat (same N+M outputs as the GB variant).")
    ;
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

