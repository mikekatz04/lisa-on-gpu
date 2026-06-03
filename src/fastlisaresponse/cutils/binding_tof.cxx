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

void GBTDIonTheFlyWrap::run_fd_wave_tdi_wrap(
    array_type<std::complex<double>> X_het,
    array_type<int>    k_f0_out,
    array_type<double> f0_grid_out,
    array_type<double> params,
    double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels
)
{
    gb_run_fd_wave_tdi_wrap(
        waveform,
        (cmplx*) return_pointer_and_check_length(X_het, "X_het",
                     N_sparse, num_bin * nchannels),
        return_pointer_and_check_length(k_f0_out, "k_f0_out", num_bin, 1),
        return_pointer_and_check_length(f0_grid_out, "f0_grid_out",
                     num_bin, 1),
        return_pointer_and_check_length(params, "params",
                     n_params, num_bin),
        t_start, Tobs,
        N_sparse, num_bin, n_params, nchannels
    );
}


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


#if 0  // === WaveletLookupTableWrap::* + gb_wdm_spline_* impls disabled at Phase 3L (2026-06-02) ===
void WaveletLookupTableWrap::get_w_mn_arr(
    array_type<double> out,
    array_type<double> amp_arr,
    array_type<double> phi_arr,
    array_type<double> f_arr,
    array_type<double> fdot_arr,
    array_type<int> m_arr,
    array_type<int> n_arr,
    int N)
{
    double *out_p   = return_pointer_and_check_length(out,   "out",   N, 1);
    double *amp_p   = return_pointer_and_check_length(amp_arr, "amp",  N, 1);
    double *phi_p   = return_pointer_and_check_length(phi_arr, "phi",  N, 1);
    double *f_p     = return_pointer_and_check_length(f_arr,  "f",    N, 1);
    double *fdot_p  = return_pointer_and_check_length(fdot_arr,"fdot",N, 1);
    int    *m_p     = return_pointer_and_check_length(m_arr,  "m",    N, 1);
    int    *n_p     = return_pointer_and_check_length(n_arr,  "n",    N, 1);

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    // GPU build: would need a kernel. Not needed for the unit test.
    throw std::runtime_error("get_w_mn_arr is CPU-only");
#else
    for (int i = 0; i < N; ++i) {
        cmplx tdi(amp_p[i] * cos(phi_p[i]), amp_p[i] * sin(phi_p[i]));
        out_p[i] = wdm_lookup->get_wdm_in_channel_over_layers(tdi, f_p[i], fdot_p[i], m_p[i], n_p[i]);
    }
#endif
}











// ---- Spline-path bindings ---------------------------------------------------

void GBComputationGroupWrap::gb_wdm_spline_fill_global(array_type<double>template_fill, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<double>factors_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt)
{
    gb_wdm_spline_fill_global_wrap(
        return_pointer_and_check_length(template_fill, "template_fill", wdm_wrap->wdm->Nf_active * wdm_wrap->wdm->Nt_active, 3),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
}

void GBComputationGroupWrap::gb_wdm_spline_get_ll(array_type<double>d_h_out, array_type<double>h_h_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt)
{
    gb_wdm_spline_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
}

void GBComputationGroupWrap::gb_wdm_spline_swap_ll(array_type<double>d_h_add_out, array_type<double>d_h_remove_out, array_type<double>add_add_out, array_type<double>remove_remove_out, array_type<double>add_remove_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_add_all, array_type<double>params_remove_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt)
{
    gb_wdm_spline_swap_ll_wrap(
        return_pointer_and_check_length(d_h_add_out,       "d_h_add_out",       num_bin, 1),
        return_pointer_and_check_length(d_h_remove_out,    "d_h_remove_out",    num_bin, 1),
        return_pointer_and_check_length(add_add_out,       "add_add_out",       num_bin, 1),
        return_pointer_and_check_length(remove_remove_out, "remove_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_remove_out,    "add_remove_out",    num_bin, 1),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_add_all,    "params_add_all",    nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin, 1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin, 1),
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
}

void GBComputationGroupWrap::gb_wdm_spline_get_ll_grad(array_type<double>grad_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, array_type<double>param_eps, int num_bin, int nparams, double T, double t_ref, int tdi_type, double coarse_dt)
{
    gb_wdm_spline_get_ll_grad_wrap(
        return_pointer_and_check_length(grad_out,          "grad_out",          nparams,  num_bin),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_all,        "params_all",        nparams,  num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin,  1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin,  1),
        return_pointer_and_check_length(param_eps,         "param_eps",         nparams,  1),
        num_bin, nparams, T, t_ref, tdi_type, coarse_dt);
}

void GBComputationGroupWrap::gb_wdm_spline_eval_inputs(
    OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    array_type<double> params_all, array_type<double> tn_arr,
    int num_bin, int nparams, int num_t, int nchannels,
    double T, double t_ref,
    double t_window_start, double coarse_dt,
    array_type<double> amp_out, array_type<double> phi_out,
    array_type<double> f_out, array_type<double> fdot_out,
    array_type<double> phase_ref_out)
{
    gb_wdm_spline_eval_inputs_wrap(
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(tn_arr, "tn_arr", num_t, 1),
        num_bin, nparams, num_t, nchannels,
        T, t_ref,
        t_window_start, coarse_dt,
        return_pointer_and_check_length(amp_out,       "amp_out",       num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(phi_out,       "phi_out",       num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(f_out,         "f_out",         num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(fdot_out,      "fdot_out",      num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(phase_ref_out, "phase_ref_out", num_bin * num_t, 1));
}
#endif  // === end WaveletLookupTableWrap::* + gb_wdm_spline_* impls disabled ===

void GBComputationGroupWrap::gb_fd_fill_global(
    array_type<std::complex<double>> template_fill,
    OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_all, array_type<int> data_index_all,
    array_type<double> factors_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels)
{
    int n_rfft = fd_wrap->fd->n_rfft;
    int num_data = fd_wrap->fd->num_data;
    gb_fd_fill_global_wrap(
        (cmplx*) return_pointer_and_check_length(template_fill,
            "template_fill", n_rfft * nchannels * num_data, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels);
}

void GBComputationGroupWrap::gb_fd_get_ll(
    array_type<double> d_h_out, array_type<double> h_h_out,
    OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    gb_fd_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type);
}

void GBComputationGroupWrap::gb_fd_swap_ll(
    array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
    array_type<double> add_add_out, array_type<double> remove_remove_out,
    array_type<double> add_remove_out,
    OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_add_all, array_type<double> params_remove_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    gb_fd_swap_ll_wrap(
        return_pointer_and_check_length(d_h_add_out, "d_h_add_out", num_bin, 1),
        return_pointer_and_check_length(d_h_remove_out, "d_h_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_add_out, "add_add_out", num_bin, 1),
        return_pointer_and_check_length(remove_remove_out, "remove_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_remove_out, "add_remove_out", num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_add_all, "params_add_all", nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type);
}

void GBComputationGroupWrap::gb_fd_get_ll_grad(
    array_type<double> grad_out,
    OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    array_type<double> param_eps,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    gb_fd_get_ll_grad_wrap(
        return_pointer_and_check_length(grad_out,        "grad_out",        nparams, num_bin),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_all,      "params_all",      nparams, num_bin),
        return_pointer_and_check_length(data_index_all,  "data_index_all",  num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(param_eps,       "param_eps",       nparams, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type);
}

void GBComputationGroupWrap::gb_fd_swap_ll_grad(
    array_type<double> grad_add_out, array_type<double> grad_remove_out,
    OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_add_all, array_type<double> params_remove_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    array_type<double> param_eps_add, array_type<double> param_eps_remove,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    gb_fd_swap_ll_grad_wrap(
        return_pointer_and_check_length(grad_add_out,      "grad_add_out",      nparams, num_bin),
        return_pointer_and_check_length(grad_remove_out,   "grad_remove_out",   nparams, num_bin),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_add_all,    "params_add_all",    nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin, 1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin, 1),
        return_pointer_and_check_length(param_eps_add,     "param_eps_add",     nparams, 1),
        return_pointer_and_check_length(param_eps_remove,  "param_eps_remove",  nparams, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type);
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


// ===========================================================================
// Chunked-heterodyne wrappers: thin pybind shims that unpack numpy arrays
// into raw pointers and forward to the C++ host wrappers in TDIonTheFly.cu.
//
// Both GB and SOBBH share an identical method body modulo source-class name;
// we keep them separate (not a template) for readability of the pybind layer.
// ===========================================================================

void GBComputationGroupWrap::gb_wdm_het_fill_global(
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
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int m_band_half_width)
{
    const int Nf = wdm_settings_wrap->wdm_settings->Nf;
    const int Nt = wdm_settings_wrap->wdm_settings->Nt;
    gb_wdm_het_fill_global_wrap(
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
        T_chunk, dt, T, t_ref, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        m_band_half_width);
}

void GBComputationGroupWrap::gb_wdm_het_get_ll(
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
    array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
    int m_band_half_width)
{
    // Group-array lengths: binary_perm is num_bin; the four group_*
    // arrays are length max(n_groups, 1) (caller passes length-1 stubs
    // when n_groups == 0).
    const int gn = (n_groups > 0) ? n_groups : 1;
    // Sizes used only for the data_d / invC length checks; pulled from
    // the WDMSettings the caller already constructed.
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    gb_wdm_het_get_ll_wrap(
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
        n_groups, m_band_half_width);
}

void GBComputationGroupWrap::gb_wdm_het_swap_ll(
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
    array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b,
    int m_band_half_width)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    const int Nf_active = wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    gb_wdm_het_swap_ll_wrap(
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
        return_pointer_and_check_length(pair_m_hi_b, "pair_m_hi_b", num_bin, 1),
        m_band_half_width);
}


void GBComputationGroupWrap::gb_wdm_het_get_fstat_ll(
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
    gb_wdm_het_get_fstat_ll_wrap(
        // outputs: (num_bin, 4) for N, (num_bin, 10) for M (upper-triangle)
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


// ---- Signal-heterodyne (v2 polyphase) pybind shim --------------------------
//
// Stage 1: takes precomputed FD (rfft(Tukey * td)) per binary, plus the
// reference's c0_sparse and bin-folded A0/A1/B0/B1. Production target moves
// FD generation into the kernel via a sparse-spline absolute-FD source class.
void GBComputationGroupWrap::gb_signal_het_get_ll(
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<std::complex<double>> fd_rfft_all,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    int nchannels, int tdi_type,
    int n_rfft)
{
    (void) Nt_layer;
    const size_t b_xyz = (size_t) num_data * nchannels * nchannels
                       * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    gb_signal_het_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            fd_rfft_all, "fd_rfft_all",
            (size_t) num_bin * nchannels * n_rfft, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all",
            (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all",
            (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels, tdi_type,
        n_rfft);
}


// Stage 2a: sparse-FD entry. X_het is the candidate's heterodyned sparse FD
// (length N_sparse_fd per (binary, channel)); k_f0 is the absolute-FD bin
// index of each binary's snapped carrier. Polyphase fold iterates only the
// N_sparse_fd nonzero bins -- the rest are implicit zero.
void GBComputationGroupWrap::gb_signal_het_get_ll_sparse(
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<std::complex<double>> X_het_all,
    array_type<int> k_f0_all,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    int nchannels, int tdi_type,
    int N_sparse_fd)
{
    (void) Nt_layer;
    const size_t b_xyz = (size_t) num_data * nchannels * nchannels
                       * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    gb_signal_het_get_ll_sparse_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            X_het_all, "X_het_all",
            (size_t) num_bin * nchannels * N_sparse_fd, 1)),
        return_pointer_and_check_length(k_f0_all, "k_f0_all", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels, tdi_type,
        N_sparse_fd);
}


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
    .def(py::init<OrbitsWrap_responselisa *, TDIConfigWrap *, double, double>(), 
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"), py::arg("t_ref"))
    // Bind member functions
    .def("run_wave_tdi_wrap", &GBTDIonTheFlyWrap::run_wave_tdi_wrap, "Preform TDI combinations.")
    .def("run_fd_wave_tdi_wrap", &GBTDIonTheFlyWrap::run_fd_wave_tdi_wrap,
         "Heterodyne FD GB TDI on a sparse time grid.")
    .def("get_buffer_size", &GBTDIonTheFlyWrap::get_buffer_size, "Get needed buffer size.")
    .def("get_fd_buffer_size", &GBTDIonTheFlyWrap::get_fd_buffer_size,
         "Get shared-memory size for the heterodyne FD kernel.")
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
    .def(py::init<Orbits *, TDIConfig*, double, double>(),
         py::arg("orbits"), py::arg("tdi_config"), py::arg("Tobs"), py::arg("t_ref"))
    ;

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

    #if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<GBComputationGroupWrap>(m, "GBComputationGroupWrapGPU")
#else
    py::class_<GBComputationGroupWrap>(m, "GBComputationGroupWrapCPU")
#endif
    .def(py::init<>())
    .def("gb_fd_fill_global", &GBComputationGroupWrap::gb_fd_fill_global,
         "FD analog of gb_wdm_fill_global: scatter per-source heterodyne FD onto a "
         "global rfft-grid template (cmplx, shape (num_data, nchannels, n_rfft)).")
    .def("gb_fd_get_ll", &GBComputationGroupWrap::gb_fd_get_ll,
         "FD analog of gb_wdm_get_ll: (d|h) and (h|h) per binary using the "
         "lisatools FD inner product (4 Re sum conj(d) h invC * df).  tdi_type "
         "selects between TDI_XYZ (cross-channel 3x3 invC) and TDI_AET/TDI_AE "
         "(diagonal invC).")
    .def("gb_fd_swap_ll", &GBComputationGroupWrap::gb_fd_swap_ll,
         "FD analog of gb_wdm_swap_ll: returns the five inner products "
         "<d|h_add>, <d|h_remove>, <h_add|h_add>, <h_remove|h_remove>, "
         "<h_add|h_remove> needed for an RJMCMC swap proposal.")
    .def("gb_fd_get_ll_grad", &GBComputationGroupWrap::gb_fd_get_ll_grad,
         "FD analog of gb_wdm_get_ll_grad: chain-rule parameter gradient of "
         "L = -1/2 <d-h|d-h> evaluated in the sparse-FD heterodyne pipeline. "
         "param_eps[k] is the central-FD step for theta_k (pass <= 0 to "
         "freeze).")
    .def("gb_fd_swap_ll_grad", &GBComputationGroupWrap::gb_fd_swap_ll_grad,
         "FD analog of gb_wdm_swap_ll_grad: returns (grad_add, grad_remove), "
         "the per-binary derivatives of ll_diff = L(after swap) - L(before "
         "swap) with respect to theta_add and theta_remove respectively.")
    // ---- Spline-path mirrors disabled at Phase 3L (2026-06-02) ----------
    // The gb_wdm_spline_* methods used WaveletLookupTableWrap; lookup-
    // table path is retiring. Chunked-heterodyne (gb_wdm_het_*) below is
    // the supported WDM path.
    // ---- Chunked-heterodyne (no lookup table) ----------------------
    .def("gb_wdm_het_fill_global", &GBComputationGroupWrap::gb_wdm_het_fill_global,
         "Chunked-heterodyne fill_global. Builds the WDM-domain GB template by "
         "iterating over precomputed chunks (chunk_t_starts / keep_lo / keep_hi / "
         "n_global_offset). Each block (chunk) walks all binaries so per-chunk "
         "PSD/data slabs are reused. grid_dim picks the launch grid (use "
         "chunked_het_grid_dim).")
    .def("gb_wdm_het_get_ll", &GBComputationGroupWrap::gb_wdm_het_get_ll,
         "Chunked-heterodyne get_ll. Returns <d|h> and <h|h> per binary, "
         "matching gb_wdm_get_ll up to numerical precision.")
    .def("gb_wdm_het_swap_ll", &GBComputationGroupWrap::gb_wdm_het_swap_ll,
         "Chunked-heterodyne swap_ll. Returns the same five inner products as "
         "gb_wdm_swap_ll: <d|h_add>, <d|h_remove>, <h_add|h_add>, "
         "<h_remove|h_remove>, <h_add|h_remove>.")
    .def("gb_wdm_het_get_fstat_ll", &GBComputationGroupWrap::gb_wdm_het_get_fstat_ll,
         "Chunked-heterodyne F-stat. Returns per-binary N_arr (4,) = "
         "<d|A_i> and M_mat (10,) = <A_i|A_j> upper-triangle (4 basis "
         "filters per Cornish & Crowder '05). Python computes "
         "F = N^T M^{-1} N / 2 from these. Imag outputs always 0 "
         "(WDM coefficients are real).")
    .def("gb_signal_het_get_ll", &GBComputationGroupWrap::gb_signal_het_get_ll,
         "Signal-heterodyne (v2 polyphase) get_ll. Takes precomputed "
         "rfft(Tukey*td) per binary plus reference c0_sparse/A0/A1/B0/B1; "
         "returns per-binary <d|h>, <h|h> via the bin-folded inner-product "
         "accumulator. Stage 1: CPU only, FD as input. Stage 2 will move FD "
         "generation in-kernel via a sparse-spline absolute-FD source.")
    .def("gb_signal_het_get_ll_sparse", &GBComputationGroupWrap::gb_signal_het_get_ll_sparse,
         "Stage 2a sparse-FD signal-het get_ll. Consumes X_het (length "
         "N_sparse_fd per binary per channel) + per-binary k_f0. Polyphase "
         "fold iterates only the N_sparse_fd nonzero bins. Stage 2b will fill "
         "X_het in-kernel from the source-class heterodyned sparse rfft.")
    ;

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

