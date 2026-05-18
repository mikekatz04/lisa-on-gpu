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


void GBComputationGroupWrap::gb_wdm_fill_global(array_type<double>template_fill, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<double>factors_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t)
{
    // from the parent class

    gb_wdm_fill_global_wrap(
        return_pointer_and_check_length(template_fill, "template_fill", wdm_wrap->wdm->Nf_active * wdm_wrap->wdm->Nt_active, 3),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t);
}

void GBComputationGroupWrap::gb_wdm_get_ll(array_type<double>d_h_out, array_type<double>h_h_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t)
{
    // from the parent class
    gb_wdm_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t);
}

void GBComputationGroupWrap::gb_wdm_swap_ll(array_type<double>d_h_add_out, array_type<double>d_h_remove_out, array_type<double>add_add_out, array_type<double>remove_remove_out, array_type<double>add_remove_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_add_all, array_type<double>params_remove_all, array_type<int>data_index_all, array_type<int>noise_index_all, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t)
{
    // from the parent class
    gb_wdm_swap_ll_wrap(
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
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t);
}


void GBComputationGroupWrap::gb_wdm_get_ll_grad(array_type<double>grad_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_all, array_type<int>data_index_all, array_type<int>noise_index_all, array_type<double>param_eps, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t)
{
    gb_wdm_get_ll_grad_wrap(
        return_pointer_and_check_length(grad_out,          "grad_out",          nparams,  num_bin),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_all,        "params_all",        nparams,  num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin,  1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin,  1),
        return_pointer_and_check_length(param_eps,         "param_eps",         nparams,  1),
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t);
}

void GBComputationGroupWrap::gb_wdm_swap_ll_grad(array_type<double>grad_add_out, array_type<double>grad_remove_out, OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap, WaveletLookupTableWrap* wdm_lookup_wrap, WDMDomainWrap* wdm_wrap, array_type<double>params_add_all, array_type<double>params_remove_all, array_type<int>data_index_all, array_type<int>noise_index_all, array_type<double>param_eps_add, array_type<double>param_eps_remove, int num_bin, int nparams, double T, double t_ref, int tdi_type, double deriv_delta_t)
{
    gb_wdm_swap_ll_grad_wrap(
        return_pointer_and_check_length(grad_add_out,      "grad_add_out",      nparams, num_bin),
        return_pointer_and_check_length(grad_remove_out,   "grad_remove_out",   nparams, num_bin),
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        wdm_lookup_wrap->wdm_lookup,
        wdm_wrap->wdm,
        return_pointer_and_check_length(params_add_all,    "params_add_all",    nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin, 1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin, 1),
        return_pointer_and_check_length(param_eps_add,     "param_eps_add",     nparams, 1),
        return_pointer_and_check_length(param_eps_remove,  "param_eps_remove",  nparams, 1),
        num_bin, nparams, T, t_ref, tdi_type, deriv_delta_t);
}


void GBComputationGroupWrap::gb_wdm_eval_inputs(
    OrbitsWrap_responselisa* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    array_type<double> params_all, array_type<double> tn_arr,
    int num_bin, int nparams, int num_t, int nchannels,
    double T, double t_ref, double deriv_delta_t,
    array_type<double> amp_out, array_type<double> phi_out,
    array_type<double> f_out, array_type<double> fdot_out,
    array_type<double> phase_ref_out)
{
    gb_wdm_eval_inputs_wrap(
        orbits_wrap->orbits,
        tdi_config_wrap->tdi_config,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(tn_arr, "tn_arr", num_t, 1),
        num_bin, nparams, num_t, nchannels,
        T, t_ref, deriv_delta_t,
        return_pointer_and_check_length(amp_out,       "amp_out",       num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(phi_out,       "phi_out",       num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(f_out,         "f_out",         num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(fdot_out,      "fdot_out",      num_bin * num_t * nchannels, 1),
        return_pointer_and_check_length(phase_ref_out, "phase_ref_out", num_bin * num_t, 1));
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
    py::class_<WaveletLookupTableWrap>(m, "WaveletLookupTableWrapGPU")
#else
    py::class_<WaveletLookupTableWrap>(m, "WaveletLookupTableWrapCPU")
#endif

    // Bind the constructor
    .def(py::init<array_type<double>,array_type<double>,int,int,double,double,double,double, double, double, int, int, int, int, int, int, int, int>(),
         py::arg("c_nm_all"), py::arg("s_nm_all"), py::arg("num_f"), py::arg("num_fdot"), py::arg("df_interp"), py::arg("dfdot_interp"), py::arg("min_f"), py::arg("min_fdot"), py::arg("layer_df"), py::arg("layer_dt"), py::arg("Nf"), py::arg("Nt"), py::arg("num_channel"), py::arg("ind_min_t"), py::arg("ind_max_t"), py::arg("ind_min_f"), py::arg("ind_max_f"), py::arg("m_ref"))
    // Bind member functions
    .def("get_w_mn_arr", &WaveletLookupTableWrap::get_w_mn_arr,
         py::arg("out"), py::arg("amp"), py::arg("phi"), py::arg("f"), py::arg("fdot"),
         py::arg("m"), py::arg("n"), py::arg("N"),
         "For each i in [0,N), compute the C-side w_mn for the given (amp,phi,f,fdot,m,n). "
         "tdi_channel_val is constructed as amp*(cos(phi) + i*sin(phi)) — matching the post conj/exp(-Iπ/2) representation used in the kernel.")

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
    .def(py::init<double*,double*,int,int,double,double,double,double, double, double, int, int, int, int, int, int, int, int>(),
         py::arg("c_nm_all"), py::arg("s_nm_all"), py::arg("num_f"), py::arg("num_fdot"), py::arg("df_interp"), py::arg("dfdot_interp"), py::arg("min_f"), py::arg("min_fdot"), py::arg("layer_df"), py::arg("layer_dt"), py::arg("Nf"), py::arg("Nt"), py::arg("num_channel"), py::arg("ind_min_t"), py::arg("ind_max_t"), py::arg("ind_min_f"), py::arg("ind_max_f"), py::arg("m_ref"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<WDMDomainWrap>(m, "WDMDomainWrapGPU")
#else
    py::class_<WDMDomainWrap>(m, "WDMDomainWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<array_type<double>,array_type<double>, double, double, int, int, int, int, int, int, int, int, int>(), 
         py::arg("wdm_data"), py::arg("wdm_noise"), py::arg("layer_df"), py::arg("layer_dt"), py::arg("Nf"), py::arg("Nt"), py::arg("num_channel"), py::arg("ind_min_t"), py::arg("ind_max_t"), py::arg("ind_min_f"), py::arg("ind_max_f"), py::arg("num_data"), py::arg("num_noise"))
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
    .def(py::init<double*,double*, double, double, int, int, int, int, int, int, int, int, int>(), 
         py::arg("wdm_data"), py::arg("wdm_noise"), py::arg("layer_df"), py::arg("layer_dt"), py::arg("Nf"), py::arg("Nt"), py::arg("num_channel"), py::arg("ind_min_t"), py::arg("ind_max_t"), py::arg("ind_min_f"), py::arg("ind_max_f"), py::arg("num_data"), py::arg("num_noise"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<FDDomainWrap>(m, "FDDomainWrapGPU")
#else
    py::class_<FDDomainWrap>(m, "FDDomainWrapCPU")
#endif
    .def(py::init<array_type<std::complex<double>>, array_type<double>,
                  int, int, int, int, int, int, double>(),
         py::arg("fd_data"), py::arg("fd_invC"),
         py::arg("n_rfft"), py::arg("num_channel"),
         py::arg("num_data"), py::arg("num_noise"),
         py::arg("ind_min"), py::arg("ind_max"), py::arg("df"))
    .def_readwrite("fd", &FDDomainWrap::fd)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<FDDomain>(m, "FDDomainGPU")
#else
    py::class_<FDDomain>(m, "FDDomainCPU")
#endif
    .def(py::init<cmplx*, double*, int, int, int, int, int, int, double>())
    ;

    #if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<GBComputationGroupWrap>(m, "GBComputationGroupWrapGPU")
#else
    py::class_<GBComputationGroupWrap>(m, "GBComputationGroupWrapCPU")
#endif
    .def(py::init<>())
    .def("gb_wdm_get_ll", &GBComputationGroupWrap::gb_wdm_get_ll, "Log-likelihood computation.")
    .def("gb_wdm_swap_ll", &GBComputationGroupWrap::gb_wdm_swap_ll,
         "Swap-likelihood computation. Returns the five inner products <d|h_add>, "
         "<d|h_remove>, <h_add|h_add>, <h_remove|h_remove>, <h_add|h_remove> needed "
         "for an RJMCMC swap proposal between an 'add' and a 'remove' template.")
    .def("gb_wdm_get_ll_grad", &GBComputationGroupWrap::gb_wdm_get_ll_grad,
         "Chain-rule parameter gradient of gb_wdm_get_ll. Per-binary central-difference "
         "derivative of L = -1/2 <d-h|d-h> with respect to the 9 galactic-binary "
         "parameters. param_eps[k] is the FD step for theta_k (pass <= 0 to freeze).")
    .def("gb_wdm_swap_ll_grad", &GBComputationGroupWrap::gb_wdm_swap_ll_grad,
         "Chain-rule parameter gradient of gb_wdm_swap_ll. Returns "
         "(grad_add, grad_remove): per-binary central-difference derivatives of "
         "the swap log-likelihood ratio ll_diff = L(after swap) - L(before swap) "
         "with respect to theta_add and theta_remove respectively.")
    .def("gb_wdm_fill_global", &GBComputationGroupWrap::gb_wdm_fill_global, "Generate a global template.")
    .def("gb_wdm_eval_inputs", &GBComputationGroupWrap::gb_wdm_eval_inputs,
         py::arg("orbits"), py::arg("tdi_config"),
         py::arg("params_all"), py::arg("tn_arr"),
         py::arg("num_bin"), py::arg("nparams"), py::arg("num_t"), py::arg("nchannels"),
         py::arg("T"), py::arg("t_ref"), py::arg("deriv_delta_t"),
         py::arg("amp_out"), py::arg("phi_out"),
         py::arg("f_out"), py::arg("fdot_out"),
         py::arg("phase_ref_out"),
         "Diagnostic: evaluate the per-pixel inputs (|M|, arg(M_mod), f, fdot, phase_ref) "
         "that the kernels feed into the WDM lookup, without doing the lookup itself.")
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
    // ---- Spline-path mirrors. `coarse_dt` (seconds) sets the coarse-grid
    // spacing used by the cubic-spline window builder. ---------------------
    .def("gb_wdm_spline_fill_global", &GBComputationGroupWrap::gb_wdm_spline_fill_global,
         "Spline-path mirror of gb_wdm_fill_global. `coarse_dt` (seconds) is "
         "the coarse-grid spacing.")
    .def("gb_wdm_spline_get_ll", &GBComputationGroupWrap::gb_wdm_spline_get_ll,
         "Spline-path mirror of gb_wdm_get_ll.")
    .def("gb_wdm_spline_swap_ll", &GBComputationGroupWrap::gb_wdm_spline_swap_ll,
         "Spline-path mirror of gb_wdm_swap_ll.")
    .def("gb_wdm_spline_get_ll_grad", &GBComputationGroupWrap::gb_wdm_spline_get_ll_grad,
         "Spline-path mirror of gb_wdm_get_ll_grad. Same chain-rule formula, "
         "frozen layer_m_c, central differences with param_eps[k]; memory is "
         "constant in nparams (three spline slots in shared).")
    .def("gb_wdm_spline_eval_inputs", &GBComputationGroupWrap::gb_wdm_spline_eval_inputs,
         py::arg("orbits"), py::arg("tdi_config"),
         py::arg("params_all"), py::arg("tn_arr"),
         py::arg("num_bin"), py::arg("nparams"), py::arg("num_t"), py::arg("nchannels"),
         py::arg("T"), py::arg("t_ref"),
         py::arg("t_window_start"), py::arg("coarse_dt"),
         py::arg("amp_out"), py::arg("phi_out"),
         py::arg("f_out"), py::arg("fdot_out"),
         py::arg("phase_ref_out"),
         "Spline diagnostic: builds one WDM_SPLINE_L-point spline window "
         "starting at t_window_start with spacing coarse_dt, then evaluates "
         "at every tn in tn_arr. Output layout matches gb_wdm_eval_inputs.")
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

