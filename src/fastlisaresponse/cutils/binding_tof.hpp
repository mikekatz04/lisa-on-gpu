#ifndef __BINDING_TOF_HPP__
#define __BINDING_TOF_HPP__

#include "TDIonTheFly.hh"
#include "LISAResponse.hh"
#include "Detector.hpp"
#include <string>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "binding.hpp"
#include "gbt_binding.hpp"
#include "gbt_global.h"
#include "binding_flr.hpp"

namespace nb = nanobind;


#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#endif
// GBTDIonTheFlyWrap + GBComputationGroupWrap aliases moved to GBGPU at Phase 3L.7g (2026-06-04).
// SOBBHTDIonTheFlyWrap + SOBBHComputationGroupWrap aliases moved to BBHx at Phase 3L.8 (2026-06-04).
// FDSplineTDIWaveformWrap + TDSplineTDIWaveformWrap aliases moved to LAT at Phase 3L.6.
// WaveletLookupTableWrap disabled at Phase 3L (2026-06-02) -- lookup-table path retiring.
// Phase 3L (2026-06-02): FDDomainWrap + WDMSettingsWrap + WDMDomainWrap moved
// to LAT. The class definitions + CPU/GPU aliases now live in
// binding_fd_domain.hpp / binding_wdm_settings.hpp / binding_wdm_domain.hpp.
#include "binding_fd_domain.hpp"
#include "binding_wdm_settings.hpp"
#include "binding_wdm_domain.hpp"
// Phase 3L.6: LISATDIonTheFlyWrap + FDSpline/TDSpline TDIWaveformWrap
// moved to LAT. Class defs live in binding_lat_spline_tdi.hpp; the
// underlying GBTDIonTheFlyWrap + SOBBHTDIonTheFlyWrap below (still in
// this repo) inherit from LISATDIonTheFlyWrap via this include.
#include "binding_lat_spline_tdi.hpp"


// LISATDIonTheFlyWrap class moved to LAT at Phase 3L.6 (2026-06-03).
// Definition lives in lisatools/cutils/binding_lat_spline_tdi.hpp;
// included above.

// FDSplineTDIWaveformWrap class moved to LAT at Phase 3L.6 (2026-06-03).
// Definition + inline run_wave_tdi_wrap body live in
// lisatools/cutils/binding_lat_spline_tdi.hpp; included above.

// TDSplineTDIWaveformWrap class moved to LAT at Phase 3L.6 (2026-06-03).
// Definition + inline run_wave_tdi_wrap body live in
// lisatools/cutils/binding_lat_spline_tdi.hpp; included above.


// class GBTDIonTheFlyWrap moved to GBGPU at Phase 3L.7g (2026-06-04). Definition
// + run_*_wrap method bodies live in
// GBGPU/src/gbgpu/cutils/binding_gbgpu.{hpp,cxx}; pybind11 registration moved
// to GBGPU's cgbgpu module. The static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)
// in binding_tof.cxx still enforces single-registrant for the LAT-owned base.


// class SOBBHTDIonTheFlyWrap moved to BBHx at Phase 3L.8 (2026-06-04). Definition
// + run_wave_tdi_wrap method body live in
// BBHx/src/bbhx/cutils/binding_bbhx.{hpp,cxx}; pybind11 registration moved
// to BBHx's cbbhx module.


#if 0  // === WaveletLookupTableWrap disabled at Phase 3L (2026-06-02) -- lookup-table path retiring ===
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

    WaveletLookupTableWrap(array_type<double>c_nm_all_, array_type<double>s_nm_all_, int num_f_, int num_fdot_, double df_interp_, double dfdot_interp_, double min_f_, double min_fdot_, double layer_df_, double layer_dt_, int Nf_, int Nt_, int num_channel_, int ind_min_t_, int ind_max_t_, int ind_min_f_, int ind_max_f_, int m_ref_, int n_ref_, int kind_)
    {
        // PER_N tables are (Nt, num_fdot, num_f); N_REF_ONLY tables are
        // (num_fdot, num_f). Size-check the input arrays accordingly.
        size_t expected_len = (kind_ == LOOKUP_N_REF_ONLY)
            ? (size_t)num_fdot_ * (size_t)num_f_
            : (size_t)Nt_ * (size_t)num_fdot_ * (size_t)num_f_;
        wdm_lookup = new WaveletLookupTable(
            return_pointer_and_check_length(c_nm_all_, "c_nm_all", expected_len, 1),
            return_pointer_and_check_length(s_nm_all_, "s_nm_all", expected_len, 1),
            num_f_, num_fdot_, df_interp_, dfdot_interp_, min_f_, min_fdot_, layer_df_, layer_dt_, Nf_, Nt_, num_channel_, ind_min_t_, ind_max_t_, ind_min_f_, ind_max_f_, m_ref_, n_ref_, kind_
        );
    };
    ~WaveletLookupTableWrap(){
        delete wdm_lookup;
    };

    // Diagnostic: directly evaluate w_mn for caller-supplied (amp, phi, f, fdot, m, n).
    // Mirrors the C kernel post-conj/shift: tdi_channel_val = amp * (cos(phi) + i*sin(phi)).
    // CPU-only — used for unit tests against the Python lookup.
    void get_w_mn_arr(
        array_type<double> out,
        array_type<double> amp_arr,
        array_type<double> phi_arr,
        array_type<double> f_arr,
        array_type<double> fdot_arr,
        array_type<int> m_arr,
        array_type<int> n_arr,
        int N);
};
#endif  // === end WaveletLookupTableWrap disabled ===


// WDMSettingsWrap class moved to LAT at Phase 3L (2026-06-02). Definition
// lives in lisatools/cutils/binding_wdm_settings.hpp; included at the top
// of this header.


// WDMDomainWrap class moved to LAT at Phase 3L (2026-06-02). Definition
// lives in lisatools/cutils/binding_wdm_domain.hpp; included at the top
// of this header.


// FDDomainWrap: thin pybind11 holder for FDDomain, mirroring WDMDomainWrap.
// invC array layout depends on tdi_type:
//   tdi_type == TDI_XYZ   : (num_noise, num_channel, num_channel, n_rfft)
//   tdi_type == TDI_AET/AE: (num_noise, num_channel, n_rfft)
// FDDomainWrap class moved to LAT at Phase 3L (2026-06-02). Definition lives
// in lisatools/cutils/binding_fd_domain.hpp; included at the top of this header.


// class GBComputationGroupWrap moved to GBGPU at Phase 3L.7g (2026-06-04).
// Definition + FD / WDM-het / signal-het method bodies live in
// GBGPU/src/gbgpu/cutils/binding_gbgpu.{hpp,cxx}; pybind11 registration moved
// to GBGPU's cgbgpu module. SOBBHComputationGroupWrap below stays in this
// repo until Phase 3L.8 moves it to BBHx.




// class SOBBHComputationGroupWrap moved to BBHx at Phase 3L.8 (2026-06-04).
// Definition + sobbh_wdm_het_* method bodies live in
// BBHx/src/bbhx/cutils/binding_bbhx.{hpp,cxx}; pybind11 registration moved
// to BBHx's cbbhx module.


#endif // __BINDING_TOF_HPP__
