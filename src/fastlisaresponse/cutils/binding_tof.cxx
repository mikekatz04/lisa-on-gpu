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
#endif

namespace nb = nanobind;

// GBTDIonTheFlyWrap::run_*_wrap method bodies moved to GBGPU at
// Phase 3L.7g (2026-06-04). See GBGPU/src/gbgpu/cutils/binding_gbgpu.cxx.
// SOBBHTDIonTheFlyWrap::run_wave_tdi_wrap method body moved to BBHx at
// Phase 3L.8 (2026-06-04). See BBHx/src/bbhx/cutils/binding_bbhx.cxx.

// moved to GBGPU at Phase 3L.7g (2026-06-04). See

// GBGPU/src/gbgpu/cutils/binding_gbgpu.cxx. SOBBH equivalents below stay.

// SOBBHComputationGroupWrap::sobbh_wdm_het_* method bodies moved to BBHx
// at Phase 3L.8 (2026-06-04). See BBHx/src/bbhx/cutils/binding_bbhx.cxx.


std::string get_module_path_tdionthefly() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    nb::gil_scoped_acquire acquire;
    // Import the module by its name
    nb::object module = nb::module_::import_("tdionthefly");
    try {
        std::string path = nb::cast<std::string>(module.attr("__file__"));
        return path;
    } catch (const nb::python_error& e) {
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}


// NB_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void tdionthefly_part(nb::module_ &m) {

    m.attr("TDI_XYZ") = TDI_XYZ;
    m.attr("TDI_AET") = TDI_AET;
    m.attr("TDI_AE") = TDI_AE;
    
    // Phase 3L.6 (2026-06-03): FDSplineTDIWaveform[Wrap] + TDSplineTDIWaveform[Wrap]
    // pybind11 registrations moved to LAT's binding_flr.cxx (registered in
    // pycppdetector via response_part(m)). The
    // static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...) at the top of this TU
    // guards against any future re-registration here.


// nb::class_<GBTDIonTheFlyWrap> + nb::class_<GBTDIonTheFly> registrations

// moved to GBGPU's cgbgpu module at Phase 3L.7g (2026-06-04).


// nb::class_<SOBBHTDIonTheFlyWrap> + nb::class_<SOBBHTDIonTheFly>
// registrations moved to BBHx's cbbhx module at Phase 3L.8 (2026-06-04).


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

// nb::class_<GBComputationGroupWrap> registration moved to GBGPU's

// cgbgpu module at Phase 3L.7g (2026-06-04).

// nb::class_<SOBBHComputationGroupWrap> registration moved to BBHx's cbbhx
// module at Phase 3L.8 (2026-06-04).

}



NB_MODULE(tdionthefly, m) {
     m.doc() = "TDI on the Fly."; // Optional module docstring

    // Call initialization functions from other files
    tdionthefly_part(m);
    
    m.def("get_module_path_cpp", &get_module_path_tdionthefly, "Returns the file path of the module");

    // Optionally, get the path during module initialization and store it
    // This can cause an AttributeError if not handled carefully, as m.attr("__file__")
    // might not be fully set during the initial call if the module is loaded in
    // a specific way (e.g., via pythonw or as a namespace package).
    try {
        std::string path_at_init = nb::cast<std::string>(m.attr("__file__"));
        // std::cout << "Module loaded from: " << path_at_init << std::endl;
        m.attr("module_dir") = nb::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (nb::python_error &e) {
         // Handle potential error here, e.g., by logging or setting a default value
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore(); // Restore exception state for proper Python handling
        PyErr_Clear();
    }
}

