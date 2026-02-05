# =============================
# ==== BACKEND COMPILATION ====
# =============================

# TODO: how can we avoid moving source and header files in this directory ?
#include_directories(BEFORE "include")
#include_directories(BEFORE "src")

# In the project root CMakeLists.txt, we defined a "fastlisaresponse" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(FASTLISA_WITH_CPU fastlisaresponse WITH_CPU)
get_target_property(FASTLISA_WITH_GPU fastlisaresponse WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
# TODO: is this needed ?
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${fastlisaresponse_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE GBT_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(GBT_CMAKE_TOOLS "${GBT_CUTILS}/cmake_functions.cmake")

# Check if the file exists
if(EXISTS "${GBT_CMAKE_TOOLS}")
    message("INFO: ${GBT_CMAKE_TOOLS} exists.")
else()
    message(FATAL_ERROR "${GBT_CMAKE_TOOLS} not found.")
    # Handle the case where the file is missing
endif()

include(${GBT_CMAKE_TOOLS})
set(HERE_CUDA_ARCH ${FASTLISA_CUDA_ARCH})

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- responselisa ---
# ----------------


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    _tmp = lisatools.__file__
    print(_tmp.split('/__init__.py')[0])"
  OUTPUT_VARIABLE LISATOOLS_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    print(lisatools._version.version)"
  OUTPUT_VARIABLE LISATOOLS_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)


message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*")
message(STATUS "LISATOOLS Source files: ${MY_FILES}")
message(STATUS "LISATOOLS version: ${LISATOOLS_VERSION}")

# I. Process fastlisaresponse_utils_wrap.pyx into a C++ file

# II. Declare the CPU backend
if(FASTLISA_WITH_CPU)
  add_custom_command(
    OUTPUT "LISAResponse.cxx"
    COMMENT "Copy LISAResponse.cu to LISAResponse.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/LISAResponse.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/LISAResponse.cxx"
    DEPENDS "LISAResponse.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_DIR}/cutils/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "${LISATOOLS_DIR}/cutils/Detector.cu"
    VERBATIM)

  if(NOT EXISTS "${LISATOOLS_DIR}/cutils/binding.hpp")
    message(FATAL_ERROR "${LISATOOLS_DIR}/cutils/binding.hpp does not exist.")
  endif()

  add_library(fastlisaresponse_cpu_response_static STATIC Detector.cxx LISAResponse.cxx)
  apply_cpu_backend_common_options(response fastlisaresponse fastlisaresponse true )
  target_include_directories(fastlisaresponse_cpu_response_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  target_sources(fastlisaresponse_cpu_response_static PUBLIC FILE_SET HEADERS FILES
                                        LISAResponse.hh)

  pybind11_add_module(fastlisaresponse_cpu_responselisa "${LISATOOLS_DIR}/cutils/binding.cxx" binding_flr.cxx)
  apply_cpu_backend_common_options(responselisa fastlisaresponse fastlisaresponse false)
  
  # Link your target to the imported target name
  target_link_libraries(fastlisaresponse_cpu_responselisa PUBLIC fastlisaresponse_cpu_response_static)
  target_include_directories(fastlisaresponse_cpu_responselisa PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  target_sources(fastlisaresponse_cpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                          binding_flr.hpp)


endif()

# III. Declare the GPU backend
if(FASTLISA_WITH_GPU)

  add_library(fastlisaresponse_gpu_response_gpu_static STATIC "${LISATOOLS_DIR}/cutils/Detector.cu" LISAResponse.cu)
  apply_gpu_backend_common_options(response_gpu fastlisaresponse fastlisaresponse true)
  target_sources(fastlisaresponse_gpu_response_gpu_static PUBLIC FILE_SET HEADERS FILES
                                         LISAResponse.hh)

  target_include_directories(fastlisaresponse_gpu_response_gpu_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})
  pybind11_add_module(fastlisaresponse_gpu_responselisa "${LISATOOLS_DIR}/cutils/binding.cxx" binding_flr.cxx)
  apply_gpu_backend_common_options(responselisa fastlisaresponse fastlisaresponse false)
  target_sources(fastlisaresponse_gpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         LISAResponse.hh
                                         )
  target_link_libraries(fastlisaresponse_gpu_responselisa PUBLIC fastlisaresponse_gpu_response_gpu_static)
  target_include_directories(fastlisaresponse_gpu_responselisa PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  
endif()


