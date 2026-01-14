# =============================
# ==== BACKEND COMPILATION ====
# =============================

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
set(GBT_LAPACK_DETECT_WITH ${FASTLISA_LAPACK_DETECT_WITH})

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
    import gpubackendtools
    _tmp = gpubackendtools.__file__
    print(_tmp.split('/__init__.py')[0])"
  OUTPUT_VARIABLE GBT_DIR
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
# message(STATUS "LISATOOLS_DETECTOR_CPU: ${LISATOOLS_DETECTOR_CPU}")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools._version.version)"
  OUTPUT_VARIABLE GBT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from lisatools_backend_cpu import pycppdetector
    print(pycppdetector.__file__)"
  OUTPUT_VARIABLE LISATOOLS_DETECTOR_CPU
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from gbt_backend_cpu import interp
    print(interp.__file__)"
  OUTPUT_VARIABLE GBT_INTERP_CPU
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "TEST CHECK 1: ${GBT_INTERP_CPU}")

message(STATUS "GBT version: ${GBT_VERSION}")

message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
message(STATUS "GBT location is ${GBT_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*" "${GBT_DIR}/cutils/*")
message(STATUS "Source files: ${MY_FILES}")

if(NOT EXISTS "${LISATOOLS_DIR}/cutils/binding.hpp")
  message(FATAL_ERROR "${LISATOOLS_DIR}/cutils/binding.hpp does not exist.")
endif()

if(NOT EXISTS "${GBT_DIR}/cutils/gbt_binding.hpp")
  message(FATAL_ERROR "${GBT_DIR}/cutils/gbt_binding.hpp does not exist.")
endif()

get_filename_component(LISATOOLS_DETECTOR_CPU_DIR ${LISATOOLS_DETECTOR_CPU} DIRECTORY)
get_filename_component(LISATOOLS_DETECTOR_CPU_LIB_NAME ${LISATOOLS_DETECTOR_CPU} NAME)
message(STATUS "dir: ${LISATOOLS_DETECTOR_CPU_DIR}")

find_library(LISATOOLS_DETECTOR_STATIC_LIB_CPU
    NAMES detector # The name of the library (without 'lib' prefix or '.a' extension)
    PATHS ${LISATOOLS_DETECTOR_CPU_DIR} # The specific folder to check
    NO_DEFAULT_PATH_LOGIC # Optional: Restricts search to only the provided PATHS
)

if(LISATOOLS_DETECTOR_STATIC_LIB_CPU)
    message("Found static library at: ${LISATOOLS_DETECTOR_STATIC_LIB_CPU}")
    # You can now use the library, e.g., add_library(MyImportedLib STATIC IMPORTED)
    # and set its location property. Or link directly in target_link_libraries
else()
    message(FATAL_ERROR "LISA Analysis Tools detector static library not found in the specified folder!")
endif()

message(STATUS "TEST CHECK: ${GBT_INTERP_CPU}")
get_filename_component(GBT_INTERP_CPU_DIR ${GBT_INTERP_CPU} DIRECTORY)
get_filename_component(GBT_INTERP_CPU_LIB_NAME ${GBT_INTERP_CPU} NAME)
message(STATUS "dir: ${GBT_INTERP_CPU_DIR}")

find_library(GBT_INTERP_STATIC_LIB_CPU
    NAMES interp # The name of the library (without 'lib' prefix or '.a' extension)
    PATHS ${GBT_INTERP_CPU_DIR} # The specific folder to check
    NO_DEFAULT_PATH_LOGIC # Optional: Restricts search to only the provided PATHS
)

if(GBT_INTERP_STATIC_LIB_CPU)
    message("Found static library at: ${GBT_INTERP_STATIC_LIB_CPU}")
    # You can now use the library, e.g., add_library(MyImportedLib STATIC IMPORTED)
    # and set its location property. Or link directly in target_link_libraries
else()
    message(FATAL_ERROR "GBT interp static library not found in the specified folder!")
endif()

# interp needs to link against LAPACKE. USe the following method to detect
# it and obtain ${GBT_LAPACKE_LIBS}. In case of failure, it will stop the
# processing execution.
enable_language(Fortran)
get_lapacke()

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

  add_library(fastlisaresponse_cpu_response_static STATIC Detector.cxx LISAResponse.cxx)
  apply_cpu_backend_common_options(response fastlisaresponse fastlisaresponse true )
  target_include_directories(fastlisaresponse_cpu_response_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  # target_link_libraries(fastlisaresponse_cpu_response_static PUBLIC ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  # target_link_directories(fastlisaresponse_cpu_response_static PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})
  target_sources(fastlisaresponse_cpu_response_static PUBLIC FILE_SET HEADERS FILES
                                        LISAResponse.hh)

  pybind11_add_module(fastlisaresponse_cpu_responselisa "${LISATOOLS_DIR}/cutils/binding.cxx" binding_flr.cxx)
  apply_cpu_backend_common_options(responselisa fastlisaresponse fastlisaresponse false)
  
  # Link your target to the imported target name
  target_link_libraries(fastlisaresponse_cpu_responselisa PUBLIC fastlisaresponse_cpu_response_static)  #  ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  # target_link_directories(fastlisaresponse_cpu_responselisa PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})  
  target_include_directories(fastlisaresponse_cpu_responselisa PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  target_sources(fastlisaresponse_cpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                          binding_flr.hpp)

endif()

# III. Declare the GPU backend
if(FASTLISA_WITH_GPU)

  execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from lisatools_backend_cuda12x import pycppdetector
    print(pycppdetector.__file__)"
  OUTPUT_VARIABLE LISATOOLS_DETECTOR_GPU
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  message(STATUS "LISATOOLS_DETECTOR_GPU: ${LISATOOLS_DETECTOR_GPU}")

  get_filename_component(LISATOOLS_DETECTOR_GPU_DIR ${LISATOOLS_DETECTOR_GPU} DIRECTORY)
  get_filename_component(LISATOOLS_DETECTOR_GPU_LIB_NAME ${LISATOOLS_DETECTOR_GPU} NAME)
  message(STATUS "dir: ${LISATOOLS_DETECTOR_GPU_DIR}")
  
  file(GLOB MY_SHARED_GPU "${LISATOOLS_DETECTOR_GPU_DIR}/*.a")
  message(STATUS "LISATOOLS lib files: ${MY_SHARED_GPU}")

  find_library(LISATOOLS_DETECTOR_STATIC_LIB_GPU
      NAMES detector_gpu # The name of the library (without 'lib' prefix or '.a' extension)
      PATHS ${LISATOOLS_DETECTOR_GPU_DIR} # The specific folder to check
      NO_DEFAULT_PATH_LOGIC # Optional: Restricts search to only the provided PATHS
  )

  if(LISATOOLS_DETECTOR_STATIC_LIB_GPU)
      message("Found static library at: ${LISATOOLS_DETECTOR_STATIC_LIB_GPU}")
      # You can now use the library, e.g., add_library(MyImportedLib STATIC IMPORTED)
      # and set its location property. Or link directly in target_link_libraries
  else()
      message(FATAL_ERROR "LISA Analysis Tools detector (gpu) static library not found in the specified folder!")
  endif()

  set(DETECTOR_GPU_OBJ "${LISATOOLS_DETECTOR_GPU_DIR}/Detector.cu.o")

  if(EXISTS "${DETECTOR_GPU_OBJ}")
    message("INFO: The file ${DETECTOR_GPU_OBJ} exists.")
    # Proceed with configuration, e.g., add the file to a target
    # add_executable(MyTarget ${DETECTOR_GPU_OBJ})
  else()
    message(STATUS "The file ${DETECTOR_GPU_OBJ} does not exist. We will unpack the installed library.")
    execute_process (
      COMMAND bash -c "cd ${LISATOOLS_DETECTOR_GPU_DIR} && ar -x libdetector_gpu.a Detector.cu.o && cd -"
    )
    # Handle the missing file, e.g., set a variable, issue a warning, or a fatal error
    # message(FATAL_ERROR "Required file not found!")
    if(NOT EXISTS "${DETECTOR_GPU_OBJ}")
      message(FATAL_ERROR "${DETECTOR_GPU_OBJ} creation not working.")
    endif()
  endif()

  add_library(fastlisaresponse_gpu_response_gpu_static STATIC "${LISATOOLS_DIR}/cutils/Detector.cu" LISAResponse.cu)
  apply_gpu_backend_common_options(response_gpu fastlisaresponse fastlisaresponse true)
  target_sources(fastlisaresponse_gpu_response_gpu_static PUBLIC FILE_SET HEADERS FILES
                                         LISAResponse.hh)

  target_include_directories(fastlisaresponse_gpu_response_gpu_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})
  # target_link_libraries(fastlisaresponse_gpu_response_gpu_static PUBLIC ${DETECTOR_GPU_OBJ}) 
  
  pybind11_add_module(fastlisaresponse_gpu_responselisa "${LISATOOLS_DIR}/cutils/binding.cxx" binding_flr.cxx)
  apply_gpu_backend_common_options(responselisa fastlisaresponse fastlisaresponse false)
  target_sources(fastlisaresponse_gpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         LISAResponse.hh
                                         )
  target_link_libraries(fastlisaresponse_gpu_responselisa PUBLIC fastlisaresponse_gpu_response_gpu_static)
  # target_link_directories(fastlisaresponse_gpu_responselisa PUBLIC ${LISATOOLS_DETECTOR_GPU_DIR})  
  target_include_directories(fastlisaresponse_gpu_responselisa PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  
endif()

#####################
## tdi on the fly  ##
#####################


# II. Declare the CPU backend
if(FASTLISA_WITH_CPU)
  add_custom_command(
    OUTPUT "TDIonTheFly.cxx"
    COMMENT "Copy TDIonTheFly.cu to TDIonTheFly.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/TDIonTheFly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/TDIonTheFly.cxx"
    DEPENDS "TDIonTheFly.cu"
    VERBATIM)

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

  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy Interpolate.cu to Interpolate.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${GBT_DIR}/cutils/Interpolate.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "${GBT_DIR}/cutils/Interpolate.cu"
    VERBATIM)

  add_library(fastlisaresponse_cpu_tdionthefly_static STATIC Interpolate.cxx Detector.cxx LISAResponse.cxx TDIonTheFly.cxx)
  apply_cpu_backend_common_options(tdionthefly fastlisaresponse fastlisaresponse true )
  message(STATUS "GBT_LAPACKE_INCLUDE_DIR: ${GBT_LAPACKE_INCLUDE_DIR}")
  target_include_directories(fastlisaresponse_cpu_tdionthefly_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS} ${GBT_LAPACKE_INCLUDE_DIR})  
  # target_link_libraries(fastlisaresponse_cpu_tdionthefly_static PUBLIC ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  # target_link_directories(fastlisaresponse_cpu_tdionthefly_static PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})
  target_sources(fastlisaresponse_cpu_tdionthefly_static PUBLIC FILE_SET HEADERS FILES
                                        TDIonTheFly.hh)
  target_link_libraries(fastlisaresponse_cpu_tdionthefly_static LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})

  pybind11_add_module(fastlisaresponse_cpu_tdionthefly "${LISATOOLS_DIR}/cutils/binding.cxx" "${GBT_DIR}/cutils/gbt_binding.cxx" binding_flr.cxx binding_tof.cxx)
  apply_cpu_backend_common_options(tdionthefly fastlisaresponse fastlisaresponse false)
  
  # Link your target to the imported target name
  target_link_libraries(fastlisaresponse_cpu_tdionthefly PUBLIC fastlisaresponse_cpu_tdionthefly_static)  #  ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  # target_link_directories(fastlisaresponse_cpu_tdionthefly PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})  
  target_include_directories(fastlisaresponse_cpu_tdionthefly PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS} ${GBT_LAPACKE_INCLUDE_DIR})  
  target_sources(fastlisaresponse_cpu_tdionthefly PUBLIC FILE_SET HEADERS FILES
                                          binding_tof.hpp)
  target_link_libraries(fastlisaresponse_cpu_tdionthefly LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})



endif()

# IV. Remove downloaded files. 
# file(REMOVE "./src/fastlisaresponse/cutils/Detector.cu")
# file(REMOVE "./src/fastlisaresponse/cutils/Detector.hpp")
# file(REMOVE "./src/fastlisaresponse/cutils/global.hpp")
# file(REMOVE "./src/fastlisaresponse/cutils/Interpolate.cu")
# file(REMOVE "./src/fastlisaresponse/cutils/interpolate.hh")



