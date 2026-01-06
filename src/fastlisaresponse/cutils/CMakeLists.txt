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

execute_process(
COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
  from lisatools_backend_cpu import pycppdetector
  print(pycppdetector.__file__)"
OUTPUT_VARIABLE LISATOOLS_DETECTOR_CPU
OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*")
message(STATUS "LISATOOLS Source files: ${MY_FILES}")
message(STATUS "LISATOOLS version: ${LISATOOLS_VERSION}")
message(STATUS "LISATOOLS_DETECTOR_CPU: ${LISATOOLS_DETECTOR_CPU}")

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

  add_library(fastlisaresponse_cpu_response_static STATIC LISAResponse.cxx)
  apply_cpu_backend_common_options(response fastlisaresponse fastlisaresponse true )
  target_include_directories(fastlisaresponse_cpu_response_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  target_link_libraries(fastlisaresponse_cpu_response_static PUBLIC ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  target_link_directories(fastlisaresponse_cpu_response_static PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})
  target_sources(fastlisaresponse_cpu_response_static PUBLIC FILE_SET HEADERS FILES
                                        LISAResponse.hh)

  pybind11_add_module(fastlisaresponse_cpu_responselisa binding.cxx)
  apply_cpu_backend_common_options(responselisa fastlisaresponse fastlisaresponse false)
  
  file(GLOB MY_SHARED "${LISATOOLS_DETECTOR_CPU_DIR}/*.a")
  message(STATUS "LISATOOLS lib files: ${MY_SHARED}")

  # Link your target to the imported target name
  target_link_libraries(fastlisaresponse_cpu_responselisa PUBLIC fastlisaresponse_cpu_response_static ${LISATOOLS_DETECTOR_STATIC_LIB_CPU})
  target_link_directories(fastlisaresponse_cpu_responselisa PUBLIC ${LISATOOLS_DETECTOR_CPU_DIR})  
  target_include_directories(fastlisaresponse_cpu_responselisa PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  target_sources(fastlisaresponse_cpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                          binding.hpp)


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

  add_library(fastlisaresponse_gpu_response_gpu_static STATIC LISAResponse.cu)
  apply_gpu_backend_common_options(response_gpu fastlisaresponse fastlisaresponse true)
  target_sources(fastlisaresponse_gpu_response_gpu_static PUBLIC FILE_SET HEADERS FILES
                                         LISAResponse.hh)

  target_include_directories(fastlisaresponse_gpu_response_gpu_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})
  target_link_libraries(fastlisaresponse_gpu_response_gpu_static PUBLIC ${DETECTOR_GPU_OBJ}) 
  
  pybind11_add_module(fastlisaresponse_gpu_responselisa binding.cxx)
  apply_gpu_backend_common_options(responselisa fastlisaresponse fastlisaresponse false)
  target_sources(fastlisaresponse_gpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         LISAResponse.hh
                                         )
  target_link_libraries(fastlisaresponse_gpu_responselisa PUBLIC ${LISATOOLS_DETECTOR_STATIC_LIB_GPU} fastlisaresponse_gpu_response_gpu_static)
  target_link_directories(fastlisaresponse_gpu_responselisa PUBLIC ${LISATOOLS_DETECTOR_GPU_DIR})  
  target_include_directories(fastlisaresponse_gpu_responselisa PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  
endif()


