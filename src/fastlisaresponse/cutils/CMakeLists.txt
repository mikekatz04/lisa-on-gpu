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

# apply_gpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to GPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "fastlisaresponse_gpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the GPU backend directory (e.g.
#   'fastlisaresponse_backend_cuda12x')
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
# * Ensures the target links against CUDA libraries (cuBLAS, cuSPARSE, ...)
# * Defines the CUDA_ARCHITECTURE property
#
# Usage example: apply_gpu_backend_common_options(fastlisaresponse_utils_wrap)
function(apply_gpu_backend_common_options libname)
  set(target_name "fastlisaresponse_gpu_${libname}")
  set(backend_name "fastlisaresponse_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${backend_name}")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION ${backend_name})

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
  target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES
                                              ${FASTLISA_CUDA_ARCH})
  message(STATUS "This is the cuda archs to be run: ${FASTLISA_CUDA_ARCH}")
endfunction()

function(apply_gpu_backend_common_options_static libname)
  set(target_name "fastlisaresponse_gpu_${libname}_static")
  set(backend_name "fastlisaresponse_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${backend_name}")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION ${backend_name})

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
  target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES
                                              ${FASTLISA_CUDA_ARCH})
  message(STATUS "This is the cuda archs to be run: ${FASTLISA_CUDA_ARCH}")
endfunction()

function(apply_cpu_backend_common_options libname)
  set(target_name "fastlisaresponse_cpu_${libname}")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/fastlisaresponse_backend_cpu")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION fastlisaresponse_backend_cpu)

  get_target_property(FASTLISA_CXX_MARCH_OPT fastlisaresponse CXX_MARCH)
  if(FASTLISA_CXX_MARCH_OPT)
    target_compile_options(${target_name} PRIVATE "${FASTLISA_CXX_MARCH_OPT}")
  endif()

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
endfunction()


function(apply_cpu_backend_common_options_static libname)
  set(target_name "fastlisaresponse_cpu_${libname}_static")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/fastlisaresponse_backend_cpu")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION fastlisaresponse_backend_cpu)

  get_target_property(FASTLISA_CXX_MARCH_OPT fastlisaresponse CXX_MARCH)
  if(FASTLISA_CXX_MARCH_OPT)
    target_compile_options(${target_name} PRIVATE "${FASTLISA_CXX_MARCH_OPT}")
  endif()

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
endfunction()

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

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

  add_library(fastlisaresponse_cpu_response_static STATIC LISAResponse.cxx)
  apply_cpu_backend_common_options_static(response)
  target_link_directories(fastlisaresponse_cpu_response_static PRIVATE ${LISATOOLS_DETECTOR_CPU_DIR})
  target_include_directories(fastlisaresponse_cpu_response_static PRIVATE "${LISATOOLS_DIR}/cutils/")  
  target_link_libraries(fastlisaresponse_cpu_response_static PUBLIC detector)
  target_sources(fastlisaresponse_cpu_response_static PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp LISAResponse.hh)


  pybind11_add_module(fastlisaresponse_cpu_responselisa LISAResponse.cxx)
  apply_cpu_backend_common_options(responselisa)
  
  file(GLOB MY_SHARED "${LISATOOLS_DETECTOR_CPU_DIR}/*.a")
  message(STATUS "LISATOOLS lib files: ${MY_SHARED}")

  # Link your target to the imported target name
  target_link_libraries(fastlisaresponse_cpu_responselisa PUBLIC detector fastlisaresponse_cpu_response_static)
  target_link_directories(fastlisaresponse_cpu_responselisa PRIVATE ${LISATOOLS_DETECTOR_CPU_DIR})  
  target_include_directories(fastlisaresponse_cpu_responselisa PRIVATE "${LISATOOLS_DIR}/cutils/")  
  target_sources(fastlisaresponse_cpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                          cuda_complex.hpp LISAResponse.hh)


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

  add_library(fastlisaresponse_gpu_response_gpu_static STATIC LISAResponse.cu)
  apply_gpu_backend_common_options_static(response_gpu)
  target_sources(fastlisaresponse_gpu_response_gpu_static PUBLIC FILE_SET HEADERS FILES
                                         LISAResponse.hh)

  target_link_libraries(fastlisaresponse_gpu_response_gpu_static PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  target_link_directories(fastlisaresponse_gpu_response_gpu_static PRIVATE ${LISATOOLS_DETECTOR_GPU_DIR})
  target_include_directories(fastlisaresponse_gpu_response_gpu_static PRIVATE "${LISATOOLS_DIR}/cutils/")  
  target_link_libraries(fastlisaresponse_gpu_response_gpu_static PUBLIC detector_gpu)
  
  message(STATUS "This is the cuda archs to be run: ${FASTLISA_CUDA_ARCH}")
  
  pybind11_add_module(fastlisaresponse_gpu_responselisa LISAResponse.cu)
  apply_gpu_backend_common_options(responselisa)
  target_sources(fastlisaresponse_gpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp LISAResponse.hh
                                         )
  target_link_libraries(fastlisaresponse_gpu_responselisa PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  target_link_libraries(fastlisaresponse_gpu_responselisa PUBLIC detector_gpu fastlisaresponse_gpu_response_static)
  target_link_directories(fastlisaresponse_gpu_responselisa PRIVATE ${LISATOOLS_DETECTOR_GPU_DIR})  
  target_include_directories(fastlisaresponse_gpu_responselisa PRIVATE "${LISATOOLS_DIR}/cutils/")  
  
  #set_target_properties(fastlisaresponse_gpu_responselisa PROPERTIES
  #                    CUDA_ARCHITECTURES ${FASTLISA_CUDA_ARCH}
  #                    CUDA_RESOLVE_DEVICE_SYMBOLS ON # <=================== Magic happens here <|
  #                    POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_ARCHITECTURES ${FASTLISA_CUDA_ARCH})
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY POSITION_INDEPENDENT_CODE ON)
  message(STATUS "CHECK: ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/fastlisaresponse")

endif()


