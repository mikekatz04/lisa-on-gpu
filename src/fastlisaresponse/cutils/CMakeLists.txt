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

message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*")
message(STATUS "LISATOOLS Source files: ${MY_FILES}")
message(STATUS "LISATOOLS version: ${LISATOOLS_VERSION}")

# I. Process fastlisaresponse_utils_wrap.pyx into a C++ file
add_custom_command(
  OUTPUT "responselisa.cxx"
  COMMENT "Cythonize responselisa.pyx into responselisa.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/responselisa.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/responselisa.cxx" -3 -+ --module-name
    "responselisa" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "responselisa.pyx"
  VERBATIM)

  message(STATUS "CHECKING: ${LISATOOLS_DIR}/cutils/global.hpp  ${CMAKE_CURRENT_SOURCE_DIR}/global.hpp")
  

  
set(FILES_TO_COPY 
  "${LISATOOLS_DIR}/cutils/global.hpp"
  "${LISATOOLS_DIR}/cutils/Detector.hpp"
  "${LISATOOLS_DIR}/cutils/Detector.cu"
)
# MY_LIST will be "string1;string2;another string"

foreach(FILE_TO_COPY IN LISTS FILES_TO_COPY)
  get_filename_component(NEW_FILE "${FILE_TO_COPY}" NAME)
  file(COPY_FILE "${FILE_TO_COPY}" "${CMAKE_CURRENT_SOURCE_DIR}/${NEW_FILE}"
     RESULT copy_result)
  if(NOT copy_result EQUAL 0)
      message(FATAL_ERROR "Failed to copy file: ${copy_result} (${FILE_TO_COPY})")
  endif()
endforeach()

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
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "Detector.cu"
    VERBATIM)

  python_add_library(fastlisaresponse_cpu_responselisa MODULE WITH_SOABI Detector.cxx responselisa.cxx LISAResponse.cxx)
  apply_cpu_backend_common_options(responselisa)

  target_sources(fastlisaresponse_cpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         global.hpp Detector.hpp cuda_complex.hpp LISAResponse.hh)
endif()

# III. Declare the GPU backend
if(FASTLISA_WITH_GPU)


  message(STATUS "This is the cuda archs to be run: ${FASTLISA_CUDA_ARCH}")
  
  python_add_library(fastlisaresponse_gpu_responselisa MODULE WITH_SOABI responselisa.cxx Detector.cu LISAResponse.cu)
  apply_gpu_backend_common_options(responselisa)
  target_sources(fastlisaresponse_gpu_responselisa PUBLIC FILE_SET HEADERS FILES
                                         global.hpp Detector.hpp cuda_complex.hpp LISAResponse.hh
                                         )
  target_link_libraries(fastlisaresponse_gpu_responselisa PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  #set_target_properties(fastlisaresponse_gpu_responselisa PROPERTIES
  #                    CUDA_ARCHITECTURES ${FASTLISA_CUDA_ARCH}
  #                    CUDA_RESOLVE_DEVICE_SYMBOLS ON # <=================== Magic happens here <|
  #                    POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_ARCHITECTURES ${FASTLISA_CUDA_ARCH})
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET fastlisaresponse_gpu_responselisa PROPERTY POSITION_INDEPENDENT_CODE ON)
  message(STATUS "CHECK: ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/fastlisaresponse")

  install(TARGETS fastlisaresponse_gpu_responselisa
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/fastlisaresponse)
endif()


