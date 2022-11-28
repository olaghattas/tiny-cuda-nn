cmake_minimum_required(VERSION 3.22)
project(matlab_bindings CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(tiny-cuda-nn REQUIRED)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Matlab REQUIRED)
set(MATLAB_INCLUDE_DIR $ENV{MATLAB_ROOT}/extern/include)
set(MATLAB_MEX_LIBRARY $ENV{MATLAB_ROOT}/bin/glnxa64)

if (WIN32) # 32-bit or 64-bit mex
  if (CMAKE_CL_64)
    SET(MEX_SUFFIX .mexw64)
  else (CMAKE_CL_64)
    SET(MEX_SUFFIX .mexw32)
  endif (CMAKE_CL_64)
else (WIN32)
  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
    SET(MEX_SUFFIX .mexa64)
  else (CMAKE_SIZEOF_VOID_P MATCHES "8")
    SET(MEX_SUFFIX .mexglx)
  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif (WIN32)


set(SRC_FILES matlab_bindings.cu)
add_library(binding_mex ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
target_link_libraries(binding_mex
    ${MATLAB_LIBRARIES}
    ${tiny-cuda-nn_LIBRARIES}
    )

target_include_directories(binding_mex PUBLIC "$ENV{MATLAB_ROOT}/extern/include")
target_include_directories(binding_mex PUBLIC "$ENV{MATLAB_ROOT}/toolbox/parallel/gpu/extern/include")
target_include_directories(binding_mex PUBLIC usr/local/include)
target_compile_options(binding_mex PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

set_target_properties(binding_mex PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(binding_mex PUBLIC ${TCNN_DEFINITIONS})
add_definitions(/DMATLAB_MEX_FILE)
add_definitions(/DMX_COMPAT_32)

install(TARGETS binding_mex DESTINATION ${CMAKE_SOURCE_DIR}/mex)
