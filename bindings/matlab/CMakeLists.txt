cmake_minimum_required(VERSION 3.22)
project(matlab_bindings)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(tiny-cuda-nn REQUIRED)
#find_package(CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Matlab REQUIRED)
set(MATLAB_INCLUDE_DIR $ENV{MATLAB_ROOT}/extern/include)
set(MATLAB_MEX_LIBRARY $ENV{MATLAB_ROOT}/bin/glnxa64)

add_library(binding_mex STATIC matlab_bindings.cu ${CMAKE_SOURCE_DIR}/Matlabdef.def)
target_include_directories(binding_mex PUBLIC "$ENV{MATLAB_ROOT}/extern/include")
target_include_directories(binding_mex PUBLIC "$ENV{MATLAB_ROOT}/toolbox/parallel/gpu/extern/include")
target_include_directories(binding_mex PUBLIC usr/local/include)
#target_include_directories(binding_mex PUBLIC /usr/local/cuda/lib64)
target_link_libraries(binding_mex
    ${MATLAB_LIBRARIES}
    ${tiny-cuda-nn_LIBRARIES}
#    ${CUDA_LIBRARIES}
#    ${CUDA_CUBLAS_LIBRARIES}
    )


target_compile_options(binding_mex PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

set_target_properties(binding_mex PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(binding_mex PUBLIC ${TCNN_DEFINITIONS} ) # -DMATLAB_MEX_FILE
#set_target_properties(binding_mex PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)

# 32-bit or 64-bit mex
if(WIN32)
  if (CMAKE_CL_64)
    set_target_properties(binding_mex PROPERTIES SUFFIX .mexw64)
  else(CMAKE_CL_64)
    set_target_properties(binding_mex PROPERTIES SUFFIX .mexw32)
  endif(CMAKE_CL_64)
else(WIN32)
  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
    set_target_properties(binding_mex PROPERTIES SUFFIX .mexa64 PREFIX "")
  else(CMAKE_SIZEOF_VOID_P MATCHES "8")
    set_target_properties(binding_mex PROPERTIES SUFFIX .mexglx PREFIX "")
  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif(WIN32)
add_definitions(/DMATLAB_MEX_FILE)
add_definitions(/DMX_COMPAT_32)

install(TARGETS binding_mex DESTINATION ${CMAKE_SOURCE_DIR}/mex)


get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
  message(STATUS "${_variableName}=${${_variableName}}")
endforeach ()
