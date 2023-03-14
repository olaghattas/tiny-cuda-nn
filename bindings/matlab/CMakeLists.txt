cmake_minimum_required(VERSION 3.22)
project(matlab_bindings)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
###############################################################################
# CUDA compiler setup
###############################################################################
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_ARCHITECTURES 86)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_EXTENSIONS OFF)
#set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

#find_package(CUDA)
#include(FindCUDA/select_compute_arch)
#CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
#string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
#string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")


find_package(CUDA)
find_package(tiny-cuda-nn REQUIRED)


#set(CMAKE_CUDA_COMPILE_OPTIONS_PIC -Xcompiler=-fPIC)
#set(CMAKE_CUDA_COMPILE_OPTIONS_PIE -Xcompiler=-fPIE)


add_definitions(/DMATLAB_MEX_FILE) #define matlab macros
add_definitions(/DMX_COMPAT_32)
add_definitions(${TCNN_DEFINITIONS})

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Matlab REQUIRED)
set(MATLAB_INCLUDE_DIR $ENV{MATLAB_ROOT}/extern/include)
set(MATLAB_MEX_LIBRARY $ENV{MATLAB_ROOT}/bin/glnxa64)

find_package(CUDALocal REQUIRED)


# set up cuda & matlab libraries
set(CU_FILE matlab_bindings)
set(CU_FILE_NAME ${CU_FILE}.cu)

#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_86,code="sm_86,compute_86" -DMATLAB_MEX_FILE -G -g -O0)
#list(APPEND CUDA_NVCC_FLAGS "-Xcompiler")
#list(APPEND CUDA_NVCC_FLAGS "-fPIC")

#list(APPEND CUDA_NVCC_FLAGS -DMATLAB_MEX_FILE)

#set(CUDA_NVCC_FLAGS -Xcompiler -fPIC -use_fast_math -gencode=arch=compute_86,code="sm_86,compute_86" --ptxas-options=-v -DMATLAB_MEX_FILE -G -g -O0)


list(APPEND MATLAB_INCLUDE_DIR "$ENV{MATLAB_ROOT}/toolbox/parallel/gpu/extern/include")
list(APPEND MATLAB_INCLUDE_DIR "$ENV{MATLAB_ROOT}/sys/cuda/glnxa64/cuda/include")

list(APPEND MATLAB_LIBRARIES "$ENV{MATLAB_ROOT}/bin/glnxa64/libmwgpu.so")
list(APPEND MATLAB_LIBRARIES "$ENV{MATLAB_ROOT}/bin/glnxa64/libMatlabDataArray.so")
list(APPEND MATLAB_LIBRARIES "$ENV{MATLAB_ROOT}/bin/glnxa64/libmat.so")
#list(APPEND MATLAB_LIBRARIES ${CUDA_CUDART_LIBRARY})
include_directories(${MATLAB_INCLUDE_DIR})
#include_directories(binding_mex PUBLIC )

cuda_compile(${CU_FILE} ${CU_FILE_NAME} SHARED)
add_library(${CU_FILE} SHARED ${${CU_FILE}} ${CU_FILE_NAME} ${CMAKE_SOURCE_DIR}/Matlabdef.def)
target_link_libraries(${CU_FILE}
    ${MATLAB_LIBRARIES}
#    ${CUDA_LIBRARIES}
    ${tiny-cuda-nn_LIBRARIES}
    fmt::fmt
    tiny-cuda-nn::tiny-cuda-nn
#    ${CUDA_CUDART_LIBRARY}
#    /usr/local/lib/libtiny-cuda-nn.a
    )

set_target_properties(${CU_FILE} PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
target_compile_options(${CU_FILE} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# 32-bit or 64-bit mex
if(WIN32)
  if (CMAKE_CL_64)
    set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexw64)
  else(CMAKE_CL_64)
    set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexw32)
  endif(CMAKE_CL_64)
else(WIN32)
  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
    set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexa64 PREFIX "")
  else(CMAKE_SIZEOF_VOID_P MATCHES "8")
    set_target_properties(${CU_FILE} PROPERTIES SUFFIX .mexglx PREFIX "")
  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif(WIN32)

install(TARGETS ${CU_FILE} DESTINATION ${CMAKE_SOURCE_DIR}/mex)

#
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
  message(STATUS "${_variableName}=${${_variableName}}")
endforeach ()



#add_library(binding_mex STATIC matlab_bindings.cu ${CMAKE_SOURCE_DIR}/Matlabdef.def)
#target_include_directories(binding_mex PUBLIC "$ENV{MATLAB_ROOT}/extern/include")
#target_include_directories(binding_mex PUBLIC "$ENV{MATLAB_ROOT}/toolbox/parallel/gpu/extern/include")
#target_include_directories(binding_mex PUBLIC usr/local/include)
##target_include_directories(binding_mex PUBLIC /usr/local/cuda/lib64)
#target_link_libraries(binding_mex
#    ${MATLAB_LIBRARIES}
#    ${tiny-cuda-nn_LIBRARIES}
##    ${CUDA_LIBRARIES}
##    ${CUDA_CUBLAS_LIBRARIES}
#    )
#
#
#target_compile_options(binding_mex PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
#
#set_target_properties(binding_mex PROPERTIES
#    CUDA_SEPARABLE_COMPILATION ON)
#target_compile_definitions(binding_mex PUBLIC ${TCNN_DEFINITIONS} ) # -DMATLAB_MEX_FILE
##set_target_properties(binding_mex PROPERTIES PREFIX "" LINKER_LANGUAGE CXX)
#
## 32-bit or 64-bit mex
#if(WIN32)
#  if (CMAKE_CL_64)
#    set_target_properties(binding_mex PROPERTIES SUFFIX .mexw64)
#  else(CMAKE_CL_64)
#    set_target_properties(binding_mex PROPERTIES SUFFIX .mexw32)
#  endif(CMAKE_CL_64)
#else(WIN32)
#  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
#    set_target_properties(binding_mex PROPERTIES SUFFIX .mexa64 PREFIX "")
#  else(CMAKE_SIZEOF_VOID_P MATCHES "8")
#    set_target_properties(binding_mex PROPERTIES SUFFIX .mexglx PREFIX "")
#  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
#endif(WIN32)
#add_definitions(/DMATLAB_MEX_FILE)
#add_definitions(/DMX_COMPAT_32)
#
