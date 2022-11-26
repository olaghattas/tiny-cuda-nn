project(matlab_bindings)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(MATLAB_INCLUDE_DIR $ENV{MATLAB_ROOT}/extern/include)
set(MATLAB_MEX_LIBRARY $ENV{MATLAB_ROOT}/bin/glnxa64)

#message(SDbdfbdfbdsb ${MATLAB_INCLUDE_DIR})

find_package(Matlab REQUIRED)

#set(CMAKE_PREFIX_PATH /usr/local/lib)
#set(CMAKE_PREFIX_PATH /usr/local/lib)
find_package(tiny-cuda-nn REQUIRED)


#ADD_DEFINITIONS(-DUSE_MEX)
#INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})
#message("MATLAB_INCLUDE_DIR   " ${MATLAB_INCLUDE_DIR})
#message("MATLAB_LIBRARIES   " ${MATLAB_LIBRARIES})

if(WIN32) # 32-bit or 64-bit mex
  if (CMAKE_CL_64)
    SET( MEX_SUFFIX .mexw64 )
  else(CMAKE_CL_64)
    SET( MEX_SUFFIX .mexw32 )
  endif(CMAKE_CL_64)
else(WIN32)
  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
    SET( MEX_SUFFIX .mexa64 )
  else(CMAKE_SIZEOF_VOID_P MATCHES "8")
    SET( MEX_SUFFIX .mexglx )
  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
endif(WIN32)

#SET( MEX_SUFFIX .mexa64 )





FIND_PACKAGE(CUDA REQUIRED)
enable_language(CUDA)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif ()


#add definition USE_MEX for C++ code, to be able to use preprocessor if CUDA is not used
#ADD_DEFINITIONS(-DUSE_CUDA)
#set(CUDA_CUBLAS_LIBRARIES "/media/paul/MATLAB/installation/MATLAB/R2022a/bin/glnxa64/libcublas.so.11")
#set(CUDA_CUFFT_LIBRARIES "/media/paul/MATLAB/installation/MATLAB/R2022a/bin/glnxa64/libcufft.so.10")
SET(CUDA_LIBS CUDA_CUBLAS_LIBRARIES CUDA_CUFFT_LIBRARIES)

foreach(libname ${CUDA_LIBS})
  message("libs!! " ${libname} : ${${libname}})
  LIST(APPEND CUDA_LIBS_SEP ${${libname}})
endforeach(${CUDA_LIBS_SEP})

#compiler flags for cuda
#compiler flags for cuda
#SET(CUDA_NVCC_FLAGS
#    -Xcompiler #Explicitly specify the language for the input files, rather than letting the compiler choose a default based on the file name suffix. Allowed values for this option: c, c++, cu.
#    -fPIC
#    -use_fast_math
#    #	-gencode=arch=compute_20,code="sm_20,compute_20"
#    --ptxas-options=-v
#    #--device-debug #possibility to debug device code; uncomment for speedup
#    #--debug #possibility to debug host code; uncomment for speedup
#    -O3 #specify optimization level for host code
#    -Wno-deprecated-gpu-targets #to suppress warning for deprecated architectures
#    #-D_FORCE_INLINES #uncommet if cuda<8.0 and some linker error of memcpy in string.h is thrown during build
#    )


#add_executable(cudaForward
#    dr_cuda_for.cu)

set(SRC_FILES dr_cuda_for.cu)

#CUDA_COMPILE(cudaForward_mex ${SRC_FILES} SHARED)
add_library(cudaForward_mex ${SRC_FILES} ${CMAKE_SOURCE_DIR}/Matlabdef.def)


message(STATUS "LINKING CUDA LIBRARIES: " ${CUDA_LIBS_SEP})
target_link_libraries(cudaForward_mex
  ${MATLAB_LIBRARIES}
  ${CUDA_LIBRARIES} #Cudart library.
  ${CUDA_LIBS_SEP} #own specified libraries from above. Note the double "${${}}", this is because above the CUDA_CUBLAS_LIBRARIES is unknown and first known after Cuda was found
)
target_include_directories(cudaForward_mex PRIVATE "/media/paul/MATLAB/installation/MATLAB/R2022a/extern/include")
target_include_directories(cudaForward_mex PRIVATE "/media/paul/MATLAB/installation/MATLAB/R2022a/toolbox/parallel/gpu/extern/include")

set_target_properties(
    cudaForward_mex
  PROPERTIES
  CUDA_SEPERABLE_COMPILATION ON
  LINKER_LANGUAGE
  CUDA
)

add_definitions(/DMATLAB_MEX_FILE)
add_definitions(/DMX_COMPAT_32)

#SET_TARGET_PROPERTIES(cudaForward_mex PROPERTIES LINKER_LANGUAGE CUDA)


install(TARGETS cudaForward_mex DESTINATION ${CMAKE_SOURCE_DIR}/mex)
