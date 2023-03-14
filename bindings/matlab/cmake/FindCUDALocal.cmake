message("setting custom matlab cuda paths")

set(MATLAB_CUDA_LIB_DIR /media/paul/MATLAB/installation/MATLAB/R2022a/bin/glnxa64)

set(CUDA_64_BIT_DEVICE_CODE ON)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_BUILD_CUBIN OFF)
set(CUDA_BUILD_EMULATION OFF)
set(CUDA_GENERATED_OUTPUT_DIR "")
set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_HOST_COMPILER /usr/bin/cc)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CUDA_NVCC_EXECUTABLE /media/paul/MATLAB/installation/MATLAB/R2022a/sys/cuda/glnxa64/cuda/bin/nvcc)
set(CUDA_NVCC_FLAGS
    -Xcompiler=-mf16c
    -Xcompiler=-Wno-float-conversion
    -Xcompiler=-fno-strict-aliasing
    -Xcudafe=--diag_suppress=unrecognized_gcc_pragma
    --extended-lambda
    --expt-relaxed-constexpr
    -Xcompiler
    -fPIC
    -gencode=arch=compute_86,code="sm_86,compute_86"
    -DMATLAB_MEX_FILE
    -G -g -O0)
set(CUDA_NVCC_FLAGS_DEBUG "")
set(CUDA_NVCC_FLAGS_MINSIZEREL "")
set(CUDA_NVCC_FLAGS_RELEASE "")
set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "")
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)

##set(CUDA_TOOLKIT_INCLUDE /usr/local/cuda/include)
##set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
##set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL /usr/local/cuda)
##set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL /usr/local/cuda)

set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CUDA_VERBOSE_BUILD OFF)
set(CUDA_VERSION 11.2)

set(CUDA_CUDART_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcudart.so)
set(CUDA_CUDA_LIBRARY /usr/local/cuda-11.8/targets/x86_64-linux/lib/stubs/libcuda.so)
set(CUDA_OpenCL_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libOpenCL.so)
set(CUDA_cublas_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcublas.so)
set(CUDA_cudadevrt_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcudadevrt.a)
set(CUDA_cudart_static_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcudart_static.a)
set(CUDA_cufft_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcufft.so)
set(CUDA_curand_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcurand.so)
set(CUDA_cusolver_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcusolver.so)
set(CUDA_cusparse_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libcusparse.so)
set(CUDA_nppc_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppc.so)
set(CUDA_nppial_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppial.so)
set(CUDA_nppicc_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppicc.so)
set(CUDA_nppidei_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppidei.so)
set(CUDA_nppif_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppif.so)
set(CUDA_nppig_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppig.so)
set(CUDA_nppim_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppim.so)
set(CUDA_nppist_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppist.so)
set(CUDA_nppisu_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppisu.so)
set(CUDA_nppitc_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnppitc.so)
set(CUDA_npps_LIBRARY=${MATLAB_CUDA_LIB_DIR}/libnpps.so)
set(CUDA_nvToolsExt_LIBRARY ${MATLAB_CUDA_LIB_DIR}/libnvToolsExt.so)

#set(CUDA_rt_LIBRARY /usr/lib/x86_64-linux-gnu/librt.a)