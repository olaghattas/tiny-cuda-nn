/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   bindings.cu
 *  @author Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel, NVIDIA
 */

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x) \
  do { if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); } while(0)

#include <tiny-cuda-nn/cpp_api.h>
#include <nlohmann/json.hpp>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/config.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"


#define CHECK_INPUT(x) CHECK_THROW(x.device().is_cuda()); CHECK_THROW(x.is_contiguous())


mxClassID matlab_type(tcnn::cpp::EPrecision precision) {
  switch (precision) {
    case tcnn::cpp::EPrecision::Fp32:
      return mxClassID::mxSINGLE_CLASS;
    case tcnn::cpp::EPrecision::Fp16:
      throw std::runtime_error{"MATLAB does not support float16 in mex."};
    default:
      throw std::runtime_error{"Unknown precision tcnn->torch"};
  }
}

//void *void_data_ptr(torch::Tensor &tensor) {
//  switch (tensor.scalar_type()) {
//    case torch::kFloat32:
//      return tensor.data_ptr<float>();
//    case torch::kHalf:
//      return tensor.data_ptr<torch::Half>();
//    default:
//      throw std::runtime_error{"Unknown precision torch->void"};
//  }
//}


class Module {
public:
  Module(tcnn::cpp::Module *module) : m_module{module} {}

  mxGPUArray *fwd(mxGPUArray const *input, mxGPUArray const *params) {
//    CHECK_INPUT(input);
//    CHECK_INPUT(params);
//
////    mxGPUCreateGPUArray();
//    mxREAL;
//
//    // Types
//    CHECK_THROW(input.scalar_type() == torch::kFloat32);
//    CHECK_THROW(params.scalar_type() == mx_param_precision());
//
//    // Sizes
//    CHECK_THROW(input.size(1) == n_input_dims());
//    CHECK_THROW(params.size(0) == n_params());
//
//    // Device
//    at::Device device = input.device();
//    CHECK_THROW(device == params.device());
//
//    const at::cuda::CUDAGuard device_guard{device};
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
    cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.


    uint32_t batch_size = mxGPUGetDimensions(input)[0];

//    torch::Tensor output = torch::empty({batch_size, n_output_dims()},
//                                        torch::TensorOptions().dtype(mx_output_precision()).device(device));

    mxGPUArray *output;
    {
      mwSize dims[2];
      dims[0] = batch_size;
      dims[1] = n_output_dims();
      output = mxGPUCreateGPUArray(2, dims, mx_param_precision(), mxREAL, MX_GPU_INITIALIZE_VALUES);
    }

//    tcnn::cpp::Context ctx;
//    if (!input.requires_grad() && !params.requires_grad()) {
    ctx = m_module->forward(stream, batch_size, (const float *) mxGPUGetDataReadOnly(input), mxGPUGetData(output),
                            mxGPUGetDataReadOnly(params), true);
//    } else {
//      ctx = m_module->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params),
//                              input.requires_grad());
//    }
    cudaStreamDestroy(stream);

    return output;
  }

  std::tuple<mxGPUArray *, mxGPUArray *>
  bwd(mxGPUArray *input, mxGPUArray *params, mxGPUArray *output,
      mxGPUArray *dL_doutput) {
    if (!ctx.ctx) {
      throw std::runtime_error{
          "Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
    }

//    CHECK_INPUT(input);
//    CHECK_INPUT(params);
//    CHECK_INPUT(output);
//    CHECK_INPUT(dL_doutput);
//
//    // Types
//    CHECK_THROW(input.scalar_type() == torch::kFloat32);
//    CHECK_THROW(params.scalar_type() == mx_param_precision());
//    CHECK_THROW(output.scalar_type() == mx_output_precision());
//    CHECK_THROW(dL_doutput.scalar_type() == mx_output_precision());
//
//    // Sizes
//    CHECK_THROW(input.size(1) == n_input_dims());
//    CHECK_THROW(output.size(1) == n_output_dims());
//    CHECK_THROW(params.size(0) == n_params());
//    CHECK_THROW(output.size(0) == input.size(0));
//    CHECK_THROW(dL_doutput.size(0) == input.size(0));
//
//    // Device
//    at::Device device = input.device();
//    CHECK_THROW(device == params.device());
//    CHECK_THROW(device == output.device());
//    CHECK_THROW(device == dL_doutput.device());

    //    const at::cuda::CUDAGuard device_guard{device};
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
    cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.


    uint32_t batch_size = mxGPUGetDimensions(input)[0];


    mxGPUArray *dL_dinput;
//    if (input.requires_grad()) {
//      dL_dinput = torch::empty({batch_size, input.size(1)},
//                               torch::TensorOptions().dtype(torch::kFloat32).device(device));
//    }
    {
      mwSize dims[2];
      dims[0] = n_params();
      dims[1] = mxGPUGetDimensions(input)[1];
      dL_dinput = mxGPUCreateGPUArray(2, dims, mx_param_precision(), mxREAL, MX_GPU_INITIALIZE_VALUES);
    }

    mxGPUArray *dL_dparams;
//    if (params.requires_grad()) {
//      dL_dparams = torch::empty({n_params()}, torch::TensorOptions().dtype(mx_param_precision()).device(device));
//    }

    {
      mwSize dims[1];
      dims[0] = n_params();
      dL_dparams = mxGPUCreateGPUArray(1, dims, mx_param_precision(), mxREAL, MX_GPU_INITIALIZE_VALUES);
    }

//    if (input.requires_grad() || params.requires_grad()) {
    m_module->backward(
        stream,
        ctx,
        batch_size,
        (float *) mxGPUGetData(dL_dinput),
        mxGPUGetData(dL_doutput),
        mxGPUGetData(dL_dparams),
        (float *) mxGPUGetData(input),
        mxGPUGetData(output),
        mxGPUGetData(params)
    );
    cudaStreamDestroy(stream);
//    }

    return {dL_dinput, dL_dparams};
  }

  std::tuple<mxGPUArray *, mxGPUArray *, mxGPUArray *>
  bwd_bwd_input(mxGPUArray *input, mxGPUArray *params, mxGPUArray *dL_ddLdinput,
                mxGPUArray *dL_doutput) {
    // from: dL_ddLdinput
    // to:   dL_ddLdoutput, dL_dparams, dL_dinput

    if (!ctx.ctx) {
      throw std::runtime_error{
          "Module::bwd_bwd_input: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
    }

//    CHECK_INPUT(input);
//    CHECK_INPUT(params);
//    CHECK_INPUT(dL_ddLdinput);
//    CHECK_INPUT(dL_doutput);

    // Types TODO error handling
//    CHECK_THROW(input.scalar_type() == torch::kFloat32);
//    CHECK_THROW(dL_ddLdinput.scalar_type() == torch::kFloat32);
//    CHECK_THROW(params.scalar_type() == mx_param_precision());
//    CHECK_THROW(dL_doutput.scalar_type() == mx_output_precision());
//
//    // Sizes
//    CHECK_THROW(input.size(1) == n_input_dims());
//    CHECK_THROW(dL_doutput.size(1) == n_output_dims());
//    CHECK_THROW(dL_ddLdinput.size(1) == n_input_dims());
//    CHECK_THROW(params.size(0) == n_params());
//    CHECK_THROW(dL_doutput.size(0) == input.size(0));
//    CHECK_THROW(dL_ddLdinput.size(0) == input.size(0));
//
//    // Device
//    at::Device device = input.device();
//    CHECK_THROW(device == params.device());
//    CHECK_THROW(device == dL_ddLdinput.device());
//    CHECK_THROW(device == dL_doutput.device());


//    const at::cuda::CUDAGuard device_guard{device};
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
    cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.


    uint32_t batch_size = mxGPUGetDimensions(input)[0];

    mxGPUArray *dL_ddLdoutput;
//    if (dL_doutput.requires_grad()) {
//      dL_ddLdoutput = torch::zeros({ batch_size, n_output_dims() }, torch::TensorOptions().dtype(mx_output_precision()).device(device));
    {
      mwSize dims[2];
      dims[0] = batch_size;
      dims[1] = n_output_dims();
      dL_ddLdoutput = mxGPUCreateGPUArray(2, dims, mx_param_precision(), mxREAL, MX_GPU_INITIALIZE_VALUES);
    }

    mxGPUArray *dL_dparams;
//    if (params.requires_grad()) {
//      dL_dparams = torch::zeros({ n_params() }, torch::TensorOptions().dtype(mx_param_precision()).device(device));
    {
      mwSize dims[1];
      dims[0] = n_params();
      dL_dparams = mxGPUCreateGPUArray(1, dims, mx_param_precision(), mxREAL, MX_GPU_INITIALIZE_VALUES);
    }

    mxGPUArray *dL_dinput;
//    if (input.requires_grad()) {
//      dL_dinput = torch::zeros({ batch_size, n_input_dims() }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    {
      mwSize dims[2];
      dims[0] = batch_size;
      dims[1] = n_input_dims();
      dL_dinput = mxGPUCreateGPUArray(2, dims, mxClassID::mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    }

//    if (dL_doutput.requires_grad() || params.requires_grad()) {
    m_module->backward_backward_input(
        stream,
        ctx,
        batch_size,
        (float *) mxGPUGetData(dL_ddLdinput),
        (float *) mxGPUGetData(input),
        mxGPUGetData(dL_doutput),
        mxGPUGetData(dL_dparams),
        mxGPUGetData(dL_ddLdoutput),
        (float *) mxGPUGetData(dL_dinput),
        mxGPUGetData(params)
    );
    cudaStreamDestroy(stream);
//    }

    return {dL_ddLdoutput, dL_dparams, dL_dinput};
  }

  mxGPUArray *initial_params(size_t seed) {
    mwSize dim = n_params();
//    *(dims) = 0;
    mxArray *tmp = mxCreateNumericMatrix(dim, 1, mxClassID::mxSINGLE_CLASS, mxREAL);
//    mxGPUArray *output = mxGPUCreateGPUArray(0, &dim, mxClassID::mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    mxGPUArray *output = mxGPUCopyFromMxArray(tmp);
//mxGPUArray *output = mxGPUCreateGPUArray(1, dims, mxClassID::mxSINGLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
//auto tmp = (float *) mxGPUGetData(output);
    m_module->initialize_params(seed, (float *) mxGPUGetData(output));
    return output;
  }

  uint32_t n_input_dims() const {
    return m_module->n_input_dims();
  }

  uint32_t n_params() const {
    return (uint32_t) m_module->n_params();
  }

  tcnn::cpp::EPrecision param_precision() const {
    return m_module->param_precision();
  }

  mxClassID mx_param_precision() const {
    return matlab_type(param_precision());
  }

  uint32_t n_output_dims() const {
    return m_module->n_output_dims();
  }

  tcnn::cpp::EPrecision output_precision() const {
    return m_module->output_precision();
  }

  mxClassID mx_output_precision() const {
    return matlab_type(output_precision());
  }

  nlohmann::json hyperparams() const {
    return m_module->hyperparams();
  }

  std::string name() const {
    return m_module->name();
  }

private:
  std::unique_ptr<tcnn::cpp::Module> m_module;
  tcnn::cpp::Context ctx;
};

int init(mxArray *plhs[], int nrhs, const mxArray *const *prhs);

#if !defined(TCNN_NO_NETWORKS)

Module *create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims) {
  nlohmann::json network = {
      {"otype",             "FullyFusedMLP"},
      {"activation",        "ReLU"},
      {"output_activation", "None"},
      {"n_neurons",         64},
      {"n_hidden_layers",   2}
  };
  nlohmann::json encoding = {
      {"otype",                "HashGrid"},
      {"n_levels",             16},
      {"n_features_per_level", 2},
      {"log2_hashmap_size",    19},
      {"base_resolution",      16},
      {"per_level_scale",      2.0}
  };

  return new Module(tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding, network));
}

Module *create_network(uint32_t n_input_dims, uint32_t n_output_dims) {//}, const nlohmann::json &network) {
  nlohmann::json network = {
      {"otype",             "FullyFusedMLP"},
      {"activation",        "ReLU"},
      {"output_activation", "None"},
      {"n_neurons",         64},
      {"n_hidden_layers",   2}
  };
  return new Module(tcnn::cpp::create_network(n_input_dims, n_output_dims, network));
}

#endif

Module *create_encoding(uint32_t n_input_dims) {
  nlohmann::json encoding = {
      {"otype",                "HashGrid"},
      {"n_levels",             16},
      {"n_features_per_level", 2},
      {"log2_hashmap_size",    19},
      {"base_resolution",      16},
      {"per_level_scale",      2.0}
  };
  return new Module(tcnn::cpp::create_encoding(n_input_dims, encoding, tcnn::cpp::EPrecision::Fp32));
}

bool mxCheckGPUArrays(mxArray const *arrays[], int len) {
  for (int i = 0; i < len; i++) {
    if (!(mxIsGPUArray(arrays[i]))) {
      char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
      char const *const errMsg = "Not all inputs are on the GPU";
      mexErrMsgIdAndTxt(errId, errMsg);
      return false;
    }
  }
  return true;
}

// operations
const int INIT = 0;
const int DESTROY = 1;
const int BATCH_SIZE_GRANULARITY = 2;
const int FREE_TEMPORARY_MEMORY = 3;
const int FWD = 4;
const int BWD = 5;
const int BWD_BWD_INPUT = 6;
const int INITIAL_PARAMS = 7;
const int N_INPUT_DIMS = 8;
const int N_PARAMS = 9;
const int N_OUTPUT_DIMS = 10;
const int HYPERPARAMS = 11;
const int NAME = 12;

// network types
const int CREATE_NETWORK_WITH_INPUT_ENCODING = 13;
const int CREATE_NETWORK = 14;
const int CREATE_ENCODING = 15;

//m.def("create_network_with_input_encoding", &create_network_with_input_encoding);
//m.def("create_network", &create_network);
//m.def("create_encoding", &create_encoding);

//m.def("batch_size_granularity", &tcnn::cpp::batch_size_granularity);
//m.def("free_temporary_memory", &tcnn::cpp::free_temporary_memory);
//.def("fwd", &Module::fwd)
//.def("bwd", &Module::bwd)
//.def("bwd_bwd_input", &Module::bwd_bwd_input)
//.def("initial_params", &Module::initial_params)
//.def("n_input_dims", &Module::n_input_dims)
//.def("n_params", &Module::n_params)
//.def("n_output_dims", &Module::n_output_dims)
//.def("output_precision", &Module::output_precision)
//.def("hyperparams", &Module::hyperparams)
//.def("name", &Module::name)


Module *getModulePtr(const mxArray *pa) {
  double *pointer0 = mxGetPr(pa);
  mxAssert(pointer0 != NULL, "input must be a valid network module pointer\n");
  intptr_t pointer1 = (intptr_t) pointer0[0];
  return (Module *) pointer1;
}


void init(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  mxAssert(nrhs >= 2, "operation requires: (op, network_type)");
  mxAssert(nlhs == 1, "only one output expected");
  int network_type = mxGetScalar(prhs[1]);
  Module *module;
  switch (network_type) {
    case CREATE_NETWORK_WITH_INPUT_ENCODING: {
      mxAssert(nrhs == 4, "operation requires: (op, network_type, num_input_dims, num_output_dims)");
      int num_input_dims = mxGetScalar(prhs[2]);
      int num_output_dims = mxGetScalar(prhs[3]);
      module = create_network_with_input_encoding(num_input_dims, num_output_dims);
    }
      break;
    case CREATE_NETWORK: {
      mxAssert(nrhs == 4, "operation requires: (op, network_type, num_input_dims, num_output_dims)");
      int num_input_dims = mxGetScalar(prhs[2]);
      int num_output_dims = mxGetScalar(prhs[3]);
      module = create_network(num_input_dims, num_output_dims);
    }
      break;
    case CREATE_ENCODING: {
      mxAssert(nrhs == 3, "operation requires: (op, network_type, num_input_dims)");
      int num_input_dims = mxGetScalar(prhs[2]);
      module = create_encoding(num_input_dims);
    }
      break;
    default:
      mxAssert(false, "invalid network type ");
      break;
  }

  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *module_ptr = mxGetPr(plhs[0]);
  module_ptr[0] = (intptr_t) module;
}

void destroy(int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[]) {
  mxAssert(nrhs == 2, "Only two input argument expected");
  mxAssert(nlhs == 0, "no output arguments expected");
  Module *module = getModulePtr(prhs[1]);
  delete module;
  printf("deleted module\n");
}

void batch_size_granularity(int nlhs, mxArray *plhs[],
                            int nrhs, const mxArray *prhs[]) {
  mxAssert(nrhs == 2, "Only one input argument expected");
  mxAssert(nlhs == 1, "only one output expected");
  Module *module = getModulePtr(prhs[1]);
  plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  double *double_ptr = mxGetPr(plhs[0]);
  *double_ptr = tcnn::cpp::batch_size_granularity();
}

void free_temporary_memory(int nlhs, mxArray *plhs[],
                           int nrhs, const mxArray *prhs[]) {
  mxAssert(nrhs == 1, "Only one input argument expected");
  mxAssert(nlhs == 0, "no outputs expected");
  tcnn::cpp::free_temporary_memory();
}

void fwd(int nlhs, mxArray *plhs[],
         int nrhs, const mxArray *prhs[]) {
  mxAssert(nrhs == 4, "Only four input argument expected");
  mxAssert(nlhs == 1, "one output argument expected");
  Module *module = getModulePtr(prhs[1]);
  mxGPUArray const *input = mxGPUCreateFromMxArray(prhs[2]);
  mxGPUArray const *params = mxGPUCreateFromMxArray(prhs[3]);

  mxGPUArray *output = module->fwd(input, params);
  plhs[0] = mxGPUCreateMxArrayOnGPU(output);
  mxGPUDestroyGPUArray(output);

  mxGPUDestroyGPUArray(input);
  mxGPUDestroyGPUArray(params);
}

void initial_params(int nlhs, mxArray *plhs[],
                    int nrhs, const mxArray *prhs[]) {
  mxAssert(nrhs == 2, "Only four input argument expected");
  mxAssert(nlhs == 1, "one output argument expected");
  Module *module = getModulePtr(prhs[1]);
  mxGPUArray *params = module->initial_params(0);
  plhs[0] = mxGPUCreateMxArrayOnGPU(params);
  mxGPUDestroyGPUArray(params);

}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Perform error checks
  mxAssert(nrhs > 0, "function requires input arguments");

  // Initialize the MathWorks GPU API
  mxInitGPU();

  int val = mxGetScalar(prhs[0]);
  mxAssert(val >= 0 && val <= 12, "validation operations are between 0 and 12");
  switch (val) {
    case INIT:
      init(nlhs, plhs, nrhs, prhs);
      break;
    case DESTROY:
      destroy(nlhs, plhs, nrhs, prhs);
      break;
    case BATCH_SIZE_GRANULARITY:
      batch_size_granularity(nlhs, plhs, nrhs, prhs);
      break;
    case FREE_TEMPORARY_MEMORY:
      batch_size_granularity(nlhs, plhs, nrhs, prhs);
      break;
    case FWD:
      fwd(nlhs, plhs, nrhs, prhs);
      break;
    case INITIAL_PARAMS:
      initial_params(nlhs, plhs, nrhs, prhs);
      break;
    default:
      mxAssert(false, "operation value is invalid");
      break;
  }

//  if (nrhs != 10) {
//    char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
//    char const *const errMsg = "Wrong number of inputs";
//    mexErrMsgIdAndTxt(errId, errMsg);
//  }

//  for (int i = 0; i < 5; i++) {
//    if (!(mxIsGPUArray(prhs[i]))) {
//      char const *const errId = "parallel:gpu:mexGPUExample:InvalidInput";
//      char const *const errMsg = "Not all inputs are on the GPU";
//      mexErrMsgIdAndTxt(errId, errMsg);
//    }
//  }


}








//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//py::enum_<tcnn::cpp::EPrecision>(m, "Precision")
//.value("Fp32", tcnn::cpp::EPrecision::Fp32)
//.value("Fp16", tcnn::cpp::EPrecision::Fp16)
//.export_values()
//;
//
//m.def("preferred_precision", &tcnn::cpp::preferred_precision);
//m.def("batch_size_granularity", &tcnn::cpp::batch_size_granularity);
//m.def("free_temporary_memory", &tcnn::cpp::free_temporary_memory);
//
//// Encapsulates an abstract context of an operation
//// (commonly the forward pass) to be passed on to other
//// operations (commonly the backward pass).
//py::class_<tcnn::cpp::Context>(m, "Context");
//
//// The python bindings expose TCNN's C++ API through
//// a single "Module" class that can act as the encoding,
//// the neural network, or a combined encoding + network
//// under the hood. The bindings don't need to concern
//// themselves with these implementation details, though.
//py::class_<Module>(m, "Module")
//.def("fwd", &Module::fwd)
//.def("bwd", &Module::bwd)
//.def("bwd_bwd_input", &Module::bwd_bwd_input)
//.def("initial_params", &Module::initial_params)
//.def("n_input_dims", &Module::n_input_dims)
//.def("n_params", &Module::n_params)
//.def("param_precision", &Module::param_precision)
//.def("n_output_dims", &Module::n_output_dims)
//.def("output_precision", &Module::output_precision)
//.def("hyperparams", &Module::hyperparams)
//.def("name", &Module::name)
//;
//
//#if !defined(TCNN_NO_NETWORKS)
//m.def("create_network_with_input_encoding", &create_network_with_input_encoding);
//m.def("create_network", &create_network);
//#endif
//
//m.def("create_encoding", &create_encoding);
//}
