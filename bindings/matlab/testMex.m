clear functions
clear all


% operations
INIT = 0;
DESTROY = 1;
BATCH_SIZE_GRANULARITY = 2;
FREE_TEMPORARY_MEMORY = 3;
FWD = 4;
BWD = 5;
BWD_BWD_INPUT = 6;
INITIAL_PARAMS = 7;
N_INPUT_DIMS = 8;
N_PARAMS = 9;
N_OUTPUT_DIMS = 10;
HYPERPARAMS = 11;
NAME = 12;

% network types
CREATE_NETWORK_WITH_INPUT_ENCODING = 13;
CREATE_NETWORK = 14;
CREATE_ENCODING = 15;


addpath('mex/')

ptr = matlab_bindings(INIT, CREATE_NETWORK_WITH_INPUT_ENCODING, 2, 3)

batch_size_granularity = matlab_bindings(BATCH_SIZE_GRANULARITY, ptr)

params =  matlab_bindings(INITIAL_PARAMS, ptr)

in = gpuArray(zeros(2, batch_size_granularity));


% out = matlab_bindings(FWD, ptr, in, params)




matlab_bindings(DESTROY, ptr);





