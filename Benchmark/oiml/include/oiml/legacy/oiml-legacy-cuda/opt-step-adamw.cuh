#include "common.cuh"

#define CUDA_OPT_STEP_ADAMW_BLOCK_SIZE 256

void oiml_cuda_opt_step_adamw(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
