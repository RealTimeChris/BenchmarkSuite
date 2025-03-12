#include "common.cuh"

#define CUDA_CLAMP_BLOCK_SIZE 256

void oiml_cuda_op_clamp(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
