#include "common.cuh"

#define CUDA_UPSCALE_BLOCK_SIZE 256

void oiml_cuda_op_upscale(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
