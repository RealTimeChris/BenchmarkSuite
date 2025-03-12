#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void oiml_cuda_op_scale(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
