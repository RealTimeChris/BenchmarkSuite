#include "common.cuh"

#define CUDA_POOL2D_BLOCK_SIZE 256

void oiml_cuda_op_pool2d(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
