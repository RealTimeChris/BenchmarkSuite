#include "common.cuh"

#define CUDA_ARANGE_BLOCK_SIZE 256

void oiml_cuda_op_arange(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
