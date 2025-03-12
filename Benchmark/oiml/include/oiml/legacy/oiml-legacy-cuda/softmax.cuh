#include "common.cuh"

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void oiml_cuda_op_soft_max(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_soft_max_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
