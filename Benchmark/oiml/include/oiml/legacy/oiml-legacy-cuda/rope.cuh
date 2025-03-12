#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void oiml_cuda_op_rope(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_rope_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
