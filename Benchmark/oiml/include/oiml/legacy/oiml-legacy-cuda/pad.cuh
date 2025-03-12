#include "common.cuh"

#define CUDA_PAD_BLOCK_SIZE 256

void oiml_cuda_op_pad(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
