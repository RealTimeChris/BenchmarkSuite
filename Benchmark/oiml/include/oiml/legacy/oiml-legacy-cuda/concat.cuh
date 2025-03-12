#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void oiml_cuda_op_concat(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
