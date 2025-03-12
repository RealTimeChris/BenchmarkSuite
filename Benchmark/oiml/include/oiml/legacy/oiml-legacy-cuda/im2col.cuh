#include "common.cuh"

#define CUDA_IM2COL_BLOCK_SIZE 256

void oiml_cuda_op_im2col(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
