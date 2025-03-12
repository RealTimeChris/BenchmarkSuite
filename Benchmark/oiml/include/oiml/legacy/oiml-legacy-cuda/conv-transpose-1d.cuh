#include "common.cuh"

#define CUDA_CONV_TRANPOSE_1D_BLOCK_SIZE 256

void oiml_cuda_op_conv_transpose_1d(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
