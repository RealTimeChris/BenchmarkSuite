#include "common.cuh"

#define CUDA_DIAG_MASK_INF_BLOCK_SIZE 32

void oiml_cuda_op_diag_mask_inf(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
