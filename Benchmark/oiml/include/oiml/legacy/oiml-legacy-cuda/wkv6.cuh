#include "common.cuh"

#define CUDA_WKV_BLOCK_SIZE 64

void oiml_cuda_op_rwkv_wkv6(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
