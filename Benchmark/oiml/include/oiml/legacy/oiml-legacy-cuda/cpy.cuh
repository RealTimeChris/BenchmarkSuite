#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void oiml_cuda_cpy(oiml_backend_cuda_context& ctx, const oiml_tensor* src0, oiml_tensor* src1);

void oiml_cuda_dup(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void* oiml_cuda_cpy_fn(const oiml_tensor* src0, oiml_tensor* src1);
