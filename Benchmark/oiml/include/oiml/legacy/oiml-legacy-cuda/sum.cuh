#include "common.cuh"

void sum_f32_cuda(oiml_cuda_pool& pool, const float* x, float* dst, const int64_t ne, cudaStream_t stream);

void oiml_cuda_op_sum(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
