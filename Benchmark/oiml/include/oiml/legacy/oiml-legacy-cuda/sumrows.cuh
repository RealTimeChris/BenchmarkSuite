#include "common.cuh"

void sum_rows_f32_cuda(const float* x, float* dst, const int ncols, const int nrows, cudaStream_t stream);

void oiml_cuda_op_sum_rows(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
