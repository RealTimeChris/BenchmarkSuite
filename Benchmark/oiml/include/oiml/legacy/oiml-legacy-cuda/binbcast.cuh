#include "common.cuh"

void oiml_cuda_op_repeat(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
void oiml_cuda_op_add(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
void oiml_cuda_op_sub(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
void oiml_cuda_op_mul(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
void oiml_cuda_op_div(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_repeat_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
