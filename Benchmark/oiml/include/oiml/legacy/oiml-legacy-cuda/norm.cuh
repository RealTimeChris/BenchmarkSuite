#include "common.cuh"

void oiml_cuda_op_norm(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_group_norm(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_rms_norm(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_rms_norm_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
