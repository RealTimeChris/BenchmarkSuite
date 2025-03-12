#include "common.cuh"

#define CUDA_ACC_BLOCK_SIZE 256

void oiml_cuda_op_acc(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
