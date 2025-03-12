#include "common.cuh"

#define CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE 256

void oiml_cuda_cross_entropy_loss(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_cross_entropy_loss_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
