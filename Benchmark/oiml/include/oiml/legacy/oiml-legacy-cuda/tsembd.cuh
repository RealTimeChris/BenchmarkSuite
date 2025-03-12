#include "common.cuh"

#define CUDA_TIMESTEP_EMBEDDING_BLOCK_SIZE 256

void oiml_cuda_op_timestep_embedding(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
