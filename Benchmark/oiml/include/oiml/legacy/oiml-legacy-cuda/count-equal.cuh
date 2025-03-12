#include "common.cuh"

#define CUDA_COUNT_EQUAL_CHUNK_SIZE 128

void oiml_cuda_count_equal(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
