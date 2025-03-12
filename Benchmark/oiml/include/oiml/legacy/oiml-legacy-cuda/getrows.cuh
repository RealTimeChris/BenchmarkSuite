#include "common.cuh"

#define CUDA_GET_ROWS_BLOCK_SIZE 256
#define CUDA_GET_ROWS_BACK_BLOCK_SIZE 256

void oiml_cuda_op_get_rows(oiml_backend_cuda_context& ctx, oiml_tensor* dst);

void oiml_cuda_op_get_rows_back(oiml_backend_cuda_context& ctx, oiml_tensor* dst);
