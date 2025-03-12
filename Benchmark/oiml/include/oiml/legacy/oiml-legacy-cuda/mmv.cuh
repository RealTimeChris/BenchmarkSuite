#include "common.cuh"

// maximum number of src0 rows with which to use mul_mat_vec over cuBLAS if FP16 tensor cores are available
#define MMV_MAX_ROWS 512

void oiml_cuda_mul_mat_vec(oiml_backend_cuda_context& ctx, const oiml_tensor* src0, const oiml_tensor* src1, oiml_tensor* dst);

void oiml_cuda_op_mul_mat_vec(oiml_backend_cuda_context& ctx, const oiml_tensor* src0, const oiml_tensor* src1, oiml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
	const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size, cudaStream_t stream);
