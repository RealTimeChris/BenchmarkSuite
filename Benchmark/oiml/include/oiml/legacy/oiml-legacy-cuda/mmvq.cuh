#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8// Max. batch size for which to use MMVQ kernels.

void oiml_cuda_op_mul_mat_vec_q(oiml_backend_cuda_context& ctx, const oiml_tensor* src0, const oiml_tensor* src1, oiml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
	const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size, cudaStream_t stream);
