#include "diagmask.cuh"

static __global__ void diag_mask_inf_f32(const float* x, float* dst, const int ncols, const int rows_per_channel, const int n_past) {
	const int col = blockDim.y * blockIdx.y + threadIdx.y;
	const int row = blockDim.x * blockIdx.x + threadIdx.x;

	if (col >= ncols) {
		return;
	}

	const int i = row * ncols + col;
	//dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
	//dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
	dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

static void diag_mask_inf_f32_cuda(const float* x, float* dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream) {
	const dim3 block_dims(1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1);
	const int block_num_x = (ncols_x + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
	const dim3 block_nums(nrows_x, block_num_x, 1);
	diag_mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols_x, rows_per_channel, n_past);
}

void oiml_cuda_op_diag_mask_inf(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];
	const float* src0_d		= ( const float* )src0->data;
	float* dst_d			= ( float* )dst->data;
	cudaStream_t stream		= ctx.stream();

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);

	const int64_t ne00 = src0->ne[0];
	const int64_t ne01 = src0->ne[1];
	const int nrows0   = oiml_nrows(src0);

	const int n_past = (( int32_t* )dst->op_params)[0];

	diag_mask_inf_f32_cuda(src0_d, dst_d, ne00, nrows0, ne01, n_past, stream);
}
