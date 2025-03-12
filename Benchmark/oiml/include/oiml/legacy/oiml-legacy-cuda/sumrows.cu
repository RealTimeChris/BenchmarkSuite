#include "sumrows.cuh"

static __global__ void k_sum_rows_f32(const float* x, float* dst, const int ncols) {
	const int row = blockIdx.x;
	const int col = threadIdx.x;

	float sum = 0.0f;
	for (int i = col; i < ncols; i += blockDim.x) {
		sum += x[row * ncols + i];
	}

	sum = warp_reduce_sum(sum);

	if (col == 0) {
		dst[row] = sum;
	}
}

void sum_rows_f32_cuda(const float* x, float* dst, const int ncols, const int nrows, cudaStream_t stream) {
	const dim3 block_dims(WARP_SIZE, 1, 1);
	const dim3 block_nums(nrows, 1, 1);
	k_sum_rows_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
}

void oiml_cuda_op_sum_rows(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];
	const float* src0_d		= ( const float* )src0->data;
	float* dst_d			= ( float* )dst->data;
	cudaStream_t stream		= ctx.stream();

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(oiml_is_contiguous(src0));

	const int64_t ncols = src0->ne[0];
	const int64_t nrows = oiml_nrows(src0);

	sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
}
