#include "scale.cuh"

static __global__ void scale_f32(const float* x, float* dst, const float scale, const int k) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= k) {
		return;
	}

	dst[i] = scale * x[i];
}

static void scale_f32_cuda(const float* x, float* dst, const float scale, const int k, cudaStream_t stream) {
	const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
	scale_f32<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

void oiml_cuda_op_scale(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];
	const float* src0_d		= ( const float* )src0->data;
	float* dst_d			= ( float* )dst->data;
	cudaStream_t stream		= ctx.stream();

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);

	float scale;
	memcpy(&scale, dst->op_params, sizeof(float));

	scale_f32_cuda(src0_d, dst_d, scale, oiml_nelements(src0), stream);
}
