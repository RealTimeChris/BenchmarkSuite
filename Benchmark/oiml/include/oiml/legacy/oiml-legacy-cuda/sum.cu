#if !defined(OIML_USE_HIP) && !defined(OIML_USE_MUSA) && CUDART_VERSION >= 11070
	#define USE_CUB
#endif// !defined(OIML_USE_HIP) && !defined(OIML_USE_MUSA) && CUDART_VERSION >= 11070

#ifdef USE_CUB
	#include <cub/cub.cuh>
using namespace cub;
#endif// USE_CUB

#include "sumrows.cuh"
#include "sum.cuh"

#include <cstdint>

void sum_f32_cuda(oiml_cuda_pool& pool, const float* x, float* dst, const int64_t ne, cudaStream_t stream) {
#ifdef USE_CUB
	size_t tmp_size = 0;
	DeviceReduce::Sum(nullptr, tmp_size, x, dst, ne, stream);
	oiml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
	DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, x, dst, ne, stream);
#else
	// Use (inefficient) sum_rows implementation as a fallback.
	// For AMD there is rocPRIM which could be used as a drop-in replacement via hipcub but this would require C++11 -> C++14.
	sum_rows_f32_cuda(x, dst, ne, 1, stream);
	OIML_UNUSED(pool);
#endif// USE_CUB
}

void oiml_cuda_op_sum(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* src0 = dst->src[0];

	OIML_ASSERT(src0->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(oiml_is_contiguous(src0));

	const float* src0_d = ( const float* )src0->data;
	float* dst_d		= ( float* )dst->data;

	const int64_t ne = oiml_nelements(src0);

	oiml_cuda_pool& pool = ctx.pool();
	cudaStream_t stream	 = ctx.stream();

	sum_f32_cuda(pool, src0_d, dst_d, ne, stream);
}
