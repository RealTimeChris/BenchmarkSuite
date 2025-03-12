#include "common.cuh"

static __device__ __forceinline__ void dequantize_q8_0(const void* vx, const int64_t ib, const int iqs, dfloat2& v) {
	const oiml::block_q8_0<oiml_half_cuda>* x = ( const oiml::block_q8_0<oiml_half_cuda>* )vx;

	const dfloat d = x[ib].d;

	v.x = x[ib].qs[iqs + 0];
	v.y = x[ib].qs[iqs + 1];

#ifdef OIML_CUDA_F16
	v = __hmul2(v, { d, d });
#else
	v.x *= d;
	v.y *= d;
#endif// OIML_CUDA_F16
}
