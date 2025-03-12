#include "common.cuh"
#include <cstdint>

static __device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
	const uint16_t* x16 = ( const uint16_t* )x;// assume at least 2 byte alignment

	int x32 = x16[2 * i32 + 0] << 0;
	x32 |= x16[2 * i32 + 1] << 16;

	return x32;
}

static __device__ __forceinline__ int get_int_b4(const void* x, const int& i32) {
	return (( const int* )x)[i32];// assume at least 4 byte alignment
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template<typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(const int* v, const int* u, const T& d8_0, const T& d8_1) {
	int sumi = 0;

#pragma unroll
	for (int i = 0; i < vdr; ++i) {
		// SIMD dot product of quantized values
		sumi = oiml_cuda_dp4a(v[i], u[i], sumi);
	}

	return d8_0 * d8_1 * (( T )sumi);
}
