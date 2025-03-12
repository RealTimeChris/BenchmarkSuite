#include "common.cuh"
#include "count-equal.cuh"

#include <cstdint>

template<typename T> static __global__ void count_equal(const T* __restrict__ x, const T* __restrict__ y, int64_t* __restrict__ dst, const int64_t dk, const int64_t k) {
	const int64_t i0 = ( int64_t )blockIdx.x * dk;
	const int64_t i1 = min(i0 + dk, k);

	int nequal = 0;

	for (int64_t i = i0 + threadIdx.x; i < i1; i += WARP_SIZE) {
		const T xi = x[i];
		const T yi = y[i];
		nequal += xi == yi;
	}

	nequal = warp_reduce_sum(nequal);

	if (threadIdx.x != 0) {
		return;
	}

	atomicAdd(( int* )dst, nequal);
}
