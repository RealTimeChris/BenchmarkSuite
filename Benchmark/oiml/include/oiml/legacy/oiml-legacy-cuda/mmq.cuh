#pragma once

#include "common.cuh"
#include "vecdotq.cuh"
#include "mma.cuh"

#include <climits>
#include <cstdint>

using namespace oiml_cuda_mma;

#define MMQ_DP4A_MAX_BATCH_SIZE 64// Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.
#define MMQ_ITER_K 256
#define MMQ_NWARPS 8

typedef void (*load_tiles_mmq_t)(const char* __restrict x, int* x_tile, const int& kbx0, const int& i_max, const int& stride);
typedef void (*vec_dot_mmq_t)(const int* __restrict x, const int* __restrict y, float* __restrict sum, const int& k00);
typedef void (*mmq_write_back_t)(const float* __restrict sum, float* __restrict dst, const int& stride, const int& i_max, const int& j_max);

enum mmq_q8_1_ds_layout {
	MMQ_Q8_1_DS_LAYOUT_D4,
	MMQ_Q8_1_DS_LAYOUT_DS4,
	MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
	// The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
	// The y float data is first grouped as blocks of 128 values.
	// These blocks are then treated as individual data values and transposed.
	//
	// To avoid shared memory bank conflicts each block is padded with 16 bytes.
	// This padding is also used to store block scales/partial sums.
	// The scales multiplied with the quantized data are equal to the unquantized values.
	// The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
	//     and are only needed for performance reasons.
	//
	// The exact data stored depends on the x data type.
	union {
		float d4[4];// 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
		half2 ds4[4];// 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
		half d2s6[8];// 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
			//     stored as d0,d1,s1,s2,s3,s4,s5
	};
	int8_t qs[4 * oiml::Q_SIZE];// 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4 * oiml::Q_SIZE + 4 * sizeof(half2), "Unexpected block_q8_1_mmq size");

static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const oiml_representation_types type_x) {
	switch (type_x) {
		case oiml::oiml_representation_types::q8_0:
			return MMQ_Q8_1_DS_LAYOUT_D4;
		default:
			OIML_ABORT("fatal error");
			break;
	}
}

struct tile_x_sizes {
	uint64_t qs;
	uint64_t dm;
	uint64_t sc;
};

static int get_mmq_x_max_host(const int cc) {
	return new_mma_available(cc)																	? 128
		: oiml_cuda_highest_compiled_arch(cc) >= OIML_CUDA_CC_VOLTA && cc < OIML_CUDA_CC_OFFSET_AMD ?
#ifdef OIML_CUDA_FORCE_MMQ
																									128
																									: 64;
#else
																									MMQ_DP4A_MAX_BATCH_SIZE
																									: 64;
#endif// OIML_CUDA_FORCE_MMQ
}

inline static constexpr __device__ int get_mmq_x_max_device() {
#ifdef NEW_MMA_AVAILABLE
	return 128;
#else// NEW_MMA_AVAILABLE

	#if defined(OIML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
	return 128;
	#else// defined(OIML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)

		#if __CUDA_ARCH__ >= OIML_CUDA_CC_VOLTA
			#ifdef OIML_CUDA_FORCE_MMQ
	return 128;
			#else// OIML_CUDA_FORCE_MMQ
	return MMQ_DP4A_MAX_BATCH_SIZE;
			#endif// OIML_CUDA_FORCE_MMQ
		#else// __CUDA_ARCH__ >= OIML_CUDA_CC_VOLTA

	return 64;
		#endif// __CUDA_ARCH__ >= OIML_CUDA_CC_VOLTA

	#endif// defined(OIML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#endif// NEW_MMA_AVAILABLE
}

static int get_mmq_y_host(const int cc) {
	return cc >= OIML_CUDA_CC_OFFSET_AMD ? (OIML_CUDA_CC_IS_RDNA1(cc) ? 64 : 128) : (oiml_cuda_highest_compiled_arch(cc) >= OIML_CUDA_CC_VOLTA ? 128 : 64);
}

static constexpr __device__ int get_mmq_y_device() {
#if defined(OIML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
	#if defined(RDNA1)
	return 64;
	#else
	return 128;
	#endif// defined RDNA1
#else
	#if __CUDA_ARCH__ >= OIML_CUDA_CC_VOLTA
	return 128;
	#else
	return 64;
	#endif// __CUDA_ARCH__ >= OIML_CUDA_CC_VOLTA
#endif// defined(OIML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
}

#define MMQ_DP4A_TXS_Q8_0 tile_x_sizes{ mmq_y * WARP_SIZE * 2 + mmq_y, mmq_y * WARP_SIZE * 2 / oiml::QI8_0 + mmq_y / (oiml::QI8_0 / 2), 0 }
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{ mmq_y * WARP_SIZE * 2 + mmq_y, mmq_y * WARP_SIZE * 4 / oiml::QI8_0 + mmq_y / (oiml::QI8_0 / 4), 0 }

static constexpr __host__ __device__ tile_x_sizes mmq_get_dp4a_tile_x_sizes(oiml_representation_types type, uint64_t mmq_y) {
	return MMQ_DP4A_TXS_Q8_0;
}

#define MMQ_MMA_TILE_X_K_Q8_0 (2 * WARP_SIZE + 2 * WARP_SIZE / oiml::QI8_0 + 4)
#define MMQ_MMA_TILE_X_K_Q8_1 (2 * WARP_SIZE + 2 * WARP_SIZE / oiml::QI8_0 + 4)
#define MMQ_MMA_TILE_X_K_Q2_K (2 * WARP_SIZE + WARP_SIZE + 4)
#define MMQ_MMA_TILE_X_K_Q3_K (2 * WARP_SIZE + WARP_SIZE / 2 + 4)
#define MMQ_MMA_TILE_X_K_Q6_K (2 * WARP_SIZE + WARP_SIZE / QI6_K + WARP_SIZE / 8 + 7)

static_assert(MMQ_MMA_TILE_X_K_Q2_K % 8 == 4, "Wrong padding.");
static_assert(MMQ_MMA_TILE_X_K_Q3_K % 8 == 4, "Wrong padding.");

static constexpr __host__ __device__ int mmq_get_mma_tile_x_k(oiml_representation_types type) {
	return MMQ_MMA_TILE_X_K_Q8_0;
}

#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE / oiml::QI8_0)

static int mmq_get_granularity_host(const int mmq_x, const int cc) {
	return new_mma_available(cc) && mmq_x >= 48 ? 16 : 8;
}

#ifdef NEW_MMA_AVAILABLE
static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
	return mmq_x >= 48 ? 16 : 8;
}
#else
static constexpr __device__ int mmq_get_granularity_device(const int /* mmq_x */) {
	return 8;
}
#endif// NEW_MMA_AVAILABLE

// ------------------------------------------------------------

template<int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_q8_0(const char* __restrict x, int* __restrict x_tile, const int& kbx0, const int& i_max, const int& stride) {
#ifdef NEW_MMA_AVAILABLE
	int* x_qs	= ( int* )x_tile;
	float* x_df = ( float* )(x_tile + 2 * WARP_SIZE);
#else
	constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(oiml::oiml_representation_types::q8_0, mmq_y);
	int* x_qs				   = ( int* )x_tile;
	float* x_df				   = ( float* )(x_qs + txs.qs);
#endif// NEW_MMA_AVAILABLE

	const int kbx  = threadIdx.x / oiml::QI8_0;
	const int kqsx = threadIdx.x % oiml::QI8_0;

#pragma unroll
	for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
		int i = i0 + threadIdx.y;

		if (need_check) {
			i = min(i, i_max);
		}

		const oiml::block_q8_0<oiml_half_cuda>* bxi = ( const oiml::block_q8_0<oiml_half_cuda>* )x + kbx0 + i * stride + kbx;

#ifdef NEW_MMA_AVAILABLE
		x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 0 + threadIdx.x]		  = get_int_b2(bxi[0].qs, kqsx);
		x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE / oiml::QI8_0].qs, kqsx);
#else
		x_qs[i * (2 * WARP_SIZE + 1) + 0 + threadIdx.x]			= get_int_b2(bxi[0].qs, kqsx);
		x_qs[i * (2 * WARP_SIZE + 1) + WARP_SIZE + threadIdx.x] = get_int_b2(bxi[WARP_SIZE / oiml::QI8_0].qs, kqsx);
#endif// NEW_MMA_AVAILABLE
	}

	const int blocks_per_tile_x_row = 2 * WARP_SIZE / oiml::QI8_0;
	const int kbxd					= threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
	for (int i0 = 0; i0 < mmq_y; i0 += nwarps * oiml::QI8_0 / 2) {
		int i = i0 + threadIdx.y * (oiml::QI8_0 / 2) + threadIdx.x / blocks_per_tile_x_row;

		if (need_check) {
			i = min(i, i_max);
		}

		const oiml::block_q8_0<oiml_half_cuda>* bxi = ( const oiml::block_q8_0<oiml_half_cuda>* )x + kbx0 + i * stride + kbxd;

#ifdef NEW_MMA_AVAILABLE
		x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = bxi->d;
#else
		x_df[i * (2 * WARP_SIZE / oiml::QI8_0) + i / (oiml::QI8_0 / 2) + kbxd] = bxi->d;
#endif// NEW_MMA_AVAILABLE
	}
}

template<typename T> static __device__ __forceinline__ void load_ldmatrix(tile<8, 8, T>& t, const T* __restrict__ xs0, const int stride) {
#ifdef NEW_MMA_AVAILABLE
	int* xi		  = ( int* )t.x;
	const int* xs = ( const int* )xs0 + (threadIdx.x % t.I) * stride + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(xi[0]), "=r"(xi[1]) : "l"(xs));
#else
	load_generic(t, xs0, stride);
#endif// NEW_MMA_AVAILABLE
}

template<typename T> static __device__ __forceinline__ void load_ldmatrix(tile<16, 4, T>& t, const T* __restrict__ xs0, const int stride) {
#ifdef NEW_MMA_AVAILABLE
	int* xi		  = ( int* )t.x;
	const int* xs = ( const int* )xs0 + (threadIdx.x % t.I) * stride;
	asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];" : "=r"(xi[0]), "=r"(xi[1]) : "l"(xs));
#else
	load_generic(xs0, stride);
#endif// NEW_MMA_AVAILABLE
}

template<typename T> static __device__ __forceinline__ void load_ldmatrix(tile<16, 8, T>& t, const T* __restrict__ xs0, const int stride) {
#ifdef NEW_MMA_AVAILABLE
	int* xi		  = ( int* )t.x;
	const int* xs = ( const int* )xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];" : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3]) : "l"(xs));
#else
	load_generic(t, xs0, stride);
#endif// NEW_MMA_AVAILABLE
}

template<typename T> static __device__ __forceinline__ void load_ldmatrix_trans(tile<16, 8, T>& t, const T* __restrict__ xs0, const int stride) {
#ifdef NEW_MMA_AVAILABLE
	int* xi		  = ( int* )t.x;
	const int* xs = ( const int* )xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
	asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];" : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3]) : "l"(xs));
#else
	OIML_UNUSED(t);
	OIML_UNUSED(xs0);
	OIML_UNUSED(stride);
	NO_DEVICE_CODE;
#endif// NEW_MMA_AVAILABLE
}

template<int mmq_x, int mmq_y, int nwarps, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(const int* __restrict__ x, const int* __restrict__ y, float* __restrict__ sum, const int& k00) {
	typedef tile<16, 8, int> tile_A;
	typedef tile<8, 8, int> tile_B;
	typedef tile<16, 8, int> tile_C;

	constexpr int granularity	= mmq_get_granularity_device(mmq_x);
	constexpr int rows_per_warp = 2 * granularity;
	constexpr int ntx			= rows_per_warp / tile_C::I;// Number of x minitiles per warp.

	y += (threadIdx.y % ntx) * (tile_B::I * MMQ_TILE_Y_K);

	const int* x_qs	  = ( const int* )x;
	const float* x_df = ( const float* )x_qs + 2 * WARP_SIZE;
	const int* y_qs	  = ( const int* )y + 4;
	const float* y_df = ( const float* )y;
	const half2* y_ds = ( const half2* )y;

	tile_A A[ntx][WARP_SIZE / oiml::QI8_0];
	float dA[ntx][tile_C::ne / 2][WARP_SIZE / oiml::QI8_0];

	const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
	for (int n = 0; n < ntx; ++n) {
#pragma unroll
		for (int k01 = 0; k01 < WARP_SIZE; k01 += oiml::QI8_0) {
			const int k0 = k00 + k01;

			load_ldmatrix(A[n][k01 / oiml::QI8_0], x_qs + (i0 + n * tile_A::I) * MMQ_MMA_TILE_X_K_Q8_0 + k0, MMQ_MMA_TILE_X_K_Q8_0);
		}

#pragma unroll
		for (int l = 0; l < tile_C::ne / 2; ++l) {
			const int i = i0 + n * tile_A::I + tile_C::get_i(2 * l);

#pragma unroll
			for (int k01 = 0; k01 < WARP_SIZE; k01 += oiml::QI8_0) {
				const int k0 = k00 + k01;

				dA[n][l][k01 / oiml::QI8_0] = x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + k0 / oiml::QI8_0];
			}
		}
	}

#pragma unroll
	for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
		for (int k01 = 0; k01 < WARP_SIZE; k01 += oiml::QI8_0) {
			tile_B B;
			float dB[tile_C::ne / 2];

			load_generic(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);// faster than load_ldmatrix

#pragma unroll
			for (int l = 0; l < tile_C::ne / 2; ++l) {
				const int j = j0 + tile_C::get_j(l);

				if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
					dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / oiml::QI8_0];
				} else {
					dB[l] = __low2float(y_ds[j * MMQ_TILE_Y_K + k01 / oiml::QI8_0]);
				}
			}

#pragma unroll
			for (int n = 0; n < ntx; ++n) {
				tile_C C;
				mma(C, A[n][k01 / oiml::QI8_0], B);

#pragma unroll
				for (int l = 0; l < tile_C::ne; ++l) {
					sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l] * dA[n][l / 2][k01 / oiml::QI8_0] * dB[l % 2];
				}
			}
		}
	}
}

template<int mmq_x, int mmq_y, int nwarps, bool need_check, oiml_representation_types type> struct mmq_type_traits;

template<int mmq_x, int mmq_y, int nwarps, bool need_check> struct mmq_type_traits<mmq_x, mmq_y, nwarps, need_check, oiml::oiml_representation_types::q8_0> {
	static constexpr int vdr					 = VDR_Q8_0_Q8_1_MMQ;
	static constexpr load_tiles_mmq_t load_tiles = load_tiles_q8_0<mmq_y, nwarps, need_check>;
};

// The mul_mat_q kernel implements "stream-k" work partitioning as described in https://arxiv.org/abs/2301.03598
