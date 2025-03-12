#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE 256
#define HALF_MAX_HALF __float2half(65504.0f / 2)// Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f// Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

typedef void (*fattn_kernel_t)(const char* __restrict__ Q, const char* __restrict__ K, const char* __restrict__ V, const char* __restrict__ mask, float* __restrict__ dst,
	float2* __restrict__ dst_meta, const float scale, const float max_bias, const float m0, const float m1, const uint32_t n_head_log2, const float logit_softcap, const int ne00,
	const int ne01, const int ne02, const int ne03, const int ne10, const int ne11, const int ne12, const int ne13, const int ne31, const int nb31, const int nb01, const int nb02,
	const int nb03, const int nb11, const int nb12, const int nb13, const int nb21, const int nb22, const int nb23, const int ne0, const int ne1, const int ne2, const int ne3);

typedef half (*vec_dot_KQ_f16_t)(const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds);
typedef float (*vec_dot_KQ_f32_t)(const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8, const void* __restrict__ Q_ds);

template<typename T, int D> static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q8_0(const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8,
	const void* __restrict__ Q_ds_v) {
	const oiml::block_q8_0<oiml_half_cuda>* K_q8_0 = ( const oiml::block_q8_0<oiml_half_cuda>* )K_c;
	constexpr int warp_size						   = oiml_cuda_get_physical_warp_size();
	OIML_UNUSED(Q_v);

	T sum = 0.0f;

#pragma unroll
	for (int k_KQ_0 = 0; k_KQ_0 < D / sizeof(int); k_KQ_0 += warp_size) {
		const int k_KQ = k_KQ_0 + threadIdx.x;

		const int ib  = k_KQ / oiml::QI8_0;
		const int iqs = k_KQ % oiml::QI8_0;

		const int v = get_int_b2(K_q8_0[ib].qs, iqs);

		T Q_d;
		if (std::is_same<T, half>::value) {
			const half2* Q_ds = ( const half2* )Q_ds_v;
			Q_d				  = __low2half(Q_ds[k_KQ_0 / warp_size]);
		} else {
			const float2* Q_ds = ( const float2* )Q_ds_v;
			Q_d				   = Q_ds[k_KQ_0 / warp_size].x;
		}

		sum += vec_dot_q8_0_q8_1_impl<T, 1>(&v, &Q_q8[k_KQ_0 / warp_size], K_q8_0[ib].d, Q_d);
	}

	return sum;
}

template<typename T, int D> static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_f16(const char* __restrict__ K_c, const void* __restrict__ Q_v, const int* __restrict__ Q_q8,
	const void* __restrict__ Q_ds_v) {
	const half2* K_h2		= ( const half2* )K_c;
	constexpr int warp_size = oiml_cuda_get_physical_warp_size();
	OIML_UNUSED(Q_q8);
	OIML_UNUSED(Q_ds_v);

#ifdef FP16_AVAILABLE
	if (std::is_same<T, half>::value) {
		const half2* Q_h2 = ( const half2* )Q_v;

		half2 sum2 = make_half2(0.0f, 0.0f);

	#pragma unroll
		for (int k_KQ_0 = 0; k_KQ_0 < D / 2; k_KQ_0 += warp_size) {
			const int k_KQ = k_KQ_0 + threadIdx.x;

			const half2 K_ik = K_h2[k_KQ];
			sum2 += K_ik * Q_h2[k_KQ_0 / warp_size];
		}

		return __low2half(sum2) + __high2half(sum2);
	}
#endif// FP16_AVAILABLE

	const float2* Q_f2 = ( const float2* )Q_v;

	float sum = 0.0f;

#pragma unroll
	for (int k_KQ_0 = 0; k_KQ_0 < D / 2; k_KQ_0 += warp_size) {
		const int k_KQ = k_KQ_0 + threadIdx.x;

		const half2 K_ik = K_h2[k_KQ];
		sum += __low2float(K_ik) * Q_f2[k_KQ_0 / warp_size].x;
		sum += __high2float(K_ik) * Q_f2[k_KQ_0 / warp_size].y;
	}

	return sum;
}

template<typename T> static __device__ __forceinline__ T dequantize_1_q8_0(const void* __restrict__ vx, const int64_t i) {
	const oiml::block_q8_0<oiml_half_cuda>* x = ( const oiml::block_q8_0<oiml_half_cuda>* )vx;

	const int64_t ib = i / oiml::Q_SIZE;
	const int iqs	 = i % oiml::Q_SIZE;

	const T d	= x[ib].d;
	const int q = x[ib].qs[iqs];

#ifdef FP16_AVAILABLE
	if (std::is_same<T, half>::value) {
		return (( half )d) * (( half )q);
	}
#endif// FP16_AVAILABLE

	return (( float )d) * (( float )q);
}

template<typename T> static __device__ __forceinline__ T dequantize_1_f16(const void* __restrict__ vx, const int64_t i) {
	const half* x = ( const half* )vx;

	return x[i];
}

template<int D> constexpr __device__ vec_dot_KQ_f16_t get_vec_dot_KQ_f16(oiml_representation_types type_K) {
	return vec_dot_fattn_vec_KQ_q8_0<half, D>;
}

template<int D> constexpr __device__ vec_dot_KQ_f32_t get_vec_dot_KQ_f32(oiml_representation_types type_K) {
	return vec_dot_fattn_vec_KQ_q8_0<float, D>;
}

static void on_no_fattn_vec_case(const int D) {
	if (D == 64) {
		fprintf(stderr, "Unsupported KV type combination for head_size 64.\n");
		fprintf(stderr, "By default only f16 KV cache is supported.\n");
		fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for V cache quantization support.\n");
		OIML_ABORT("fatal error");
	} else if (D == 128) {
		fprintf(stderr, "Unsupported KV type combination for head_size 128.\n");
		fprintf(stderr, "Supported combinations:\n");
		fprintf(stderr, "  - K == q4_0, V == q4_0,  4.50 BPV\n");
		fprintf(stderr, "  - K == q8_0, V == q8_0,  8.50 BPV\n");
		fprintf(stderr, "  - K == f16,  V == f16,  16.00 BPV\n");
		fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for all combinations of q4_0, q4_1, q5_0, q5_1, q8_0, and f16.\n");
		OIML_ABORT("fatal error");
	} else {
		fprintf(stderr, "Unsupported KV type combination for head_size 256.\n");
		fprintf(stderr, "Only f16 is supported.\n");
		OIML_ABORT("fatal error");
	}
}

template<int D, int parallel_blocks>// D == head size
#if !(defined(OIML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif// !(defined(OIML_USE_HIP) && defined(__HIP_PLATFORM_AMD__))
	static __global__ void flash_attn_combine_results(const float* __restrict__ VKQ_parts, const float2* __restrict__ VKQ_meta, float* __restrict__ dst) {
	VKQ_parts += parallel_blocks * D * gridDim.y * blockIdx.x;
	VKQ_meta += parallel_blocks * gridDim.y * blockIdx.x;
	dst += D * gridDim.y * blockIdx.x;

	const int tid = threadIdx.x;
	__builtin_assume(tid < D);

	__shared__ float2 meta[parallel_blocks];
	if (tid < 2 * parallel_blocks) {
		(( float* )meta)[threadIdx.x] = (( const float* )VKQ_meta)[blockIdx.y * (2 * parallel_blocks) + tid];
	}

	__syncthreads();

	float kqmax = meta[0].x;
#pragma unroll
	for (int l = 1; l < parallel_blocks; ++l) {
		kqmax = max(kqmax, meta[l].x);
	}

	float VKQ_numerator	  = 0.0f;
	float VKQ_denominator = 0.0f;
#pragma unroll
	for (int l = 0; l < parallel_blocks; ++l) {
		const float diff		 = meta[l].x - kqmax;
		const float KQ_max_scale = expf(diff);
		const uint32_t ftz_mask	 = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
		*(( uint32_t* )&KQ_max_scale) &= ftz_mask;

		VKQ_numerator += KQ_max_scale * VKQ_parts[l * gridDim.y * D + blockIdx.y * D + tid];
		VKQ_denominator += KQ_max_scale * meta[l].y;
	}

	dst[blockIdx.y * D + tid] = VKQ_numerator / VKQ_denominator;
}

template<int D, int ncols1, int ncols2, int KQ_stride>// D == head size
__launch_bounds__(D, 1) static __global__
	void flash_attn_stream_k_fixup(float* __restrict__ dst, const float2* __restrict__ dst_fixup, const int ne01, const int ne02, const int ne11) {
	constexpr int ncols = ncols1 * ncols2;

	const int bidx0 = blockIdx.x;
	const int j		= blockIdx.y;
	const int c		= blockIdx.z;
	const int jc	= j * ncols2 + c;
	const int tid	= threadIdx.x;

	const float* dst_fixup_data = (( const float* )dst_fixup) + gridDim.x * (2 * 2 * ncols);

	const int iter_k = ne11 / FATTN_KQ_STRIDE;
	const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

	const int kbc0		= (bidx0 + 0) * iter_k * iter_j * (ne02 / ncols2) / gridDim.x;
	const int kbc0_stop = (bidx0 + 1) * iter_k * iter_j * (ne02 / ncols2) / gridDim.x;

	const bool did_not_have_any_data   = kbc0 == kbc0_stop;
	const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
	const bool did_not_write_last	   = kbc0 / iter_k == kbc0_stop / iter_k && kbc0_stop % iter_k != 0;
	if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
		return;
	}

	const int channel = kbc0 / (iter_k * iter_j);
	const int jt	  = (kbc0 - channel * iter_k * iter_j) / iter_k;

	if (jt * ncols1 + j >= ne01) {
		return;
	}

	dst += jt * ne02 * (ncols1 * D) + channel * (ncols2 * D) + (j * ne02 + c) * D + tid;

	// Load the partial result that needs a fixup:
	float dst_val = 0.0f;
	float max_val = 0.0f;
	float rowsum  = 0.0f;
	{
		dst_val = *dst;

		const float2 tmp = dst_fixup[bidx0 * ncols + jc];
		max_val			 = tmp.x;
		rowsum			 = tmp.y;
	}

	// Iterate over previous blocks and compute the combined results.
	// All CUDA blocks that get here must have a previous block that needs a fixup.
	int bidx	 = bidx0 - 1;
	int kbc_stop = kbc0;
	while (true) {
		const int kbc = bidx * iter_k * iter_j * (ne02 / ncols2) / gridDim.x;
		if (kbc == kbc_stop) {// Did not have any data.
			bidx--;
			kbc_stop = kbc;
			continue;
		}

		const float dst_add = dst_fixup_data[bidx * ncols * D + jc * D + tid];

		const float2 tmp = dst_fixup[(gridDim.x + bidx) * ncols + jc];

		// Scale the current and new value accumulators depending on the max. values.
		const float max_val_new = fmaxf(max_val, tmp.x);

		const float diff_val = max_val - max_val_new;
		const float diff_add = tmp.x - max_val_new;

		const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
		const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

		dst_val = scale_val * dst_val + scale_add * dst_add;
		rowsum	= scale_val * rowsum + scale_add * tmp.y;

		max_val = max_val_new;

		// If this block started in a previous tile we are done and don't need to combine additional partial results.
		if (kbc % iter_k == 0 || kbc / iter_k < kbc0 / iter_k) {
			break;
		}
		bidx--;
		kbc_stop = kbc;
	}

	// Write back final result:
	*dst = dst_val / rowsum;
}

// parallel_blocks == 0 is stream-k decomposition
template<int D, int ncols1, int ncols2, int parallel_blocks, int KQ_stride> void launch_fattn(oiml_backend_cuda_context& ctx, oiml_tensor* dst, fattn_kernel_t fattn_kernel,
	const int nwarps, const size_t nbytes_shared, const bool need_f16_K, const bool need_f16_V) {
	constexpr int ncols = ncols1 * ncols2;

	const oiml_tensor* Q = dst->src[0];
	const oiml_tensor* K = dst->src[1];
	const oiml_tensor* V = dst->src[2];

	const oiml_tensor* mask = dst->src[3];

	oiml_tensor* KQV = dst;

	OIML_ASSERT(Q->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(KQV->type == oiml::oiml_representation_types::float_32);

	OIML_ASSERT(!mask || mask->type == oiml::oiml_representation_types::float_16);
	OIML_ASSERT(!mask || mask->ne[1] >= OIML_PAD(Q->ne[1], 16) && "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

	OIML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

	OIML_ASSERT(Q->ne[3] == 1);

	const int warp_size = oiml_cuda_info().devices[ctx.device].warp_size;

	oiml_cuda_pool& pool	 = ctx.pool();
	cudaStream_t main_stream = ctx.stream();
	const int id			 = oiml_cuda_get_device();
	const int cc			 = oiml_cuda_info().devices[id].cc;
	const int nsm			 = oiml_cuda_info().devices[id].nsm;

	oiml_cuda_pool_alloc<half> K_f16(pool);
	oiml_cuda_pool_alloc<half> V_f16(pool);
	oiml_cuda_pool_alloc<float> dst_tmp(pool);
	oiml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

	const char* K_data = ( const char* )K->data;
	size_t nb11		   = K->nb[1];
	size_t nb12		   = K->nb[2];
	size_t nb13		   = K->nb[3];

	const char* V_data = ( const char* )V->data;
	size_t nb21		   = V->nb[1];
	size_t nb22		   = V->nb[2];
	size_t nb23		   = V->nb[3];

	if (need_f16_K && K->type != oiml::oiml_representation_types::float_16) {
		K_f16.alloc(oiml_nelements(K));
		to_fp16_cuda_t to_fp16 = oiml_get_to_fp16_cuda(K->type);
		to_fp16(K_data, K_f16.ptr, oiml_nelements(K), main_stream);
		K_data = ( char* )K_f16.ptr;

		const size_t bs = oiml_blck_size(K->type);
		const size_t ts = oiml_type_size(K->type);

		nb11 = nb11 * bs * sizeof(half) / ts;
		nb12 = nb12 * bs * sizeof(half) / ts;
		nb13 = nb13 * bs * sizeof(half) / ts;
	}

	if (need_f16_V && V->type != oiml::oiml_representation_types::float_16) {
		V_f16.alloc(oiml_nelements(V));
		to_fp16_cuda_t to_fp16 = oiml_get_to_fp16_cuda(V->type);
		to_fp16(V_data, V_f16.ptr, oiml_nelements(V), main_stream);
		V_data = ( char* )V_f16.ptr;

		const size_t bs = oiml_blck_size(V->type);
		const size_t ts = oiml_type_size(V->type);

		nb21 = nb21 * bs * sizeof(half) / ts;
		nb22 = nb22 * bs * sizeof(half) / ts;
		nb23 = nb23 * bs * sizeof(half) / ts;
	}

	const int ntiles_x	   = ((Q->ne[1] + ncols1 - 1) / ncols1);
	const int ntiles_total = ntiles_x * (Q->ne[2] / ncols2) * Q->ne[3];

	const dim3 block_dim(warp_size, nwarps, 1);
	dim3 blocks_num;
	if (parallel_blocks == 0) {
		// For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
		const int max_blocks			   = 2 * nsm;
		const int tiles_nwaves			   = (ntiles_total + max_blocks - 1) / max_blocks;
		const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks * tiles_nwaves);

		const int nblocks_stream_k = max_blocks;

		const bool use_stream_k = cc >= OIML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;

		blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
		blocks_num.y = 1;
		blocks_num.z = 1;

		dst_tmp_meta.alloc(blocks_num.x * ncols * (2 * 2 + D) * sizeof(float));
	} else {
		blocks_num.x = parallel_blocks * ntiles_x;
		blocks_num.y = Q->ne[2];
		blocks_num.z = Q->ne[3];

		if (parallel_blocks > 1) {
			dst_tmp.alloc(parallel_blocks * oiml_nelements(KQV));
			dst_tmp_meta.alloc(parallel_blocks * oiml_nrows(KQV));
		}
	}

	float scale			= 1.0f;
	float max_bias		= 0.0f;
	float logit_softcap = 0.0f;

	memcpy(&scale, ( const float* )KQV->op_params + 0, sizeof(float));
	memcpy(&max_bias, ( const float* )KQV->op_params + 1, sizeof(float));
	memcpy(&logit_softcap, ( const float* )KQV->op_params + 2, sizeof(float));

	if (logit_softcap != 0.0f) {
		scale /= logit_softcap;
	}

	const uint32_t n_head	   = Q->ne[2];
	const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

	const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
	const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

	OIML_ASSERT(block_dim.x % warp_size == 0);
	OIML_ASSERT(!OIML_CUDA_CC_IS_AMD(cc) || block_dim.x * block_dim.y <= 4 * ( unsigned int )warp_size);
	fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(( const char* )Q->data, K_data, V_data, mask ? (( const char* )mask->data) : nullptr,
		(parallel_blocks) > 1 ? dst_tmp.ptr : ( float* )KQV->data, dst_tmp_meta.ptr, scale, max_bias, m0, m1, n_head_log2, logit_softcap, Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3],
		K->ne[0], K->ne[1], K->ne[2], K->ne[3], mask ? mask->ne[1] : 0, mask ? mask->nb[1] : 0, Q->nb[1], Q->nb[2], Q->nb[3], nb11, nb12, nb13, nb21, nb22, nb23, KQV->ne[0],
		KQV->ne[1], KQV->ne[2], KQV->ne[3]);
	CUDA_CHECK(cudaGetLastError());

	if constexpr (parallel_blocks == 0) {
		if (ntiles_total % blocks_num.x != 0) {// Fixup is only needed if the SMs work on fractional tiles.
			const dim3 block_dim_combine(D, 1, 1);
			const dim3 blocks_num_combine = { blocks_num.x, ncols1, ncols2 };

			flash_attn_stream_k_fixup<D, ncols1, ncols2, KQ_stride>
				<<<blocks_num_combine, block_dim_combine, 0, main_stream>>>(( float* )KQV->data, dst_tmp_meta.ptr, Q->ne[1], Q->ne[2], K->ne[1]);
		}
	} else if constexpr (parallel_blocks > 1) {
		const dim3 block_dim_combine(D, 1, 1);
		const dim3 blocks_num_combine(Q->ne[1], blocks_num.y, blocks_num.z);

		flash_attn_combine_results<D, parallel_blocks><<<blocks_num_combine, block_dim_combine, 0, main_stream>>>(dst_tmp.ptr, dst_tmp_meta.ptr, ( float* )KQV->data);
	}
	CUDA_CHECK(cudaGetLastError());
}
