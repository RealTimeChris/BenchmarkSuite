#include "convert.cuh"
#include "dequantize.cuh"

#define CUDA_Q8_0_NE_ALIGN 2048

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k) {
	const int64_t i = ( int64_t )2 * (blockDim.x * blockIdx.x + threadIdx.x);

	if (i >= k) {
		return;
	}

	const int64_t ib	   = i / qk;// block index
	const int64_t iqs	   = (i % qk) / qr;// quant index
	const int64_t iybs	   = i - i % qk;// y block start index
	const int64_t y_offset = qr == 1 ? 1 : qk / 2;

	// dequantize
	dfloat2 v;
	dequantize_kernel(vx, ib, iqs, v);

	y[iybs + iqs + 0]		 = v.x;
	y[iybs + iqs + y_offset] = v.y;
}

template<bool need_check> static __global__ void dequantize_block_q8_0_f16(const void* __restrict__ vx, half* __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= OIML_CUDA_CC_PASCAL
	constexpr int nint = CUDA_Q8_0_NE_ALIGN / sizeof(int) + WARP_SIZE;

	const int64_t i0 = CUDA_Q8_0_NE_ALIGN * blockIdx.x;
	const int* x0	 = (( int* )vx) + blockIdx.x * nint;
	half2* y2		 = ( half2* )(y + i0);

	__shared__ int vals[nint];

	#pragma unroll
	for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
		if (need_check &&
			i0 * sizeof(oiml::block_q8_0<oiml_half_cuda>) / oiml::Q_SIZE + sizeof(int) * (ix0 + threadIdx.x) >= k * sizeof(oiml::block_q8_0<oiml_half_cuda>) / oiml::Q_SIZE) {
			break;
		}

		const int ix = ix0 + threadIdx.x;
		vals[ix]	 = x0[ix];
	}

	__syncthreads();

	#pragma unroll
	for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2 * WARP_SIZE) {
		if (need_check && i0 + iy + 2 * threadIdx.x >= k) {
			return;
		}

		const half* b0 = (( const half* )vals) + (sizeof(oiml::block_q8_0<oiml_half_cuda>) / sizeof(half)) * ((iy + 2 * threadIdx.x) / oiml::Q_SIZE);
		const half d   = *b0;
		const char2 qs = (( const char2* )(b0 + 1))[threadIdx.x % (oiml::Q_SIZE / 2)];

		y2[iy / 2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
	}
#else
	OIML_UNUSED(vx);
	OIML_UNUSED(y);
	OIML_UNUSED(k);
#endif// __CUDA_ARCH__ >= OIML_CUDA_CC_PASCAL
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k, cudaStream_t stream) {
	const int num_blocks = (k + 2 * CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
	dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

static void dequantize_block_q8_0_f16_cuda(const void* __restrict__ vx, half* __restrict__ y, const int64_t k, cudaStream_t stream) {
	const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
	if (k % CUDA_Q8_0_NE_ALIGN == 0) {
		const bool need_check = false;
		dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
	} else {
		const bool need_check = true;
		dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
	}
}

template<typename dst_t> static void dequantize_row_q2_K_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_q3_K_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_q4_0_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb32 = k / 32;
	const int nb   = (k + 255) / 256;
	dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t> static void dequantize_row_q4_1_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb32 = k / 32;
	const int nb   = (k + 255) / 256;
	dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t> static void dequantize_row_q4_K_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_q5_K_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_q6_K_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq2_xxs_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq2_xs_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq2_s_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq3_xxs_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq3_s_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq1_s_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq4_nl_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = (k + oiml::QK_K - 1) / oiml::QK_K;
	dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq1_m_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = k / oiml::QK_K;
	dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t> static void dequantize_row_iq4_xs_cuda(const void* vx, dst_t* y, const int64_t k, cudaStream_t stream) {
	const int nb = (k + oiml::QK_K - 1) / oiml::QK_K;
	dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename src_t, typename dst_t> static __global__ void convert_unary(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k) {
	const int64_t i = ( int64_t )blockDim.x * blockIdx.x + threadIdx.x;

	if (i >= k) {
		return;
	}

	const src_t* x = ( src_t* )vx;

	y[i] = x[i];
}

template<typename src_t, typename dst_t> static void convert_unary_cuda(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k, cudaStream_t stream) {
	const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
	convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(vx, y, k);
}

to_fp16_cuda_t oiml_get_to_fp16_cuda(oiml_representation_types type) {
	switch (type) {
		case oiml::oiml_representation_types::q8_0:
			if (fp16_available(oiml_cuda_info().devices[oiml_cuda_get_device()].cc)) {
				return dequantize_block_q8_0_f16_cuda;
			}
		case oiml::oiml_representation_types::float_32:
			return convert_unary_cuda<float>;
		default:
			return nullptr;
	}
}

to_fp32_cuda_t oiml_get_to_fp32_cuda(oiml_representation_types type) {
	switch (type) {
		case oiml::oiml_representation_types::q8_0:
			return dequantize_block_cuda<oiml::Q_SIZE, oiml::QR8_0, dequantize_q8_0>;
		case oiml::oiml_representation_types::float_16:
			return convert_unary_cuda<half>;
		case oiml::oiml_representation_types::brain_float_16:
			return convert_unary_cuda<nv_bfloat16>;
		default:
			return nullptr;
	}
}
