#include <oiml/legacy/oiml-legacy-common/oiml.hpp>
#include "common.cuh"
#include "mmv.cuh"

template<typename T, typename type_acc, int block_size> static __global__ void mul_mat_vec(const T* __restrict__ x, const float* __restrict__ y, float* __restrict__ dst,
	const int64_t ncols2, const int64_t stride_row, const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
	const int64_t sample_ratio, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst) {
	const int64_t row		= blockIdx.x;
	const int64_t channel	= blockIdx.y;
	const int64_t sample	= blockIdx.z;
	const int tid			= threadIdx.x;
	constexpr int warp_size = oiml_cuda_get_physical_warp_size();

	x += (sample / sample_ratio) * stride_sample_x + (channel / channel_ratio) * stride_channel_x + row * stride_row;
	y += sample * stride_sample_y + channel * stride_channel_y;
	dst += sample * stride_sample_dst + channel * stride_channel_dst;

	const float2* y2 = ( const float2* )y;

	extern __shared__ char data_mmv[];
	float* buf_iw = ( float* )data_mmv;

	if (block_size > warp_size) {
		if (tid < warp_size) {
			buf_iw[tid] = 0.0f;
		}
		__syncthreads();
	}

	float sumf;

	if constexpr (std::is_same<T, half>::value) {
		const half2* x2 = ( const half2* )x;

		if (std::is_same<type_acc, float>::value) {
			sumf = 0.0f;

			for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
				const float2 tmpx = __half22float2(x2[col2]);
				const float2 tmpy = y2[col2];
				sumf += tmpx.x * tmpy.x;
				sumf += tmpx.y * tmpy.y;
			}
		} else {
#ifdef FP16_AVAILABLE
			half2 sumh2 = make_half2(0.0f, 0.0f);

			for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
				const float2 tmp = y2[col2];
				sumh2 += x2[col2] * make_half2(tmp.x, tmp.y);
			}

			sumf = __low2float(sumh2) + __high2float(sumh2);
#else
#endif// FP16_AVAILABLE
		}
	} else if constexpr (std::is_same<T, nv_bfloat16>::value) {
		const int* x2 = ( const int* )x;
		sumf		  = 0.0f;

		for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
			const int tmpx	  = x2[col2];
			const float2 tmpy = y2[col2];
			sumf += float(reinterpret_cast<const nv_bfloat16*>(&tmpx)[0]) * tmpy.x;
			sumf += float(reinterpret_cast<const nv_bfloat16*>(&tmpx)[1]) * tmpy.y;
		}
	} else {
		static_assert(std::is_same<T, void>::value, "unsupported type");
	}

	sumf = warp_reduce_sum<warp_size>(sumf);

	if (block_size > warp_size) {
		buf_iw[tid / warp_size] = sumf;
		__syncthreads();
		if (tid >= warp_size) {
			return;
		}
		sumf = buf_iw[tid];
		sumf = warp_reduce_sum<warp_size>(sumf);
	}

	if (tid != 0) {
		return;
	}

	dst[row] = sumf;
}

template<typename T, typename type_acc> static void launch_mul_mat_vec_cuda(const T* x, const float* y, float* dst, const int64_t ncols, const int64_t nrows,
	const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y, const int64_t stride_channel_x, const int64_t stride_channel_y,
	const int64_t stride_channel_dst, const int64_t nsamples_x, const int64_t nsamples_y, const int64_t stride_sample_x, const int64_t stride_sample_y,
	const int64_t stride_sample_dst, cudaStream_t stream) {
	OIML_ASSERT(ncols % 2 == 0);
	OIML_ASSERT(stride_row % 2 == 0);
	OIML_ASSERT(nchannels_y % nchannels_x == 0);
	OIML_ASSERT(nsamples_y % nsamples_x == 0);
	const int64_t channel_ratio = nchannels_y / nchannels_x;
	const int64_t sample_ratio	= nsamples_y / nsamples_x;
	int device;
	int warp_size;

	CUDA_CHECK(cudaGetDevice(&device));
	warp_size = oiml_cuda_info().devices[device].warp_size;

	int64_t block_size_best = warp_size;
	int64_t niter_best		= (ncols + 2 * warp_size - 1) / (2 * warp_size);
	int64_t max_block_size	= 256;
	if (oiml_cuda_info().devices[device].cc > OIML_CUDA_CC_OFFSET_AMD && oiml_cuda_info().devices[device].cc < OIML_CUDA_CC_RDNA1) {
		max_block_size = 128;
	}
	for (int64_t block_size = 2 * warp_size; block_size <= max_block_size; block_size += warp_size) {
		const int64_t niter = (ncols + 2 * block_size - 1) / (2 * block_size);
		if (niter < niter_best) {
			niter_best		= niter;
			block_size_best = block_size;
		}
	}

	const int smem = warp_size * sizeof(float);
	const dim3 block_nums(nrows, nchannels_y, nsamples_y);
	const dim3 block_dims(block_size_best, 1, 1);
	switch (block_size_best) {
		case 32: {
			mul_mat_vec<T, type_acc, 32><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		case 64: {
			mul_mat_vec<T, type_acc, 64><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		case 96: {
			mul_mat_vec<T, type_acc, 96><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		case 128: {
			mul_mat_vec<T, type_acc, 128><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		case 160: {
			mul_mat_vec<T, type_acc, 160><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		case 192: {
			mul_mat_vec<T, type_acc, 192><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		case 224: {
			mul_mat_vec<T, type_acc, 224><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		case 256: {
			mul_mat_vec<T, type_acc, 256><<<block_nums, block_dims, smem, stream>>>(x, y, dst, ncols / 2, stride_row, channel_ratio, stride_channel_x, stride_channel_y,
				stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst);
		} break;
		default: {
			OIML_ABORT("fatal error");
		} break;
	}
}

template<typename T> static void mul_mat_vec_cuda(const T* x, const float* y, float* dst, const int64_t ncols, const int64_t nrows, const int64_t stride_row,
	const int64_t nchannels_x, const int64_t nchannels_y, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
	const int64_t nsamples_x, const int64_t nsamples_y, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst, enum oiml_prec prec,
	cudaStream_t stream) {
	switch (prec) {
		case OIML_PREC_DEFAULT: {
			launch_mul_mat_vec_cuda<T, half>(x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x,
				nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
		} break;
		case OIML_PREC_F32: {
			launch_mul_mat_vec_cuda<T, float>(x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x,
				nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, stream);
		} break;
	}
}

void oiml_cuda_mul_mat_vec(oiml_backend_cuda_context& ctx, const oiml_tensor* src0, const oiml_tensor* src1, oiml_tensor* dst) {
	OIML_ASSERT(src1->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);

	OIML_TENSOR_BINARY_OP_LOCALS;

	const size_t ts_src0 = oiml_type_size(src0->type);
	const size_t ts_src1 = oiml_type_size(src1->type);
	const size_t ts_dst	 = oiml_type_size(dst->type);

	OIML_ASSERT(ne11 == 1);
	OIML_ASSERT(ne12 == ne2);
	OIML_ASSERT(ne13 == ne3);

	OIML_ASSERT(nb00 == ts_src0);
	OIML_ASSERT(nb10 == ts_src1);
	OIML_ASSERT(nb0 == ts_dst);

	const int cc			  = oiml_cuda_info().devices[oiml_cuda_get_device()].cc;
	const enum oiml_prec prec = fast_fp16_available(cc) ? oiml_prec(dst->op_params[0]) : OIML_PREC_F32;

	const float* src1_d = ( const float* )src1->data;
	float* dst_d		= ( float* )dst->data;

	const int64_t s01 = src0->nb[1] / ts_src0;
	const int64_t s02 = src0->nb[2] / ts_src0;
	const int64_t s12 = src1->nb[2] / ts_src1;
	const int64_t s2  = dst->nb[2] / ts_dst;
	const int64_t s03 = src0->nb[3] / ts_src0;
	const int64_t s13 = src1->nb[3] / ts_src1;
	const int64_t s3  = dst->nb[3] / ts_dst;

	switch (src0->type) {
		case oiml::oiml_representation_types::float_16: {
			const half* src0_d = ( const half* )src0->data;
			mul_mat_vec_cuda(src0_d, src1_d, dst_d, ne00, ne01, s01, ne02, ne12, s02, s12, s2, ne03, ne13, s03, s13, s3, prec, ctx.stream());
		} break;
		case oiml::oiml_representation_types::brain_float_16: {
			const nv_bfloat16* src0_d = ( const nv_bfloat16* )src0->data;
			mul_mat_vec_cuda(src0_d, src1_d, dst_d, ne00, ne01, s01, ne02, ne12, s02, s12, s2, ne03, ne13, s03, s13, s3, prec, ctx.stream());
		} break;
		default:
			OIML_ABORT("unsupported type: %s", oiml_type_name(src0->type));
	}
}

void oiml_cuda_op_mul_mat_vec(oiml_backend_cuda_context& ctx, const oiml_tensor* src0, const oiml_tensor* src1, oiml_tensor* dst, const char* src0_dd_i, const float* src1_ddf_i,
	const char* src1_ddq_i, float* dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_row_size, cudaStream_t stream) {
	OIML_ASSERT(src1->type == oiml::oiml_representation_types::float_32);
	OIML_ASSERT(dst->type == oiml::oiml_representation_types::float_32);

	const int64_t ne00	   = src0->ne[0];
	const int64_t row_diff = row_high - row_low;

	OIML_ASSERT(src1_ncols == 1);

	const int cc			  = oiml_cuda_info().devices[oiml_cuda_get_device()].cc;
	const enum oiml_prec prec = fast_fp16_available(cc) ? oiml_prec(dst->op_params[0]) : OIML_PREC_F32;


	// oiml_cuda_op provides single, contiguous matrices
	const int64_t stride_row		 = ne00;
	const int64_t nchannels_x		 = 1;
	const int64_t nchannels_y		 = 1;
	const int64_t stride_channel_x	 = 0;
	const int64_t stride_channel_y	 = 0;
	const int64_t stride_channel_dst = 0;
	const int64_t nsamples_x		 = 1;
	const int64_t nsamples_y		 = 1;
	const int64_t stride_sample_x	 = 0;
	const int64_t stride_sample_y	 = 0;
	const int64_t stride_sample_dst	 = 0;

	switch (src0->type) {
		case oiml::oiml_representation_types::float_16: {
			const half* src0_d = ( const half* )src0_dd_i;
			mul_mat_vec_cuda(src0_d, src1_ddf_i, dst_dd_i, ne00, row_diff, stride_row, nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x,
				nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream);
		} break;
		case oiml::oiml_representation_types::brain_float_16: {
			const nv_bfloat16* src0_d = ( const nv_bfloat16* )src0_dd_i;
			mul_mat_vec_cuda(src0_d, src1_ddf_i, dst_dd_i, ne00, row_diff, stride_row, nchannels_x, nchannels_y, stride_channel_x, stride_channel_y, stride_channel_dst, nsamples_x,
				nsamples_y, stride_sample_x, stride_sample_y, stride_sample_dst, prec, stream);
		} break;
		default:
			OIML_ABORT("unsupported type: %s", oiml_type_name(src0->type));
	}

	OIML_UNUSED(ctx);
	OIML_UNUSED(src1);
	OIML_UNUSED(dst);
	OIML_UNUSED(src1_ddq_i);
	OIML_UNUSED(src1_ncols);
	OIML_UNUSED(src1_padded_row_size);
}
