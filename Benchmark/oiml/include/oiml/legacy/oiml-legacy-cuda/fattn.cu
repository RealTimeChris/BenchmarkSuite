#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16.cuh"
#include "fattn-vec-f32.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

template<int D, int ncols2> static void oiml_cuda_flash_attn_ext_mma_f16_switch_ncols1(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* Q = dst->src[0];

	if (Q->ne[1] <= 8 / ncols2) {
		oiml_cuda_flash_attn_ext_mma_f16_case<D, 8 / ncols2, ncols2>(ctx, dst);
		return;
	}

	if (Q->ne[1] <= 16 / ncols2) {
		oiml_cuda_flash_attn_ext_mma_f16_case<D, 16 / ncols2, ncols2>(ctx, dst);
		return;
	}

	if (Q->ne[1] <= 32 / ncols2) {
		oiml_cuda_flash_attn_ext_mma_f16_case<D, 32 / ncols2, ncols2>(ctx, dst);
		return;
	}

	oiml_cuda_flash_attn_ext_mma_f16_case<D, 64 / ncols2, ncols2>(ctx, dst);
}

template<int ncols2> static void oiml_cuda_flash_attn_ext_mma_f16_switch_hs(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* Q = dst->src[0];

	switch (Q->ne[0]) {
		case 64:
			oiml_cuda_flash_attn_ext_mma_f16_switch_ncols1<64, ncols2>(ctx, dst);
			break;
		case 80:
			oiml_cuda_flash_attn_ext_mma_f16_switch_ncols1<80, ncols2>(ctx, dst);
			break;
		case 96:
			oiml_cuda_flash_attn_ext_mma_f16_switch_ncols1<96, ncols2>(ctx, dst);
			break;
		case 112:
			oiml_cuda_flash_attn_ext_mma_f16_switch_ncols1<112, ncols2>(ctx, dst);
			break;
		case 128:
			oiml_cuda_flash_attn_ext_mma_f16_switch_ncols1<128, ncols2>(ctx, dst);
			break;
		case 256:
			oiml_cuda_flash_attn_ext_mma_f16_switch_ncols1<256, ncols2>(ctx, dst);
			break;
		default:
			OIML_ABORT("fatal error");
			break;
	}
}

static void oiml_cuda_flash_attn_ext_mma_f16(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* KQV	= dst;
	const oiml_tensor* Q	= dst->src[0];
	const oiml_tensor* K	= dst->src[1];
	const oiml_tensor* mask = dst->src[3];

	float max_bias = 0.0f;
	memcpy(&max_bias, ( const float* )KQV->op_params + 1, sizeof(float));

	const float use_gqa_opt = mask && max_bias == 0.0f;

	OIML_ASSERT(Q->ne[2] % K->ne[2] == 0);
	const int gqa_ratio = Q->ne[2] / K->ne[2];

	if (use_gqa_opt && gqa_ratio % 8 == 0) {
		oiml_cuda_flash_attn_ext_mma_f16_switch_hs<8>(ctx, dst);
		return;
	}

	if (use_gqa_opt && gqa_ratio == 4) {
		oiml_cuda_flash_attn_ext_mma_f16_switch_hs<4>(ctx, dst);
		return;
	}

	if (use_gqa_opt && gqa_ratio == 2) {
		oiml_cuda_flash_attn_ext_mma_f16_switch_hs<2>(ctx, dst);
		return;
	}

	oiml_cuda_flash_attn_ext_mma_f16_switch_hs<1>(ctx, dst);
}

#define FATTN_VEC_F16_CASE(D, type_K, type_V) \
	if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) { \
		oiml_cuda_flash_attn_ext_vec_f16_case<D, type_K, type_V>(ctx, dst); \
		return; \
	}

static void oiml_cuda_flash_attn_ext_vec_f16(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_tensor* Q = dst->src[0];
	oiml_tensor* K = dst->src[1];
	oiml_tensor* V = dst->src[2];

#ifdef OIML_CUDA_FA_ALL_QUANTS
	FATTN_VEC_F16_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_0)
	FATTN_VEC_F16_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_1)
	FATTN_VEC_F16_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_0)
	FATTN_VEC_F16_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_1)
	FATTN_VEC_F16_CASE(64, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F16_CASE(64, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)

	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q4_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q4_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q4_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q4_0)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q4_0)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_0)

	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q4_1)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q4_1)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q4_1)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q4_1)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q4_1)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_1)

	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q5_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q5_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q5_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q5_0)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q5_0)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_0)

	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q5_1)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q5_1)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q5_1)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q5_1)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q5_1)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_1)

	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_0, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_1, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_0, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_1, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::q8_0, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::q8_0)

	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_0, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q4_1, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_0, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F16_CASE(128, OIML_TYPE_Q5_1, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::q8_0, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)

	FATTN_VEC_F16_CASE(256, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
#else

	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::q8_0, oiml::oiml_representation_types::q8_0)

	FATTN_VEC_F16_CASE(64, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F16_CASE(128, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F16_CASE(256, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
#endif// OIML_CUDA_FA_ALL_QUANTS

	on_no_fattn_vec_case(Q->ne[0]);
}

#define FATTN_VEC_F32_CASE(D, type_K, type_V) \
	if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) { \
		oiml_cuda_flash_attn_ext_vec_f32_case<D, type_K, type_V>(ctx, dst); \
		return; \
	}

static void oiml_cuda_flash_attn_ext_vec_f32(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	oiml_tensor* Q = dst->src[0];
	oiml_tensor* K = dst->src[1];
	oiml_tensor* V = dst->src[2];

#ifdef OIML_CUDA_FA_ALL_QUANTS
	FATTN_VEC_F32_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_0)
	FATTN_VEC_F32_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_1)
	FATTN_VEC_F32_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_0)
	FATTN_VEC_F32_CASE(64, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_1)
	FATTN_VEC_F32_CASE(64, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F32_CASE(64, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)

	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q4_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q4_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q4_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q4_0)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q4_0)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_0)

	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q4_1)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q4_1)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q4_1)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q4_1)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q4_1)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q4_1)

	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q5_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q5_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q5_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q5_0)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q5_0)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_0)

	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_0, OIML_TYPE_Q5_1)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_1, OIML_TYPE_Q5_1)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_0, OIML_TYPE_Q5_1)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_1, OIML_TYPE_Q5_1)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::q8_0, OIML_TYPE_Q5_1)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::float_16, OIML_TYPE_Q5_1)

	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_0, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_1, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_0, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_1, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::q8_0, oiml::oiml_representation_types::q8_0)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::q8_0)

	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_0, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q4_1, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_0, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F32_CASE(128, OIML_TYPE_Q5_1, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::q8_0, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)

	FATTN_VEC_F32_CASE(256, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
#else

	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::q8_0, oiml::oiml_representation_types::q8_0)

	FATTN_VEC_F32_CASE(64, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F32_CASE(128, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
	FATTN_VEC_F32_CASE(256, oiml::oiml_representation_types::float_16, oiml::oiml_representation_types::float_16)
#endif// OIML_CUDA_FA_ALL_QUANTS

	on_no_fattn_vec_case(Q->ne[0]);
}

void oiml_cuda_flash_attn_ext(oiml_backend_cuda_context& ctx, oiml_tensor* dst) {
	const oiml_tensor* KQV	= dst;
	const oiml_tensor* Q	= dst->src[0];
	const oiml_tensor* K	= dst->src[1];
	const oiml_tensor* V	= dst->src[2];
	const oiml_tensor* mask = dst->src[3];

	oiml_cuda_set_device(ctx.device);
	const int cc			  = oiml_cuda_info().devices[oiml_cuda_get_device()].cc;
	const int warp_size		  = oiml_cuda_info().devices[oiml_cuda_get_device()].warp_size;
	const enum oiml_prec prec = oiml_flash_attn_ext_get_prec(KQV);

	if (cc >= OIML_CUDA_CC_OFFSET_AMD) {
#if defined(OIML_HIP_ROCWMMA_FATTN)
		if (fp16_mma_available(cc)) {
			oiml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
			return;
		}
#endif// defined(OIML_HIP_ROCWMMA_FATTN)

		// On AMD the tile kernels perform poorly, use the vec kernel instead:
		if (prec == OIML_PREC_DEFAULT && fast_fp16_available(cc)) {
			oiml_cuda_flash_attn_ext_vec_f16(ctx, dst);
		} else {
			oiml_cuda_flash_attn_ext_vec_f32(ctx, dst);
		}
		return;
	}

	if (!fast_fp16_available(cc)) {
		if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
			oiml_cuda_flash_attn_ext_vec_f32(ctx, dst);
		} else {
			oiml_cuda_flash_attn_ext_tile_f32(ctx, dst);
		}
		return;
	}

	if (!fp16_mma_available(cc)) {
		if (prec == OIML_PREC_DEFAULT) {
			if (Q->ne[1] <= 8) {
				oiml_cuda_flash_attn_ext_vec_f16(ctx, dst);
			} else {
				oiml_cuda_flash_attn_ext_tile_f16(ctx, dst);
			}
		} else {
			if (Q->ne[1] <= 8) {
				oiml_cuda_flash_attn_ext_vec_f32(ctx, dst);
			} else {
				oiml_cuda_flash_attn_ext_tile_f32(ctx, dst);
			}
		}
		return;
	}

	const int gqa_ratio			= Q->ne[2] / K->ne[2];
	const bool mma_fast_for_bs1 = fp16_mma_available(cc) && gqa_ratio % 2 == 0 && K->type == oiml::oiml_representation_types::float_16 && V->type == oiml::oiml_representation_types::float_16 && mask;
	if (Q->ne[1] == 1 && Q->ne[0] % (2 * warp_size) == 0 && !mma_fast_for_bs1) {
		if (prec == OIML_PREC_DEFAULT) {
			oiml_cuda_flash_attn_ext_vec_f16(ctx, dst);
			return;
		} else if (Q->ne[0] <= 128) {
			oiml_cuda_flash_attn_ext_vec_f32(ctx, dst);
			return;
		}
	}

	// The MMA implementation needs Turing or newer, use the old WMMA code for Volta:
	if (fp16_mma_available(cc) && !new_mma_available(cc)) {
		oiml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
		return;
	}

	oiml_cuda_flash_attn_ext_mma_f16(ctx, dst);
}
