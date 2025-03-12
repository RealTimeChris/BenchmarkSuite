#pragma once

#include <oiml/common/util_functions.hpp>
#include <oiml-cpu/detect_isa.hpp>
#include <oiml/common/common.hpp>
#include <oiml-cpu/common.hpp>
#include <assert.h>
#include <cassert>

#if defined(OIML_IS_ARM64)

namespace oiml {

	template<> struct function_dispatcher<function_type::from_float, oiml_representation_types::q8_0, 1> {
		OIML_FORCE_INLINE static void impl(const float* __restrict x, block_q8_0<oiml_half>* __restrict y, int64_t k) {
			assert(QK8_0 == 32);
			assert(k % QK8_0 == 0);
			const int nb = k / QK8_0;

			for (int i = 0; i < nb; i++) {
				float32x4_t srcv[8];
				float32x4_t asrcv[8];
				float32x4_t amaxv[8];

				for (int j = 0; j < 8; j++)
					srcv[j] = vld1q_f32(x + i * 32 + 4 * j);
				for (int j = 0; j < 8; j++)
					asrcv[j] = vabsq_f32(srcv[j]);

				for (int j = 0; j < 4; j++)
					amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
				for (int j = 0; j < 2; j++)
					amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
				for (int j = 0; j < 1; j++)
					amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

				const float amax = vmaxvq_f32(amaxv[0]);

				const float d  = amax / ((1 << 7) - 1);
				const float id = d ? 1.0f / d : 0.0f;

				y[i].d = oiml_fp32_to_fp16(d);

				for (int j = 0; j < 8; j++) {
					const float32x4_t v = vmulq_n_f32(srcv[j], id);
					const int32x4_t vi	= vcvtnq_s32_f32(v);

					y[i].qs[4 * j + 0] = vgetq_lane_s32(vi, 0);
					y[i].qs[4 * j + 1] = vgetq_lane_s32(vi, 1);
					y[i].qs[4 * j + 2] = vgetq_lane_s32(vi, 2);
					y[i].qs[4 * j + 3] = vgetq_lane_s32(vi, 3);
				}
			}
		}
	};

	template<> struct function_dispatcher<function_type::to_float, oiml_representation_types::q8_0, 1> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, float* __restrict y, int64_t k) {
			static const int qk = QK8_0;

			assert(k % qk == 0);

			const int nb = k / qk;

			for (int i = 0; i < nb; i++) {
				const float d = oiml_fp16_to_fp32(x[i].d);

				float32x4_t scale = vdupq_n_f32(d);

				for (int j = 0; j < qk; j += 16) {
					int8x16_t quantized = vld1q_s8(x[i].qs + j);

					int16x8_t low  = vmovl_s8(vget_low_s8(quantized));
					int16x8_t high = vmovl_s8(vget_high_s8(quantized));

					int32x4_t low_low	= vmovl_s16(vget_low_s16(low));
					int32x4_t low_high	= vmovl_s16(vget_high_s16(low));
					int32x4_t high_low	= vmovl_s16(vget_low_s16(high));
					int32x4_t high_high = vmovl_s16(vget_high_s16(high));

					float32x4_t float_low_low	= vcvtq_f32_s32(low_low);
					float32x4_t float_low_high	= vcvtq_f32_s32(low_high);
					float32x4_t float_high_low	= vcvtq_f32_s32(high_low);
					float32x4_t float_high_high = vcvtq_f32_s32(high_high);

					float_low_low	= vmulq_f32(float_low_low, scale);
					float_low_high	= vmulq_f32(float_low_high, scale);
					float_high_low	= vmulq_f32(float_high_low, scale);
					float_high_high = vmulq_f32(float_high_high, scale);

					vst1q_f32(y + i * qk + j, float_low_low);
					vst1q_f32(y + i * qk + j + 4, float_low_high);
					vst1q_f32(y + i * qk + j + 8, float_high_low);
					vst1q_f32(y + i * qk + j + 12, float_high_high);
				}
			}
		}
	};

	template<> struct function_dispatcher<function_type::vec_dot, oiml_representation_types::q8_0, 1> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, const block_q8_0<oiml_half>* __restrict y, float* __restrict z, int64_t k) {
			const int qk = QK8_0;
			const int nb = k / qk;

			assert(k % qk == 0);

			int ib	   = 0;
			float sumf = 0;

			float32x4_t sumv0 = vdupq_n_f32(0.0f);
			float32x4_t sumv1 = vdupq_n_f32(0.0f);

			for (; ib + 1 < nb; ib += 2) {
				const block_q8_0<oiml_half>* __restrict x0 = &x[ib + 0];
				const block_q8_0<oiml_half>* __restrict x1 = &x[ib + 1];
				const block_q8_0<oiml_half>* __restrict y0 = &y[ib + 0];
				const block_q8_0<oiml_half>* __restrict y1 = &y[ib + 1];

				const int8x16_t x0_0 = vld1q_s8(x0->qs);
				const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
				const int8x16_t x1_0 = vld1q_s8(x1->qs);
				const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

				const int8x16_t y0_0 = vld1q_s8(y0->qs);
				const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
				const int8x16_t y1_0 = vld1q_s8(y1->qs);
				const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

				sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(oiml_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0), oiml_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))),
					oiml_fp16_to_fp32(x0->d) * oiml_fp16_to_fp32(y0->d));

				sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(oiml_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0), oiml_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))),
					oiml_fp16_to_fp32(x1->d) * oiml_fp16_to_fp32(y1->d));
			}

			sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
			for (; ib < nb; ++ib) {
				int sumi = 0;

				for (int j = 0; j < qk; j++) {
					sumi += x[ib].qs[j] * y[ib].qs[j];
				}

				sumf += sumi * (oiml_fp16_to_fp32(x[ib].d) * oiml_fp16_to_fp32(y[ib].d));
			}

			*z = sumf;
		}
	};	

}

#endif