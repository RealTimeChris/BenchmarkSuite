#pragma once

#include <oiml/common/util_functions.hpp>
#include <oiml-cpu/detect_isa.hpp>
#include <oiml/common/common.hpp>
#include <oiml-cpu/common.hpp>
#include <assert.h>

#include <cassert>

#if defined(OIML_IS_X86_64)

namespace oiml {

	template<> struct function_dispatcher<function_type::from_float, oiml_representation_types::q8_0, 2> {
		OIML_FORCE_INLINE static void impl(const float* __restrict x, block_q8_0<oiml_half>* __restrict y, int64_t k) {
			assert(QK8_0 == 32);
			assert(k % QK8_0 == 0);
			const int nb = k / QK8_0;

			for (int i = 0; i < nb; i++) {
				__m256 v0 = _mm256_load_ps(x);
				__m256 v1 = _mm256_load_ps(x + 8);
				__m256 v2 = _mm256_load_ps(x + 16);
				__m256 v3 = _mm256_load_ps(x + 24);
				x += 32;

				const __m256 signBit = _mm256_set1_ps(-0.0f);
				__m256 maxAbs		 = _mm256_andnot_ps(signBit, v0);
				maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
				maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
				maxAbs				 = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

				__m128 max4			  = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1), _mm256_castps256_ps128(maxAbs));
				max4				  = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
				max4				  = _mm_max_ss(max4, _mm_movehdup_ps(max4));
				const float maxScalar = _mm_cvtss_f32(max4);

				const float d	 = maxScalar / 127.f;
				y[i].d			 = oiml_fp32_to_fp16(d);
				const float id	 = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
				const __m256 mul = _mm256_set1_ps(id);

				v0 = _mm256_mul_ps(v0, mul);
				v1 = _mm256_mul_ps(v1, mul);
				v2 = _mm256_mul_ps(v2, mul);
				v3 = _mm256_mul_ps(v3, mul);

				v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
				v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
				v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
				v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

				__m256i i0 = _mm256_cvtps_epi32(v0);
				__m256i i1 = _mm256_cvtps_epi32(v1);
				__m256i i2 = _mm256_cvtps_epi32(v2);
				__m256i i3 = _mm256_cvtps_epi32(v3);

				i0 = _mm256_packs_epi32(i0, i1);
				i2 = _mm256_packs_epi32(i2, i3);
				i0 = _mm256_packs_epi16(i0, i2);

				const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
				i0				   = _mm256_permutevar8x32_epi32(i0, perm);

				_mm256_storeu_si256(( __m256i* )y[i].qs, i0);
			}
		}
	};

	template<> struct function_dispatcher<function_type::to_float, oiml_representation_types::q8_0, 2> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, float* __restrict y, int64_t k) {
			static const int qk = QK8_0;

			assert(k % qk == 0);

			const int nb = k / qk;

			for (int i = 0; i < nb; i++) {
				const float d = oiml_fp16_to_fp32(x[i].d);

				__m256 scale = _mm256_set1_ps(d);

				for (int j = 0; j < qk; j += 32) {
					__m256i quantized = _mm256_load_si256(( const __m256i* )(x[i].qs + j));

					__m256i low	 = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(quantized));
					__m256i high = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(quantized, 1));

					__m256 float_low  = _mm256_cvtepi32_ps(low);
					__m256 float_high = _mm256_cvtepi32_ps(high);

					float_low  = _mm256_mul_ps(float_low, scale);
					float_high = _mm256_mul_ps(float_high, scale);

					_mm256_storeu_ps(y + i * qk + j, float_low);
					_mm256_storeu_ps(y + i * qk + j + 8, float_high);
				}
			}
		}
	};

	template<> struct function_dispatcher<function_type::vec_dot, oiml_representation_types::q8_0, 2> {
		OIML_FORCE_INLINE static void impl(const block_q8_0<oiml_half>* __restrict x, const block_q8_0<oiml_half>* __restrict y, float* __restrict z, int64_t k) {
			static const int qk = QK8_0;

			assert(k % qk == 0);
			const int nb = k / qk;
			uint64_t ib	 = 0;
			float sumf	 = 0;
			__m256 acc	 = _mm256_setzero_ps();

			for (; ib < nb; ++ib) {
				const __m256 d = _mm256_set1_ps(oiml::oiml_lookup_fp16_to_fp32(x[ib].d) * oiml::oiml_lookup_fp16_to_fp32(y[ib].d));
				__m256i qx	   = _mm256_load_si256(( const __m256i* )x[ib].qs);
				__m256i qy	   = _mm256_load_si256(( const __m256i* )y[ib].qs);

				const __m256 q = oiml::mul_sum_i8_pairs_float(qx, qy);

				acc = _mm256_fmadd_ps(d, q, acc);
			}

			sumf = hsum_float_8(acc);

			for (; ib < nb; ++ib) {
				int sumi = 0;

				for (int j = 0; j < qk; j++) {
					sumi += x[ib].qs[j] * y[ib].qs[j];
				}

				sumf += sumi * (oiml::oiml_lookup_fp16_to_fp32(x[ib].d) * oiml::oiml_lookup_fp16_to_fp32(y[ib].d));
			}

			*z = sumf;
		}
	};	

}

#endif
