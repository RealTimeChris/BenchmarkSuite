#pragma once

#include <oiml/common/util_functions.hpp>
#include <oiml/cpu/detect_isa.hpp>
#include <oiml/common/common.hpp>
#include <oiml-cpu/common.hpp>
#include <assert.h>
#include <cassert>

#if defined(OIML_IS_X86_64)

namespace oiml {

	OIML_FORCE_INLINE void oiml_quantize_row_q8_0_avx2(size_t k, const float* x, block_q8_0<oiml_half>* vy) {
		int nb = k / Q_SIZE;

		block_q8_0<oiml_half>* y = vy;

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

	

	OIML_FORCE_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
		// Perform multiplication and create 16-bit values
		const __m256i dot = _mm256_maddubs_epi16(ax, sy);
		return oiml::sum_i16_pairs_float(dot);
	}

	// multiply int8_t, add results pairwise twice and return as float vector
	OIML_FORCE_INLINE static __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
		// Get absolute values of x vectors
		const __m256i ax = _mm256_sign_epi8(x, x);
		// Sign the values of the y vectors
		const __m256i sy = _mm256_sign_epi8(y, x);
		return oiml::mul_sum_us8_pairs_float(ax, sy);
	}

	OIML_FORCE_INLINE static float hsum_float_8(const __m256 x) {
		__m128 res = _mm256_extractf128_ps(x, 1);
		res		   = _mm_add_ps(res, _mm256_castps256_ps128(x));
		res		   = _mm_add_ps(res, _mm_movehl_ps(res, res));
		res		   = _mm_add_ss(res, _mm_movehdup_ps(res));
		return _mm_cvtss_f32(res);
	}

	using oiml_quantize_row_q8_0_type = decltype(&oiml_quantize_row_q8_0_avx2);

	inline static constexpr oiml_array<oiml_quantize_row_q8_0_type, 3> oiml_quantize_row_q8_0_type_funcs{ { oiml_quantize_row_q8_0_avx2, oiml_quantize_row_q8_0_avx2,
		oiml_quantize_row_q8_0_avx2 } };
	inline static const oiml_quantize_row_q8_0_type oiml_quantize_row_q8_0_function_ptr{ get_work_func(oiml_quantize_row_q8_0_type_funcs, cpu_arch_index) };

	OIML_FORCE_INLINE void oiml_vec_dot_q8_0_q8_0_avx2(size_t n, float* s, const block_q8_0<oiml_half>* vx, const block_q8_0<oiml_half>* vy) {
		static constexpr size_t qk = Q_SIZE;
		size_t nb				   = n / qk;

		assert(n % qk == 0);

		const block_q8_0<oiml_half>* x = vx;
		const block_q8_0<oiml_half>* y = vy;

		size_t ib  = 0;
		float sumf = 0;
		__m256 acc = _mm256_setzero_ps();

		for (; ib < nb; ++ib) {
			const __m256 d = _mm256_set1_ps(oiml_fp16_to_fp32(x[ib].d) * oiml_fp16_to_fp32(y[ib].d));
			__m256i qx	   = _mm256_load_si256(( const __m256i* )x[ib].qs);
			__m256i qy	   = _mm256_load_si256(( const __m256i* )y[ib].qs);

			const __m256 q = mul_sum_i8_pairs_float(qx, qy);

			acc = _mm256_fmadd_ps(d, q, acc);
		}

		sumf = oiml::hsum_float_8(acc);

		for (; ib < nb; ++ib) {
			size_t sumi = 0;

			for (size_t j = 0; j < qk; j++) {
				sumi += x[ib].qs[j] * y[ib].qs[j];
			}

			sumf += sumi * (oiml_fp16_to_fp32(x[ib].d) * oiml_fp16_to_fp32(y[ib].d));
		}

		*s = sumf;
	}

	using oiml_vec_dot_q8_0_q8_0_type = decltype(&oiml_vec_dot_q8_0_q8_0_avx2);

	inline static constexpr oiml_array<oiml_vec_dot_q8_0_q8_0_type, 3> oiml_vec_dot_q8_0_q8_0_type_funcs{ { oiml_vec_dot_q8_0_q8_0_avx2, oiml_vec_dot_q8_0_q8_0_avx2,
		oiml_vec_dot_q8_0_q8_0_avx2 } };
	inline static const oiml_vec_dot_q8_0_q8_0_type oiml_vec_dot_q8_0_q8_0_function_ptr{ get_work_func(oiml_vec_dot_q8_0_q8_0_type_funcs, cpu_arch_index) };

	OIML_FORCE_INLINE void oiml_vec_dot_q8_0_f32_avx512(size_t n, float* s, const block_q8_0<oiml_half>* x_new, const float* y) {

	#if defined(__AVX512F__)

		static constexpr size_t qk = Q_SIZE;
		size_t nb				   = n / qk;

		float sumf = 0;

		const block_q8_0<oiml_half>* x = static_cast<const block_q8_0<oiml_half>*>(x_new);

		const __m512 zero = _mm512_setzero_ps();
		__m512 sumv0	  = _mm512_setzero_ps();
		__m512 sumv1	  = _mm512_setzero_ps();
		__m512 total_sum  = _mm512_setzero_ps();

		for (size_t ib = 0; ib < nb - 4; ib += 4) {
			__m512 d_broad = _mm512_set1_ps(oiml_fp16_to_fp32(x->d));

			__m256i qs = _mm256_load_si256(( __m256i* )(x->qs));
			__m512i x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_load_ps(y), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_load_ps(y + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			y += 32;

			d_broad = _mm512_set1_ps(oiml_fp16_to_fp32(x->d));

			qs = _mm256_load_si256(( __m256i* )(x->qs));
			x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_load_ps(y), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_load_ps(y + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			y += 32;

			d_broad = _mm512_set1_ps(oiml_fp16_to_fp32(x->d));

			qs = _mm256_load_si256(( __m256i* )(x->qs));
			x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_load_ps(y), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_load_ps(y + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			y += 32;

			d_broad = _mm512_set1_ps(oiml_fp16_to_fp32(x->d));

			qs = _mm256_load_si256(( __m256i* )(x->qs));
			x0 = _mm512_cvtepi8_epi16(qs);

			sumv0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(x0))), _mm512_load_ps(y), zero);
			sumv1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(x0, 1))), _mm512_load_ps(y + 16), sumv0);

			total_sum = _mm512_fmadd_ps(sumv1, d_broad, total_sum);

			++x;
			y += 32;
		}

		__m256 sum = _mm256_add_ps(_mm512_castps512_ps256(total_sum), _mm512_extractf32x8_ps(total_sum, 1));

		{
			// hiQuad = ( x7, x6, x5, x4 )
			const __m128 hiQuad = _mm256_extractf128_ps(sum, 1);
			// loQuad = ( x3, x2, x1, x0 )
			const __m128 loQuad = _mm256_castps256_ps128(sum);
			// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
			const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
			// loDual = ( -, -, x1 + x5, x0 + x4 )
			const __m128 loDual = sumQuad;
			// hiDual = ( -, -, x3 + x7, x2 + x6 )
			const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
			// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
			const __m128 sumDual = _mm_add_ps(loDual, hiDual);
			// lo = ( -, -, -, x0 + x2 + x4 + x6 )
			const __m128 lo = sumDual;
			// hi = ( -, -, -, x1 + x3 + x5 + x7 )
			const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
			// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
			const __m128 sum = _mm_add_ss(lo, hi);
			sumf			 = _mm_cvtss_f32(sum);
		}

		*s = sumf;

	#endif

	}

	OIML_FORCE_INLINE void oiml_vec_dot_q8_0_f32_avx2(size_t n, float* s, const block_q8_0<oiml_half>* x_new, const float* y_new) {
		static constexpr size_t qk = Q_SIZE;
		size_t nb				   = n / qk;

		float sumf = 0;

	#if defined(__AVX2__)

		const block_q8_0<oiml_half>* x = x_new;
		const float* y		= y_new;

		__m256 total_sum = _mm256_setzero_ps();

		for (size_t ib = 0; ib < nb - 4; ib += 4) {
			__m256 local_sum = _mm256_setzero_ps();
			__m256 d_broad00 = _mm256_set1_ps(oiml_fp16_to_fp32(x->d));

			__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
			__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));
			++x;

			__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			__m256 y0	 = _mm256_load_ps(y);
			local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_load_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_load_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_load_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));

			d_broad00 = _mm256_set1_ps(oiml_fp16_to_fp32(x->d));
			y += 32;
			++x;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_load_ps(y);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_load_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_load_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_load_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));

			d_broad00 = _mm256_set1_ps(oiml_fp16_to_fp32(x->d));
			y += 32;
			++x;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_load_ps(y);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_load_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_load_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_load_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			local_sum = _mm256_setzero_ps();

			x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
			x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));

			d_broad00 = _mm256_set1_ps(oiml_fp16_to_fp32(x->d));
			y += 32;
			++x;

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
			y0		  = _mm256_load_ps(y);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
			y0		  = _mm256_load_ps(y + 8);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
			y0		  = _mm256_load_ps(y + 16);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

			temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
			y0		  = _mm256_load_ps(y + 24);
			local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);
			total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
			y += 32;
		}

		__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

		sum = _mm_hadd_ps(sum, sum);
		sum = _mm_hadd_ps(sum, sum);

		sumf = _mm_cvtss_f32(sum);

	#endif

		*s = sumf;
	}

	using oiml_vec_dot_q8_0_f32_type = decltype(&oiml_vec_dot_q8_0_f32_avx2);

	inline static constexpr oiml_array<oiml_vec_dot_q8_0_f32_type, 3> oiml_vec_dot_q8_0_f32_type_funcs{ { &oiml_vec_dot_q8_0_f32_avx2, &oiml_vec_dot_q8_0_f32_avx2,
		&oiml_vec_dot_q8_0_f32_avx512 } };
	inline static const oiml_vec_dot_q8_0_f32_type oiml_vec_dot_q8_0_f32_function_ptr{ get_work_func(oiml_vec_dot_q8_0_f32_type_funcs, cpu_arch_index) };

	OIML_FORCE_INLINE static void oiml_vec_dot_f32_avx2(size_t n, float* s, const float* x, const float* y) {
		float sumf = 0.0f;
		size_t np  = (n & ~(32 - 1));

		__m256 sum[(32 / 8)] = { _mm256_setzero_ps() };

		__m256 ax[(32 / 8)];
		__m256 ay[(32 / 8)];

		for (size_t i = 0; i < np; i += 32) {
			for (size_t j = 0; j < (32 / 8); j++) {
				ax[j] = _mm256_load_ps(x + i + j * 8);
				ay[j] = _mm256_load_ps(y + i + j * 8);

				sum[j] = _mm256_fmadd_ps(sum[j], ax[j], ay[j]);
			}
		}

		static constexpr int32_t offset = (32 / 8) >> 1;
		for (size_t i = 0; i < offset; ++i) {
			sum[i] = _mm256_add_ps(sum[i], sum[offset + i]);
		}
		static constexpr int32_t offset02 = offset >> 1;
		for (size_t i = 0; i < offset02; ++i) {
			sum[i] = _mm256_add_ps(sum[i], sum[offset + i]);
		}
		static constexpr int32_t offset03 = offset02 >> 1;
		for (size_t i = 0; i < offset03; ++i) {
			sum[i] = _mm256_add_ps(sum[i], sum[offset + i]);
		}
		const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sum[0]), _mm256_extractf128_ps(sum[0], 1));
		const __m128 t1 = _mm_hadd_ps(t0, t0);
		sumf			= ( oiml_float )_mm_cvtss_f32(_mm_hadd_ps(t1, t1));

		for (size_t i = np; i < n; ++i) {
			sumf += x[i] * y[i];
		}

		*s = sumf;
	}

	using oiml_vec_dot_f32_type = decltype(&oiml_vec_dot_f32_avx2);

	inline static constexpr oiml_array<oiml_vec_dot_f32_type, 3> oiml_vec_dot_f32_type_funcs{ { &oiml_vec_dot_f32_avx2, &oiml_vec_dot_f32_avx2, &oiml_vec_dot_f32_avx2 } };
	inline static const oiml_vec_dot_f32_type oiml_vec_dot_f32_function_ptr{ oiml::get_work_func(oiml_vec_dot_f32_type_funcs, cpu_arch_index) };

	OIML_FORCE_INLINE void oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2(size_t n, float* s, const int8_t* x_quants, const float* x_scales,
		const int8_t* y_quants, const float* y_scales) {
		static constexpr size_t stepk = Q_SIZE * 8;
		size_t nb					  = n / stepk;

		const __m256i* xq = reinterpret_cast<const __m256i*>(x_quants);
		const __m256i* yq = reinterpret_cast<const __m256i*>(y_quants);
		const float* xs	  = reinterpret_cast<const float*>(x_scales);
		const float* ys	  = reinterpret_cast<const float*>(y_scales);

		__m256 acc = _mm256_setzero_ps();

		float dx, dy;
		__m256 d, q;
		__m256i qx, qy;

	#define UNROLLED_STEP() \
		dx	= *xs; \
		dy	= *ys; \
		d	= _mm256_set1_ps(dx * dy); \
		qx	= _mm256_load_si256(xq); \
		qy	= _mm256_load_si256(yq); \
		q	= mul_sum_i8_pairs_float(qx, qy); \
		acc = _mm256_fmadd_ps(d, q, acc); \
		++xs; \
		++ys; \
		++xq; \
		++yq;

		// Main loop
		for (size_t ib = 0; ib < nb; ++ib) {
			UNROLLED_STEP();
			UNROLLED_STEP();
			UNROLLED_STEP();
			UNROLLED_STEP();
			UNROLLED_STEP();
			UNROLLED_STEP();
			UNROLLED_STEP();
			UNROLLED_STEP();
		}

	#undef UNROLLED_STEP

		*s = oiml::hsum_float_8(acc);
	}

	OIML_FORCE_INLINE void oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx(size_t n, float* s, const int8_t* x_quants, const float* x_scales,
		const int8_t* y_quants, const float* y_scales) {
		float sumf = 0;
		*s		   = sumf;
	}

	using oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_type = decltype(&oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx);
	inline static constexpr oiml_array<oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_type, 3> oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_type_funcs{ { &oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx,
		&oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2, &oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2 } };

	inline static const oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_type oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_function_ptr{ get_work_func(oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_type_funcs,
		cpu_arch_index) };
}

#endif
