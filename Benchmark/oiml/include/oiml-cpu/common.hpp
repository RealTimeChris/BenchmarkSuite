#pragma once

#include <oiml/common/representation_traits.hpp>
#include <iostream>

namespace oiml {

	enum class function_type { vec_dot = 0, from_float = 1, to_float = 2 };

	template<function_type function_type, oiml_representation_types rep_type, size_t index> struct function_dispatcher;

	OIML_FORCE_INLINE consteval int oiml_cpu_get_sve_cnt() {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
		return 1;//oiml_arm_arch_features.sve_cnt;
#else
		return 0;
#endif
	}

#if defined(OIML_IS_X86_64)
	#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1);

	OIML_FORCE_INLINE static float hsum_float_8(const __m256 x) {
		__m128 res = _mm256_extractf128_ps(x, 1);
		res		   = _mm_add_ps(res, _mm256_castps256_ps128(x));
		res		   = _mm_add_ps(res, _mm_movehl_ps(res, res));
		res		   = _mm_add_ss(res, _mm_movehdup_ps(res));
		return _mm_cvtss_f32(res);
	}

	OIML_FORCE_INLINE static __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
		const __m128i ones			= _mm_set1_epi16(1);
		const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
		const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
		const __m256i summed_pairs	= MM256_SET_M128I(summed_pairsh, summed_pairsl);
		return _mm256_cvtepi32_ps(summed_pairs);
	}

	OIML_FORCE_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
		const __m128i axl  = _mm256_castsi256_si128(ax);
		const __m128i axh  = _mm256_extractf128_si256(ax, 1);
		const __m128i syl  = _mm256_castsi256_si128(sy);
		const __m128i syh  = _mm256_extractf128_si256(sy, 1);
		const __m128i dotl = _mm_maddubs_epi16(axl, syl);
		const __m128i doth = _mm_maddubs_epi16(axh, syh);
		return sum_i16_pairs_float(doth, dotl);
	}

	OIML_FORCE_INLINE static __m256 sum_i16_pairs_float(const __m256i x) {
		const __m256i ones		   = _mm256_set1_epi16(1);
		const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
		return _mm256_cvtepi32_ps(summed_pairs);
	}

	OIML_FORCE_INLINE static __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
		const __m128i xl   = _mm256_castsi256_si128(x);
		const __m128i xh   = _mm256_extractf128_si256(x, 1);
		const __m128i yl   = _mm256_castsi256_si128(y);
		const __m128i yh   = _mm256_extractf128_si256(y, 1);
		const __m128i axl  = _mm_sign_epi8(xl, xl);
		const __m128i axh  = _mm_sign_epi8(xh, xh);
		const __m128i syl  = _mm_sign_epi8(yl, xl);
		const __m128i syh  = _mm_sign_epi8(yh, xh);
		const __m128i dotl = _mm_maddubs_epi16(axl, syl);
		const __m128i doth = _mm_maddubs_epi16(axh, syh);
		return sum_i16_pairs_float(doth, dotl);
	}

	#if defined(_MSC_VER)

		#define m512bh(p) p
		#define m512i(p) p

	#else

		#define m512bh(p) (__m512bh)(p)
		#define m512i(p) (__m512i)(p)

	#endif

	// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
	#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
		#ifndef __FMA__
			#define __FMA__
		#endif
		#ifndef __F16C__
			#define __F16C__
		#endif
	#endif

	// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
	#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
		#ifndef __SSE3__
			#define __SSE3__
		#endif
		#ifndef __SSSE3__
			#define __SSSE3__
		#endif
	#endif

#else

	#if defined(__ARM_FEATURE_SVE)
		#include <arm_sve.h>
	#endif

	#if defined(__ARM_NEON)
		#include <arm_neon.h>

		#ifdef _MSC_VER

	typedef uint16_t oiml_fp16_internal_t;

			#define oiml_vld1q_u32(w, x, y, z) { ((w) + (( uint64_t )(x) << 32)), ((y) + (( uint64_t )(z) << 32)) }

		#else

	typedef __fp16 oiml_fp16_internal_t;

			#define oiml_vld1q_u32(w, x, y, z) { (w), (x), (y), (z) }

		#endif

		#if !defined(__aarch64__)

	OIML_FORCE_INLINE static int32_t vaddlvq_s16(int16x8_t v) {
		int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
		return vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
	}

	OIML_FORCE_INLINE static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
		int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
		int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
		return vcombine_s16(a0, b0);
	}

	OIML_FORCE_INLINE static int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
		int32x2_t a0 = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
		int32x2_t b0 = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
		return vcombine_s32(a0, b0);
	}

	OIML_FORCE_INLINE static int32_t vaddvq_s32(int32x4_t v) {
		return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
	}

	OIML_FORCE_INLINE static float vaddvq_f32(float32x4_t v) {
		return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
	}

	OIML_FORCE_INLINE static float vmaxvq_f32(float32x4_t v) {
		return MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)), MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
	}

	OIML_FORCE_INLINE static int32x4_t vcvtnq_s32_f32(float32x4_t v) {
		int32x4_t res;

		res[0] = roundf(vgetq_lane_f32(v, 0));
		res[1] = roundf(vgetq_lane_f32(v, 1));
		res[2] = roundf(vgetq_lane_f32(v, 2));
		res[3] = roundf(vgetq_lane_f32(v, 3));

		return res;
	}

	OIML_FORCE_INLINE static uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
		uint8x8_t res;

		res[0] = a[0];
		res[1] = b[0];
		res[2] = a[1];
		res[3] = b[1];
		res[4] = a[2];
		res[5] = b[2];
		res[6] = a[3];
		res[7] = b[3];

		return res;
	}

	OIML_FORCE_INLINE static uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
		uint8x8_t res;

		res[0] = a[4];
		res[1] = b[4];
		res[2] = a[5];
		res[3] = b[5];
		res[4] = a[6];
		res[5] = b[6];
		res[6] = a[7];
		res[7] = b[7];

		return res;
	}

	typedef struct oiml_int16x8x2_t {
		int16x8_t val[2];
	} oiml_int16x8x2_t;

	OIML_FORCE_INLINE static oiml_int16x8x2_t oiml_vld1q_s16_x2(const int16_t* ptr) {
		oiml_int16x8x2_t res;

		res.val[0] = vld1q_s16(ptr + 0);
		res.val[1] = vld1q_s16(ptr + 8);

		return res;
	}

	typedef struct oiml_uint8x16x2_t {
		uint8x16_t val[2];
	} oiml_uint8x16x2_t;

	OIML_FORCE_INLINE static oiml_uint8x16x2_t oiml_vld1q_u8_x2(const int8_t* ptr) {
		oiml_uint8x16x2_t res;

		res.val[0] = vld1q_u8(ptr + 0);
		res.val[1] = vld1q_u8(ptr + 16);

		return res;
	}

	typedef struct oiml_uint8x16x4_t {
		uint8x16_t val[4];
	} oiml_uint8x16x4_t;

	OIML_FORCE_INLINE static oiml_uint8x16x4_t oiml_vld1q_u8_x4(const int8_t* ptr) {
		oiml_uint8x16x4_t res;

		res.val[0] = vld1q_u8(ptr + 0);
		res.val[1] = vld1q_u8(ptr + 16);
		res.val[2] = vld1q_u8(ptr + 32);
		res.val[3] = vld1q_u8(ptr + 48);

		return res;
	}

	typedef struct oiml_int8x16x2_t {
		int8x16_t val[2];
	} oiml_int8x16x2_t;

	OIML_FORCE_INLINE static oiml_int8x16x2_t oiml_vld1q_s8_x2(const int8_t* ptr) {
		oiml_int8x16x2_t res;

		res.val[0] = vld1q_s8(ptr + 0);
		res.val[1] = vld1q_s8(ptr + 16);

		return res;
	}

	typedef struct oiml_int8x16x4_t {
		int8x16_t val[4];
	} oiml_int8x16x4_t;

	OIML_FORCE_INLINE static oiml_int8x16x4_t oiml_vld1q_s8_x4(const int8_t* ptr) {
		oiml_int8x16x4_t res;

		res.val[0] = vld1q_s8(ptr + 0);
		res.val[1] = vld1q_s8(ptr + 16);
		res.val[2] = vld1q_s8(ptr + 32);
		res.val[3] = vld1q_s8(ptr + 48);

		return res;
	}

	OIML_FORCE_INLINE static int8x16_t oiml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
		int8x16_t res;

		res[0]	= a[b[0]];
		res[1]	= a[b[1]];
		res[2]	= a[b[2]];
		res[3]	= a[b[3]];
		res[4]	= a[b[4]];
		res[5]	= a[b[5]];
		res[6]	= a[b[6]];
		res[7]	= a[b[7]];
		res[8]	= a[b[8]];
		res[9]	= a[b[9]];
		res[10] = a[b[10]];
		res[11] = a[b[11]];
		res[12] = a[b[12]];
		res[13] = a[b[13]];
		res[14] = a[b[14]];
		res[15] = a[b[15]];

		return res;
	}

	OIML_FORCE_INLINE static uint8x16_t oiml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
		uint8x16_t res;

		res[0]	= a[b[0]];
		res[1]	= a[b[1]];
		res[2]	= a[b[2]];
		res[3]	= a[b[3]];
		res[4]	= a[b[4]];
		res[5]	= a[b[5]];
		res[6]	= a[b[6]];
		res[7]	= a[b[7]];
		res[8]	= a[b[8]];
		res[9]	= a[b[9]];
		res[10] = a[b[10]];
		res[11] = a[b[11]];
		res[12] = a[b[12]];
		res[13] = a[b[13]];
		res[14] = a[b[14]];
		res[15] = a[b[15]];

		return res;
	}

		#else

			#define oiml_int16x8x2_t int16x8x2_t
			#define oiml_uint8x16x2_t uint8x16x2_t
			#define oiml_uint8x16x4_t uint8x16x4_t
			#define oiml_int8x16x2_t int8x16x2_t
			#define oiml_int8x16x4_t int8x16x4_t

			#define oiml_vld1q_s16_x2 vld1q_s16_x2
			#define oiml_vld1q_u8_x2 vld1q_u8_x2
			#define oiml_vld1q_u8_x4 vld1q_u8_x4
			#define oiml_vld1q_s8_x2 vld1q_s8_x2
			#define oiml_vld1q_s8_x4 vld1q_s8_x4
			#define oiml_vqtbl1q_s8 vqtbl1q_s8
			#define oiml_vqtbl1q_u8 vqtbl1q_u8

		#endif

		#if !defined(__ARM_FEATURE_DOTPROD)

	OIML_FORCE_INLINE static int32x4_t oiml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
		const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
		const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

		return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
	}

		#else

			#define oiml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

		#endif

	#endif
#endif


}