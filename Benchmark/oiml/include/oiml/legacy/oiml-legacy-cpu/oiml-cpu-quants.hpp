#pragma once

#define OIML_COMMON_DECL_CPP
#include <oiml/legacy/oiml-legacy-common/oiml-common.hpp>
#include <oiml/common/config.hpp>
#include <oiml/common/common.hpp>

#include <oiml/legacy/oiml-legacy-common/oiml-quants.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>
#include <oiml/legacy/oiml-legacy-cpu/oiml-cpu-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-cpu.hpp>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h>// for qsort
#include <stdio.h>// for OIML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#if defined(_MSC_VER)
	// disable "possible loss of data" to aOIML_FORCE_INLINE void warnings for hundreds of casts
	// we should just be careful :)
	#pragma warning(disable : 4244 4267)
#endif

#define UNUSED OIML_UNUSED



#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
OIML_FORCE_INLINE static __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
	// Get absolute values of x vectors
	const __m128i ax = _mm_sign_epi8(x, x);
	// Sign the values of the y vectors
	const __m128i sy = _mm_sign_epi8(y, x);
	// Perform multiplication and create 16-bit values
	const __m128i dot  = _mm_maddubs_epi16(ax, sy);
	const __m128i ones = _mm_set1_epi16(1);
	return _mm_madd_epi16(ones, dot);
}

	#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
// horizontally add 8 floats
OIML_FORCE_INLINE static float hsum_float_8(const __m256 x) {
	__m128 res = _mm256_extractf128_ps(x, 1);
	res		   = _mm_add_ps(res, _mm256_castps256_ps128(x));
	res		   = _mm_add_ps(res, _mm_movehl_ps(res, res));
	res		   = _mm_add_ss(res, _mm_movehdup_ps(res));
	return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
OIML_FORCE_INLINE static int hsum_i32_8(const __m256i a) {
	const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
	const __m128i hi64	 = _mm_unpackhi_epi64(sum128, sum128);
	const __m128i sum64	 = _mm_add_epi32(hi64, sum128);
	const __m128i hi32	 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
	return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
OIML_FORCE_INLINE static int hsum_i32_4(const __m128i a) {
	const __m128i hi64	= _mm_unpackhi_epi64(a, a);
	const __m128i sum64 = _mm_add_epi32(hi64, a);
	const __m128i hi32	= _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
	return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

		#if defined(__AVX2__) || defined(__AVX512F__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
OIML_FORCE_INLINE static __m256i bytes_from_bits_32(const int8_t* x) {
	uint32_t x32;
	memcpy(&x32, x, sizeof(uint32_t));
	const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
	__m256i bytes			= _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
	const __m256i bit_mask	= _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
	bytes					= _mm256_or_si256(bytes, bit_mask);
	return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
OIML_FORCE_INLINE static __m256i bytes_from_nibbles_32(const int8_t* rsi) {
	const __m128i tmp	  = _mm_loadu_si128(( const __m128i* )rsi);
	const __m256i bytes	  = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
	const __m256i lowMask = _mm256_set1_epi8(0xF);
	return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
OIML_FORCE_INLINE static __m256 sum_i16_pairs_float(const __m256i x) {
	const __m256i ones		   = _mm256_set1_epi16(1);
	const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
	return _mm256_cvtepi32_ps(summed_pairs);
}

OIML_FORCE_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
			#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
	const __m256i zero		   = _mm256_setzero_si256();
	const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
	return _mm256_cvtepi32_ps(summed_pairs);
			#elif defined(__AVXVNNI__)
	const __m256i zero		   = _mm256_setzero_si256();
	const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
	return _mm256_cvtepi32_ps(summed_pairs);
			#else
	// Perform multiplication and create 16-bit values
	const __m256i dot = _mm256_maddubs_epi16(ax, sy);
	return sum_i16_pairs_float(dot);
			#endif
}

OIML_FORCE_INLINE static __m128i packNibbles(__m256i bytes) {
			// Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
			#if __AVX512F__
	const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);// 0000_0000_abcd_0000
	bytes					   = _mm256_or_si256(bytes, bytes_srli_4);// 0000_abcd_abcd_efgh
	return _mm256_cvtepi16_epi8(bytes);// abcd_efgh
			#else
	const __m256i lowByte = _mm256_set1_epi16(0xFF);
	__m256i high		  = _mm256_andnot_si256(lowByte, bytes);
	__m256i low			  = _mm256_and_si256(lowByte, bytes);
	high				  = _mm256_srli_epi16(high, 4);
	bytes				  = _mm256_or_si256(low, high);

	// Compress uint16_t lanes into bytes
	__m128i r0 = _mm256_castsi256_si128(bytes);
	__m128i r1 = _mm256_extracti128_si256(bytes, 1);
	return _mm_packus_epi16(r0, r1);
			#endif
}
		#elif defined(__AVX__)
OIML_FORCE_INLINE static __m128i packNibbles(__m128i bytes1, __m128i bytes2) {
	// Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
	const __m128i lowByte = _mm_set1_epi16(0xFF);
	__m128i high		  = _mm_andnot_si128(lowByte, bytes1);
	__m128i low			  = _mm_and_si128(lowByte, bytes1);
	high				  = _mm_srli_epi16(high, 4);
	bytes1				  = _mm_or_si128(low, high);
	high				  = _mm_andnot_si128(lowByte, bytes2);
	low					  = _mm_and_si128(lowByte, bytes2);
	high				  = _mm_srli_epi16(high, 4);
	bytes2				  = _mm_or_si128(low, high);

	return _mm_packus_epi16(bytes1, bytes2);
}

OIML_FORCE_INLINE static __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
	const __m128i ax = _mm_sign_epi8(x, x);
	const __m128i sy = _mm_sign_epi8(y, x);
	return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
OIML_FORCE_INLINE static __m256i bytes_from_bits_32(const int8_t* x) {
	uint32_t x32;
	memcpy(&x32, x, sizeof(uint32_t));
	const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
	const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
	__m128i bytesl			 = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
	__m128i bytesh			 = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
	const __m128i bit_mask	 = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
	bytesl					 = _mm_or_si128(bytesl, bit_mask);
	bytesh					 = _mm_or_si128(bytesh, bit_mask);
	bytesl					 = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
	bytesh					 = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
	return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
OIML_FORCE_INLINE static __m256i bytes_from_nibbles_32(const int8_t* rsi) {
	// Load 16 bytes from memory
	__m128i tmpl		  = _mm_loadu_si128(( const __m128i* )rsi);
	__m128i tmph		  = _mm_srli_epi16(tmpl, 4);
	const __m128i lowMask = _mm_set1_epi8(0xF);
	tmpl				  = _mm_and_si128(lowMask, tmpl);
	tmph				  = _mm_and_si128(lowMask, tmph);
	return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
OIML_FORCE_INLINE static __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
	const __m128i ones			= _mm_set1_epi16(1);
	const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
	const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
	const __m256i summed_pairs	= MM256_SET_M128I(summed_pairsh, summed_pairsl);
	return _mm256_cvtepi32_ps(summed_pairs);
}

OIML_FORCE_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
	const __m128i axl = _mm256_castsi256_si128(ax);
	const __m128i axh = _mm256_extractf128_si256(ax, 1);
	const __m128i syl = _mm256_castsi256_si128(sy);
	const __m128i syh = _mm256_extractf128_si256(sy, 1);
	// Perform multiplication and create 16-bit values
	const __m128i dotl = _mm_maddubs_epi16(axl, syl);
	const __m128i doth = _mm_maddubs_epi16(axh, syh);
	return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
OIML_FORCE_INLINE static __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
	const __m128i xl = _mm256_castsi256_si128(x);
	const __m128i xh = _mm256_extractf128_si256(x, 1);
	const __m128i yl = _mm256_castsi256_si128(y);
	const __m128i yh = _mm256_extractf128_si256(y, 1);
	// Get absolute values of x vectors
	const __m128i axl = _mm_sign_epi8(xl, xl);
	const __m128i axh = _mm_sign_epi8(xh, xh);
	// Sign the values of the y vectors
	const __m128i syl = _mm_sign_epi8(yl, xl);
	const __m128i syh = _mm_sign_epi8(yh, xh);
	// Perform multiplication and create 16-bit values
	const __m128i dotl = _mm_maddubs_epi16(axl, syl);
	const __m128i doth = _mm_maddubs_epi16(axh, syh);
	return sum_i16_pairs_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
OIML_FORCE_INLINE static __m256 mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0, const __m128i x_2_1, const __m128i y_1_0, const __m128i y_1_1,
	const __m128i y_2_0, const __m128i y_2_1) {
	const __m128i mone = _mm_set1_epi16(1);

	const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
	const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
	const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
	const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
	const __m128i p_1_0	  = _mm_madd_epi16(p16_1_0, mone);
	const __m128i p_1_1	  = _mm_madd_epi16(p16_1_1, mone);
	const __m128i p_2_0	  = _mm_madd_epi16(p16_2_0, mone);
	const __m128i p_2_1	  = _mm_madd_epi16(p16_2_1, mone);
	const __m128i p_1	  = _mm_add_epi32(p_1_0, p_1_1);
	const __m128i p_2	  = _mm_add_epi32(p_2_0, p_2_1);
	return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
OIML_FORCE_INLINE static __m256 quad_fp16_delta_float(const float x0, const float y0, const float x1, const float y1) {
	// oiml::oiml_lookup_fp16_to_fp32 is faster than Intel F16C
	return _mm256_set_m128(_mm_set1_ps(oiml::oiml_lookup_fp16_to_fp32(x1) * oiml::oiml_lookup_fp16_to_fp32(y1)),
		_mm_set1_ps(oiml::oiml_lookup_fp16_to_fp32(x0) * oiml::oiml_lookup_fp16_to_fp32(y0)));
}
		#endif
	#elif defined(__SSSE3__)
// horizontally add 4x4 floats
OIML_FORCE_INLINE static float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
	__m128 res_0 = _mm_hadd_ps(a, b);
	__m128 res_1 = _mm_hadd_ps(c, d);
	__m128 res	 = _mm_hadd_ps(res_0, res_1);
	res			 = _mm_hadd_ps(res, res);
	res			 = _mm_hadd_ps(res, res);

	return _mm_cvtss_f32(res);
}
	#endif// __AVX__ || __AVX2__ || __AVX512F__
#endif// defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

#if defined(__ARM_NEON) || defined(__wasm_simd128__) || defined(__POWER9_VECTOR__)
	#define B1(c, s, n) 0x##n##c, 0x##n##s
	#define B2(c, s, n) B1(c, s, n##c), B1(c, s, n##s)
	#define B3(c, s, n) B2(c, s, n##c), B2(c, s, n##s)
	#define B4(c, s, n) B3(c, s, n##c), B3(c, s, n##s)
	#define B5(c, s, n) B4(c, s, n##c), B4(c, s, n##s)
	#define B6(c, s, n) B5(c, s, n##c), B5(c, s, n##s)
	#define B7(c, s, n) B6(c, s, n##c), B6(c, s, n##s)
	#define B8(c, s) B7(c, s, c), B7(c, s, s)

// precomputed tables for expanding 8bits to 8 bytes:
static constexpr uint64_t table_b2b_0[1 << 8] = { B8(00, 10) };// ( b) << 4
static constexpr uint64_t table_b2b_1[1 << 8] = { B8(10, 00) };// (!b) << 4
#endif
