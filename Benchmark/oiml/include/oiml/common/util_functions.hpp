#pragma once

#include <oiml/common/config.hpp>
#include <oiml/common/common.hpp>
#include <oiml/common/array.hpp>
#include <cstring>
#include <cstdint>
#include <math.h>
#include <bit>

#if defined(_WIN32) || defined(OIML_MSVC) || defined(OIML_LINUX)
	#include <immintrin.h>
#endif

namespace oiml {

#if defined(__ARM_NEON)
	#if defined(_MSC_VER) || (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11)
	using oiml_fp16_internal_t = uint16_t;
	#else
	using oiml_fp16_internal_t = __fp16;
	#endif
#endif

#if defined(__ARM_NEON) && !defined(_MSC_VER) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11)

	OIML_FORCE_INLINE static float oiml_compute_fp16_to_fp32(oiml_fp16_t h) {
		oiml_fp16_internal_t tmp;
		memcpy(&tmp, &h, sizeof(oiml_fp16_t));
		return ( float )tmp;
	}

	OIML_FORCE_INLINE static oiml_fp16_t oiml_compute_fp32_to_fp16(float f) {
		oiml_fp16_t res;
		oiml_fp16_internal_t tmp = f;
		memcpy(&res, &tmp, sizeof(oiml_fp16_t));
		return res;
	}

#elif defined(__F16C__)

	#ifdef _MSC_VER
		#define oiml_compute_fp16_to_fp32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
		#define oiml_compute_fp32_to_fp16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
	#else
		#define oiml_compute_fp16_to_fp32(x) _cvtsh_ss(x)
		#define oiml_compute_fp32_to_fp16(x) _cvtss_sh(x, 0)
	#endif

#else

	// FP16 <-> FP32
	// ref: https://github.com/Maratyszcza/FP16

	OIML_FORCE_INLINE static constexpr float fp32_from_bits(uint32_t w) {
		return std::bit_cast<float>(w);
	}

	OIML_FORCE_INLINE static constexpr uint32_t fp32_to_bits(float f) {
		return std::bit_cast<uint32_t>(f);
	}

	OIML_FORCE_INLINE static constexpr float oiml_compute_fp16_to_fp32(uint16_t h) {
		const uint32_t w	 = static_cast<uint32_t>(h) << 16;
		const uint32_t sign	 = w & 0x80000000u;
		const uint32_t two_w = w + w;

		constexpr uint32_t exp_offset = 0xE0u << 23;
		constexpr float exp_scale	  = fp32_from_bits(0x7800000u);
		const float normalized_value  = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

		constexpr uint32_t magic_mask  = 126u << 23;
		constexpr float magic_bias	   = 0.5f;
		const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

		constexpr uint32_t denormalized_cutoff = 1u << 27;
		const uint32_t result				   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
		return fp32_from_bits(result);
	}

	OIML_FORCE_INLINE constexpr float fabsf(float x) {
		return (x < 0.0f) ? -x : x;
	}

	OIML_FORCE_INLINE static constexpr oiml_fp16_t oiml_compute_fp32_to_fp16(float f) {
		static constexpr float scale_to_inf	 = fp32_from_bits(0x77800000u);
		static constexpr float scale_to_zero = fp32_from_bits(0x08800000u);
		float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

		const uint32_t w	  = fp32_to_bits(f);
		const uint32_t shl1_w = w + w;
		const uint32_t sign	  = w & 0x80000000u;
		uint32_t bias		  = shl1_w & 0xFF000000u;
		if (bias < 0x71000000u) {
			bias = 0x71000000u;
		}

		base						 = fp32_from_bits((bias >> 1) + 0x07800000u) + base;
		const uint32_t bits			 = fp32_to_bits(base);
		const uint32_t exp_bits		 = (bits >> 13) & 0x00007C00u;
		const uint32_t mantissa_bits = bits & 0x00000FFFu;
		const uint32_t nonsign		 = exp_bits + mantissa_bits;
		return (sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign);
	}

#endif// defined(__ARM_NEON) && (!defined(__MSC_VER)

	inline static oiml::oiml_array<float, (1 << 16)> oiml_table_f32_f16{ [] {
		oiml::oiml_array<float, (1 << 16)> returnValues{};
		for (uint32_t x = 0; x < (1 << 16); ++x) {
			returnValues[x] = oiml::oiml_compute_fp16_to_fp32(static_cast<uint16_t>(x));
		}
		return returnValues;
	}() };

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into oiml_lookup_fp16_to_fp32,
// so we define oiml_lookup_fp16_to_fp32 and oiml_compute_fp32_to_fp16 elsewhere for NEON.
// This is also true for POWER9.
#if !defined(oiml_lookup_fp16_to_fp32)
	OIML_FORCE_INLINE static float oiml_lookup_fp16_to_fp32(oiml_fp16_t f) {
		return oiml::oiml_table_f32_f16[f];
	}
#endif

	OIML_FORCE_INLINE static constexpr float oiml_fp16_to_fp32(uint16_t val) {
		return oiml_table_f32_f16[val];
	}

	OIML_FORCE_INLINE static void oi_prefetch(const void* ptr) noexcept {
#if defined(OIML_MAC) && defined(__arm64__)
		__builtin_prefetch(ptr, 0, 0);
#elif defined(OIML_MSVC) || defined(OIML_GNUCXX) || defined(OIML_CLANG)
		_mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#else
	#error "Compiler or architecture not supported for prefetching"
#endif
	}

	OIML_FORCE_INLINE static float oiml_compute_bf16_to_fp32(oiml_bf16_t h) {
		uint32_t temp = static_cast<uint32_t>(h) << 16;
		return std::bit_cast<float>(temp);
	}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */

	OIML_FORCE_INLINE static oiml_bf16_t oiml_compute_fp32_to_bf16(float s) {
		uint32_t i = std::bit_cast<uint32_t>(s);
		oiml_bf16_t h;

		if ((i & 0x7fffffff) > 0x7f800000) { /* nan */
			h = (i >> 16) | 64; /* force to quiet */
			return h;
		}
		h = (i + (0x7fff + ((i >> 16) & 1))) >> 16;
		return h;
	}

	OIML_FORCE_INLINE oiml_fp16_t oiml_fp32_to_fp16(float x) {
		return oiml_compute_fp32_to_fp16(x);
	}

	OIML_FORCE_INLINE float oiml_bf16_to_fp32(oiml_bf16_t x) {
		return oiml_compute_bf16_to_fp32(x);// it just left shifts
	}

	OIML_FORCE_INLINE oiml_bf16_t oiml_fp32_to_bf16(float x) {
		return oiml_compute_fp32_to_bf16(x);
	}

	OIML_FORCE_INLINE void oiml_fp16_to_fp32_row(const oiml_fp16_t* x, float* y, int64_t n) {
		for (int64_t i = 0; i < n; i++) {
			y[i] = oiml_lookup_fp16_to_fp32(x[i]);
		}
	}

	// FIXME: these functions must detect the instruction set at runtime, since they are part of the core oiml library
	//        currently, the oiml_cpu_has_* functions are entirely compile-time
	OIML_FORCE_INLINE void oiml_fp32_to_fp16_row(const float* x, oiml_fp16_t* y, int64_t n) {
		int64_t i = 0;
#if defined(__F16C__)
		//if (oiml_cpu_has_f16c()) {
		for (; i + 7 < n; i += 8) {
			__m256 x_vec  = _mm256_loadu_ps(x + i);
			__m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storeu_si128(( __m128i* )(y + i), y_vec);
		}
		for (; i + 3 < n; i += 4) {
			__m128 x_vec  = _mm_loadu_ps(x + i);
			__m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
			_mm_storel_epi64(( __m128i* )(y + i), y_vec);
		}
		//}
#endif
		for (; i < n; i++) {
			y[i] = oiml::oiml_compute_fp32_to_fp16(x[i]);
		}
	}

	OIML_FORCE_INLINE void oiml_bf16_to_fp32_row(const oiml_bf16_t* x, float* y, int64_t n) {
		int64_t i = 0;
#if defined(__AVX512F__)
		//if (oiml_cpu_has_avx512()) {
		for (; i + 16 <= n; i += 16) {
			_mm512_storeu_ps(y + i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(( const __m256i* )(x + i))), 16)));
		}
		//}
#endif
#if defined(__AVX2__)
		//if (oiml_cpu_has_avx2()) {
		for (; i + 8 <= n; i += 8) {
			_mm256_storeu_ps(y + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(( const __m128i* )(x + i))), 16)));
		}
		//}
#endif
		for (; i < n; i++) {
			y[i] = oiml::oiml_compute_bf16_to_fp32(x[i]);
		}
	}

	OIML_FORCE_INLINE void oiml_fp32_to_bf16_row_ref(const float* x, oiml_bf16_t* y, int64_t n) {
		for (int i = 0; i < n; i++) {
			y[i] = oiml::oiml_compute_fp32_to_bf16(x[i]);
		}
	}

	OIML_FORCE_INLINE void oiml_fp32_to_bf16_row(const float* x, oiml_bf16_t* y, int64_t n) {
		int i = 0;
#if defined(__AVX512BF16__)
		// subnormals are flushed to zero on this platform
		for (; i + 32 <= n; i += 32) {
			_mm512_storeu_si512(( __m512i* )(y + i), m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(x + i))));
		}
#endif
		for (; i < n; i++) {
			y[i] = oiml::oiml_compute_fp32_to_bf16(x[i]);
		}
	}	
}