#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <cstdio>

#include <cmath>
// or
#include <math.h>

#if defined(_WIN32)
	#include <windows.h>
#else
	#include <sys/time.h>
#endif

#if defined(BNCH_SWT_MSVC)
	#pragma optimize("", off)
void doNotOptimize(const void* value) {
	( void )value;
};
	#pragma optimize("", on)
#else
void doNotOptimize(const void* value) {
	#if defined(BNCH_SWT_CLANG)
	asm volatile("" : "+r,m"(value) : : "memory");
	#elif defined(BNCH_SWT_GNUCXX)
	asm volatile("" : "+m,r"(value) : : "memory");
	#endif
}
#endif

static inline float fp32_from_bits(uint32_t w) {
	union {
		uint32_t as_bits;
		float as_value;
	} fp32;
	fp32.as_bits = w;
	return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
	union {
		float as_value;
		uint32_t as_bits;
	} fp32;
	fp32.as_value = f;
	return fp32.as_bits;
}

typedef uint16_t ggml_fp16_t;

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
	const uint32_t w	 = ( uint32_t )h << 16;
	const uint32_t sign	 = w & UINT32_C(0x80000000);
	const uint32_t two_w = w + w;

	const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
	const float exp_scale = 0x1.0p-112f;
#else
	const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
	const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	const uint32_t magic_mask	   = UINT32_C(126) << 23;
	const float magic_bias		   = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
	const uint32_t result			   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
	const float scale_to_inf  = 0x1.0p+112f;
	const float scale_to_zero = 0x1.0p-110f;
#else
	const float scale_to_inf  = fp32_from_bits(UINT32_C(0x77800000));
	const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
	float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

	const uint32_t w	  = fp32_to_bits(f);
	const uint32_t shl1_w = w + w;
	const uint32_t sign	  = w & UINT32_C(0x80000000);
	uint32_t bias		  = shl1_w & UINT32_C(0xFF000000);
	if (bias < UINT32_C(0x71000000)) {
		bias = UINT32_C(0x71000000);
	}

	base						 = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
	const uint32_t bits			 = fp32_to_bits(base);
	const uint32_t exp_bits		 = (bits >> 13) & UINT32_C(0x00007C00);
	const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
	const uint32_t nonsign		 = exp_bits + mantissa_bits;
	return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

float ggml_table_f32_f16[1 << 16];

inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
	uint16_t s;
	memcpy(&s, &f, sizeof(uint16_t));
	return ggml_table_f32_f16[s];
}

int32_t main() {
	for (int i = 0; i < (1 << 16); ++i) {
		union {
			uint16_t u16;
			ggml_fp16_t fp16;
		} u					  = { i };
		ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
	}
	printf("Value: %f\n", ggml_lookup_fp16_to_fp32(23223));
	return 0;
}