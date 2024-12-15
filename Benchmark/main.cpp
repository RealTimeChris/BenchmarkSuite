#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

#include <array>
#include <vector>
#include <cassert>
#include <numeric>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

static constexpr int n = 8192;

struct block_q8_0 {
	int16_t d;
	int8_t qs[n / 4];
};

struct alignas(32) block_q8_0_new {
	alignas(32) int8_t qs[n / 4];
	int16_t d;
};

JSONIFIER_INLINE void ggml_vec_dot_q8_0_f32_unaligned_float(float* s, const block_q8_0* x, const void* vy) {
	static constexpr int qk = 32;
	static constexpr int nb = n / qk;

	assert(n % qk == 0);

	const float* y{ ( const float* )(vy) };

	__m256 total_sum = _mm256_setzero_ps();

	for (int32_t ib = 0; ib < nb - 4; ib += 4) {
		__m256 local_sum = _mm256_setzero_ps();
		__m256 d_broad00 = _mm256_set1_ps((float)(x->d));

		__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));
		++x;

		__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		__m256 y0	 = _mm256_loadu_ps(y);
		local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps((float)(x->d));
		y += 32;
		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_loadu_ps(y);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps((float)(x->d));
		y += 32;
		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_loadu_ps(y);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps((float)(x->d));
		y += 32;
		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_loadu_ps(y);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_loadu_ps(y + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_loadu_ps(y + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);
		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		y += 32;
	}

	__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);

	*s = _mm_cvtss_f32(sum);
}

JSONIFIER_INLINE void ggml_vec_dot_q8_0_f32_unaligned_int8(float* s, const block_q8_0* x, const void* vy) {
	static constexpr int qk = 32;
	static constexpr int nb = n / qk;
	assert(n % qk == 0);

	 __m256 total_sum = _mm256_setzero_ps();

	const __m256i* y = ( const __m256i* )vy;

	for (int32_t ib = 0; ib < nb - 4; ib += 4) {
		__m256 local_sum = _mm256_setzero_ps();
		__m256 d_broad00 = _mm256_set1_ps((float)(x->d));

		__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));
		++x;

		__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		__m256 y0	 = _mm256_castsi256_ps(_mm256_loadu_si256(y));
		local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps((float)(x->d));
		y += 4;

		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps((float)(x->d));
		y += 4;

		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps((float)(x->d));
		y += 4;

		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_loadu_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps((float)(x->d));
		y += 4;
	}

	__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);

	*s = _mm_cvtss_f32(sum);
}

JSONIFIER_INLINE void ggml_vec_dot_q8_0_f32_aligned_float(float* s, const block_q8_0_new* x, const void* vy) {
	static constexpr int qk = 32;
	static constexpr int nb = n / qk;

	assert(n % qk == 0);

	const float* y			= ( const float* )vy;

	__m256 total_sum = _mm256_setzero_ps();

	for (int32_t ib = 0; ib < nb - 4; ib += 4) {
		__m256 local_sum = _mm256_setzero_ps();
		__m256 d_broad00 = _mm256_set1_ps(( float )(x->d));

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

		d_broad00 = _mm256_set1_ps(( float )(x->d));
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

		d_broad00 = _mm256_set1_ps(( float )(x->d));
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

		d_broad00 = _mm256_set1_ps(( float )(x->d));
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

	*s = _mm_cvtss_f32(sum);
}

JSONIFIER_INLINE void ggml_vec_dot_q8_0_f32_aligned_int8(float* s, const block_q8_0_new* x, const void* vy) {
	static constexpr int qk = 32;
	static constexpr int nb = n / qk;
	assert(n % qk == 0);

	__m256 total_sum = _mm256_setzero_ps();

	const __m256i* y = ( const __m256i* )vy;

	for (int32_t ib = 0; ib < nb - 4; ib += 4) {
		__m256 local_sum = _mm256_setzero_ps();
		__m256 d_broad00 = _mm256_set1_ps(( float )(x->d));

		__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
		__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));
		++x;

		__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		__m256 y0	 = _mm256_castsi256_ps(_mm256_load_si256(y));
		local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps(( float )(x->d));
		y += 4;

		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps(( float )(x->d));
		y += 4;

		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps(( float )(x->d));
		y += 4;

		++x;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 1));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 2));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_castsi256_ps(_mm256_load_si256(y + 3));
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x->qs + 16)));

		d_broad00 = _mm256_set1_ps(( float )(x->d));
		y += 4;
	}

	__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);

	*s = _mm_cvtss_f32(sum);
}

static constexpr uint64_t maxIterations{ 500 };
static constexpr uint64_t measuredIterations{ 20 };

struct test_data {
	std::unique_ptr<block_q8_0[]> orig_x = std::make_unique<block_q8_0[]>(n / 32);
	alignas(32) std::array<char, n * 4> data{};
	alignas(32) std::unique_ptr<block_q8_0_new[]> orig_y = std::make_unique<block_q8_0_new[]>(n / 32);

	test_data() {
		for (size_t x = 0; x < n * 4; ++x) {
			data[x] = bnch_swt::random_generator::generateValue<char>();
		}
		for (size_t x = 0; x < n / 32; ++x) {
			orig_x[x].d = bnch_swt::random_generator::generateValue<int16_t>();
			std::generate(orig_x[x].qs, orig_x[x].qs + 128, [] {
				return bnch_swt::random_generator::generateValue<int8_t>();
			});
		}
		for (size_t x = 0; x < n / 32; ++x) {
			orig_y[x].d = bnch_swt::random_generator::generateValue<int16_t>();
			std::generate(orig_y[x].qs, orig_y[x].qs + 128, [] {
				return bnch_swt::random_generator::generateValue<int8_t>();
			});
		}
	}
	test_data(const test_data&)			   = delete;
	test_data& operator=(const test_data&) = delete;
	test_data(test_data&&)				   = delete;
	test_data& operator=(test_data&&)	   = delete;
};

template<bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() noexcept {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	std::vector<test_data> testData{ maxIterations };
	std::vector<float> testDataResults01{};
	testDataResults01.resize(maxIterations);
	std::vector<float> testDataResults02{};
	testDataResults02.resize(maxIterations);
	size_t currentIndex = 0;

	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"unaligned-float", "cyan">([&] {
		uint64_t bytesProcessed{};
		ggml_vec_dot_q8_0_f32_unaligned_float(&testDataResults01[currentIndex], testData[currentIndex].orig_x.get(), testData[currentIndex].data.data());
		bytesProcessed += n;
		bnch_swt::doNotOptimizeAway(testDataResults01[currentIndex]);
		++currentIndex;
		return bytesProcessed;
	});
	currentIndex = 0;
	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"unaligned-int8", "cyan">([&] {
		uint64_t bytesProcessed{};
		ggml_vec_dot_q8_0_f32_unaligned_int8(&testDataResults01[currentIndex], testData[currentIndex].orig_x.get(), testData[currentIndex].data.data());
		bytesProcessed += n;
		bnch_swt::doNotOptimizeAway(testDataResults01[currentIndex]);
		++currentIndex;
		return bytesProcessed;
	});

	currentIndex = 0;
	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"aligned-float", "cyan">([&] {
		uint64_t bytesProcessed{};
		ggml_vec_dot_q8_0_f32_aligned_float(&testDataResults01[currentIndex], testData[currentIndex].orig_y.get(), testData[currentIndex].data.data());
		bnch_swt::doNotOptimizeAway(testDataResults02[currentIndex]);
		bytesProcessed += n;
		++currentIndex;
		return bytesProcessed;
	});

	currentIndex = 0;
	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"aligned-int8", "cyan">([&] {
		uint64_t bytesProcessed{};
		ggml_vec_dot_q8_0_f32_aligned_int8(&testDataResults01[currentIndex], testData[currentIndex].orig_y.get(), testData[currentIndex].data.data());
		bnch_swt::doNotOptimizeAway(testDataResults02[currentIndex]);
		bytesProcessed += n;
		++currentIndex;
		return bytesProcessed;
	});

	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::printResults(true, true);
}


int main() {
	testFunction<"reza-new-impl-test">();
	return 0;
}