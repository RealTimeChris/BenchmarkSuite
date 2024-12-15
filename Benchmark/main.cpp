#include <BnchSwt/BenchmarkSuite.hpp>
#include <immintrin.h>

constexpr size_t QK8_0 = 32;

BNCH_SWT_INLINE constexpr static float fp32_from_bits(uint32_t w) {
	return std::bit_cast<float>(w);
}

BNCH_SWT_INLINE constexpr static uint32_t fp32_to_bits(float f) {
	return std::bit_cast<uint32_t>(f);
}

BNCH_SWT_INLINE constexpr static float ggml_compute_fp16_to_fp32(uint16_t h) {
	const uint32_t w	 = ( uint32_t )h << 16;
	const uint32_t sign	 = w & UINT32_C(0x80000000);
	const uint32_t two_w = w + w;

	constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
	constexpr float exp_scale	  = fp32_from_bits(UINT32_C(0x7800000));
	const float normalized_value  = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	constexpr const uint32_t magic_mask = UINT32_C(126) << 23;
	constexpr const float magic_bias	= 0.5f;
	const float denormalized_value		= fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
	const uint32_t result				   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

BNCH_SWT_INLINE static __m256 sum_i16_pairs_float(const __m256i x) {
	const __m256i ones		   = _mm256_set1_epi16(1);
	const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
	return _mm256_cvtepi32_ps(summed_pairs);
}

BNCH_SWT_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
	// Perform multiplication and create 16-bit values
	const __m256i dot = _mm256_maddubs_epi16(ax, sy);
	return sum_i16_pairs_float(dot);
}

// multiply int8_t, add results pairwise twice and return as float vector
BNCH_SWT_INLINE static __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
	// Get absolute values of x vectors
	const __m256i ax = _mm256_sign_epi8(x, x);
	// Sign the values of the y vectors
	const __m256i sy = _mm256_sign_epi8(y, x);
	return mul_sum_us8_pairs_float(ax, sy);
}

BNCH_SWT_INLINE static float hsum_float_8(const __m256 x) {
	__m128 res = _mm256_extractf128_ps(x, 1);
	res		   = _mm_add_ps(res, _mm256_castps256_ps128(x));
	res		   = _mm_add_ps(res, _mm_movehl_ps(res, res));
	res		   = _mm_add_ss(res, _mm_movehdup_ps(res));
	return _mm_cvtss_f32(res);
}

template<size_t n> BNCH_SWT_INLINE void oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2(float* s, const uint8_t* xq, const float* xs, const uint8_t* yq, const float* ys) {
	constexpr int32_t stepk = QK8_0 * 8;// each iteration is 8x unrolled
	static constexpr std::array<size_t, 8> second_indices_quants{ [] {
		std::array<size_t, 8> returnValues{};
		for (size_t x = 0; x < 8; ++x) {
			returnValues[x] = x * 32;
		}
		return returnValues;
	}() };

	constexpr int64_t nb = n / stepk;

	alignas(32) std::array<uint8_t, 256> temp_buffer_x_quants{};
	alignas(32) std::array<uint8_t, 256> temp_buffer_y_quants{};
	alignas(4) std::array<float, 8> temp_buffer_x_scales{};
	alignas(4) std::array<float, 8> temp_buffer_y_scales{};
	// Initialize accumulator with zeros
	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (uint64_t ib = 0; ib < nb; ++ib) {
		const size_t* second_index{ &second_indices_quants[0] };
		size_t first_index_quants{ ib * 256ull };
		size_t first_index_scales{ ib * 8ull };
		std::memcpy(temp_buffer_x_quants.data(), &xq[first_index_quants], 256);
		std::memcpy(temp_buffer_y_quants.data(), &yq[first_index_quants], 256);
		std::memcpy(temp_buffer_x_scales.data(), &xs[first_index_scales], sizeof(float) * 8);
		std::memcpy(temp_buffer_y_scales.data(), &ys[first_index_scales], sizeof(float) * 8);
		// 0
		float dx   = temp_buffer_x_scales[0];
		float dy   = temp_buffer_y_scales[0];
		__m256 d   = _mm256_set1_ps(dx * dy);
		__m256i qx = _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data()));
		__m256i qy = _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data()));
		__m256 q   = mul_sum_i8_pairs_float(qx, qy);
		acc		   = _mm256_fmadd_ps(d, q, acc);
		++second_index;
		// 1
		dx	= temp_buffer_x_scales[1];
		dy	= temp_buffer_y_scales[1];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data() + *second_index));
		qy	= _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data() + *second_index));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 2
		dx	= temp_buffer_x_scales[2];
		dy	= temp_buffer_y_scales[2];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data() + *second_index));
		qy	= _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data() + *second_index));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 3
		dx	= temp_buffer_x_scales[3];
		dy	= temp_buffer_y_scales[3];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data() + *second_index));
		qy	= _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data() + *second_index));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 4
		dx	= temp_buffer_x_scales[4];
		dy	= temp_buffer_y_scales[4];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data() + *second_index));
		qy	= _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data() + *second_index));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 5
		dx	= temp_buffer_x_scales[5];
		dy	= temp_buffer_y_scales[5];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data() + *second_index));
		qy	= _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data() + *second_index));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 6
		dx	= temp_buffer_x_scales[6];
		dy	= temp_buffer_y_scales[6];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data() + *second_index));
		qy	= _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data() + *second_index));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 7
		dx	= temp_buffer_x_scales[7];
		dy	= temp_buffer_y_scales[7];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(temp_buffer_x_quants.data() + *second_index));
		qy	= _mm256_load_si256(( const __m256i* )(temp_buffer_y_quants.data() + *second_index));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;
	}
	*s = hsum_float_8(acc);
}

int32_t main() {
	float dst{};
	uint8_t values01[512 * 16]{};
	uint8_t values02[512 * 16]{};
	float values03[(512 * 16) / 32]{};
	float values04[(512 * 16) / 32]{};
	for (size_t x = 0; x < 512 * 16; ++x) {
		values01[x]		 = static_cast<uint8_t>(rand());
		values02[x]		 = static_cast<uint8_t>(rand());
		values03[x % 32] = static_cast<float>(rand());
		values04[x % 32] = static_cast<float>(rand());
	}
	oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2<512>(&dst, values01, values03, values02, values04);
	bnch_swt::doNotOptimizeAway(dst);
	return 0;
}