#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstdint>

constexpr size_t QK8_0 = 32;

struct block_q8_0 {
	int8_t qs[QK8_0];
	uint16_t d;
};

struct block_q8_0_aligned_quants {
	alignas(32) int8_t qs[QK8_0];
};

struct block_q8_0_aligned_float {
	float d;
};

constexpr size_t QK8_0_MEGA_D  = 8;
constexpr size_t QK8_0_MEGA_QS = QK8_0_MEGA_D * QK8_0;

struct block_q8_0_mega_quants {
	alignas(32) int8_t qs[QK8_0_MEGA_QS];
};

struct block_q8_0_mega_float  {
	float d[QK8_0_MEGA_D];
};

constexpr size_t dim				= 2048 * 2048;
constexpr size_t num_blocks			= dim / sizeof(block_q8_0);
constexpr size_t num_blocks_aligned = dim / sizeof(block_q8_0_aligned_quants);
constexpr size_t num_blocks_mega	= dim / QK8_0_MEGA_QS;

struct orig {
	std::unique_ptr<block_q8_0[]> x{ std::make_unique<block_q8_0[]>(num_blocks) };
	std::unique_ptr<block_q8_0[]> y{ std::make_unique<block_q8_0[]>(num_blocks) };
	orig() {
		for (int i = 0; i < num_blocks; ++i) {
			for (size_t j = 0; j < QK8_0; ++j) {
				x[i].qs[j] = bnch_swt::random_generator::generateValue<uint8_t>();
				y[i].qs[j] = bnch_swt::random_generator::generateValue<uint8_t>();
			}
		}
	}
};

struct aligned {
	alignas(32) std::unique_ptr<block_q8_0_aligned_quants[]> w{ std::make_unique<block_q8_0_aligned_quants[]>(num_blocks_aligned) };
	alignas(32) std::unique_ptr<block_q8_0_aligned_quants[]> x{ std::make_unique<block_q8_0_aligned_quants[]>(num_blocks_aligned) };
	alignas(32) std::unique_ptr<block_q8_0_aligned_float[]> y{ std::make_unique<block_q8_0_aligned_float[]>(num_blocks_aligned) };
	alignas(32) std::unique_ptr<block_q8_0_aligned_float[]> z{ std::make_unique<block_q8_0_aligned_float[]>(num_blocks_aligned) };
	aligned() {
		for (int i = 0; i < num_blocks_aligned; ++i) {
			for (size_t j = 0; j < QK8_0; ++j) {
				w[i].qs[j] = bnch_swt::random_generator::generateValue<uint8_t>();
				x[i].qs[j] = bnch_swt::random_generator::generateValue<uint8_t>();
			}
		}
		for (int i = 0; i < num_blocks_aligned; ++i) {
			y[i].d = bnch_swt::random_generator::generateValue<float>();
			z[i].d = bnch_swt::random_generator::generateValue<float>();
		}
	}
};

struct mega_blocks {
	alignas(32) std::vector<block_q8_0_mega_quants, jsonifier::internal::alloc_wrapper<block_q8_0_mega_quants>> w{};
	alignas(32) std::vector<block_q8_0_mega_quants, jsonifier::internal::alloc_wrapper<block_q8_0_mega_quants>> x{};
	alignas(32) std::vector<block_q8_0_mega_float, jsonifier::internal::alloc_wrapper<block_q8_0_mega_float>> y{};
	alignas(32) std::vector<block_q8_0_mega_float, jsonifier::internal::alloc_wrapper<block_q8_0_mega_float>> z{};
	mega_blocks() {
		w.resize(num_blocks_mega);
		x.resize(num_blocks_mega);
		y.resize(num_blocks_mega);
		z.resize(num_blocks_mega);
		for (int i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
				w[i].qs[j] = bnch_swt::random_generator::generateValue<uint8_t>();
				x[i].qs[j] = bnch_swt::random_generator::generateValue<uint8_t>();
			}
		}
		for (int i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_D; ++j) {
				y[i].d[j] = bnch_swt::random_generator::generateValue<float>();
				z[i].d[j] = bnch_swt::random_generator::generateValue<float>();
			}
		}
	}
};

struct mega {
	alignas(32) std::vector<uint8_t, jsonifier::internal::alloc_wrapper<uint8_t>> w{};
	alignas(32) std::vector<uint8_t, jsonifier::internal::alloc_wrapper<uint8_t>> x{};
	alignas(32) std::vector<float, jsonifier::internal::alloc_wrapper<float>> y{};
	alignas(32) std::vector<float, jsonifier::internal::alloc_wrapper<float>> z{};
	mega() {
		w.resize(num_blocks_mega * QK8_0_MEGA_QS);
		x.resize(num_blocks_mega * QK8_0_MEGA_QS);
		y.resize(num_blocks_mega * QK8_0_MEGA_D);
		z.resize(num_blocks_mega * QK8_0_MEGA_D);
		for (int i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
				w[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<uint8_t>();
				x[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<uint8_t>();
			}
		}
		for (int i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_D; ++j) {
				y[i * QK8_0_MEGA_D + j] = bnch_swt::random_generator::generateValue<float>();
				z[i * QK8_0_MEGA_D + j] = bnch_swt::random_generator::generateValue<float>();
			}
		}
	}
};

struct mega_stack {
	alignas(32) std::array<uint8_t, dim> w{};
	alignas(32) std::array<uint8_t, dim> x{};
	alignas(32) std::array<float, num_blocks_mega * QK8_0_MEGA_D> y{};
	alignas(32) std::array<float, num_blocks_mega * QK8_0_MEGA_D> z{};
	mega_stack() {
		for (int i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
				w[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<uint8_t>();
				x[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<uint8_t>();
			}
		}
		for (int i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_D; ++j) {
				y[i * QK8_0_MEGA_D + j] = bnch_swt::random_generator::generateValue<float>();
				z[i * QK8_0_MEGA_D + j] = bnch_swt::random_generator::generateValue<float>();
			}
		}
	}
};

JSONIFIER_INLINE constexpr static float fp32_from_bits(uint32_t w) {
	return std::bit_cast<float>(w);
}

JSONIFIER_INLINE constexpr static uint32_t fp32_to_bits(float f) {
	return std::bit_cast<uint32_t>(f);
}

JSONIFIER_INLINE constexpr static float ggml_compute_fp16_to_fp32(uint16_t h) {
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

JSONIFIER_INLINE static __m256 sum_i16_pairs_float(const __m256i x) {
	const __m256i ones		   = _mm256_set1_epi16(1);
	const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
	return _mm256_cvtepi32_ps(summed_pairs);
}

JSONIFIER_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
	// Perform multiplication and create 16-bit values
	const __m256i dot = _mm256_maddubs_epi16(ax, sy);
	return sum_i16_pairs_float(dot);
}

// multiply int8_t, add results pairwise twice and return as float vector
JSONIFIER_INLINE static __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
	// Get absolute values of x vectors
	const __m256i ax = _mm256_sign_epi8(x, x);
	// Sign the values of the y vectors
	const __m256i sy = _mm256_sign_epi8(y, x);
	return mul_sum_us8_pairs_float(ax, sy);
}

JSONIFIER_INLINE static float hsum_float_8(const __m256 x) {
	__m128 res = _mm256_extractf128_ps(x, 1);
	res		   = _mm_add_ps(res, _mm256_castps256_ps128(x));
	res		   = _mm_add_ps(res, _mm_movehl_ps(res, res));
	res		   = _mm_add_ss(res, _mm_movehdup_ps(res));
	return _mm_cvtss_f32(res);
}

JSONIFIER_INLINE void oi_vec_dot_q8_0_q8_0(const int ne, float* dst, const block_q8_0* __restrict x, const block_q8_0* __restrict y) {
	const int nb = ne / QK8_0;

	// Initialize accumulator with zeros
	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {
		// Compute combined scale for the block
		const float xd = ggml_compute_fp16_to_fp32(x[ib].d);
		const float yd = ggml_compute_fp16_to_fp32(y[ib].d);
		const __m256 d = _mm256_set1_ps(xd * yd);

		__m256i qx = _mm256_load_si256(( const __m256i* )x[ib].qs);
		__m256i qy = _mm256_load_si256(( const __m256i* )y[ib].qs);

		const __m256 q = mul_sum_i8_pairs_float(qx, qy);

		// Multiply q with scale and accumulate
		acc = _mm256_fmadd_ps(d, q, acc);
	}

	*dst = hsum_float_8(acc);
}

JSONIFIER_INLINE void oi_vec_dot_q8_0_q8_0_aligned(const int ne, float* dst, const block_q8_0_aligned_quants* __restrict x, const block_q8_0_aligned_quants* __restrict y,
	const block_q8_0_aligned_float* x_x, const block_q8_0_aligned_float* y_x) {
	const int nb = ne / QK8_0;

	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {
		const float xd = x_x[ib].d;
		const float yd = y_x[ib].d;
		const __m256 d = _mm256_set1_ps(xd * yd);

		__m256i qx = _mm256_load_si256(( const __m256i* )x[ib].qs);
		__m256i qy = _mm256_load_si256(( const __m256i* )y[ib].qs);

		const __m256 q = mul_sum_i8_pairs_float(qx, qy);

		// Multiply q with scale and accumulate
		acc = _mm256_fmadd_ps(d, q, acc);
	}

	*dst = hsum_float_8(acc);
}

JSONIFIER_INLINE void oi_vec_dot_q8_0_q8_0_mega_blocks(const int ne, float* dst, const block_q8_0_mega_quants* __restrict x, const block_q8_0_mega_quants* __restrict y,
	const block_q8_0_mega_float* __restrict x_x, const block_q8_0_mega_float* __restrict y_x) {
	const int nb = ne / QK8_0_MEGA_QS;
	float sum{};
	// Main loop
	for (int ib = 0; ib < nb; ++ib) {

		float dx   = x_x[ib].d[0];
		float dy   = y_x[ib].d[0];
		__m256i qx = _mm256_load_si256(( const __m256i* )x[ib].qs);
		__m256i qy = _mm256_load_si256(( const __m256i* )y[ib].qs);

		// 1
		dx	= x_x[ib].d[1];
		dy	= y_x[ib].d[1];
		qx	= _mm256_load_si256(( const __m256i* )&x[ib].qs[1 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[ib].qs[1 * 32]);

		// 2
		dx	= x_x[ib].d[2];
		dy	= y_x[ib].d[2];
		qx	= _mm256_load_si256(( const __m256i* )&x[ib].qs[2 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[ib].qs[2 * 32]);

		// 3
		dx	= x_x[ib].d[3];
		dy	= y_x[ib].d[3];
		qx	= _mm256_load_si256(( const __m256i* )&x[ib].qs[3 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[ib].qs[3 * 32]);

		// 4
		dx	= x_x[ib].d[0];
		dy	= y_x[ib].d[0];
		qx	= _mm256_load_si256(( const __m256i* )&x[ib].qs[4 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[ib].qs[4 * 32]);

		// 5
		dx	= x_x[ib].d[1];
		dy	= y_x[ib].d[1];
		qx	= _mm256_load_si256(( const __m256i* )&x[ib].qs[5 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[ib].qs[5 * 32]);

		// 6
		dx	= x_x[ib].d[2];
		dy	= y_x[ib].d[2];
		qx	= _mm256_load_si256(( const __m256i* )&x[ib].qs[6 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[ib].qs[6 * 32]);

		// 7
		dx	= x_x[ib].d[3];
		dy	= y_x[ib].d[3];
		qx	= _mm256_load_si256(( const __m256i* )&x[ib].qs[7 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[ib].qs[7 * 32]);
		sum += qy.m256i_u64[0];
	}

	*dst = sum;
}

JSONIFIER_INLINE void oi_vec_dot_q8_0_q8_0_mega(const int ne, float* dst, const uint8_t* __restrict x, const uint8_t* __restrict y, const float* __restrict x_x,
	const float* __restrict y_x) {
	const int nb = ne / QK8_0_MEGA_QS;

	float sum{};

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {

		float dx   = x_x[ib * 8];
		float dy   = y_x[ib * 8];
		__m256i qx = _mm256_load_si256(( const __m256i* )&x[ib * 32]);
		__m256i qy = _mm256_load_si256(( const __m256i* )&y[ib * 32]);

		// 1
		dx	= x_x[ib * 8 + 1];
		dy	= y_x[ib * 8 + 1];
		qx	= _mm256_load_si256(( const __m256i* )&x[(ib + 1) * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[(ib + 1) * 32]);

		// 2
		dx	= x_x[ib * 8 + 2];
		dy	= y_x[ib * 8 + 2];
		qx	= _mm256_load_si256(( const __m256i* )&x[(ib + 2) * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[(ib + 2) * 32]);

		// 3
		dx	= x_x[ib * 8 + 3];
		dy	= y_x[ib * 8 + 3];
		qx	= _mm256_load_si256(( const __m256i* )&x[(ib + 3) * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[(ib + 3) * 32]);

		// 4
		dx	= x_x[ib * 8 + 0];
		dy	= y_x[ib * 8 + 0];
		qx	= _mm256_load_si256(( const __m256i* )&x[(ib + 4) * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[(ib + 4) * 32]);

		// 5
		dx	= x_x[ib * 8 + 1];
		dy	= y_x[ib * 8 + 1];
		qx	= _mm256_load_si256(( const __m256i* )&x[(ib + 5) * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[(ib + 5) * 32]);

		// 6
		dx	= x_x[ib * 8 + 2];
		dy	= y_x[ib * 8 + 2];
		qx	= _mm256_load_si256(( const __m256i* )&x[(ib + 6) * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[(ib + 6) * 32]);

		// 7
		dx	= x_x[ib * 8 + 3];
		dy	= y_x[ib * 8 + 3];
		qx	= _mm256_load_si256(( const __m256i* )&x[(ib + 7) * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&y[(ib + 7) * 32]);
		sum += qy.m256i_u64[0];
	}

	*dst = sum;
}

static constexpr uint64_t maxIterations{ 100};

template<bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	size_t currentIndex{};
	float dst{};
	{
		std::array<mega, maxIterations> testDataMega{};
		currentIndex = 0ull;
		bnch_swt::benchmark_stage<testName, maxIterations, 20>::template runBenchmark<"mega", "CYAN">([&] {
			uint64_t bytesProcessed{};
			oi_vec_dot_q8_0_q8_0_mega(dim, &dst, testDataMega[currentIndex].w.data(), testDataMega[currentIndex].x.data(), testDataMega[currentIndex].y.data(),
				testDataMega[currentIndex].z.data());
			bytesProcessed += dim;
			bnch_swt::doNotOptimizeAway(dst);
			++currentIndex;
			return bytesProcessed;
		});
	}
	std::cout << "Passed test 01" << std::endl;
	{
		std::array<mega_blocks, maxIterations> testDataMega{};
		currentIndex = 0ull;
		bnch_swt::benchmark_stage<testName, maxIterations, 20>::template runBenchmark<"mega-blocks", "CYAN">([&] {
			uint64_t bytesProcessed{};
			oi_vec_dot_q8_0_q8_0_mega_blocks(dim, &dst, testDataMega[currentIndex].w.data(), testDataMega[currentIndex].x.data(), testDataMega[currentIndex].y.data(),
				testDataMega[currentIndex].z.data());
			bytesProcessed += dim;
			bnch_swt::doNotOptimizeAway(dst);
			++currentIndex;
			return bytesProcessed;
		});
	}
	std::cout << "Passed test 02" << std::endl;
	{
		std::array<mega, maxIterations> testDataMega{};
		currentIndex = 0ull;
		bnch_swt::benchmark_stage<testName, maxIterations, 20>::template runBenchmark<"mega-blocks-casted", "CYAN">([&] {
			uint64_t bytesProcessed{};
			oi_vec_dot_q8_0_q8_0_mega_blocks(dim, &dst, ( const block_q8_0_mega_quants* )testDataMega[currentIndex].w.data(),
				( const block_q8_0_mega_quants* )testDataMega[currentIndex].x.data(), ( const block_q8_0_mega_float* )testDataMega[currentIndex].y.data(),
				( const block_q8_0_mega_float* )testDataMega[currentIndex].z.data());
			bytesProcessed += dim;
			bnch_swt::doNotOptimizeAway(dst);
			++currentIndex;
			return bytesProcessed;
		});
	}
	std::cout << "Passed test 02" << std::endl;

	bnch_swt::benchmark_stage<testName, maxIterations, 20>::printResults(true, true);
}

int main() {
	testFunction<"orig-vs-aligned-vs-mega">();
	return 0;
}