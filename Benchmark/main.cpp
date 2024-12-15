#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

struct block_q8_0_orig {
	int16_t simulated_half{};
	alignas(16) int8_t qs[32]{};
};

struct block_q8_0_d_new {
	alignas(16) int16_t simulated_half[8]{};
};

struct block_q8_0_qs_new {
	alignas(16) int8_t qs[16]{};
};

JSONIFIER_INLINE static void oi_test_q8_existing_impl(const int n, float* dst, const block_q8_0_orig* x, const block_q8_0_orig* y) {
	const int steps = n / 32;
	__m256 sum		= _mm256_setzero_ps();
	for (int s = 0; s < steps; ++s, ++x, ++y) {
		const __m256 dx		  = _mm256_set1_ps(( float )(x->simulated_half));
		const __m256 dy		  = _mm256_set1_ps(( float )(x->simulated_half));
		const __m256i x0	  = _mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )x->qs));
		const __m256i x1	  = _mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )x->qs + 16));
		const __m256i y0	  = _mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )y->qs));
		const __m256i y1	  = _mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )y->qs + 16));
		const __m256 partial0 = _mm256_mul_ps(_mm256_cvtepi32_ps(x0), _mm256_cvtepi32_ps(y0));
		const __m256 partial1 = _mm256_mul_ps(_mm256_cvtepi32_ps(x1), _mm256_cvtepi32_ps(y0));
		const __m256 partial  = _mm256_mul_ps(partial0, partial1);
		sum					  = _mm256_fmadd_ps(_mm256_mul_ps(partial, dx), dy, sum);
	}
	float temp[8];
	_mm256_store_ps(temp, sum);
	*dst = temp[0];
}

JSONIFIER_INLINE static void oi_test_q8_new_impl(const int n, float* dst, const block_q8_0_d_new* xd, const block_q8_0_qs_new* xqs, const block_q8_0_d_new* yd,
	const block_q8_0_qs_new* yqs) {
	const int steps = n / 128;
	__m256 sum		= _mm256_setzero_ps();
	float dx32x8[8] = { 0 };
	float dy32x8[8] = { 0 };
	for (int s = 0; s < steps; ++s, ++xd, ++yd) {
		_mm_prefetch(( const char* )(xd + 1), _MM_HINT_T0);
		_mm_prefetch(( const char* )(yd + 1), _MM_HINT_T0);
		_mm_prefetch(( const char* )(xqs + 8), _MM_HINT_T0);
		_mm_prefetch(( const char* )(yqs + 8), _MM_HINT_T0);
		const __m256 dx_8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )xd->simulated_half)));
		const __m256 dy_8 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )yd->simulated_half)));
		_mm256_store_ps(dx32x8, dx_8);
		_mm256_store_ps(dy32x8, dy_8);
		for (int id = 0; id < 8; ++id) {
			const __m256 dx		 = _mm256_set1_ps(dx32x8[id]);
			const __m256 dy		 = _mm256_set1_ps(dy32x8[id]);
			const __m256i x0	 = _mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )xqs->qs));
			const __m256i y0	 = _mm256_cvtepi16_epi32(_mm_load_si128(( __m128i* )yqs->qs));
			const __m256 partial = _mm256_mul_ps(_mm256_cvtepi32_ps(x0), _mm256_cvtepi32_ps(y0));
			sum					 = _mm256_fmadd_ps(_mm256_mul_ps(partial, dx), dy, sum);
			++xqs;
			++yqs;
		}
	}
	float temp[8];
	_mm256_store_ps(temp, sum);
	*dst = temp[0];
}

static constexpr uint64_t maxIterations{ 200 };
static constexpr uint64_t measuredIterations{ 20 };

struct test_data {
	const int n				   = 8192;
	alignas(16) block_q8_0_orig* orig_x	   = ( block_q8_0_orig* )malloc((n / 32) * sizeof(block_q8_0_orig));
	alignas(16) block_q8_0_orig* orig_y	   = ( block_q8_0_orig* )malloc((n / 32) * sizeof(block_q8_0_orig));
	alignas(16) block_q8_0_d_new* new_xd   = ( block_q8_0_d_new* )malloc((n / 128) * sizeof(block_q8_0_d_new));
	alignas(16) block_q8_0_d_new* new_yd   = ( block_q8_0_d_new* )malloc((n / 128) * sizeof(block_q8_0_d_new));
	alignas(16) block_q8_0_qs_new* new_xqs = ( block_q8_0_qs_new* )malloc((n / 16) * sizeof(block_q8_0_qs_new));
	alignas(16) block_q8_0_qs_new* new_yqs = ( block_q8_0_qs_new* )malloc((n / 16) * sizeof(block_q8_0_qs_new));
	test_data() {
		for (size_t x = 0; x < n / 32; ++x) {
			if (orig_x) {
				orig_x[x].simulated_half = bnch_swt::random_generator::generateValue<int16_t>();
				for (size_t y = 0; y < 32; ++y) {
					orig_x[x].qs[y] = bnch_swt::random_generator::generateValue<int8_t>();
				}
			}
			if (orig_y) {
				orig_y[x].simulated_half = bnch_swt::random_generator::generateValue<int16_t>();
				for (size_t y = 0; y < 32; ++y) {
					orig_y[x].qs[y] = bnch_swt::random_generator::generateValue<int8_t>();
				}
			}
		}
		for (size_t x = 0; x < n / 128; ++x) {
			for (size_t y = 0; y < 8; ++y) {
				if (new_xd) {
					new_xd[x].simulated_half[y] = bnch_swt::random_generator::generateValue<int16_t>();
				}
				if (new_yd) {
					new_yd[x].simulated_half[y] = bnch_swt::random_generator::generateValue<int16_t>();
				}
			}
		}
		for (size_t x = 0; x < n / 16; ++x) {
			if (new_xqs) {
				for (size_t y = 0; y < 16; ++y) {
					new_xqs[x].qs[y] = bnch_swt::random_generator::generateValue<int8_t>();
				}
			}
			if (new_yqs) {
				for (size_t y = 0; y < 16; ++y) {
					new_yqs[x].qs[y] = bnch_swt::random_generator::generateValue<int8_t>();
				}
			}
		}
	}
	test_data(const test_data&)			   = delete;
	test_data& operator=(const test_data&) = delete;
	test_data(test_data&&)				   = delete;
	test_data& operator=(test_data&&)	   = delete;
	~test_data() {
		free(orig_x);
		free(orig_y);
		free(new_xd);
		free(new_yd);
		free(new_xqs);
		free(new_yqs);
	}
};

template<bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	std::array<test_data, maxIterations> testData{};
	std::array<float, maxIterations> testDataResults01{};
	std::array<float, maxIterations> testDataResults02{};
	size_t currentIndex{};

	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"oi_test_q8_existing_impl", "CYAN">([&] {
		uint64_t bytesProcessed{};
		oi_test_q8_existing_impl(testData[currentIndex].n, &testDataResults01[currentIndex], testData[currentIndex].orig_x, testData[currentIndex].orig_y);
		bytesProcessed += testData[currentIndex].n;
		bnch_swt::doNotOptimizeAway(testDataResults01[currentIndex]);
		++currentIndex;
		return bytesProcessed;
	});

	currentIndex = 0;
	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"oi_test_q8_new_impl", "CYAN">([&] {
		uint64_t bytesProcessed{};
		oi_test_q8_new_impl(testData[currentIndex].n, &testDataResults02[currentIndex], testData[currentIndex].new_xd, testData[currentIndex].new_xqs,
			testData[currentIndex].new_yd, testData[currentIndex].new_yqs);
		bnch_swt::doNotOptimizeAway(testDataResults02[currentIndex]);
		bytesProcessed += testData[currentIndex].n;
		++currentIndex;
		return bytesProcessed;
	});

	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::printResults(true, true);
}

int main() {
	testFunction<"reza-new-impl-test">();
	return 0;
}