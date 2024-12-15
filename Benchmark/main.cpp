#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstdint>

constexpr size_t QK8_0		   = 32;
constexpr size_t QK8_0_MEGA_D  = 8;
constexpr size_t QK8_0_MEGA_QS = QK8_0_MEGA_D * QK8_0;

constexpr size_t dim			 = 3072 * 3072;
constexpr size_t num_blocks_mega = dim / QK8_0_MEGA_QS;

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
		for (int32_t i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
				w[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<uint8_t>();
				x[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<uint8_t>();
			}
		}
		for (int32_t i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_D; ++j) {
				y[i * QK8_0_MEGA_D + j] = static_cast<float>(bnch_swt::random_generator::generateValue<uint16_t>());
				z[i * QK8_0_MEGA_D + j] = static_cast<float>(bnch_swt::random_generator::generateValue<uint16_t>());
			}
		}
	}
};

JSONIFIER_INLINE static __m256 sum_i16_pairs_float(const __m256i x) {
	const __m256i ones		   = _mm256_set1_epi16(1);
	const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
	return _mm256_cvtepi32_ps(summed_pairs);
}

JSONIFIER_INLINE static __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
	const __m256i dot = _mm256_maddubs_epi16(ax, sy);
	return sum_i16_pairs_float(dot);
}

JSONIFIER_INLINE static __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
	const __m256i ax = _mm256_sign_epi8(x, x);
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

JSONIFIER_INLINE void oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2_orig(int32_t n, float* s, const void* x_quants, const void* x_scales, const void* y_quants, const void* y_scales) {
	constexpr int qk	= QK8_0;
	constexpr int stepk = QK8_0 * 8;// each iteration is 8x unrolled

	const int nb = n / stepk;

	const uint8_t* xq = reinterpret_cast<const uint8_t*>(x_quants);
	const uint8_t* yq = reinterpret_cast<const uint8_t*>(y_quants);
	const float* xs	  = reinterpret_cast<const float*>(x_scales);
	const float* ys	  = reinterpret_cast<const float*>(y_scales);

	// Initialize accumulator with zeros
	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (int ib = 0; ib < nb; ++ib) {
		// prefetch the quants for the next iteration of the loop
		jsonifier::jsonifierPrefetchImpl(&yq[(ib + 1) * stepk]);
		jsonifier::jsonifierPrefetchImpl(&xq[(ib + 1) * stepk]);

		// 0
		float dx   = xs[ib * 8];
		float dy   = ys[ib * 8];
		__m256 d   = _mm256_set1_ps(dx * dy);
		__m256i qx = _mm256_load_si256(( const __m256i* )&xq[ib * stepk]);
		__m256i qy = _mm256_load_si256(( const __m256i* )&yq[ib * stepk]);
		__m256 q   = mul_sum_i8_pairs_float(qx, qy);
		acc		   = _mm256_fmadd_ps(d, q, acc);

		// 1
		dx	= xs[ib * 8 + 1];
		dy	= ys[ib * 8 + 1];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )&xq[ib * stepk + 1 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&yq[ib * stepk + 1 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 2
		dx	= xs[ib * 8 + 2];
		dy	= ys[ib * 8 + 2];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )&xq[ib * stepk + 2 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&yq[ib * stepk + 2 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 3
		dx	= xs[ib * 8 + 3];
		dy	= ys[ib * 8 + 3];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )&xq[ib * stepk + 3 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&yq[ib * stepk + 3 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 4
		dx	= xs[ib * 8 + 4];
		dy	= ys[ib * 8 + 4];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )&xq[ib * stepk + 4 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&yq[ib * stepk + 4 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 5
		dx	= xs[ib * 8 + 5];
		dy	= ys[ib * 8 + 5];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )&xq[ib * stepk + 5 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&yq[ib * stepk + 5 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 6
		dx	= xs[ib * 8 + 6];
		dy	= ys[ib * 8 + 6];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )&xq[ib * stepk + 6 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&yq[ib * stepk + 6 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);

		// 7
		dx	= xs[ib * 8 + 7];
		dy	= ys[ib * 8 + 7];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )&xq[ib * stepk + 7 * 32]);
		qy	= _mm256_load_si256(( const __m256i* )&yq[ib * stepk + 7 * 32]);
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
	}

	*s = hsum_float_8(acc);
}

template<size_t n> JSONIFIER_INLINE void oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2(float* s, const uint8_t* xq, const float* xs, const uint8_t* yq, const float* ys) {
	constexpr int32_t stepk = QK8_0 * 8;// each iteration is 8x unrolled
	static constexpr jsonifier::internal::array<size_t, 8> second_indices_quants{ [] {
		jsonifier::internal::array<size_t, 8> returnValues{};
		for (size_t x = 0; x < 8; ++x) {
			returnValues[x] = x * 32;
		}
		return returnValues;
	}() };

	constexpr int64_t nb = n / stepk;
	// Initialize accumulator with zeros
	__m256 acc = _mm256_setzero_ps();

	// Main loop
	for (uint64_t ib = 0; ib < nb; ++ib) {
		jsonifier::jsonifierPrefetchImpl(&yq[(ib + 1) * stepk]);
		jsonifier::jsonifierPrefetchImpl(&xq[(ib + 1) * stepk]);
		const size_t* second_index{ &second_indices_quants[0] };
		size_t first_index_quants{ ib * 256ull };
		size_t first_index_scales{ ib * 8ull };
		// 0
		float dx   = xs[first_index_scales];
		float dy   = ys[first_index_scales];
		__m256 d   = _mm256_set1_ps(dx * dy);
		__m256i qx = _mm256_load_si256(( const __m256i* )(&xq[first_index_quants]));
		__m256i qy = _mm256_load_si256(( const __m256i* )(&yq[first_index_quants]));
		__m256 q   = mul_sum_i8_pairs_float(qx, qy);
		acc		   = _mm256_fmadd_ps(d, q, acc);
		++second_index;
		// 1
		dx	= xs[first_index_scales + 1];
		dy	= ys[first_index_scales + 1];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(&xq[first_index_quants + *second_index]));
		qy	= _mm256_load_si256(( const __m256i* )(&yq[first_index_quants + *second_index]));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 2
		dx	= xs[first_index_scales + 2];
		dy	= ys[first_index_scales + 2];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(&xq[first_index_quants + *second_index]));
		qy	= _mm256_load_si256(( const __m256i* )(&yq[first_index_quants + *second_index]));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 3
		dx	= xs[first_index_scales + 3];
		dy	= ys[first_index_scales + 3];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(&xq[first_index_quants + *second_index]));
		qy	= _mm256_load_si256(( const __m256i* )(&yq[first_index_quants + *second_index]));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 4
		dx	= xs[first_index_scales + 4];
		dy	= ys[first_index_scales + 4];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(&xq[first_index_quants + *second_index]));
		qy	= _mm256_load_si256(( const __m256i* )(&yq[first_index_quants + *second_index]));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 5
		dx	= xs[first_index_scales + 5];
		dy	= ys[first_index_scales + 5];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(&xq[first_index_quants + *second_index]));
		qy	= _mm256_load_si256(( const __m256i* )(&yq[first_index_quants + *second_index]));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 6
		dx	= xs[first_index_scales + 6];
		dy	= ys[first_index_scales + 6];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(&xq[first_index_quants + *second_index]));
		qy	= _mm256_load_si256(( const __m256i* )(&yq[first_index_quants + *second_index]));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;

		// 7
		dx	= xs[first_index_scales + 7];
		dy	= ys[first_index_scales + 7];
		d	= _mm256_set1_ps(dx * dy);
		qx	= _mm256_load_si256(( const __m256i* )(&xq[first_index_quants + *second_index]));
		qy	= _mm256_load_si256(( const __m256i* )(&yq[first_index_quants + *second_index]));
		q	= mul_sum_i8_pairs_float(qx, qy);
		acc = _mm256_fmadd_ps(d, q, acc);
		++second_index;
	}
	*s = hsum_float_8(acc);
}

static constexpr uint64_t maxIterations{ 80 };

template<bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	float dst{};
	float dst01{};
	size_t currentIndex{};
	jsonifier::internal::array<mega, maxIterations> testDataMega{};
	{
		currentIndex = 0;
		bnch_swt::benchmark_stage<testName, maxIterations, 20>::template runBenchmark<"mega-orig", "CYAN">([&] {
			uint64_t bytesProcessed{};
			oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2_orig(dim, &dst, testDataMega[currentIndex].w.data(), testDataMega[currentIndex].y.data(), testDataMega[currentIndex].x.data(),
				testDataMega[currentIndex].z.data());
			bytesProcessed += dim;
			bnch_swt::doNotOptimizeAway(dst);
			++currentIndex;
			return bytesProcessed;
		});
	}
	std::cout << "Passed test 01" << std::endl;
	{
		currentIndex = 0;
		bnch_swt::benchmark_stage<testName, maxIterations, 20>::template runBenchmark<"mega", "CYAN">([&] {
			uint64_t bytesProcessed{};
			oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2<dim>(&dst, testDataMega[currentIndex].w.data(), testDataMega[currentIndex].y.data(), testDataMega[currentIndex].x.data(),
				testDataMega[currentIndex].z.data());
			bytesProcessed += dim;
			bnch_swt::doNotOptimizeAway(dst);
			++currentIndex;
			return bytesProcessed;
		});
	}
	std::cout << "Passed test 02" << std::endl;

	bnch_swt::benchmark_stage<testName, maxIterations, 20>::printResults(true, true);
}

int32_t main() {
	std::ifstream file(BASE_PATH, std::ios::binary);
	if (file.is_open()) {
		int n = 0, nb = 0;
		file.read(( char* )&n, sizeof(int));
		file.read(( char* )&nb, sizeof(int));
		assert(n / 32 == nb);

		struct ggml_block_q8_0 {
			uint16_t d;
			uint8_t qs[32];
		};

		std::vector<ggml_block_q8_0> x, y;
		x.resize(nb);
		y.resize(nb);

		file.read(( char* )x.data(), nb * sizeof(ggml_block_q8_0));
		file.read(( char* )y.data(), nb * sizeof(ggml_block_q8_0));

		float s = 0.f;
		file.read(( char* )&s, sizeof(float));
		mega testData{};

		testData.w.resize(n);
		testData.x.resize(n);
		testData.y.resize(nb);
		testData.z.resize(nb);
		size_t bytesTotal{};
		for (int i = 0; i < nb; ++i) {
			testData.y[i] = static_cast<float>(x[i].d);
			testData.z[i] = static_cast<float>(y[i].d);
			bytesTotal += sizeof(testData.y[i]) * 2;
			for (int j = 0; j < 32; ++j) {
				testData.w[i * 32 + j] = x[i].qs[j];
				testData.x[i * 32 + j] = y[i].qs[j];
				bytesTotal += sizeof(testData.x[i * 32 + j]) * 2;
			}
		}
		float dst01{};
		float dst02{};
		oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2_orig(dim, &dst01, testData.w.data(), testData.y.data(), testData.x.data(), testData.z.data());
		oiml_vec_dot_i8x32xf32x1_i8x32xf32x1_avx2<dim>(&dst02, testData.w.data(), testData.y.data(), testData.x.data(), testData.z.data());
		if (dst01 == dst02) {
			std::cout << "They are equal and they are: " << dst01 << ", and: " << dst02 << std::endl;
		} else {
			std::cout << "They are NOT equal and they are: " << dst01 << ", and: " << dst02 << std::endl;
		}
		testFunction<"orig-vs-aligned-vs-mega">();
	}

	return 0;
}