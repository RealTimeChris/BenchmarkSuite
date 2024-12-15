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
	int8_t qs[QK8_0]{};
	uint16_t d{};
};

constexpr size_t QK8_0_MEGA_D	 = 8;
constexpr size_t QK8_0_MEGA_QS	 = QK8_0_MEGA_D * QK8_0;
constexpr size_t dim			 = 32 * 8;
constexpr size_t num_blocks		 = dim / 32;
constexpr size_t num_blocks_mega = dim / QK8_0_MEGA_QS;

struct mega {
	alignas(32) std::vector<uint8_t, jsonifier::internal::alloc_wrapper<uint8_t>> x_qs{};
	alignas(32) std::vector<uint16_t, jsonifier::internal::alloc_wrapper<uint16_t>> x_d{};
	alignas(32) std::vector<float, jsonifier::internal::alloc_wrapper<float>> z{};
	mega() {
		x_qs.resize(num_blocks_mega * QK8_0_MEGA_QS);
		x_d.resize(num_blocks_mega * QK8_0_MEGA_QS / 32);
		z.resize(num_blocks_mega * QK8_0_MEGA_QS);
		for (int32_t i = 0; i < num_blocks_mega; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
				x_qs[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<uint8_t>();
			}
			for (size_t j = 0; j < QK8_0_MEGA_QS / 32; ++j) {
				x_d[i * (QK8_0_MEGA_QS / 32) + j] = bnch_swt::random_generator::generateValue<uint16_t>();
			}
			for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
				z[i * QK8_0_MEGA_QS + j] = bnch_swt::random_generator::generateValue<float>();
			}
		}
	}
};

struct orig {
	std::unique_ptr<block_q8_0[]> x{ std::make_unique<block_q8_0[]>(num_blocks) };
	std::unique_ptr<float[]> z{ std::make_unique<float[]>(num_blocks * 32) };
	orig& operator=(const orig& other) {
		std::copy(other.x.get(), other.x.get() + num_blocks, x.get());
		std::copy(other.z.get(), other.z.get() + num_blocks * 32, z.get());
		return *this;
	}
	orig(const orig& other) {
		*this = other;
	}
	orig() = default;
	orig(const mega& mega_new) {
		size_t index = 0;
		for (int32_t i = 0; i < num_blocks; ++i) {
			for (size_t j = 0; j < QK8_0_MEGA_QS; ++j) {
				x[i].qs[j % 32] = mega_new.x_qs[i * QK8_0_MEGA_QS + j];
			}
			for (size_t j = 0; j < 32; ++j) {
				z[i * 32 + j] = mega_new.z[i * 32 + j];
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

void ggml_vec_dot_q8_0_f32_converted(int n, float* __restrict s, const block_q8_0* __restrict vx, const float* __restrict vy) {
	const int nb = n / QK8_0;

	float sumf = 0;

	const block_q8_0* __restrict x_qs = vx;
	const float* __restrict x_d	   = vy;
	std::cout << "SECOND 32 VALUES: " << std::endl;
	for (size_t j = 0; j < 32; ++j) {
//		std::cout << "NEXT 32 VALUES: " << std::endl;
		//for (size_t i = 0; i < 32; ++i) {
		//			std::cout << x_qs[j].qs[i] << std::endl;
		//}
			
		std::cout << x_qs[j].d << std::endl;
		//std::cout << x_d[j] << std::endl;
	}

	__m256 total_sum = _mm256_setzero_ps();

	for (int32_t ib = 0; ib < nb - 4; ib += 4) {
		__m256 local_sum = _mm256_setzero_ps();
		__m256 d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(x_qs->d));

		__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs)));
		__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs + 16)));
		++x_qs;

		__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		__m256 y0	 = _mm256_load_ps(x_d);
		local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs + 16)));

		d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(x_qs->d));
		x_d += 32;
		++x_qs;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_load_ps(x_d);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs + 16)));

		d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(x_qs->d));
		x_d += 32;
		++x_qs;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_load_ps(x_d);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_qs->qs + 16)));

		d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(x_qs->d));
		x_d += 32;
		++x_qs;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_load_ps(x_d);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);
		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		x_d += 32;
	}

	__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);

	sumf = _mm_cvtss_f32(sum);

	*s = sumf;
}

void ggml_vec_dot_q8_0_f32(int n, float* __restrict s, const uint8_t* __restrict vx_u8, const uint16_t* vx_u16, const float* __restrict vy) {
	const int nb = n / 256;

	float sumf = 0;

	const uint8_t* __restrict x_u8	 = vx_u8;
	const uint16_t* __restrict x_u16 = vx_u16;
	const float* __restrict x_d		 = vy;
	std::cout << "FIRST 32 VALUES: " << std::endl;
	for (size_t j = 0; j < 32; ++j) {
		//std::cout << "NEXT 32 VALUES: " << std::endl;
		//for (size_t i = 0; i < 32; ++i) {
			//std::cout << x_u8[j * 32 + i] << std::endl;
		//}

		std::cout << x_u16[j] << std::endl;
		//std::cout << x_d[j] << std::endl;
	}
	__m256 total_sum = _mm256_setzero_ps();

	for (int32_t ib = 0; ib < nb - 4; ib += 4) {
		__m256 local_sum = _mm256_setzero_ps();
		__m256 d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(*x_u16));

		__m256i x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_u8)));
		__m256i x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )((x_u8 + 16))));
		x_u8 += 32;
		++x_u16;

		__m256 temp0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		__m256 y0	 = _mm256_load_ps(x_d);
		local_sum	 = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_u8)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )((x_u8 + 16))));

		d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(*x_u16));
		x_d += 32;
		x_u8 += 32;
		++x_u16;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_load_ps(x_d);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_u8)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )((x_u8 + 16))));

		d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(*x_u16));
		x_d += 32;
		x_u8 += 32;
		++x_u16;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_load_ps(x_d);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		local_sum = _mm256_setzero_ps();

		x0_0_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )(x_u8)));
		x0_1_0_0 = _mm256_cvtepi8_epi16(_mm_load_si128(( __m128i* )((x_u8 + 16))));

		d_broad00 = _mm256_set1_ps(ggml_compute_fp16_to_fp32(*x_u16));
		x_d += 32;
		x_u8 += 32;
		++x_u16;

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_0_0_0)));
		y0		  = _mm256_load_ps(x_d);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_0_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 8);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(x0_1_0_0)));
		y0		  = _mm256_load_ps(x_d + 16);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);

		temp0	  = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(x0_1_0_0, 1)));
		y0		  = _mm256_load_ps(x_d + 24);
		local_sum = _mm256_fmadd_ps(temp0, y0, local_sum);
		total_sum = _mm256_add_ps(_mm256_mul_ps(d_broad00, local_sum), total_sum);
		x_d += 32;
	}

	__m128 sum = _mm_add_ps(_mm256_castps256_ps128(total_sum), _mm256_extractf128_ps(total_sum, 1));

	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);

	sumf = _mm_cvtss_f32(sum);

	*s = sumf;
}

static constexpr uint64_t maxIterations{ 1};

template<bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	size_t currentIndex{};
	float dst{};
	float dst01{};
	{
		currentIndex = 0ull;
		std::array<mega, maxIterations> testDataMega{};
		bnch_swt::benchmark_stage<testName, maxIterations, 1>::template runBenchmark<"mega", "CYAN">([&] {
			uint64_t bytesProcessed{};
			ggml_vec_dot_q8_0_f32(dim, &dst, testDataMega[currentIndex].x_qs.data(), testDataMega[currentIndex].x_d.data(), testDataMega[currentIndex].z.data());
			bytesProcessed += dim;
			bnch_swt::doNotOptimizeAway(dst);
			++currentIndex;
			return bytesProcessed;
		});
		std::array<orig, maxIterations> testDataOrig{};
		for (size_t x_qs = 0; x_qs < maxIterations; ++x_qs) {
			testDataOrig[x_qs] = orig{ testDataMega[x_qs] };
		}
		currentIndex = 0ull;
		bnch_swt::benchmark_stage<testName, maxIterations, 1>::template runBenchmark<"mega-converted", "CYAN">([&] {
			uint64_t bytesProcessed{};
			ggml_vec_dot_q8_0_f32_converted(dim, &dst, testDataOrig[currentIndex].x.get(), testDataOrig[currentIndex].z.get());
			bytesProcessed += dim;
			bnch_swt::doNotOptimizeAway(dst);
			++currentIndex;
			return bytesProcessed;
		});
		if (dst01 == dst) {
			std::cout << "Yes we are equal, dst: " << dst << ", dst01: " << dst01 << std::endl;
		} else {
			std::cout << "We are not equal, dst: " << dst << ", dst01: " << dst01 << std::endl;
		}
	}

	bnch_swt::benchmark_stage<testName, maxIterations, 1>::printResults(true, true);
}

int32_t main() {
	std::cout << "VALUE: " << (2106 * 32 + 3 * 32) << std::endl;
	testFunction<"orig-vs-aligned-vs-mega">();
	return 0;
}