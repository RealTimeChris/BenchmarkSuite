#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>
#include <vector>
#include <iostream>
#include <thread>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

static constexpr uint64_t QK8_0{32};

uint64_t generateRandomIntegerByLength(uint32_t digitLength) {
	if (digitLength == 0) {
		throw std::invalid_argument("Digit length must be greater than 0.");
	}

	if (digitLength > 20) {
		throw std::invalid_argument("Digit length exceeds the limit for uint64_t (maximum 20 digits).");
	}

	uint64_t minValue = static_cast<uint64_t>(std::pow(10, digitLength - 1));
	uint64_t maxValue = static_cast<uint64_t>(std::pow(10, digitLength) - 1);

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint64_t> dist(minValue, maxValue);

	return dist(gen);
}

template<typename value_type> std::vector<value_type> generateRandomIntegersImpl(size_t count, size_t maxLength) {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint16_t> lengthGen(1, maxLength);
	std::vector<value_type> randomNumbers;

	for (size_t i = 0; i < count; ++i) {
		uint64_t newValue{ generateRandomIntegerByLength(lengthGen(gen)) };
		if (newValue >= std::numeric_limits<value_type>::max()) {
			newValue /= 10;
		}
		randomNumbers.push_back(newValue);
	}

	return randomNumbers;
}

// Generate Q8_0 test data
template<size_t count> std::array<std::vector<int8_t>, count> generateRandomQ8Data(size_t blockCount) {
	std::array<std::vector<int8_t>, count> return_values{};
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<int16_t> dist(-128, 127);

	for (size_t x = 0; x < count; ++x) {
		return_values[x].resize(blockCount * QK8_0);
		for (size_t i = 0; i < blockCount * QK8_0; ++i) {
			return_values[x][i] = dist(gen);
		}
	}
	return return_values;
}

// Generate scale factors (fp16)
template<size_t count> std::array<std::vector<uint16_t>, count> generateRandomScales(size_t blockCount) {
	std::array<std::vector<uint16_t>, count> return_values{};
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint16_t> dist(0x3000, 0x4000);// Small positive fp16 values

	for (size_t x = 0; x < count; ++x) {
		return_values[x].resize(blockCount);
		for (size_t i = 0; i < blockCount; ++i) {
			return_values[x][i] = dist(gen);
		}
	}
	return return_values;
}

BNCH_SWT_INLINE static constexpr float fp32_from_bits(uint32_t w) {
	return std::bit_cast<float>(w);
}

BNCH_SWT_INLINE static constexpr uint32_t fp32_to_bits(float f) {
	return std::bit_cast<uint32_t>(f);
}

BNCH_SWT_INLINE static constexpr float compute_fp16_to_fp32(uint16_t h) {
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

alignas(64) inline static const std::array<float, (1 << 16)> fp16_to_fp32_table{ [] {
	std::array<float, (1 << 16)> returnValues{};
	for (uint32_t x = 0; x < (1 << 16); ++x) {
		returnValues[x] = compute_fp16_to_fp32(static_cast<uint16_t>(x));
	}
	return returnValues;
}() };

BNCH_SWT_INLINE static float fp16_to_fp32(uint16_t h) {
	return *(fp16_to_fp32_table.data() + h);
}

struct block_q8_0 {
	uint16_t d;
	int8_t qs[QK8_0];
	BNCH_SWT_INLINE bool operator==(const block_q8_0& other) const {
		return d == other.d && std::memcmp(qs, other.qs, sizeof(qs)) == 0;
	}
};

BNCH_SWT_INLINE void ggml_dequantize_q8_0_simd(const void* vx, float* y, int k) {
	const block_q8_0* x = ( const block_q8_0* )vx;

	for (int i = 0; i < k / QK8_0; ++i) {
		const float scale	   = fp16_to_fp32(x[i].d);
		const __m256 scale_vec = _mm256_set1_ps(scale);

		// Prefetch next block
		if (i + 1 < k / QK8_0) {
			_mm_prefetch(( const char* )&x[i + 1], _MM_HINT_T0);
		}

		for (int j = 0; j < QK8_0; j += 8) {
			__m128i q8_int8	 = _mm_loadl_epi64(( __m128i* )&x[i].qs[j]);
			__m256i q8_int32 = _mm256_cvtepi8_epi32(q8_int8);
			__m256 q8_floats = _mm256_cvtepi32_ps(q8_int32);

			__m256 result = _mm256_mul_ps(scale_vec, q8_floats);
			_mm256_store_ps(&y[i * QK8_0 + j], result);
		}
	}
}

static constexpr auto alignas(32) int8_to_float_simple = []() {
	alignas(32) std::array<float, 256> table{};
	for (int i = 0; i < 256; ++i) {
		table[i] = static_cast<float>(static_cast<int8_t>(i));
	}
	return table;
}();


BNCH_SWT_INLINE void dequantize_q8_0_hybrid(const void* vx, float* y, int k) {
    const block_q8_0* x = (const block_q8_0*)vx;

    for (int i = 0; i < k / QK8_0; ++i) {
		__m256 scale_vec = _mm256_set1_ps(fp16_to_fp32(x[i].d));

        for (int j = 0; j < QK8_0; j += 8) {
			__m128i q8_indices_128 = _mm_loadl_epi64(( __m128i* )&x[i].qs[j]);

            __m256i q8_indices = _mm256_cvtepu8_epi32(q8_indices_128);

            __m256 q8_floats = _mm256_i32gather_ps(int8_to_float_simple.data(), q8_indices, 4);
            
            __m256 result = _mm256_mul_ps(scale_vec, q8_floats);
			_mm256_store_ps(&y[i * QK8_0 + j], result);
        }
    }
}

BNCH_SWT_INLINE void matrix_multiply_f32_avx2(const float* a, const float* b, float* c, int m, int n, int k) {
	// Matrix multiplication: C = A * B (A is m x k, B is k x n, C is m x n)
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; j += 8) {// Process 8 columns at once with AVX2
			__m256 sum = _mm256_setzero_ps();

			for (int l = 0; l < k; ++l) {
				__m256 a_val  = _mm256_set1_ps(a[i * k + l]);
				__m256 b_vals = _mm256_load_ps(&b[l * n + j]);
				sum			  = _mm256_fmadd_ps(a_val, b_vals, sum);
			}

			_mm256_store_ps(&c[i * n + j], sum);
		}
	}
}

// Combined test: dequantize weights, then multiply with activations
BNCH_SWT_INLINE void ggml_dequant_and_multiply(const block_q8_0* weights, const float* activations, float* output, int weight_count, int activation_size) {
	// Dequantize weights
	alignas(32) std::vector<float> dequant_weights(weight_count);
	ggml_dequantize_q8_0_simd(weights, dequant_weights.data(), weight_count);

	// Multiply: output = activations * dequant_weights
	// Treating it as vector dot product for simplicity
	for (int i = 0; i < std::min(weight_count, activation_size); i += 8) {
		__m256 act_vals	   = _mm256_load_ps(&activations[i]);
		__m256 weight_vals = _mm256_load_ps(&dequant_weights[i]);
		__m256 result	   = _mm256_mul_ps(act_vals, weight_vals);
		_mm256_store_ps(&output[i], result);
	}
}

BNCH_SWT_INLINE void nihilus_dequant_and_multiply(const block_q8_0* weights, const float* activations, float* output, int weight_count, int activation_size) {
	// Dequantize weights using table lookup
	alignas(32) std::vector<float> dequant_weights(weight_count);
	dequantize_q8_0_hybrid(weights, dequant_weights.data(), weight_count);

	// Multiply: output = activations * dequant_weights
	for (int i = 0; i < std::min(weight_count, activation_size); i += 8) {
		__m256 act_vals	   = _mm256_load_ps(&activations[i]);
		__m256 weight_vals = _mm256_load_ps(&dequant_weights[i]);
		__m256 result	   = _mm256_mul_ps(act_vals, weight_vals);
		_mm256_store_ps(&output[i], result);
	}
}

template<size_t count, size_t iterationCount, bnch_swt::string_literal name> BNCH_SWT_INLINE void testFunction64() {
	constexpr size_t blockCount = count / QK8_0;

	// Generate test data (using your existing functions)
	auto randomQ8Data = generateRandomQ8Data<iterationCount>(blockCount);
	auto randomScales = generateRandomScales<iterationCount>(blockCount);

	// Generate activation data (simulating model activations)
	auto randomActivations = generateRandomIntegersImpl<float>(count, 4);// Generate float "activations"
	alignas(32) std::vector<float> activations(count);
	for (size_t i = 0; i < count; ++i) {
		activations[i] = static_cast<float>(randomActivations[i]) / 1000.0f;// Normalize
	}

	// Create block arrays for each iteration
	std::array<std::vector<block_q8_0, jsonifier::internal::alloc_wrapper<block_q8_0>>, iterationCount> blockArrays;
	std::array<std::vector<float, jsonifier::internal::alloc_wrapper<float>>, iterationCount> ggmlResults;
	std::array<std::vector<float, jsonifier::internal::alloc_wrapper<float>>, iterationCount> nihilusResults;
	std::array<std::vector<float, jsonifier::internal::alloc_wrapper<float>>, iterationCount> ggmlMulResults;
	std::array<std::vector<float, jsonifier::internal::alloc_wrapper<float>>, iterationCount> nihilusMulResults;

	// Initialize data structures
	uint64_t byte_count{};
	for (size_t iter = 0; iter < iterationCount; ++iter) {
		blockArrays[iter].resize(blockCount);
		ggmlResults[iter].resize(count);
		nihilusResults[iter].resize(count);
		ggmlMulResults[iter].resize(count);
		nihilusMulResults[iter].resize(count);

		// Fill blocks with test data
		for (size_t i = 0; i < blockCount; ++i) {
			blockArrays[iter][i].d = randomScales[iter][i];
			std::memcpy(blockArrays[iter][i].qs, &randomQ8Data[iter][i * QK8_0], QK8_0);
			byte_count += QK8_0;
		}
	}

	// Benchmark dequantization + multiplication pipeline
	bnch_swt::benchmark_stage<"dequant-and-multiply-" + name, iterationCount, iterationCount / 5>::template runBenchmark<"ggml-full-pipeline">([&]() {
		for (size_t iter = 0; iter < iterationCount; ++iter) {
			ggml_dequant_and_multiply(blockArrays[iter].data(), activations.data(), ggmlMulResults[iter].data(), count, count);
		}
		bnch_swt::doNotOptimizeAway(ggmlMulResults[0][0]);
		return byte_count;
	});

	bnch_swt::benchmark_stage<"dequant-and-multiply-" + name, iterationCount, iterationCount / 5>::template runBenchmark<"nihilus-full-pipeline">([&]() {
		for (size_t iter = 0; iter < iterationCount; ++iter) {
			nihilus_dequant_and_multiply(blockArrays[iter].data(), activations.data(), nihilusMulResults[iter].data(), count, count);
		}
		bnch_swt::doNotOptimizeAway(nihilusMulResults[0][0]);
		return byte_count;
	});

	// Compare results for correctness
	bool allCorrect = true;
	for (size_t iter = 0; iter < iterationCount; ++iter) {
		for (size_t i = 0; i < count; ++i) {
			if (std::abs(ggmlResults[iter][i] - nihilusResults[iter][i]) > 1e-6f) {
				std::cout << "Mismatch at iteration " << iter << ", index " << i << ": GGML=" << ggmlResults[iter][i] << " vs Nihilus=" << nihilusResults[iter][i] << std::endl;
				allCorrect = false;
				break;
			}
		}
		if (!allCorrect)
			break;
	}

	if (allCorrect) {
		std::cout << "✅ All results match!" << std::endl;
	} else {
		std::cout << "❌ Results differ!" << std::endl;
	}

	bnch_swt::benchmark_stage<"dequant-and-multiply-" + name, iterationCount, iterationCount / 5>::printResults(true, true);
}

int main() {
	testFunction64<1000000, 100, "q8_0-dequant-test-1M">();
	return 0;
}