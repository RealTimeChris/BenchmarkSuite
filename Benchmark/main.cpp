#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>

BNCH_SWT_INLINE void mul_mat_scalar_f32(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
	const float* a_ptr = A;
	const float* b_ptr = B;
	float* c_ptr	   = C;
	for (size_t i = 0; i < M; i++) {
		for (size_t j = 0; j < N; j++) {
			float sum = 0;
			for (size_t k = 0; k < K; k++) {
				size_t a_idx = i * K + k;
				size_t b_idx = k * N + j;
				sum += a_ptr[a_idx] * b_ptr[b_idx];
			}
			size_t c_idx = i * N + j;
			c_ptr[c_idx] = sum;
		}
	}
}

BNCH_SWT_INLINE void mul_mat_linear_f32(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
	const float* a_ptr = A;
	float* c_ptr	   = C;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			const float* b_ptr = B + j;
			float sum		   = 0;
			for (int k = 0; k < K; k++) {
				sum += (*a_ptr++) * (*b_ptr);
				b_ptr += N;
			}
			*c_ptr++ = sum;
			a_ptr -= K;
		}
		a_ptr += K;
	}
}

int main() {
	static constexpr uint64_t test_iteration_count{ 20 };
	static constexpr size_t dim_a{ 512 };
	static constexpr size_t dim_b{ 512 };
	static constexpr size_t dim_c{ 512 };

	std::vector<float> input01{};
	input01.resize(dim_a * dim_b);
	for (size_t x = 0; x < dim_a * dim_b;++x) {
		input01[x] = bnch_swt::random_generator::generateValue<float>();
	}
	std::vector<float> input02{};
	input02.resize(dim_a * dim_b);
	for (size_t x = 0; x < dim_a * dim_b; ++x) {
		input02[x] = bnch_swt::random_generator::generateValue<float>();
	}
	std::vector<float> output{};
	output.resize(dim_a * dim_b);

	struct standard_indexing {
		BNCH_SWT_INLINE static size_t impl(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
			mul_mat_scalar_f32(A, B, C, M, K, N);
			bnch_swt::doNotOptimizeAway(C);
			return M * K * sizeof(float);
		};
	};

	struct linear_indexing {
		BNCH_SWT_INLINE static size_t impl(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
			mul_mat_linear_f32(A, B, C, M, K, N);
			bnch_swt::doNotOptimizeAway(C);
			return M * K * sizeof(float);
		};
	};

	bnch_swt::benchmark_stage<"test_stage", test_iteration_count, test_iteration_count / 10>::runBenchmark<"standard_indexing", standard_indexing>(input01.data(), input02.data(),
		output.data(), dim_a, dim_b, dim_c);
	bnch_swt::benchmark_stage<"test_stage", test_iteration_count, test_iteration_count / 10>::runBenchmark<"linear_indexing", linear_indexing>(input01.data(), input02.data(),
		output.data(), dim_a, dim_b, dim_c);

	bnch_swt::benchmark_stage<"test_stage", test_iteration_count, test_iteration_count / 10>::printResults();
	return 0;
}