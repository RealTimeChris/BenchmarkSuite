#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <array>
#include <queue>
#include <latch>
#include <bit>

void mul_mat_scalar_f32(const float* A, const float* B, float* C, size_t M, size_t K, size_t N) {
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			float sum = 0.0f;
			for (size_t k = 0; k < K; ++k) {
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

int main() {

	static constexpr size_t time_to_wait{ 10 };

	struct test_struct_no_pause {
		BNCH_SWT_INLINE static uint64_t impl() {
			auto start = std::chrono::high_resolution_clock::now();
			auto end   = std::chrono::high_resolution_clock::now();
			while ((end-start).count()<time_to_wait) {
				end = std::chrono::high_resolution_clock::now();
			}
			return 200000ull;
		};
	};

	struct test_struct_pause {
		BNCH_SWT_INLINE static uint64_t impl() {
			auto start = std::chrono::high_resolution_clock::now();
			auto end   = std::chrono::high_resolution_clock::now();
			while ((end - start).count() < time_to_wait) {
				std::this_thread::yield();
				end = std::chrono::high_resolution_clock::now();
			}
			return 200000ull;
		};
	};

	bnch_swt::benchmark_stage<"test_stage", 2, 1>::runBenchmark<"no-yield", test_struct_no_pause>();
	bnch_swt::benchmark_stage<"test_stage", 2, 1>::runBenchmark<"yield", test_struct_pause>();

	bnch_swt::benchmark_stage<"test_stage", 2, 1>::printResults();
	return 0;
}
