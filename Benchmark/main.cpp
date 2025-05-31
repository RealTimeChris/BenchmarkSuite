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
	std::vector<float> A{ 1, 2, 3, 4, 5, 6 };
	std::vector<float> B{ 7, 8, 9, 10, 11, 12 };
	std::vector<float> C{};

	mul_mat_scalar_f32(A.data(), B.data(), C.data(), 2, 3, 2);

	std::cout << A << std::endl;

	bnch_swt::benchmark_stage<"test_stage">::runBenchmark<"test01", "Cyan">([] {
		std::this_thread::sleep_for(std::chrono::milliseconds{ 1000 });
		return 20ull;
	});
	bnch_swt::benchmark_stage<"test_stage", 5, 1, false, "TEST">::printResults();
	return 0;
}
