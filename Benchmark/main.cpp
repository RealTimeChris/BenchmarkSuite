#include <BnchSwt/BenchmarkSuite.hpp>
#include <thread>
#include <iostream>
#include <vector>
#include <mutex>
#include <source_location>
#include <array>
#include <queue>
#include <latch>
#include <bit>

int main() {
	static constexpr uint64_t total_iterations{ 100 };
	static constexpr uint64_t measured_iterations{ 10 };

	struct test_struct_no_pause {
		BNCH_SWT_INLINE static uint64_t impl() {
			auto start = std::chrono::high_resolution_clock::now();
			auto end   = std::chrono::high_resolution_clock::now();
			return 200000ull;
		};
	};

	struct test_struct_pause {
		BNCH_SWT_INLINE static uint64_t impl() {
			auto start = std::chrono::high_resolution_clock::now();
			auto end   = std::chrono::high_resolution_clock::now();
			return 200000ull;
		};
	};

	bnch_swt::benchmark_stage<"test_stage", 2, 1>::runBenchmark<"no-yield", test_struct_no_pause>();
	bnch_swt::benchmark_stage<"test_stage", 2, 1>::runBenchmark<"yield", test_struct_pause>();

	bnch_swt::benchmark_stage<"test_stage", 2, 1>::printResults();
	return 0;
}
