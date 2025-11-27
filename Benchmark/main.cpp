#include <BnchSwt/Index.hpp>

static constexpr uint64_t total_iterations{ 100 };
static constexpr uint64_t measured_iterations{ 10 };

int main() {

	struct test_struct_no_pause {
		BNCH_SWT_HOST static uint64_t impl() {
			[[maybe_unused]] auto start = std::chrono::high_resolution_clock::now();
			[[maybe_unused]] auto end = std::chrono::high_resolution_clock::now();
			return 200000ull;
		}
	};

	struct test_struct_pause {
		BNCH_SWT_HOST static uint64_t impl() {
			[[maybe_unused]] auto start = std::chrono::high_resolution_clock::now();
			[[maybe_unused]] auto end = std::chrono::high_resolution_clock::now();
			return 200000ull;
		}
	};

	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations>::runBenchmark<"no-yield", test_struct_no_pause>();
	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations>::runBenchmark<"yield", test_struct_pause>();

	bnch_swt::benchmark_stage<"test_stage", total_iterations, measured_iterations>::printResults();
	return 0;
}
