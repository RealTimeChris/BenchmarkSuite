#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <thread>

int main() {
	bnch_swt::benchmark_stage<"TEST STAGE">::runBenchmark<"TEST", "CYAN">([] {
		return 0;
	});
	return 0;
}