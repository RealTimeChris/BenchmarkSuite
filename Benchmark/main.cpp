#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <string>

#define JSONIFIER_IS_INTEGER(x) ((static_cast<uint8_t>(x - '0')) < 10)

template<typename value_type> JSONIFIER_ALWAYS_INLINE bool isInteger(value_type val) {
	return ((static_cast<uint8_t>(val - '0')) < 10);
}

template<size_t length> BNCH_SWT_ALWAYS_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testStage{ "function-vs-macro" };
	std::vector<std::string> randomStrings{};
	for (size_t x = 0; x < 101 * 1001; ++x) {
		randomStrings.emplace_back(bnch_swt::test_generator::generateString(length));
	}
	uint64_t currentIndex{};
	bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<"macro", "cyan">([&] {
		for (size_t x = 0; x < 1000; ++x) {
			auto newValue = JSONIFIER_IS_INTEGER(*randomStrings[currentIndex].data());
			bnch_swt::doNotOptimizeAway(newValue);
			++currentIndex;
		}
		return length * 1000;
	});
	currentIndex = 0;
	bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<"function", "cyan">([&] {
		for (size_t x = 0; x < 1000; ++x) {
			auto newValue = isInteger(*randomStrings[currentIndex].data());
			bnch_swt::doNotOptimizeAway(newValue);
			++currentIndex;
		}
		return length * 1000;
	});
	bnch_swt::benchmark_stage<testStage>::printResults();
}

int main() {
	testFunction<4>();
	testFunction<5>();
	testFunction<6>();
	testFunction<7>();
	testFunction<8>();
	return 0;
}
