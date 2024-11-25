#include <iostream>
#include <array>
#include <simdjson.h>
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "simdjson.h"
#include "Tests/Common.hpp"
#include "Tests/Jsonifier.hpp"

template<uint64_t digitCount, uint64_t length, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void parseFunction() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	static constexpr jsonifier_internal::string_literal stringNew{ "123457678002435" };
	std::string newString{ "123457678002454" };
	const char* str = newString.data();

	auto results = bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<testName, "fast_float::loop_parse_if_eight_digits", "dodgerblue">([&]() mutable {
		static constexpr auto newLiteral{ stringNew };
		static constexpr auto newLength{ stringNew.size() };
		static constexpr auto valuesNew{ jsonifier_internal::packValues<newLiteral>() };
		for (size_t x = 0; x < 1024; ++x) {
			uint8_t intermediateVals[16];
			std::memcpy(intermediateVals, str, newLength);
			jsonifier_simd_int_128 data1{ simd_internal::gatherValues<jsonifier_simd_int_128>(intermediateVals) };
			std::memcpy(intermediateVals, valuesNew.data(), newLength);
			jsonifier_simd_int_128 data2{ simd_internal::gatherValues<jsonifier_simd_int_128>(intermediateVals) };
			bnch_swt::doNotOptimizeAway(data2);
		}
		return newLength;
	});

	results = bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<testName, "fast_float::loop_parse_if_digits", "dodgerblue">([&]() mutable {
		static constexpr auto newLiteral{ stringNew };
		static constexpr auto newLength{ stringNew.size() };
		static constexpr auto valuesNew{ jsonifier_internal::packValues<newLiteral>() };
		static constexpr auto maskValues = [] {
			std::array<uint8_t, 16> returnValues{};
			for (size_t x = 0; x < newLength; ++x) {
				returnValues[x] = 0xFF;
			}
			return returnValues;
		}();
		for (size_t x = 0; x < 1024; ++x) {
			jsonifier_simd_int_128 data1{ simd_internal::opAnd(simd_internal::gatherValues<jsonifier_simd_int_128>(maskValues.data()),
				simd_internal::gatherValuesU<jsonifier_simd_int_128>(str)) };
			jsonifier_simd_int_128 data2{ simd_internal::gatherValues<jsonifier_simd_int_128>(valuesNew.data()) };
			bnch_swt::doNotOptimizeAway(data2);
			bnch_swt::doNotOptimizeAway(data1);
		}
		return newLength;
	});
	std::cout << "ITERATION COUNT: " << results.totalIterationCount.value() << std::endl;
	bnch_swt::benchmark_stage<testStage, 100>::printResults();
}

int main() {

	parseFunction<1, 1, "fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1",
		"fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1">();

	return 0;
}