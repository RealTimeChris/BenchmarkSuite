#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Simdjson.hpp"
#include "Tests/Jsonifier.hpp"
#include "Tests/Common.hpp"
#include "Tests/Uint.hpp"
#include "Tests/Float.hpp"
#include "Tests/RoundTrip.hpp"
#include "Tests/Int.hpp"
#include "Tests/String.hpp"
#include <glaze/glaze.hpp>

template<size_t length> struct test_struct_new {
	constexpr test_struct_new(const char* valuesNew) {
		std::copy(valuesNew, valuesNew + length, values);
	}
	char values[length]{};
};

template<size_t length, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLengthParse(const std::string& dataToParse) {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	static constexpr const char* newValues{ "TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING TESTING" };
	static constexpr test_struct_new<length> newerValues{ newValues };
	static constexpr jsonifier_internal::string_literal<length, char> testLiteral01{ newerValues.values };
	static constexpr jsonifier_internal::string_literal<length, char> testLiteral02{ newerValues.values };
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time, .totalIterationCountCap = 300 }>::template runBenchmark<testName,
		"Glaze-Function", "dodgerblue">([=]() mutable {
		for (size_t x = 0; x < 1024 * 16; ++x) {
			auto newString = glz::compare<testLiteral01.size()>(testLiteral01.data(), testLiteral02.data());
			bnch_swt::doNotOptimizeAway(newString);
		}
	});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time, .totalIterationCountCap = 300 }>::template runBenchmark<testName,
		"Jsonifier-Function", "dodgerblue">([=]() mutable {
		for (size_t x = 0; x < 1024 * 16; ++x) {
			auto newString = jsonifier_internal::string_literal_comparitor<testLiteral01>::impl(testLiteral02.data());
			bnch_swt::doNotOptimizeAway(newString);
		}
	});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int32_t main() {
	std::string newString{};
	runForLengthParse<1, "string_literal_comparison-length-1", "string_literal_comparison-length-1">(newString);
	runForLengthParse<2, "string_literal_comparison-length-2", "string_literal_comparison-length-2">(newString);
	runForLengthParse<4, "string_literal_comparison-length-4", "string_literal_comparison-length-4">(newString);
	runForLengthParse<8, "string_literal_comparison-length-8", "string_literal_comparison-length-8">(newString);
	runForLengthParse<16, "string_literal_comparison-length-16", "string_literal_comparison-length-16">(newString);
	runForLengthParse<32, "string_literal_comparison-length-32", "string_literal_comparison-length-32">(newString);
	runForLengthParse<64, "string_literal_comparison-length-64", "string_literal_comparison-length-64">(newString);
	runForLengthParse<128, "string_literal_comparison-length-128", "string_literal_comparison-length-128">(newString);
	return 0;
}