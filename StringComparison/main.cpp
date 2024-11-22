#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "StrToDOld.hpp"
#include "StrToDNew.hpp"
#include "fast_float.h"

template<jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize02() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<double> newerDoubles00{};
	for (size_t x = 0; x < 1024 * 64; ++x) {
		newerDoubles00.emplace_back(test_generator::generateValue<double>());
	}

	std::vector<std::string> newDoubles{};
	for (auto value: newerDoubles00) {
		newDoubles.emplace_back(std::to_string(value));
	}

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "is_made_up_of_eight_digits_fast",
		"dodgerblue">([&]() mutable {
		for (size_t x = 0; x < 1024 * 64; ++x) {
			const auto* iter = newDoubles[x].data();
			uint64_t value{};
			std::memcpy(&value, iter, 8);
			bnch_swt::doNotOptimizeAway(fast_float::is_made_of_eight_digits_fast(value));
		}
	});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "isValidToParse64", "dodgerblue">(
		[&]() mutable {
			for (size_t x = 0; x < 1024 * 64; ++x) {
				const auto* iter = newDoubles[x].data();
				uint64_t value{};
				std::memcpy(&value, iter, 8);
				bnch_swt::doNotOptimizeAway(fast_float_new::isValidToParse64(value));
			}
		});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int main() {
	runForLengthSerialize02<"is_made_of_eight_digits_fast-vs-isValidForParse64", "is_made_of_eight_digits_fast-vs-isValidForParse64">();
	return 0;
}