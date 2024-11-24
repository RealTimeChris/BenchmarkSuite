#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "fast_float.h"
#include "fast_float_new.hpp"

template<size_t length, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void parseFunction() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<std::string> newUints{};
	std::vector<size_t> newerUints01{};
	std::vector<size_t> newerUints02{};
	for (size_t x = 0; x < 1024 * 128; ++x) {
		newUints.emplace_back(test_generator::generateRandomNumberString(length));
	}
	newerUints02.resize(1024 * 128);
	newerUints01.resize(1024 * 128);

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "fast_float::loop_parse_if_eight_digits",
		"dodgerblue">([&]() mutable {
		for (size_t x = 0; x < 1024 * 128; ++x) {
			uint64_t value{};
			const auto* iter = newUints[x].data();
			const auto* end	 = iter + newUints[x].size();
			while ((iter != end) && fast_float_orig::is_integer(*iter)) {
				uint8_t digit = uint8_t(*iter - char('0'));
				++iter;
				value = value * 10 + digit;// in rare cases, this will overflow, but that's ok
			}
			newerUints01[x] = value;
			bnch_swt::doNotOptimizeAway(value);
		}
	});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName,
		"fast_float_new::loop_parse_if_digits", "dodgerblue">([&]() mutable {
		for (size_t x = 0; x < 1024 * 128; ++x) {
			size_t value{};
			const auto* iter = newUints[x].data();
			const auto* end	 = iter + newUints[x].size();
			if (iter != end) {
				uint8_t digit;
				while (iter != end && (digit = fast_float_new::char_to_digit_table[uint8_t(*iter++)], digit != 255)) {
					value = value * 10 + uint8_t(digit);
				}
			}
			newerUints02[x] = value;
			bnch_swt::doNotOptimizeAway(value);
		}
	});
	for (size_t x = 0; x < 1024 * 128; ++x) {
		if (newerUints02[x] != newerUints01[x]) {
			std::cout << "Failed to parse at index: " << x << std::endl;
			std::cout << "Input Value: " << newUints[x] << std::endl;
			std::cout << "Intended Value: " << newerUints01[x] << std::endl;
			std::cout << "Parsed Value: " << newerUints02[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int main() {
	parseFunction<1, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-1",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-1">();
	parseFunction<2, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-2",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-2">();
	parseFunction<3, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-3",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-3">();
	parseFunction<4, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-4",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-4">();
	parseFunction<5, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-5",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-5">();
	parseFunction<6, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-6",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-6">();
	parseFunction<7, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-7",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-7">();
	parseFunction<8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-8",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-8">();
	parseFunction<9, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-9",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-9">();
	parseFunction<10, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-10",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-10">();
	parseFunction<11, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-11",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-11">();
	parseFunction<12, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-12",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-12">();
	parseFunction<13, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-13",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-13">();
	parseFunction<14, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-14",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-14">();
	parseFunction<15, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-15",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-15">();
	parseFunction<16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-16",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-16">();
	parseFunction<17, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-17",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-17">();
	parseFunction<18, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-18",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-18">();
	parseFunction<19, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-19",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-19">();
	parseFunction<20, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-20",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-20">();
	return 0;
}