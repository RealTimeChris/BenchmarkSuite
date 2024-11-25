#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "fast_float.h"
#include "fast_float_new.hpp"

template<uint64_t digitCount, uint64_t length, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void parseFunction() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<std::string> newUints{};
	std::vector<uint64_t> newerUints01{};
	std::vector<uint64_t> newerUints02{};
	for (uint64_t x = 0; x < 1024 * 128; ++x) {
		newUints.emplace_back(test_generator::generateRandomNumberString(digitCount, length));
	}
	newerUints02.resize(1024 * 128);
	newerUints01.resize(1024 * 128);

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "fast_float::loop_parse_if_eight_digits",
		"dodgerblue">([&]() mutable {
		size_t totalBytes{};
		for (uint64_t x = 0; x < 1024 * 128; ++x) {
			uint64_t value{};
			const auto* iter = newUints[x].data();
			const auto* end	 = iter + newUints[x].size();
			fast_float_orig::loop_parse_if_eight_digits(iter, end, value);
			while ((iter != end) && fast_float_orig::is_integer(*iter)) {
				uint8_t digit = uint8_t(*iter - char('0'));
				++iter;
				value = value * 10 + digit;// in rare cases, this will overflow, but that's ok
			}
			totalBytes += 8;
			newerUints01[x] = value;
			bnch_swt::doNotOptimizeAway(value);
		}
		return totalBytes;
	});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "fast_float_new::loop_parse_if_digits",
		"dodgerblue">([&]() mutable {
		for (uint64_t x = 0; x < 1024 * 128; ++x) {
			uint64_t value{};
			const auto* iter = newUints[x].data();
			const auto* end	 = iter + newUints[x].size();
			fast_float_new::loop_parse_if_digits(iter, end, value);
			newerUints02[x] = value;
			bnch_swt::doNotOptimizeAway(value);
		}
		return sizeof(uint64_t);
	});
	for (uint64_t x = 0; x < 1024 * 128; ++x) {
		if (newerUints02[x] != newerUints01[x]) {
			std::cout << "Failed to parse at index: " << x << std::endl;
			std::cout << "Input Value: " << newUints[x] << std::endl;
			std::cout << "Intended Value: " << newerUints01[x] << std::endl;
			std::cout << "Parsed Value: " << newerUints02[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int32_t main() {
	bnch_swt::event_collector counter{};
	counter.start();
	uint32_t valueNew{};
	auto results = counter.end();
	double cycles{};
	std::cout << "CURRENT RESULTS: " << results.cycles(cycles) << std::endl;
	parseFunction<1, 1, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1">();
	parseFunction<2, 2, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-2-and-digit-count-2",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-2-and-digit-count-2">();
	parseFunction<3, 3, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-3-and-digit-count-3",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-3-and-digit-count-3">();
	parseFunction<4, 4, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-4-and-digit-count-4",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-4-and-digit-count-4">();
	parseFunction<5, 5, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-5-and-digit-count-5",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-5-and-digit-count-5">();
	parseFunction<6, 6, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-6-and-digit-count-6",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-6-and-digit-count-6">();
	parseFunction<7, 7, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-7-and-digit-count-7",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-7-and-digit-count-7">();
	parseFunction<8, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-8",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-8">();
	parseFunction<9, 9, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-9-and-digit-count-9",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-9-and-digit-count-9">();
	parseFunction<10, 10, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-10-and-digit-count-10",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-10-and-digit-count-10">();
	parseFunction<11, 11, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-11-and-digit-count-11",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-11-and-digit-count-11">();
	parseFunction<12, 12, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-12-and-digit-count-12",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-12-and-digit-count-12">();
	parseFunction<13, 13, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-13-and-digit-count-13",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-13-and-digit-count-13">();
	parseFunction<14, 14, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-14-and-digit-count-14",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-14-and-digit-count-14">();
	parseFunction<15, 15, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-15-and-digit-count-15",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-15-and-digit-count-15">();
	parseFunction<16, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-16",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-16">();
	parseFunction<17, 17, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-17-and-digit-count-17",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-17-and-digit-count-17">();
	parseFunction<18, 18, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-18-and-digit-count-18",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-18-and-digit-count-18">();
	parseFunction<19, 19, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-19-and-digit-count-19",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-19-and-digit-count-19">();
	parseFunction<20, 20, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-20-and-digit-count-20",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-20-and-digit-count-20">();
	parseFunction<21, 21, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-21-and-digit-count-21",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-21-and-digit-count-21">();
	parseFunction<22, 22, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-22-and-digit-count-22",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-22-and-digit-count-22">();
	parseFunction<23, 23, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-23-and-digit-count-23",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-23-and-digit-count-23">();
	parseFunction<24, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-24",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-24">();


	parseFunction<3, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-1",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-1">();
	parseFunction<1, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-1",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-1">();
	parseFunction<2, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-2",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-2">();
	parseFunction<3, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-3",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-3">();
	parseFunction<4, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-4",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-4">();
	parseFunction<5, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-5",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-5">();
	parseFunction<6, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-6",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-6">();
	parseFunction<7, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-7",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-7">();
	parseFunction<8, 8, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-8",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-8-and-digit-count-8">();
	parseFunction<9, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-9",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-9">();
	parseFunction<10, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-10",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-10">();
	parseFunction<11, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-11",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-11">();
	parseFunction<12, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-12",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-12">();
	parseFunction<13, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-13",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-13">();
	parseFunction<14, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-14",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-14">();
	parseFunction<15, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-15",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-15">();
	parseFunction<16, 16, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-16",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-16-and-digit-count-16">();
	parseFunction<17, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-17",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-17">();
	parseFunction<18, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-18",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-18">();
	parseFunction<19, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-19",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-19">();
	parseFunction<20, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-20",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-20">();
	parseFunction<21, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-21",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-21">();
	parseFunction<22, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-22",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-22">();
	parseFunction<23, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-23",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-23">();
	parseFunction<24, 24, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-24",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-24-and-digit-count-24">();


	return 0;
}