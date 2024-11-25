#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "SimdDispatcher.hpp"

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

	bnch_swt::benchmark_stage<testStage, 5>::template runBenchmark<testName, "fast_float::loop_parse_if_eight_digits", "dodgerblue">([&]() mutable {
		size_t totalBytes{ 1024 * 1024 };
		std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
		return totalBytes;
	});

	bnch_swt::benchmark_stage<testStage, 5>::template runBenchmark<testName, "fast_float_new::loop_parse_if_digits", "dodgerblue">([&]() mutable {
		size_t totalBytes{ 1024 * 1024 };
		std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
		return totalBytes;
	});
	for (uint64_t x = 0; x < 1024 * 128; ++x) {
		if (newerUints02[x] != newerUints01[x]) {
			std::cout << "Failed to parse at index: " << x << std::endl;
			std::cout << "Input Value: " << newUints[x] << std::endl;
			std::cout << "Intended Value: " << newerUints01[x] << std::endl;
			std::cout << "Parsed Value: " << newerUints02[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, 5>::printResults();
}

#if defined(_MSC_VER) && defined(__AVX2__)
	#ifndef __POPCNT__
		#define __POPCNT__ 1
	#endif
	#ifndef __LZCNT__
		#define __LZCNT__ 1
	#endif
	#ifndef __BMI__
		#define __BMI__ 1
	#endif
#endif

int32_t main() {
#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_POPCNT)
	std::cout << "POPCNT is supported.\n";
#else
	std::cout << "POPCNT is NOT supported.\n";
#endif

#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_LZCNT)
	std::cout << "LZCNT is supported.\n";
#else
	std::cout << "LZCNT is NOT supported.\n";
#endif

#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_BMI)
	std::cout << "BMI1 (including TZCNT) is supported.\n";
#else
	std::cout << "BMI1 is NOT supported.\n";
#endif

#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_NEON)
	std::cout << "NEON DETECTED!" << std::endl;
#endif

#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_AVX)
	std::cout << "AVX DETECTED!" << std::endl;
#endif
#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_AVX2)
	std::cout << "AVX-2 DETECTED!" << std::endl;
#endif
#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_AVX512)
	std::cout << "AVX-512 DETECTED!" << std::endl;
#endif
	bnch_swt::event_collector counter{};
	counter.start();
	uint32_t valueNew{};
	auto results = counter.end();
	double cycles{};
	std::cout << "CURRENT RESULTS: " << results.cycles(cycles) << std::endl;
	parseFunction<1, 1, "fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1",
		"fast_float_new::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1">();
	/*
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
		*/

	return 0;
}