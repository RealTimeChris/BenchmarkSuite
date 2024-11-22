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

std::string generateRandomNumberString(int length) {
	std::string randomNumber = "";
	std::srand(std::time(0));
	randomNumber += std::to_string(std::rand() % 9 + 1);
	for (int i = 1; i < length; ++i) {
		randomNumber += std::to_string(std::rand() % 10);
	}
	return randomNumber;
}

template<size_t length, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize02() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<std::string> newUints{};
	std::vector<uint64_t> newerUints01{};
	std::vector<uint64_t> newerUints02{};
	for (size_t x = 0; x < 1024 * 128; ++x) {
		newUints.emplace_back(generateRandomNumberString(length));
	}
	newerUints02.resize(1024 * 128);
	newerUints01.resize(1024 * 128);

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "fast_float::loop_parse_if_eight_digits",
		"dodgerblue">([&]() mutable {
		uint64_t value{};
		for (size_t x = 0; x < 1024 * 128; ++x) {
			const auto* iter = newUints[x].data();
			const auto* end	 = newUints[x].data() + newUints[x].size();
			fast_float::loop_parse_if_eight_digits(iter, end, value);
			newerUints01[x] = value;
			bnch_swt::doNotOptimizeAway(value);
		}
	});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName,
		"fast_float_new::loop_parse_if_eight_digits", "dodgerblue">([&]() mutable {
		uint64_t value{};
		for (size_t x = 0; x < 1024 * 128; ++x) {
			const auto* iter = newUints[x].data();
			const auto* end	 = newUints[x].data() + newUints[x].size();
			fast_float_new::loop_parse_if_eight_digits(iter, end, value);
			newerUints02[x] = value;
			bnch_swt::doNotOptimizeAway(value);
		}
	});
	for (size_t x = 0; x < 1024 * 128; ++x) {
		if (newerUints02[x] != newerUints01[x]) {
			std::cout << "Failed to parse at index: " << x << std::endl;
			std::cout << "Input Value: " << newUints[x] << std::endl;
			std::cout << "Parsed Value: " << newerUints02[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int main() {
	runForLengthSerialize02<1, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-1",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-1">();
	runForLengthSerialize02<2, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-2",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-2">();
	runForLengthSerialize02<3, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-3",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-3">();
	runForLengthSerialize02<4, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-4",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-4">();
	runForLengthSerialize02<5, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-5",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-5">();
	runForLengthSerialize02<6, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-6",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-6">();
	runForLengthSerialize02<7, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-7",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-7">();
	runForLengthSerialize02<8, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-8",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-8">();
	runForLengthSerialize02<9, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-9",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-9">();
	runForLengthSerialize02<10, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-10",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-10">();
	runForLengthSerialize02<11, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-11",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-11">();
	runForLengthSerialize02<12, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-12",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-12">();
	runForLengthSerialize02<13, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-13",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-13">();
	runForLengthSerialize02<14, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-14",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-14">();
	runForLengthSerialize02<15, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-15",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-15">();
	runForLengthSerialize02<16, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-16",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-16">();
	runForLengthSerialize02<17, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-17",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-17">();
	runForLengthSerialize02<18, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-18",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-18">();
	runForLengthSerialize02<19, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-19",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-19">();
	runForLengthSerialize02<20, "fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-20",
		"fast_float_new::loop_parse_if_eight_digits-vs-fast_float::loop_parse_if_eight_digits-20">();

	return 0;
}