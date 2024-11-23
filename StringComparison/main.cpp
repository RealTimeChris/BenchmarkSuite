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

template<size_t length, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize02() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<std::string> newUints{};
	std::vector<uint64_t> newerUints01{};
	std::vector<uint64_t> newerUints02{};
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
			const auto* end	 = newUints[x].data() + newUints[x].size();
			fast_float::loop_parse_if_eight_digits(iter, end, value);
			while (end - iter > 0) {
				value = value * 10 + static_cast<uint8_t>(*iter - '0');
				++iter;
			}
			newerUints01[x] = value;
			bnch_swt::doNotOptimizeAway(value);
		}
	});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName,
		"fast_float_new::loop_parse_if_eight_digits", "dodgerblue">([&]() mutable {
		for (size_t x = 0; x < 1024 * 128; ++x) {
			uint64_t value{};
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
			std::cout << "Intended Value: " << newerUints01[x] << std::endl;
			std::cout << "Parsed Value: " << newerUints02[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

JSONIFIER_ALWAYS_INLINE uint32_t parse_eight_digits_unrolled(uint64_t val) noexcept {
	constexpr uint64_t mask{ 0x000000FF000000FF };
	constexpr uint64_t mul1{ 100 + (1000000ULL << 32) };
	constexpr uint64_t mul2{ 1 + (10000ULL << 32) };
	constexpr uint64_t subMask{ 0x3030303030303030 };
	val -= subMask;
	val = (val * 10) + (val >> 8);
	val = (((val & mask) * mul1) + (((val >> 16) & mask) * mul2)) >> 32;
	return uint32_t(val);
}

JSONIFIER_ALWAYS_INLINE uint32_t parse_four_digits_unrolled(uint64_t val) noexcept {
	constexpr uint64_t mask{ 0x000000FF000000FF };
	constexpr uint64_t mul1{ 100ULL << 32ULL };
	constexpr uint64_t mul2{ 1ULL << 32ULL };
	constexpr uint64_t subMask{ 0x0000000030303030ULL & 0x00000000FFFFFFFFULL };
	val -= subMask;
	val = (val * 10ULL) + (val >> 8ULL);
	val = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return static_cast<uint32_t>(val);
}

JSONIFIER_ALWAYS_INLINE uint32_t parse_two_digits_unrolled(uint64_t val) noexcept {
	constexpr uint64_t mask{ 0x000000FF000000FF };
	constexpr uint64_t mul1{ 1ULL << 32ULL };
	constexpr uint64_t subMask{ 0x0000000000003030ULL & 0x000000000000FFFFULL };
	val -= subMask;
	val = (val * 10ULL) + (val >> 8ULL);
	val = ((val & mask) * mul1) >> 32ULL;
	return static_cast<uint32_t>(val);
}

int main() {
	std::string newString{ "12345678" };
	std::cout << "Parse 8: " << parse_eight_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	std::cout << "Parse 7: " << parse_eight_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	std::cout << "Parse 6: " << parse_eight_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	std::cout << "Parse 5: " << parse_eight_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	std::cout << "Parse 4: " << parse_four_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	std::cout << "Parse 3: " << parse_eight_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	std::cout << "Parse 2: " << parse_two_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	std::cout << "Parse 1: " << parse_eight_digits_unrolled(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
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