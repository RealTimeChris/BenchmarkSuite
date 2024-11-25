#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "fast_float.h"
#include "fast_float_new.hpp"

template<typename UC> fastfloat_really_inline constexpr bool is_integer01(UC c) noexcept {
	return static_cast<uint8_t>(c - '0') < 10;
}

template<typename UC> fastfloat_really_inline constexpr bool is_integer02(UC c) noexcept {
	return !(c > UC('9') || c < UC('0'));
}

template<jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void parseFunction() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<std::string> newUints{};
	std::vector<size_t> newerUints01{};
	std::vector<size_t> newerUints02{};
	for (size_t x = 0; x < 1024 * 128; ++x) {
		newUints.emplace_back(test_generator::generateRandomNumberString(100));
	}
	newerUints02.resize(1024 * 128);
	newerUints01.resize(1024 * 128);

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "is_integer01",
		"dodgerblue">([&]() mutable {
		for (size_t x = 0; x < 1024 * 128; ++x) {
			size_t newValue{};
			const auto* iter = newUints[x].data();
			const auto* end	 = iter + newUints[x].size();
			while ((iter != end) && is_integer01(*iter)) {
				++iter;
				newValue += *iter;
			}
			bnch_swt::doNotOptimizeAway(newValue);
		}
	});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName,
		"is_integer02", "dodgerblue">([&]() mutable {
		for (size_t x = 0; x < 1024 * 128; ++x) {
			size_t newValue{};
			const auto* iter = newUints[x].data();
			const auto* end	 = iter + newUints[x].size();
			while ((iter != end) && is_integer02(*iter)) {
				++iter;
				newValue += *iter;
			}
			bnch_swt::doNotOptimizeAway(newValue);
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

constexpr uint32_t handle_zero(uint32_t result) {
	uint32_t mask = (result != 0) - 1;
	return (result & mask) | (std::numeric_limits<uint32_t>::max() & ~mask);
}

int main() {
	static constexpr uint8_t newValue{};
	static constexpr const uint8_t* newPtr{ &newValue };
	static constexpr auto newerPtr = handle_zero(0);
	std::cout << "IS IT NULL: " << newerPtr << std::endl;
	static constexpr auto newerPtr01 = handle_zero(2);
	std::cout << "IS IT NULL: " << newerPtr01 << std::endl;
	parseFunction<"is_integer01-vs-is_integer02", "is_integer01-vs-is_integer02">();
}