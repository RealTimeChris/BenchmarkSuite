#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>

struct index_processor_parse {
	template<size_t index, typename value_type>
	JSONIFIER_ALWAYS_INLINE static void processIndex(value_type& value) noexcept {
		value += index;
	}
};

template<size_t currentSize, typename value_type> struct index_processor_parse_map {
	template<size_t... indices> static constexpr auto generateFunctionPtrsImpl(std::index_sequence<indices...>) {
		using function_type = decltype(&index_processor_parse::template processIndex<0, value_type>);
		return std::array<function_type, sizeof...(indices)>{ &index_processor_parse::template processIndex<indices, value_type>... };
	}

	static constexpr auto generateFunctionPtrs() {
		return generateFunctionPtrsImpl(std::make_index_sequence<currentSize>{});
	}

	static constexpr auto bases{ generateFunctionPtrs() };

	constexpr index_processor_parse_map() {
	}

	JSONIFIER_ALWAYS_INLINE static constexpr void processIndexReal(size_t indexNew, value_type& value) {
		return bases[indexNew](value);
	}
};

template<size_t maxIndexCount, typename lambda> struct index_executor : lambda {
	template<typename... arg_types> JSONIFIER_ALWAYS_INLINE void executeIndices(arg_types&&... args) const {
		executeIndicesImpl(std::make_index_sequence<maxIndexCount>{}, std::forward<arg_types>(args)...);
	}

	template<typename... arg_types, size_t... indices> JSONIFIER_ALWAYS_INLINE void executeIndicesImpl(std::index_sequence<indices...>, arg_types&&... args) const {
		(this->operator()<indices>(std::forward<arg_types>(args)...), ...);
	}
};

template<size_t maxIndex, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	int32_t value{};
	bool newValue{};
	uint64_t currentValue{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "constexpr-array-of-function-ptrs", "dodgerblue">(
		[&]() mutable {
			for (size_t y = 0; y < 1024 * 16; ++y) {
				for (size_t x = 0; x < maxIndex; ++x) {
					index_processor_parse_map<maxIndex, uint64_t>::processIndexReal(x, currentValue);
					bnch_swt::doNotOptimizeAway(currentValue);
				}
			}
		});
	std::cout << "CURRENT VALUE: " << currentValue << std::endl;
	currentValue							 = 0;
	static constexpr auto processIndexLambda = []<size_t index>(uint64_t& value) {
		value += index;
	};
	static constexpr index_executor<maxIndex, decltype(processIndexLambda)> indexProcessor{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "derived-from-lambda",
		"dodgerblue">([&]() mutable {
		for (size_t y = 0; y < 1024 * 16; ++y) {
			indexProcessor.executeIndices(currentValue);
		}
	});
	std::cout << "CURRENT VALUE: " << currentValue << std::endl;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int main() {
	static constexpr auto newLambda = [] {
	};
	index_executor<2, decltype(newLambda)> newMap{};

	newMap.executeIndices();
	runForLengthSerialize<1, "Inheritance-vc-Constexpr-Array-1", "Inheritance-vc-Constexpr-Array-1">();
	runForLengthSerialize<2, "Inheritance-vc-Constexpr-Array-2", "Inheritance-vc-Constexpr-Array-2">();
	runForLengthSerialize<3, "Inheritance-vc-Constexpr-Array-3", "Inheritance-vc-Constexpr-Array-3">();
	runForLengthSerialize<4, "Inheritance-vc-Constexpr-Array-4", "Inheritance-vc-Constexpr-Array-4">();
	runForLengthSerialize<5, "Inheritance-vc-Constexpr-Array-5", "Inheritance-vc-Constexpr-Array-5">();
	runForLengthSerialize<6, "Inheritance-vc-Constexpr-Array-6", "Inheritance-vc-Constexpr-Array-6">();
	runForLengthSerialize<7, "Inheritance-vc-Constexpr-Array-7", "Inheritance-vc-Constexpr-Array-7">();
	runForLengthSerialize<8, "Inheritance-vc-Constexpr-Array-8", "Inheritance-vc-Constexpr-Array-8">();
	runForLengthSerialize<9, "Inheritance-vc-Constexpr-Array-9", "Inheritance-vc-Constexpr-Array-9">();
	runForLengthSerialize<10, "Inheritance-vc-Constexpr-Array-10", "Inheritance-vc-Constexpr-Array-10">();
	runForLengthSerialize<11, "Inheritance-vc-Constexpr-Array-11", "Inheritance-vc-Constexpr-Array-11">();
	runForLengthSerialize<12, "Inheritance-vc-Constexpr-Array-12", "Inheritance-vc-Constexpr-Array-12">();
	runForLengthSerialize<13, "Inheritance-vc-Constexpr-Array-13", "Inheritance-vc-Constexpr-Array-13">();
	runForLengthSerialize<14, "Inheritance-vc-Constexpr-Array-14", "Inheritance-vc-Constexpr-Array-14">();
	runForLengthSerialize<15, "Inheritance-vc-Constexpr-Array-15", "Inheritance-vc-Constexpr-Array-15">();
	runForLengthSerialize<16, "Inheritance-vc-Constexpr-Array-16", "Inheritance-vc-Constexpr-Array-16">();
	runForLengthSerialize<17, "Inheritance-vc-Constexpr-Array-17", "Inheritance-vc-Constexpr-Array-17">();
	runForLengthSerialize<18, "Inheritance-vc-Constexpr-Array-18", "Inheritance-vc-Constexpr-Array-18">();
	runForLengthSerialize<19, "Inheritance-vc-Constexpr-Array-19", "Inheritance-vc-Constexpr-Array-19">();
	runForLengthSerialize<20, "Inheritance-vc-Constexpr-Array-20", "Inheritance-vc-Constexpr-Array-20">();
	runForLengthSerialize<21, "Inheritance-vc-Constexpr-Array-21", "Inheritance-vc-Constexpr-Array-21">();
	runForLengthSerialize<22, "Inheritance-vc-Constexpr-Array-22", "Inheritance-vc-Constexpr-Array-22">();
	runForLengthSerialize<23, "Inheritance-vc-Constexpr-Array-23", "Inheritance-vc-Constexpr-Array-23">();
	runForLengthSerialize<24, "Inheritance-vc-Constexpr-Array-24", "Inheritance-vc-Constexpr-Array-24">();
	runForLengthSerialize<25, "Inheritance-vc-Constexpr-Array-25", "Inheritance-vc-Constexpr-Array-25">();
	runForLengthSerialize<26, "Inheritance-vc-Constexpr-Array-26", "Inheritance-vc-Constexpr-Array-26">();
	runForLengthSerialize<27, "Inheritance-vc-Constexpr-Array-27", "Inheritance-vc-Constexpr-Array-27">();
	runForLengthSerialize<28, "Inheritance-vc-Constexpr-Array-28", "Inheritance-vc-Constexpr-Array-28">();
	runForLengthSerialize<29, "Inheritance-vc-Constexpr-Array-29", "Inheritance-vc-Constexpr-Array-29">();
	runForLengthSerialize<30, "Inheritance-vc-Constexpr-Array-30", "Inheritance-vc-Constexpr-Array-30">();
	runForLengthSerialize<31, "Inheritance-vc-Constexpr-Array-31", "Inheritance-vc-Constexpr-Array-31">();
	runForLengthSerialize<32, "Inheritance-vc-Constexpr-Array-32", "Inheritance-vc-Constexpr-Array-32">();

	return 0;
}