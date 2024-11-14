#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"

struct index_processor_parse {
	template<size_t index, typename value_type> JSONIFIER_ALWAYS_INLINE static bool processIndex(value_type& value) noexcept {
		value += index;
		return true;
	}
};

template<size_t size, typename value_type> struct index_processor_parse_map {
	template<size_t... indices> static constexpr auto generateFunctionPtrsImpl(std::index_sequence<indices...>) {
		using function_type = decltype(&index_processor_parse::template processIndex<0, value_type>);
		return std::array<function_type, sizeof...(indices)>{ &index_processor_parse::template processIndex<indices, value_type>... };
	}

	static constexpr auto generateFunctionPtrs() {
		constexpr auto tupleSize = size;
		return generateFunctionPtrsImpl(std::make_index_sequence<tupleSize>{});
	}

	static constexpr auto bases{ generateFunctionPtrs() };

	constexpr index_processor_parse_map() {
	}

	JSONIFIER_ALWAYS_INLINE static constexpr bool processIndexReal(size_t indexNew, value_type& value) {
		return bases[indexNew](value);
	}
};

template<size_t size> struct parse_impl_internal {
  public:
	template<typename value_type> JSONIFIER_ALWAYS_INLINE static bool executeIndices(value_type& value) {
		return executeIndices(value, std::make_index_sequence<size>{});
	}

  protected:
	template<size_t... indices, typename value_type> JSONIFIER_ALWAYS_INLINE static bool executeIndices(value_type& value, std::index_sequence<indices...>) {
		return (impl<indices>(value), ...);
	}
	template<size_t index, typename value_type> JSONIFIER_ALWAYS_INLINE static bool impl(value_type& value) {
		value += index;
		return true;
	}
};

template<size_t index> struct switch_statement {};

template<> struct switch_statement<1> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 1;
	}
};

template<> struct switch_statement<2> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 2;
	}
};

template<> struct switch_statement<3> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 3;
	}
};

template<> struct switch_statement<4> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 4;
	}
};

template<> struct switch_statement<5> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 5;
	}
};

template<> struct switch_statement<6> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 6;
	}
};

template<> struct switch_statement<7> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 7;
	}
};

template<> struct switch_statement<8> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 8;
	}
};

template<> struct switch_statement<9> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 9;
	}
};

template<> struct switch_statement<10> {
	JSONIFIER_ALWAYS_INLINE static void impl(uint64_t& value) {
		value += 10;
	}
};

template<size_t maxIndex, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	int32_t value{};
	bool newValue{};
	uint64_t currentValue{};
	static constexpr auto newerLambda = [](const auto currentIndex, const auto maxIndexNew, uint64_t& value, auto&& newerLambda) {
		if constexpr (currentIndex < maxIndexNew) {
			value += currentIndex;
			return newerLambda(std::integral_constant<size_t, currentIndex + 1>{}, std::integral_constant<size_t, maxIndexNew>{}, value, newerLambda);
		}
	};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "switch-statement", "dodgerblue">(
		[&]() mutable {
			for (size_t y = 0; y < 1024 * 16; ++y) {
				for (size_t x = 0; x < maxIndex; ++x) {
					switch (x) {
						case 0: {
							break;
						}
						case 1: {
							currentValue += 1;
							break;
						}
						case 2: {
							currentValue += 2;
							break;
						}
						case 3: {
							currentValue += 3;
							break;
						}
						case 4: {
							currentValue += 4;
							break;
						}
						case 5: {
							currentValue += 5;
							break;
						}
						case 6: {
							currentValue += 6;
							break;
						}
						case 7: {
							currentValue += 7;
							break;
						}
						case 8: {
							currentValue += 8;
							break;
						}
						case 9: {
							currentValue += 9;
							break;
						}
						case 10: {
							currentValue += 10;
							break;
						}
						case 11: {
							currentValue += 11;
							break;
						}
						case 12: {
							currentValue += 12;
							break;
						}
						case 13: {
							currentValue += 13;
							break;
						}
						case 14: {
							currentValue += 14;
							break;
						}
						case 15: {
							currentValue += 15;
							break;
						}
						case 16: {
							currentValue += 16;
							break;
						}
						case 17: {
							currentValue += 17;
							break;
						}
						case 18: {
							currentValue += 18;
							break;
						}
						case 19: {
							currentValue += 19;
							break;
						}
						case 20: {
							currentValue += 20;
							break;
						}
						case 21: {
							currentValue += 21;
							break;
						}
					};
					bnch_swt::doNotOptimizeAway(currentValue);
				}
			}
		});
	std::cout << "CURRENT VALUE: " << currentValue << std::endl;
	currentValue = 0;

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "recursive-lambda", "dodgerblue">(
		[&]() mutable {
			for (size_t y = 0; y < 1024 * 16; ++y) {
				newerLambda(std::integral_constant<size_t, 0>{}, std::integral_constant<size_t, maxIndex>{}, currentValue, newerLambda);
				bnch_swt::doNotOptimizeAway(currentValue);
			}
		});
	std::cout << "CURRENT VALUE: " << currentValue << std::endl;
	currentValue = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "constexpr-array-of-function-ptrs",
		"dodgerblue">([&]() mutable {
		for (size_t y = 0; y < 1024 * 16; ++y) {
			for (size_t x = 0; x < maxIndex; ++x) {
				index_processor_parse_map<maxIndex, uint64_t>::processIndexReal(x, currentValue);
				bnch_swt::doNotOptimizeAway(currentValue);
			}
		}
	});
	std::cout << "CURRENT VALUE: " << currentValue << std::endl;
	currentValue = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "containing-class", "dodgerblue">(
		[&]() mutable {
			for (size_t y = 0; y < 1024 * 16; ++y) {
				parse_impl_internal<maxIndex>::executeIndices(currentValue);
				bnch_swt::doNotOptimizeAway(currentValue);
			}
		});
	std::cout << "CURRENT VALUE: " << currentValue << std::endl;
	currentValue							 = 0;
	static constexpr auto processIndexLambda = []<size_t index>(uint64_t& value) {
		value += index;
		return true;
	};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int main() {
	jsonifier::jsonifier_core parser{};
	parser.parseJson(canada_message{}, std::string{});
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