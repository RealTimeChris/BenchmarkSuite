#include <BnchSwt/BenchmarkSuite.hpp> 
#include <jsonifier/Index.hpp>
#include <thread>
//#include "Tests/Common.hpp"

template<jsonifier::parse_options options, bool minified> struct index_processor_parse {
	template<size_t index, typename buffer_type, typename value_type, typename parse_context_type>
	JSONIFIER_ALWAYS_INLINE static bool processIndex(value_type&& value, parse_context_type&& context) noexcept {
		std::cout << "CURRENT INDEX IS: " << index << std::endl;
		return true;
	}
};

template<typename buffer_type, typename value_type, typename parse_context_type, jsonifier::parse_options options, bool minified, size_t... indices>
JSONIFIER_ALWAYS_INLINE static bool processIndexImpl(value_type&& value, parse_context_type&& context, size_t index, std::index_sequence<indices...>) noexcept {
	return ((index == indices ? index_processor_parse<options, minified>::template processIndex<indices, buffer_type>(value, context) : false) || ...);
}

template<uint64_t digitCount, uint64_t length, bnch_swt::string_literal testStageNew, bnch_swt::string_literal testNameNew>
BNCH_SWT_INLINE  void parseFunction() {
	static constexpr bnch_swt::string_literal testStage{ testStageNew.values };
	static constexpr bnch_swt::string_literal testName{ testNameNew.values };
	std::string newString01{ "123457678002454" };

	auto results = bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<testName, "copy-construct", "dodgerblue">([&]() mutable {
		std::this_thread::sleep_for(std::chrono::milliseconds{ 10 });
		return 1024ull * 1024ull;
	});

	results = bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<testName, "move-construct", "dodgerblue">([=]() mutable {
		std::this_thread::sleep_for(std::chrono::milliseconds{ 20 });
		return 1024ull * 1024ull;
	});
	std::cout << "ITERATION COUNT: " << results.totalIterationCount.value() << std::endl;
	bnch_swt::benchmark_stage<testStage, 100>::printResults();
}

int main() {
	std::string newString01{ "default_profile" };
	static constexpr jsonifier_internal::string_literal sl1{ "\image\":false,\"f" };
	std::cout << "ARE WE EQUAL?-1: " << jsonifier_internal::string_literal_comparitor<decltype(sl1), sl1>::impl(newString01.data()) << std::endl;

	//std::cout << "CURRENT SIMD-TYPE: " << typeid(jsonifier_simd_int_t).name() << std::endl;
	//std::cout << "RESULT: "//
	//<< processIndexImpl<std::string, std::string, std::string, jsonifier::parse_options{}, false>(std::string{}, std::string{}, 14, std::make_index_sequence<23>{})
	//<< std::endl;
	parseFunction<1, 1, "fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1",
		"fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1">();

	return 0;
}