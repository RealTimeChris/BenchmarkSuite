#include <BnchSwt/BenchmarkSuite.hpp> 
#include <thread>

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

class simd_base {};

class simd_avx {};

class simd_avx2 {};

class simd_avx512 {};

class simd_neon {};

class parser : public simd_base {};

int main() {
	std::string newString01{ "default_profile" };

	//std::cout << "CURRENT SIMD-TYPE: " << typeid(jsonifier_simd_int_t).name() << std::endl;
	//std::cout << "RESULT: "//
	//<< processIndexImpl<std::string, std::string, std::string, jsonifier::parse_options{}, false>(std::string{}, std::string{}, 14, std::make_index_sequence<23>{})
	//<< std::endl;
	parseFunction<1, 1, "fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1",
		"fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1">();

	return 0;
}