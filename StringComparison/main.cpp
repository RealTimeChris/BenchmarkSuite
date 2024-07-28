#if defined(JSONIFIER_CPU_INSTRUCTIONS)
//#undef JSONIFIER_CPU_INSTRUCTIONS
//#define JSONIFIER_CPU_INSTRUCTIONS (JSONIFIER_AVX2 | JSONIFIER_POPCNT)
#endif
//#include "UnicodeEmoji.hpp"
#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "glaze/glaze.hpp"
#include <unordered_set>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

int main() {
	try {
		std::vector<std::vector<jsonifier::string>> newString02{ { "0.03434" }, { "0.23e23" }, {}, { "0.0344e-23" } };
		std::string newString{};
		std::string newString03{};
		jsonifier::jsonifier_core parser{};
		auto result = parser.serializeJson(newString02, newString);
		result		= parser.prettifyJson(newString, newString03);	

		bnch_swt::benchmark_stage<"test_stage", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::runBenchmark<"test-01", "library02", "magenta">([&]() mutable {
			for (size_t x = 0; x < 128; ++x) {
				//std::cout << "CURRENT RESULT: " << x << std::endl;
			auto result = glz::write_json(newString02, newString);
				bnch_swt::doNotOptimizeAway(result);
			}
		});

		bnch_swt::benchmark_stage<"test_stage", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::runBenchmark<"test-02", "library02", "magenta">([&]() mutable {
			for (size_t x = 0; x < 128; ++x) {
				//std::cout << "CURRENT RESULT: " << x << std::endl;
				auto result = glz::write_json(newString02, newString);
				bnch_swt::doNotOptimizeAway(result);
			}
		});
		std::cout << "CURRENT RESULT: " << std::endl;
		auto resultsNew = bnch_swt::benchmark_stage<"test_stage", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::results;
		for (auto& value: resultsNew) {
			std::cout << "CURRENT RESULT: " << value.libraryName << std::endl;
		}
		bnch_swt::benchmark_stage<"test_stage", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	} catch (std::runtime_error& e) {
		std::cout << e.what() << std::endl;
	} catch (std::out_of_range& e) {
		std::cout << e.what() << std::endl;
	}
	return 0;
}