#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <thread>

std::string data{ "[\"B07X51T2VK\",\"HUAWEI\",\"Honor 5X Unlocked Smartphone, 16GB Dark Grey (US Warranty) "
				  "(Renewed)\",\"https://www.amazon.com/Honor-Unlocked-Smartphone-Warranty-Renewed/dp/B07X51T2VK\",\"https://m.media-amazon.com/images/I/"
				  "71qG253LcxL._AC_UY218_SEARCH213888_FMwebp_QL75_.jpg\",4,\"https://www.amazon.com/product-reviews/B07X51T2VK\",1,\"$74.99\"]" };
using value_type = std::tuple<std::string, std::string, std::string, std::string, std::string, uint64_t, std::string, uint64_t, std::string>;

size_t calcNewValue(double value) {
	value += 2323;
	return std::pow(std::pow(value, value), std::pow(value, value));
}

int main() {
	std::variant<bool, int32_t> testVariant{ false };
	std::tuple<bool, int32_t> testTuple{ false, 23 };
	std::unordered_map<std::string, std::vector<int32_t>> mapTest{ { "terts", { 2323 } } };
	std::vector<std::string> testVector{ "TEST" };
	std::cout << testVector << std::endl;
	std::cout << testTuple << std::endl;
	std::cout << testVariant << std::endl;
	jsonifier::jsonifier_core<> parser{};
	value_type dataNew{};
	parser.parseJson(dataNew, data);
	std::cout << "CURRENT SIZE: " << std::get<8>(dataNew) << std::endl;
	std::cout << "CURRENT SIZE: " << mapTest << std::endl;
	auto result = bnch_swt::benchmark_stage<"test-stage01", 2>::template runBenchmarkWithPrep<"Test-Lib01", "cyan">(
		[] {
			for (size_t x = 0; x < 1024; ++x) {
				size_t newSz{ 234234 };
				auto newValue = calcNewValue(newSz);
				bnch_swt::doNotOptimizeAway(newValue);
			}
			return 8;
		},
		[]() {
			for (size_t x = 0; x < 1024; ++x) {
				size_t newSz{ 234234 };
				auto newValue = calcNewValue(newSz);
				bnch_swt::doNotOptimizeAway(newValue);
			}
			return 8;
		});

	result = bnch_swt::benchmark_stage<"test-stage01", 2>::template runBenchmark<"Test-Lib03", "cyan">([]() {
		for (size_t x = 0; x < 1024; ++x) {
			size_t newSz{ 234234 };
			auto newValue = calcNewValue(newSz);
			bnch_swt::doNotOptimizeAway(newValue);
		}
		return 8;
	});

	result = bnch_swt::benchmark_stage<"test-stage01", 2>::template runBenchmark<"Test-Lib02", "cyan">([]() {
		for (size_t x = 0; x < 1024; ++x) {
			size_t newSz{ 234234 };
			auto newValue = calcNewValue(newSz);
			bnch_swt::doNotOptimizeAway(newValue);
		}
		return 8;
	});
	bnch_swt::benchmark_stage<"test-stage01", 2>::printResults();
	std::cout << "CURRENT RESULT: " << result.bytesProcessed << std::endl;
	return 0;
}