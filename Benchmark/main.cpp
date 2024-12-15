#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"

static constexpr uint64_t maxIterations{ 400 };

template<uint64_t count, uint64_t length, bool minified, typename value_type, bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	std::vector<value_type> testValuesRaw{};
	std::vector<std::string> testValues{};
	jsonifier::jsonifier_core parser{};
	testValuesRaw.resize(maxIterations);
	testValues.resize(maxIterations);
	for (size_t x = 0; x < maxIterations; ++x) {
		testValuesRaw[x] = bnch_swt::random_generator::generateValue<value_type>(count);
		parser.serializeJson<jsonifier::serialize_options{ .prettify = !minified }>(testValuesRaw[x], testValues[x]);
	}
	std::vector<value_type> resultValues01{};
	resultValues01.resize(maxIterations);
	std::vector<value_type> resultValues02{};
	resultValues02.resize(maxIterations);
	if constexpr (std::is_same_v<std::string, value_type>) {
		for (size_t x = 0; x < maxIterations; ++x) {
			resultValues01[x].resize(length);
			resultValues02[x].resize(length);
		}
	}
	size_t currentIndex{};
	

	bnch_swt::benchmark_stage<testName, maxIterations>::template runBenchmark<"glz::read_json", "CYAN">([&] {
		size_t bytesProcessed{};
		glz::read<glz::opts{ .minified = minified }>(resultValues02[currentIndex], testValues[currentIndex]);
		bytesProcessed += testValues[currentIndex].size();
		bnch_swt::doNotOptimizeAway(resultValues02[currentIndex]);
		++currentIndex;
		return bytesProcessed;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		if (testValuesRaw[x] != resultValues02[x]) {
			if (testValuesRaw[x] != resultValues02[x]) {
				std::cout << "Glaze failed to match output for index: " << x << std::endl;
				std::cout << "For Data: " << testValues[x] << std::endl;
			}
		}
	}

	currentIndex = 0;
	bnch_swt::benchmark_stage<testName, maxIterations>::template runBenchmark<"jsonifier::jsonifier_core::parseJson", "CYAN">([&] {
		size_t bytesProcessed{};
		parser.parseJson<jsonifier::parse_options{ .minified = minified }>(resultValues01[currentIndex], testValues[currentIndex]);
		bytesProcessed += testValues[currentIndex].size();
		bnch_swt::doNotOptimizeAway(resultValues01[currentIndex]);
		++currentIndex;
		return bytesProcessed;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		if (testValuesRaw[x] != resultValues01[x]) { 
			if (testValuesRaw[x] != resultValues02[x]) {
				std::cout << "Jsonifier sfailed to match output for index: " << x << std::endl;
				std::cout << "For Data: " << testValues[x] << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<testName, maxIterations>::printResults(true, true);
}

int main() {

	testFunction<32, 64, true, std::vector<std::string>, "bnch_swt::test<bnch_swt::test_struct>-parse-comparisons-64-minified">();
	testFunction<32, 64, false, std::vector<std::string>, "bnch_swt::test<bnch_swt::test_struct>-parse-comparisons-64-prettified">();
	return 0;
}