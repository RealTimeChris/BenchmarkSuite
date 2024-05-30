#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <thread>

BNCH_SWT_NO_INLINE int64_t testFunction01(int64_t value) {
	auto lambda = [](int64_t valueNew) {
		valueNew += 2;
		while (valueNew < std::numeric_limits<int64_t>::max() / valueNew) {
			valueNew *= valueNew;
		}
		return valueNew;
	};
	return lambda(value);
}

BNCH_SWT_ALWAYS_INLINE int64_t testFunction02(int64_t value) {
	auto lambda = [](int64_t valueNew) {
		valueNew += 2;
		while (valueNew < std::numeric_limits<int64_t>::max() / valueNew) {
			valueNew *= valueNew;
		}
		return valueNew;
	};
	return lambda(value);
}

template<typename value_type> JSONIFIER_ALWAYS_INLINE value_type loadValue(const char* src) noexcept {
	value_type value;
	std::memcpy(&value, src, sizeof(value_type));
	return value;
}

BNCH_SWT_ALWAYS_INLINE int64_t testFunction03(int64_t value) {
	auto lambda = [](int64_t valueNew) {
		valueNew += 2;
		while (valueNew < std::numeric_limits<int64_t>::max() / valueNew) {
			valueNew *= valueNew;
		}
		return valueNew;
	};
	return lambda(value);
}

int64_t testFunction04(int64_t value) {
	auto lambda = [](int64_t valueNew) {
		valueNew += 2;
		while (valueNew < std::numeric_limits<int64_t>::max() / valueNew) {
			valueNew *= valueNew;
		}
		return valueNew;
	};
	return lambda(value);
}

template<uint64_t digitCount, uint64_t length, bnch_swt::string_literal testStageNew, bnch_swt::string_literal testNameNew> BNCH_SWT_ALWAYS_INLINE void parseFunction() {
	static constexpr bnch_swt::string_literal testStage{ testStageNew.values };
	static constexpr bnch_swt::string_literal testName{ testNameNew.values };
	std::string newString{ bnch_swt::test_generator::generateString(8) };
	auto results = bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<testName, "no-inline", "dodgerblue">([&]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			uint64_t value{ loadValue<uint64_t>(newString.data()) };
			bnch_swt::doNotOptimizeAway(value);
		}
		return 8;
	});
	results		 = bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<testName, "sometimes-inline", "dodgerblue">([=]() mutable {
		 for (size_t x = 0; x < 1024; ++x) {
			 uint64_t value;
			 std::memcpy(&value, newString.data(), sizeof(uint64_t));
			 bnch_swt::doNotOptimizeAway(value);
		 }
		 return 8;
	 });
	bnch_swt::benchmark_stage<testStage, 100>::printResults();
}

template<typename value_type>
concept has_value_type = requires() { typename value_type::value_type; };

template<has_value_type value_type> void testFunction(value_type value) {};

int main() {
	testFunction(std::vector<int32_t>{});
	
	parseFunction<1, 1, "fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1",
		"fast_float::loop_parse_if_digits-vs-fast_float::loop_parse_if_eight_digits-for-length-1-and-digit-count-1">();

	return 0;
}