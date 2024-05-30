#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Simdjson.hpp"
#include "Tests/Jsonifier.hpp"
#include "Tests/Common.hpp"
#include "Tests/Uint.hpp"
#include "Tests/Float.hpp"
#include "Tests/RoundTrip.hpp"
#include "Tests/Int.hpp"
#include "Tests/String.hpp"
#include <glaze/glaze.hpp>

template<typename data_structure, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLength(const jsonifier::string& dataToParse) {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	jsonifier::jsonifier_core parser{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Glaze-Function", "dodgerblue">(
		[=]() mutable {
			data_structure valueNew{};
			auto result = glz::read_json(valueNew, dataToParse);
			bnch_swt::doNotOptimizeAway(result);
			bnch_swt::doNotOptimizeAway(valueNew);
		});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Jsonifier-Function", "dodgerblue">(
		[=]() mutable {
			data_structure valueNew{};
			auto result = parser.parseJson<jsonifier::parse_options{ .knownOrder = true }>(valueNew, dataToParse);
			bnch_swt::doNotOptimizeAway(result);
			bnch_swt::doNotOptimizeAway(valueNew);
		});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

template<typename value_type> struct test_struct_new {
	value_type value01{ test_generator::generateValue<value_type>() };
	value_type value02{ test_generator::generateValue<value_type>() };
	value_type value03{ test_generator::generateValue<value_type>() };
	value_type value04{ test_generator::generateValue<value_type>() };
	value_type value05{ test_generator::generateValue<value_type>() };
	value_type value06{ test_generator::generateValue<value_type>() };
	value_type value07{ test_generator::generateValue<value_type>() };
	value_type value08{ test_generator::generateValue<value_type>() };
	value_type value09{ test_generator::generateValue<value_type>() };
	value_type value10{ test_generator::generateValue<value_type>() };
};

template<typename value_type_new> struct jsonifier::core<test_struct_new<value_type_new>> {
	using value_type = test_struct_new<value_type_new>;
	static constexpr auto parseValue = createValue<&value_type::value01, &value_type::value02, &value_type::value03, &value_type::value04, &value_type::value05,
		&value_type::value06, &value_type::value07, &value_type::value08, &value_type::value09, &value_type::value10>();
};

template<typename value_type_new> struct glz::meta<test_struct_new<value_type_new>> {
	using value_type				 = test_struct_new<value_type_new>;
	static constexpr auto value = object(&value_type::value01, &value_type::value02, &value_type::value03, &value_type::value04, &value_type::value05, &value_type::value06,
		&value_type::value07, &value_type::value08, &value_type::value09, &value_type::value10);
};

int32_t main() {
	/*
	runForLength<bool, "bool-false", "bool-false">("false");
	runForLength<bool, "bool-true", "bool-true">("true");
	runForLength<std::vector<bool>, "std::vector<bool>-5", "std::vector<bool>">(jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<bool>(5)));
	runForLength<std::vector<bool>, "std::vector<bool>-10", "std::vector<bool>">(jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<bool>(10)));
	runForLength<std::vector<bool>, "std::vector<bool>-20", "std::vector<bool>">(jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<bool>(20)));
	runForLength<std::vector<double>, "std::vector<double>-5", "std::vector<double>">(jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<double>(5)));
	runForLength<std::vector<double>, "std::vector<double>-10", "std::vector<double>">(jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<double>(10)));
	runForLength<std::vector<double>, "std::vector<double>-20", "std::vector<double>">(jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<double>(20)));

	runForLength<std::vector<uint64_t>, "std::vector<uint64_t>-5", "std::vector<uint64_t>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<uint64_t>(5)));
	runForLength<std::vector<uint64_t>, "std::vector<uint64_t>-10", "std::vector<uint64_t>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<uint64_t>(10)));
	runForLength<std::vector<uint64_t>, "std::vector<uint64_t>-20", "std::vector<uint64_t>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<uint64_t>(20)));

	runForLength<std::vector<std::string>, "std::vector<std::string>-5", "std::vector<std::string>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<std::string>(5)));
	runForLength<std::vector<std::string>, "std::vector<std::string>-10", "std::vector<std::string>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<std::string>(10)));
	runForLength<std::vector<std::string>, "std::vector<std::string>-20", "std::vector<std::string>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<std::string>(20)));

	runForLength<std::vector<int64_t>, "std::vector<int64_t>-5", "std::vector<int64_t>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<int64_t>(5)));
	runForLength<std::vector<int64_t>, "std::vector<int64_t>-10", "std::vector<int64_t>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<int64_t>(10)));
	runForLength<std::vector<int64_t>, "std::vector<int64_t>-20", "std::vector<int64_t>">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<int64_t>(20)));
		*/
	runForLength<test_struct_new<int64_t>, "test_struct_new<int64_t>-5", "test_struct_new<int64_t>">(jsonifier::jsonifier_core<true>{}.serializeJson(test_struct_new<int64_t>{}));
	return 0;
}