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
JSONIFIER_ALWAYS_INLINE void runForLengthParse(const std::string& dataToParse) {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };

	std::string newString{};
	newString.push_back('f');
	newString.push_back('t');
	newString.push_back('r');
	jsonifier::jsonifier_core parser{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time, .totalIterationCountCap = 300 }>::template runBenchmark<testName,
		"Glaze-Function", "dodgerblue">([=]() mutable {
		for (size_t x = 0; x < 1024 * 16; ++x) {
			newString[0] = '2';
			bnch_swt::doNotOptimizeAway(newString);
		}
	});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time, .totalIterationCountCap = 300 }>::template runBenchmark<testName,
		"Jsonifier-Function", "dodgerblue">([=]() mutable {
		for (size_t x = 0; x < 1024 * 16; ++x) {
			std::memcpy(&newString[0], "2f", 1);
			bnch_swt::doNotOptimizeAway(newString);
		}
	});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

template<typename data_structure, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLengthSerialize(const data_structure& dataToSerialize) {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	jsonifier::jsonifier_core parser{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time, .totalIterationCountCap = 300 }>::template runBenchmark<testName,
		"Glaze-Function", "dodgerblue">([=]() mutable {
		std::string dataNew{};
		auto newValue01 = glz::write_json(dataToSerialize, dataNew);
		bnch_swt::doNotOptimizeAway(newValue01);
	});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time, .totalIterationCountCap = 300 }>::template runBenchmark<testName,
		"Jsonifier-Function", "dodgerblue">([=]() mutable {
		std::string dataNew{};
		auto newValue01 = parser.serializeJson(dataToSerialize, dataNew);
		bnch_swt::doNotOptimizeAway(newValue01);
	});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int32_t main() {
	std::string newString{};
	jsonifier::jsonifier_core parser{};
	auto newVectorBool = test_generator::generateVector<bool>(5, 5);
	parser.serializeJson(newVectorBool, newString);
	//runForLengthSerialize<std::vector<bool>, "serialize-std::vector<bool>-5", "std::vector<bool>-5">(newVectorBool);
	runForLengthParse<std::vector<bool>, "parse-std::vector<bool>-5", "std::vector<bool>-5">(newString);
	auto newVectorDouble = test_generator::generateVector<double>(5, 5);
	parser.serializeJson(newVectorDouble, newString);
	//runForLengthSerialize<std::vector<double>, "serialize-std::vector<double>-5", "std::vector<double>-5">(newVectorDouble);
	runForLengthParse<std::vector<double>, "parse-std::vector<double>-5", "std::vector<double>-5">(newString);

	auto newVectorInt = test_generator::generateVector<int64_t>(5, 5);
	parser.serializeJson(newVectorInt, newString);
	//runForLengthSerialize<std::vector<int64_t>, "serialize-std::vector<int64_t>-5", "std::vector<int64_t>-5">(newVectorInt);
	runForLengthParse<std::vector<int64_t>, "parse-std::vector<int64_t>-5", "std::vector<int64_t>-5">(newString);

	auto newVectorUint = test_generator::generateVector<uint64_t>(5, 5);
	parser.serializeJson(newVectorUint, newString);
	//runForLengthSerialize<std::vector<uint64_t>, "serialize-std::vector<uint64_t>-5", "std::vector<uint64_t>-5">(newVectorUint);
	runForLengthParse<std::vector<uint64_t>, "parse-std::vector<uint64_t>-5", "std::vector<uint64_t>-5">(newString);

	auto newVectorString = test_generator::generateVector<std::string>(5, 5);
	parser.serializeJson(newVectorString, newString);
	//runForLengthSerialize<std::vector<std::string>, "serialize-std::vector<std::string>-5", "std::vector<std::string>-5">(newVectorString);
	runForLengthParse<std::vector<std::string>, "parse-std::vector<std::string>-5", "std::vector<std::string>-5">(newString);

	newVectorBool = test_generator::generateVector<bool>(15, 15);
	parser.serializeJson(newVectorBool, newString);
	//runForLengthSerialize<std::vector<bool>, "serialize-std::vector<bool>-15", "std::vector<bool>-15">(newVectorBool);
	runForLengthParse<std::vector<bool>, "parse-std::vector<bool>-15", "std::vector<bool>-15">(newString);
	newVectorDouble = test_generator::generateVector<double>(15, 15);
	parser.serializeJson(newVectorDouble, newString);
	//runForLengthSerialize<std::vector<double>, "serialize-std::vector<double>-15", "std::vector<double>-15">(newVectorDouble);
	runForLengthParse<std::vector<double>, "parse-std::vector<double>-15", "std::vector<double>-15">(newString);

	newVectorInt = test_generator::generateVector<int64_t>(15, 15);
	parser.serializeJson(newVectorInt, newString);
	//runForLengthSerialize<std::vector<int64_t>, "serialize-std::vector<int64_t>-15", "std::vector<int64_t>-15">(newVectorInt);
	runForLengthParse<std::vector<int64_t>, "parse-std::vector<int64_t>-15", "std::vector<int64_t>-15">(newString);

	newVectorUint = test_generator::generateVector<uint64_t>(15, 15);
	parser.serializeJson(newVectorUint, newString);
	//runForLengthSerialize<std::vector<uint64_t>, "serialize-std::vector<uint64_t>-15", "std::vector<uint64_t>-15">(newVectorUint);
	runForLengthParse<std::vector<uint64_t>, "parse-std::vector<uint64_t>-15", "std::vector<uint64_t>-15">(newString);

	newVectorString = test_generator::generateVector<std::string>(15, 15);
	parser.serializeJson(newVectorString, newString);
	//runForLengthSerialize<std::vector<std::string>, "serialize-std::vector<std::string>-15", "std::vector<std::string>-15">(newVectorString);
	runForLengthParse<std::vector<std::string>, "parse-std::vector<std::string>-15", "std::vector<std::string>-15">(newString);

	newVectorBool = test_generator::generateVector<bool>(30, 30);
	parser.serializeJson(newVectorBool, newString);
	//runForLengthSerialize<std::vector<bool>, "serialize-std::vector<bool>-30", "std::vector<bool>-30">(newVectorBool);
	runForLengthParse<std::vector<bool>, "parse-std::vector<bool>-30", "std::vector<bool>-30">(newString);
	newVectorDouble = test_generator::generateVector<double>(30, 30);
	parser.serializeJson(newVectorDouble, newString);
	//runForLengthSerialize<std::vector<double>, "serialize-std::vector<double>-30", "std::vector<double>-30">(newVectorDouble);
	runForLengthParse<std::vector<double>, "parse-std::vector<double>-30", "std::vector<double>-30">(newString);

	newVectorInt = test_generator::generateVector<int64_t>(30, 30);
	parser.serializeJson(newVectorInt, newString);
	//runForLengthSerialize<std::vector<int64_t>, "serialize-std::vector<int64_t>-30", "std::vector<int64_t>-30">(newVectorInt);
	runForLengthParse<std::vector<int64_t>, "parse-std::vector<int64_t>-30", "std::vector<int64_t>-30">(newString);

	newVectorUint = test_generator::generateVector<uint64_t>(30, 30);
	parser.serializeJson(newVectorUint, newString);
	//runForLengthSerialize<std::vector<uint64_t>, "serialize-std::vector<uint64_t>-30", "std::vector<uint64_t>-30">(newVectorUint);
	runForLengthParse<std::vector<uint64_t>, "parse-std::vector<uint64_t>-30", "std::vector<uint64_t>-30">(newString);

	newVectorString = test_generator::generateVector<std::string>(30, 30);
	parser.serializeJson(newVectorString, newString);
	//runForLengthSerialize<std::vector<std::string>, "serialize-std::vector<std::string>-30", "std::vector<std::string>-30">(newVectorString);
	runForLengthParse<std::vector<std::string>, "parse-std::vector<std::string>-30", "std::vector<std::string>-30">(newString);
	/*
	runForLengthParse<std::string, "parse-std::string-2", "parse-std::string-2">(2);
	runForLengthParse<std::string, "parse-std::string-3", "parse-std::string-3">(3);
	runForLengthParse<std::string, "parse-std::string-4", "parse-std::string-4">(4);
	runForLengthParse<std::string, "parse-std::string-5", "parse-std::string-5">(5);
	runForLengthParse<std::string, "parse-std::string-6", "parse-std::string-6">(6);
	runForLengthParse<std::string, "parse-std::string-7", "parse-std::string-7">(7);
	runForLengthParse<std::string, "parse-std::string-8", "parse-std::string-8">(8);
	runForLengthParse<std::string, "parse-std::string-9", "parse-std::string-9">(9);
	runForLengthParse<std::string, "parse-std::string-10", "parse-std::string-10">(10);
	runForLengthParse<std::string, "parse-std::string-11", "parse-std::string-11">(11);
	runForLengthParse<std::string, "parse-std::string-12", "parse-std::string-12">(12);
	runForLengthParse<std::string, "parse-std::string-13", "parse-std::string-13">(13);
	runForLengthParse<std::string, "parse-std::string-14", "parse-std::string-14">(14);
	runForLengthParse<std::string, "parse-std::string-15", "parse-std::string-15">(15);
	runForLengthParse<std::string, "parse-std::string-16", "parse-std::string-16">(16);
	runForLengthParse<std::string, "parse-std::string-32", "parse-std::string-32">(32);
	runForLengthParse<std::string, "parse-std::string-64", "parse-std::string-64">(64);
	runForLengthParse<std::string, "parse-std::string-128", "parse-std::string-128">(128);

	//runForLengthSerialize<std::string, "serialize-std::string-1", "serialize-std::string-1">(1);
	//runForLengthSerialize<std::string, "serialize-std::string-2", "serialize-std::string-3">(2);
	//runForLengthSerialize<std::string, "serialize-std::string-3", "serialize-std::string-3">(3);
	//runForLengthSerialize<std::string, "serialize-std::string-4", "serialize-std::string-5">(4);
	//runForLengthSerialize<std::string, "serialize-std::string-5", "serialize-std::string-5">(5);
	//runForLengthSerialize<std::string, "serialize-std::string-6", "serialize-std::string-6">(6);
	//runForLengthSerialize<std::string, "serialize-std::string-7", "serialize-std::string-7">(7);
	//runForLengthSerialize<std::string, "serialize-std::string-8", "serialize-std::string-8">(8);
	//runForLengthSerialize<std::string, "serialize-std::string-9", "serialize-std::string-9">(9);
	//runForLengthSerialize<std::string, "serialize-std::string-10", "serialize-std::string-10">(10);
	//runForLengthSerialize<std::string, "serialize-std::string-11", "serialize-std::string-11">(11);
	//runForLengthSerialize<std::string, "serialize-std::string-12", "serialize-std::string-12">(12);
	//runForLengthSerialize<std::string, "serialize-std::string-13", "serialize-std::string-13">(13);
	//runForLengthSerialize<std::string, "serialize-std::string-14", "serialize-std::string-14">(14);
	//runForLengthSerialize<std::string, "serialize-std::string-15", "serialize-std::string-15">(15);
	//runForLengthSerialize<std::string, "serialize-std::string-16", "serialize-std::string-16">(16);
	//runForLengthSerialize<std::string, "serialize-std::string-32", "serialize-std::string-32">(32);
	//runForLengthSerialize<std::string, "serialize-std::string-64", "serialize-std::string-64">(64);
	//runForLengthSerialize<std::string, "serialize-std::string-128", "serialize-std::string-128">(128);
	*/
	/*
	runForLengthParse<std::vector<uint64_t>, "parse-std::vector<uint64_t>-5", "parse-std::vector<uint64_t>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<uint64_t>(5)));
	runForLengthParse<std::vector<uint64_t>, "parse-std::vector<uint64_t>-10", "parse-std::vector<uint64_t>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<uint64_t>(10)));
	runForLengthParse<std::vector<uint64_t>, "parse-std::vector<uint64_t>-20", "parse-std::vector<uint64_t>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<uint64_t>(20)));

	
	runForLengthParse<std::vector<std::string>, "parse-std::vector<std::string>-10", "parse-std::vector<std::string>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson((10)));
	runForLengthParse<std::vector<std::string>, "parse-std::vector<std::string>-20", "parse-std::vector<std::string>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson((20)));

	runForLengthParse<std::vector<int64_t>, "parse-std::vector<int64_t>-5", "parse-std::vector<int64_t>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<int64_t>(5)));
	runForLengthParse<std::vector<int64_t>, "parse-std::vector<int64_t>-10", "parse-std::vector<int64_t>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<int64_t>(10)));
	runForLengthParse<std::vector<int64_t>, "parse-std::vector<int64_t>-20", "parse-std::vector<int64_t>-5">(
		jsonifier::jsonifier_core<true>{}.serializeJson(test_generator::generateVector<int64_t>(20)));
	std::string newString{ jsonifier::jsonifier_core<true>{}.serializeJson(test_struct_new<std::string>{}) };
	std::cout << "CURRENT JSON DATA: " << newString << std::endl;
	runForLengthParse<test_struct_new<std::string>, "parse-test_struct_new<std::string>-5", "test_struct_new<std::string>-5">(newString);

	newString = jsonifier::jsonifier_core<true>{}.serializeJson(test_struct_new<int64_t>{});
	std::cout << "CURRENT JSON DATA: " << newString << std::endl;
	runForLengthParse<test_struct_new<int64_t>, "parse-test_struct_new<test_struct_new<int64_t>>-5", "test_struct_new<test_struct_new<int64_t>>-5">(newString);

	newString = jsonifier::jsonifier_core<true>{}.serializeJson(test_struct_new<bool>{});
	std::cout << "CURRENT JSON DATA: " << newString << std::endl;
	runForLengthParse<test_struct_new<bool>, "parse-test_struct_new<bool>-5", "test_struct_new<bool>-5">(newString);
	//runForLengthSerialize<std::string, "serialize-bool-false", "serialize-bool-false">(false);
	//runForLengthSerialize<std::string, "serialize-bool-true", "serialize-bool-true">(true);
	//runForLengthSerialize<std::string, "serialize-std::vector<bool>-5", "serialize-std::vector<bool>-5">(test_generator::generateVector<bool>(5));
	//runForLengthSerialize<std::string, "serialize-std::vector<bool>-10", "serialize-std::vector<bool>-5">(test_generator::generateVector<bool>(10));
	//runForLengthSerialize<std::string, "serialize-std::vector<bool>-20", "serialize-std::vector<bool>-5">(test_generator::generateVector<bool>(20));
	//runForLengthSerialize<std::string, "serialize-std::vector<double>-5", "serialize-std::vector<double>-5">(test_generator::generateVector<double>(5));
	//runForLengthSerialize<std::string, "serialize-std::vector<double>-10", "serialize-std::vector<double>-5">(test_generator::generateVector<double>(10));
	//runForLengthSerialize<std::string, "serialize-std::vector<double>-20", "serialize-std::vector<double>-5">(test_generator::generateVector<double>(20));

	//runForLengthSerialize<std::string, "serialize-std::vector<uint64_t>-5", "serialize-std::vector<uint64_t>-5">(test_generator::generateVector<uint64_t>(5));
	//runForLengthSerialize<std::string, "serialize-std::vector<uint64_t>-10", "serialize-std::vector<uint64_t>-5">(test_generator::generateVector<uint64_t>(10));
	//runForLengthSerialize<std::string, "serialize-std::vector<uint64_t>-20", "serialize-std::vector<uint64_t>-5">(test_generator::generateVector<uint64_t>(20));

	//runForLengthSerialize<std::string, "serialize-std::vector<std::string>-5", "serialize-std::vector<std::string>-5">((5));
	//runForLengthSerialize<std::string, "serialize-std::vector<std::string>-10", "serialize-std::vector<std::string>-5">((10));
	//runForLengthSerialize<std::string, "serialize-std::vector<std::string>-20", "serialize-std::vector<std::string>-5">((20));

	//runForLengthSerialize<std::string, "serialize-std::vector<int64_t>-5", "serialize-std::vector<int64_t>-5">(test_generator::generateVector<int64_t>(5));
	//runForLengthSerialize<std::string, "serialize-std::vector<int64_t>-10", "serialize-std::vector<int64_t>-5">(test_generator::generateVector<int64_t>(10));
	//runForLengthSerialize<std::string, "serialize-std::vector<int64_t>-20", "serialize-std::vector<int64_t>-5">(test_generator::generateVector<int64_t>(20));
	//runForLengthSerialize<std::string, "serialize-test_struct_new<std::string>-5", "test_struct_new<std::string>-5">(test_struct_new<std::string>{});
	//runForLengthSerialize<std::string, "serialize-test_struct_new<test_struct_new<int64_t>>-5", "test_struct_new<test_struct_new<int64_t>>-5">(test_struct_new<int64_t>{});
	//runForLengthSerialize<std::string, "serialize-test_struct_new<bool>-5", "test_struct_new<bool>-5">(test_struct_new<bool>{});
	*/
	return 0;
}