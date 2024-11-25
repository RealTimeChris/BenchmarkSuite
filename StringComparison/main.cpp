#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "unordered_map/UnorderedSet.hpp"
#include "unordered_map/UnorderedMap.hpp"

template<uint64_t length, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void parseFunction() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<std::string> newUints{};
	std::vector<std::string> newerUints01{};
	std::vector<std::string> newerUints02{};
	for (uint64_t x = 0; x < 1024 * 128; ++x) {
		newUints.emplace_back(test_generator::generateValue<std::string>(length));
	}
	newerUints02.resize(1024 * 128);
	newerUints01.resize(1024 * 128);

	bnch_swt::benchmark_stage<testStage + "-emplace", 5>::template runBenchmark<testName, "std::unordered_map", "dodgerblue">([&]() mutable {
		uint64_t totalBytes{};
		std::unordered_map<std::string, std::string> map{};
		for (size_t x = 0; x < 1024; ++x) {
			map.emplace(newUints[x], newUints[x]);
			totalBytes += newUints[x].size() * 2;
		}
		bnch_swt::doNotOptimizeAway(map);
		return totalBytes;
	});

	bnch_swt::benchmark_stage<testStage + "-emplace", 5>::template runBenchmark<testName, "jsonifier::unordered_map", "dodgerblue">([&]() mutable {
		uint64_t totalBytes{};
		jsonifier::unordered_map<std::string, std::string> map{};
		for (size_t x = 0; x < 1024; ++x) {
			map.emplace(newUints[x], newUints[x]);
			totalBytes += newUints[x].size() * 2;
		}
		bnch_swt::doNotOptimizeAway(map);
		return totalBytes;
	});

	bnch_swt::benchmark_stage<testStage + "-find", 5>::template runBenchmark<testName, "std::unordered_map", "dodgerblue">([&]() mutable {
		uint64_t totalBytes{ 1024 * 1024 };
		std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
		return totalBytes;
	});

	bnch_swt::benchmark_stage<testStage + "-find", 5>::template runBenchmark<testName, "jsonifier::unordered_map", "dodgerblue">([&]() mutable {
		uint64_t totalBytes{ 1024 * 1024 };
		std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
		return totalBytes;
	});

	bnch_swt::benchmark_stage<testStage + "-insert", 5>::template runBenchmark<testName, "std::unordered_map", "dodgerblue">([&]() mutable {
		uint64_t totalBytes{ 1024 * 1024 };
		std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
		return totalBytes;
	});

	bnch_swt::benchmark_stage<testStage + "-insert", 5>::template runBenchmark<testName, "jsonifier::unordered_map", "dodgerblue">([&]() mutable {
		uint64_t totalBytes{ 1024 * 1024 };
		std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
		return totalBytes;
	});
	for (uint64_t x = 0; x < 1024 * 128; ++x) {
		if (newerUints02[x] != newerUints01[x]) {
			std::cout << "Failed to parse at index: " << x << std::endl;
			std::cout << "Input Value: " << newUints[x] << std::endl;
			std::cout << "Intended Value: " << newerUints01[x] << std::endl;
			std::cout << "Parsed Value: " << newerUints02[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage + "-emplace", 5>::printResults();
	bnch_swt::benchmark_stage<testStage + "-find", 5>::printResults();
	bnch_swt::benchmark_stage<testStage + "-insert", 5>::printResults();
}

struct test_new {};

void printPowersOf2() {
	for (size_t x = 0; x < 128; ++x) {
		std::cout << static_cast<uint64_t>(std::pow(2, x)) << "," << std::endl;
	}
}

template<jsonifier::concepts::variant_t variant_type> void testFunction(variant_type variant) {
}

int main() {
	jsonifier::unordered_map<std::string, uint64_t> map{};
	std::unordered_map<std::string, uint64_t> map02{};
	std::vector<std::string> testStrings{};
	//map.emplace("TESTING", 23);
	map02.emplace("TESTING", 23);
	for (size_t x = 0; x < 20; ++x) {
		testStrings.emplace_back("TEST-STRING: " + std::to_string(x));
		map.emplace("TEST-STRING: " + std::to_string(x), x);
		map02.emplace("TEST-STRING: " + std::to_string(x), x);
	}

	std::cout << "CURRENT SIZE: " << map.size() << std::endl;
	std::cout << "CURRENT CAPACITY: " << map.capacity() << std::endl;
	std::cout << "CURRENT SIZE-02: " << map02.size() << std::endl;
	std::cout << "CURRENT CAPACITY-02: " << map02.bucket_count() << std::endl;
	for (size_t x = 0; x < 20; ++x) {
		std::cout << map.find("TEST-STRING: " + std::to_string(x)).operator*().first << std::endl;
	}
	auto iter = map.find("TESTING");
	if (iter != map.end()) {
		std::cout << "CURRENT VALUE: " << map.find("TESTING").operator*().second << std::endl;
	}
	for (auto iter = map.begin(); iter != map.end(); ++iter) {
		//std::cout << "CURRENT VALUE: " << iter.operator*().second << std::endl;
	}
	std::variant<std::string> testVariant{};
	testFunction(testVariant);
	std::tuple<int32_t, std::vector<std::string>> tupleNew{};
	std::tuple<> tupleNewer{};
	int32_t newValue{};
	std::cout << map << std::endl;

	
	parseFunction<1, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-1">();/*
	parseFunction<2, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-2">();
	parseFunction<3, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-3">();
	parseFunction<4, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-4">();
	parseFunction<5, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-5">();
	parseFunction<6, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-6">();
	parseFunction<7, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-7">();
	parseFunction<8, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-8">();
	parseFunction<9, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-9">();
	parseFunction<10, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-10">();
	parseFunction<11, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-11">();
	parseFunction<12, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-12">();
	parseFunction<13, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-13">();
	parseFunction<14, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-14">();
	parseFunction<15, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-15">();
	parseFunction<16, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-16">();
	parseFunction<17, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-17">();
	parseFunction<18, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-18">();
	parseFunction<19, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-19">();
	parseFunction<20, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-20">();
	parseFunction<21, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-21">();
	parseFunction<22, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-22">();
	parseFunction<23, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-23">();
	parseFunction<24, "jsonifier::unordered_map-vs-std::unordered_map", "jsonifier::unordered_map-vs-std::unordered_map-24">();*/
	return 0;
}