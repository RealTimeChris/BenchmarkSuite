#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "StrToDOld.hpp"
#include "FastFloatNew.hpp"
#include "fast_float.h"

std::string generateIntegerString(size_t length) {
	if (length == 0) {
		throw std::invalid_argument("Length must be greater than 0");
	}

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<int> digitDist(0, 9);

	std::string result;
	result.reserve(length);
	result += '1' + digitDist(generator) % 9;
	for (size_t i = 1; i < length; ++i) {
		result += '0' + digitDist(generator);
	}

	return result;
}

std::string generateFloatingPointString(bool negative, size_t digit_count, size_t fractionalLength = 0, size_t exponentLength = 0) {
	if (fractionalLength == 0 && exponentLength == 0) {
		throw std::invalid_argument("At least one of fractionalLength or exponentLength must be greater than 0.");
	}

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<size_t> lengthDist(1, 5);

	std::string result = negative ? "-" : "";
	result += generateIntegerString(digit_count);

	if (fractionalLength > 0) {
		result += "." + generateIntegerString(fractionalLength);
	}

	if (exponentLength > 0) {
		std::uniform_int_distribution<int> exponentSignDist(0, 1);
		char exponentSign = exponentSignDist(generator) ? '+' : '-';
		result += "e" + exponentSign + generateIntegerString(exponentLength);
	}

	return result;
}

std::vector<std::string> generateValidFloatingPointStrings(size_t count, size_t digit_count, size_t fractionalLength, size_t exponentLength, bool allowNegative) {
	std::vector<std::string> validStrings;
	validStrings.reserve(count);

	while (validStrings.size() < count) {
		try {
			bool negative		  = allowNegative && (std::rand() % 2 == 0);
			std::string candidate = generateFloatingPointString(negative, digit_count, fractionalLength, exponentLength);

			char* endPtr = nullptr;
			double value = std::strtod(candidate.c_str(), &endPtr);

			if (endPtr == candidate.c_str() || *endPtr != '\0') {
				throw std::invalid_argument("strtod failed to parse the string.");
			}

			std::string convertedBack = std::to_string(value);

			validStrings.push_back(candidate);
		} catch (...) {
			continue;
		}
	}

	return validStrings;
}

constexpr auto Calc(auto x) {
	for (size_t y = 0; y < 32; ++y) {
		x = 4 * x;
		++x;
		++x;
	}
	return x;
}

consteval auto as_constant(auto value) {
	return value;
}

template<jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize02() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "non-as-constant", "dodgerblue">(
		[&]() mutable {
			for (size_t x = 0; x < 1024 * 1024; ++x) {
				size_t newValue{ Calc(245ull) };
				++newValue;
				bnch_swt::doNotOptimizeAway(newValue);
			}
		});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "as-constant", "dodgerblue">(
		[&]() mutable {
			for (size_t x = 0; x < 1024 * 1024; ++x) {
				size_t newValue{ as_constant(Calc(245ull)) };
				++newValue;
				bnch_swt::doNotOptimizeAway(newValue);
			}
		});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

template<size_t maxIndex, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	auto newFile{ bnch_swt::file_loader ::loadFile(std::string{ JSON_BASE_PATH } + "/CitmCatalogData-Prettified.json") };
	jsonifier::jsonifier_core parser{};
	std::vector<std::vector<std::vector<double>>> coordinates{};
	parser.parseJson(coordinates, newFile);
	std::vector<double> newerDoubles00{};
	for (auto& value: coordinates) {
		for (auto& valueNew: value) {
			for (auto& valueNewer: valueNew) {
				newerDoubles00.emplace_back(test_generator::generateDouble());
			}
		}
	}

	std::vector<double> newerDoubles01{};
	std::vector<std::string> newDoubles{};
	for (size_t x = 0; x < newerDoubles00.size(); ++x) {
		newDoubles.emplace_back(std::to_string(newerDoubles00[x]));
	}
	std::vector<double> newerDoubles02{};
	std::vector<double> newerDoubles03{};
	newerDoubles01.resize(newDoubles.size());
	newerDoubles03.resize(newDoubles.size());
	newerDoubles02.resize(newDoubles.size());

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "glz-from_chars", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					glz::from_chars<true>(iter, end, newDouble);
					newerDoubles02[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_old::parseFloat(iter, end, newDouble);
					newerDoubles01[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "orginal-fastfloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					fast_float::from_chars_advanced(iter, end, newDouble, fast_float::parse_options_t<char>{});
					newerDoubles01[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "new-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_new::parseFloat(iter, end, newDouble); 
					newerDoubles03[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	for (size_t x = 0; x < maxIndex; ++x) {
		if (newerDoubles03[x] != newerDoubles01[x]) {
			std::cout << "FAILED TO PARSE AT INDEX: " << x << std::endl;
			std::cout << "Input Value: " << newDoubles[x] << std::endl;
			std::cout << "Intended Value: " << newerDoubles01[x] << std::endl;
			std::cout << "Actual Value: " << newerDoubles03[x] << std::endl;
		} else {
			//std::cout << "Here's the value: " << newerDoubles01[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

template<size_t maxIndex, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize02() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	auto newFile{ bnch_swt::file_loader ::loadFile(std::string{ JSON_BASE_PATH } + "/CitmCatalogData-Prettified.json") };
	jsonifier::jsonifier_core parser{};
	std::vector<std::vector<std::vector<double>>> coordinates{};
	parser.parseJson(coordinates, newFile);
	std::vector<double> newerDoubles00{};
	for (auto& value: coordinates) {
		for (auto& valueNew: value) {
			for (auto& valueNewer: valueNew) {
				newerDoubles00.emplace_back(valueNewer);
			}
		}
	}

	std::vector<double> newerDoubles01{};
	std::vector<std::string> newDoubles{};
	for (size_t x = 0; x < newerDoubles00.size(); ++x) {
		newDoubles.emplace_back(std::to_string(newerDoubles00[x]));
	}
	std::vector<double> newerDoubles02{};
	std::vector<double> newerDoubles03{};
	newerDoubles01.resize(newDoubles.size());
	newerDoubles03.resize(newDoubles.size());
	newerDoubles02.resize(newDoubles.size());

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "glz-from_chars", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					glz::from_chars<true>(iter, end, newDouble);
					newerDoubles02[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_old::parseFloat(iter, end, newDouble);
					newerDoubles01[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "orginal-fastfloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					fast_float::from_chars_advanced(iter, end, newDouble, fast_float::parse_options_t<char>{});
					newerDoubles01[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "new-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_new::parseFloat(iter, end, newDouble); 

					newerDoubles03[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	for (size_t x = 0; x < maxIndex; ++x) {
		if (newerDoubles03[x] != newerDoubles01[x]) {
			std::cout << "FAILED TO PARSE AT INDEX: " << x << std::endl;
			std::cout << "Input Value: " << newDoubles[x] << std::endl;
			std::cout << "Intended Value: " << newerDoubles01[x] << std::endl;
			std::cout << "Actual Value: " << newerDoubles03[x] << std::endl;
		} else {
			//std::cout << "Here's the value: " << newerDoubles01[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

template<size_t maxIndex, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize03() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	auto newFile{ bnch_swt::file_loader ::loadFile(std::string{ JSON_BASE_PATH } + "/CitmCatalogData-Prettified.json") };
	jsonifier::jsonifier_core parser{};
	std::vector<std::vector<std::vector<double>>> coordinates{};
	parser.parseJson(coordinates, newFile);
	std::vector<uint64_t> newerDoubles00{};
	for (auto& value: coordinates) {
		for (auto& valueNew: value) {
			for (auto& valueNewer: valueNew) {
				newerDoubles00.emplace_back(test_generator::generateValue<uint64_t>());
			}
		}
	}

	std::vector<double> newerDoubles01{};
	std::vector<std::string> newDoubles{};
	for (size_t x = 0; x < newerDoubles00.size(); ++x) {

		std::string newString{ std::to_string(newerDoubles00[x]) };
		if (newString.size() > 19) {
			//newString.resize(19);
		}
		newDoubles.emplace_back(newString);
	}
	std::vector<double> newerDoubles02{};
	std::vector<double> newerDoubles03{};
	newerDoubles01.resize(newDoubles.size());
	newerDoubles03.resize(newDoubles.size());
	newerDoubles02.resize(newDoubles.size());

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "glz-from_chars", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					glz::from_chars<true>(iter, end, newDouble);
					newerDoubles02[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_old::parseFloat(iter, end, newDouble);
					newerDoubles01[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "orginal-fastfloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					fast_float::from_chars_advanced(iter, end, newDouble, fast_float::parse_options_t<char>{});
					newerDoubles01[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "new-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble;
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_new::parseFloat(iter, end, newDouble);

					newerDoubles03[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	for (size_t x = 0; x < maxIndex; ++x) {
		if (newerDoubles03[x] != newerDoubles01[x]) {
			std::cout << "FAILED TO PARSE AT INDEX: " << x << std::endl;
			std::cout << "Input Value: " << newDoubles[x] << std::endl;
			std::cout << "Intended Value: " << newerDoubles01[x] << std::endl;
			std::cout << "Actual Value: " << newerDoubles03[x] << std::endl;
		} else {
			//std::cout << "Here's the value: " << newerDoubles01[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

consteval int32_t newValue() {
	int32_t outPut{};
	for (size_t x = 0; x < 1024 *16; ++x) {
		outPut += x;
	}
	return outPut;
}

constexpr int32_t newValue02() {
	int32_t outPut{};
	for (size_t x = 0; x < 1024 * 16; ++x) {
		outPut += x;
	}
	return outPut;
}

template<jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize04() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "glz-from_chars", "dodgerblue">(
		[&]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				auto newerValue = newValue();
				bnch_swt::doNotOptimizeAway(newerValue);
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parseFloat", "dodgerblue">(
		[&]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				auto newerValue = newValue02();
				bnch_swt::doNotOptimizeAway(newerValue);
			}
		});

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}


void testFunction(const std::string& test = std::string{}){};
int main() {
	runForLengthSerialize04<"TEST01", "TEST02">();
	std::string newString{ "3423424" };
	const auto* iter = newString.data();
	const auto* end	 = newString.data() + newString.size();
	double newDouble{};
	runForLengthSerialize<1, "Old-FastFloat-vs-New-FastFloat-1", "Old-FastFloat-vs-New-FastFloat-1">();
	runForLengthSerialize<8, "Old-FastFloat-vs-New-FastFloat-8", "Old-FastFloat-vs-New-FastFloat-8">();
	runForLengthSerialize<64, "Old-FastFloat-vs-New-FastFloat-64", "Old-FastFloat-vs-New-FastFloat-64">();
	runForLengthSerialize<512, "Old-FastFloat-vs-New-FastFloat-512", "Old-FastFloat-vs-New-FastFloat-512">();
	runForLengthSerialize02<1, "Old-FastFloat-vs-New-FastFloat-Short-1", "Old-FastFloat-vs-New-FastFloat-Short-1">();
	runForLengthSerialize02<8, "Old-FastFloat-vs-New-FastFloat-Short-8", "Old-FastFloat-vs-New-FastFloat-Short-8">();
	runForLengthSerialize02<64, "Old-FastFloat-vs-New-FastFloat-Short-64", "Old-FastFloat-vs-New-FastFloat-Short-64">();
	runForLengthSerialize02<512, "Old-FastFloat-vs-New-FastFloat-Short-512", "Old-FastFloat-vs-New-FastFloat-Short-512">();
	/*
	runForLengthSerialize03<1, "Old-FastFloat-vs-New-FastFloat-Integer-1", "Old-FastFloat-vs-New-FastFloat-Integer-1">();
	runForLengthSerialize03<8, "Old-FastFloat-vs-New-FastFloat-Integer-8", "Old-FastFloat-vs-New-FastFloat-Integer-8">();
	runForLengthSerialize03<64, "Old-FastFloat-vs-New-FastFloat-Integer-64", "Old-FastFloat-vs-New-FastFloat-Integer-64">();
	runForLengthSerialize03<512, "Old-FastFloat-vs-New-FastFloat-Integer-512", "Old-FastFloat-vs-New-FastFloat-Integer-512">();*/
	return 0;
}