#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "StrToDOld.hpp"
#include "StrToDNew.hpp"
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
		std::string exponentSign = std::string{ exponentSignDist(generator) ? '+' : '-' };
		result += "e" + exponentSign + generateIntegerString(exponentLength);
	}

	return result;
}

std::vector<std::string> generateValidFloatingPointStrings(size_t count, size_t digit_count, size_t fractionalLength, size_t exponentLength, bool allowNegative = false) {
	if (digit_count == 0) {
		throw std::invalid_argument("digit_count must be greater than 0");
	}
	if (count == 0) {
		throw std::invalid_argument("count must be greater than 0");
	}

	std::vector<std::string> validStrings;
	validStrings.reserve(count);

	size_t maxRetries = 10000;// Limit the retries to avoid infinite loops.
	size_t retries	  = 0;

	while (validStrings.size() < count && retries < maxRetries) {
		try {
			bool negative		  = allowNegative && (std::rand() % 2 == 0);
			std::string candidate = generateFloatingPointString(negative, digit_count, fractionalLength, exponentLength);

			char* endPtr = nullptr;
			double value = std::strtod(candidate.c_str(), &endPtr);

			if (endPtr == candidate.c_str() || *endPtr != '\0') {
				throw std::invalid_argument("strtod failed to parse the string.");
			}

			validStrings.push_back(candidate);
		} catch (const std::exception& e) {
			// Log the error (optional)
			std::cerr << "Error: " << e.what() << std::endl;
			retries++;
			continue;
		}
	}

	if (validStrings.size() < count) {
		throw std::runtime_error("Failed to generate enough valid strings within the retry limit.");
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
				fast_float_new::adjusted_mantissa newValue{};
				++newValue.mantissa;
				bnch_swt::doNotOptimizeAway(newValue);
			}
		});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "as-constant", "dodgerblue">(
		[&]() mutable {
			for (size_t x = 0; x < 1024 * 1024; ++x) {
				fast_float_new::adjusted_mantissa newValue{ as_constant(fast_float_new::adjusted_mantissa{}) };
				++newValue.mantissa;
				bnch_swt::doNotOptimizeAway(newValue);
			}
		});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

template<size_t maxIndex, size_t integerLength, size_t fractionLength, size_t exponentLength, jsonifier_internal::string_literal testStageNew,
	jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLengthSerialize() {
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
	std::vector<std::string> newDoubles{ generateValidFloatingPointStrings(maxIndex, integerLength, fractionLength, exponentLength) };
	for (auto& value: newDoubles) {
		std::cout << "CURRENT DOUBLE: " << value << std::endl;
	}
	std::vector<double> newerDoubles02{};
	std::vector<double> newerDoubles03{};
	newerDoubles01.resize(newDoubles.size());
	newerDoubles03.resize(newDoubles.size());
	newerDoubles02.resize(newDoubles.size());

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "glz-from_chars", "dodgerblue">(
		[&]() mutable {
			double newDouble{};
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

	/*

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble{};
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_old::parseFloat(iter, end, newDouble);
					newerDoubles01[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});*/

	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "orginal-fastfloat", "dodgerblue">(
		[&]() mutable {
			double newDouble{};
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
			double newDouble{};
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_new::parseFloat(newDouble, iter, end);
					newerDoubles03[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	for (size_t x = 0; x < maxIndex; ++x) {
		if (newerDoubles03[x] != newerDoubles01[x]) {
			double newDouble{};
			std::cout << "FAILED TO PARSE AT INDEX: " << x << std::endl;
			const auto* iter = newDoubles[x].data();
			const auto* end	 = newDoubles[x].data() + newDoubles[x].size();
			jsonifier_internal_new::parseFloat(newDouble, iter, end);
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
			double newDouble{};
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
	/*
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble{};
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
	*/
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "orginal-fastfloat", "dodgerblue">(
		[&]() mutable {
			double newDouble{};
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
			double newDouble{};
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_new::parseFloat(newDouble, iter, end);

					newerDoubles03[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	for (size_t x = 0; x < maxIndex; ++x) {
		if (newerDoubles03[x] != newerDoubles01[x]) {
			double newDouble{};
			std::cout << "FAILED TO PARSE AT INDEX: " << x << std::endl;
			const auto* iter = newDoubles[x].data();
			const auto* end	 = newDoubles[x].data() + newDoubles[x].size();
			jsonifier_internal_new::parseFloat(newDouble, iter, end);
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
			double newDouble{};
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
	/*
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parseFloat", "dodgerblue">(
		[&]() mutable {
			double newDouble{};
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
		*/
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "orginal-fastfloat", "dodgerblue">(
		[&]() mutable {
			double newDouble{};
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
			double newDouble{};
			for (size_t x = 0; x < 10; ++x) {
				for (size_t y = 0; y < maxIndex; ++y) {
					const auto* iter = newDoubles[y].data();
					const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
					jsonifier_internal_new::parseFloat(newDouble, iter, end);

					newerDoubles03[y] = newDouble;
					bnch_swt::doNotOptimizeAway(newDouble);
				}
			}
		});

	for (size_t x = 0; x < maxIndex; ++x) {
		if (newerDoubles03[x] != newerDoubles01[x]) {
			double newDouble{};
			std::cout << "FAILED TO PARSE AT INDEX: " << x << std::endl;
			const auto* iter = newDoubles[x].data();
			const auto* end	 = newDoubles[x].data() + newDoubles[x].size();
			jsonifier_internal_new::parseFloat(newDouble, iter, end);
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
	for (size_t x = 0; x < 1024 * 16; ++x) {
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

JSONIFIER_ALWAYS_INLINE uint64_t count_digit_bytes(uint64_t val) noexcept {
	uint64_t high_bits	= val + 0x4646464646464646;
	uint64_t low_bits	= val - 0x3030303030303030;
	uint64_t mask		= (high_bits | low_bits) & 0x8080808080808080;
	uint64_t digit_mask = ~mask & 0x8080808080808080;
	digit_mask			= (digit_mask >> 7) & 0x0101010101010101;
	return popcnt(digit_mask);
}

uint64_t parse_seven_digits_unrolled(const char* chars) {
	std::uint64_t val = 0;
	std::memcpy(&val, chars, 7);
	constexpr uint64_t mask = 0x000000FF000000FF;
	constexpr uint64_t mul1 = 10ULL + (100000ULL << 32ULL);
	constexpr uint64_t mul2 = 1000ULL << 32ULL;
	val -= 0x0030303030303030ULL;
	uint8_t spare{ static_cast<uint8_t>(val >> 48ull & 0xFF) };
	val = (val * 10ULL) + (val >> 8ULL);
	val = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return static_cast<uint32_t>(val) + spare;
}

uint64_t parse_digits_unrolled(const char* chars) {
	uint64_t b1	   = chars[0] - '0';
	uint64_t b2	   = chars[1] - '0';
	uint64_t b3	   = chars[2] - '0';
	uint64_t b4	   = chars[3] - '0';
	uint64_t b5	   = chars[4] - '0';
	uint64_t b6	   = chars[5] - '0';
	uint64_t b7	   = chars[6] - '0';
	uint64_t b8	   = chars[7] - '0';
	auto firstSum  = 1000000 * (10 * b1 + b2);
	auto thirdSum  = 100 * (10 * b5 + b6);
	auto secondSum = 10 * b7 + b8;
	auto fourthSum = 10000 * (10 * b3 + b4);
	return firstSum + secondSum + thirdSum + fourthSum;
}

uint64_t parse_digits_unrolled_seven_new_working(const char* chars) {
	std::uint64_t val = 0;
	std::memcpy(&val, chars, 7);
	constexpr uint64_t mask = 0x000000FF000000FF;
	constexpr uint64_t mul1 = 10ULL + (100000ULL << 32ULL);
	constexpr uint64_t mul2 = 1000ULL << 32ULL;
	val -= 0x0030303030303030ULL;
	uint8_t spare{ static_cast<uint8_t>(val >> 48 & 0xFF) };
	val = (val * 10ULL) + (val >> 8ULL);
	val = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return static_cast<uint32_t>(val) + spare;
}

uint64_t parse_digits_unrolled_six_new_working(const char* chars) {
	std::uint64_t val = 0;
	std::memcpy(&val, chars, 6);
	constexpr uint64_t mask = 0x000000FF000000FF;
	constexpr uint64_t mul1 = 1 + (10000ULL << 32ULL);
	constexpr uint64_t mul2 = 100ULL << 32ULL;
	val -= 0x0000303030303030ULL;
	val = (val * 10ULL) + (val >> 8ULL);
	val = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return static_cast<uint32_t>(val);
}

uint64_t parse_digits_unrolled_five_new_working(const char* chars) {
	std::uint64_t val = 0;
	std::memcpy(&val, chars, 5);
	constexpr uint64_t mask = 0x000000FF000000FF;
	constexpr uint64_t mul1 = (1000ULL << 32ULL);
	constexpr uint64_t mul2 = 10ULL << 32ULL;
	val -= 0x0000003030303030ULL;
	uint8_t spare{ static_cast<uint8_t>(val >> 32 & 0xFF) };
	val = (val * 10ULL) + (val >> 8ULL);
	val = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return static_cast<uint32_t>(val) + spare;
}

uint64_t parse_digits_unrolled_four_new_working(const char* chars) {
	std::uint64_t val = 0;
	std::memcpy(&val, chars, 4);
	constexpr uint64_t mask = 0x000000FF000000FF;
	constexpr uint64_t mul1 = (100ULL << 32ULL);
	constexpr uint64_t mul2 = 1ULL << 32ULL;
	val -= 0x0000000030303030ULL;
	val = (val * 10ULL) + (val >> 8ULL);
	val = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return static_cast<uint32_t>(val);
}

uint64_t parse_digits_unrolled_three_new_working(const char* chars) {
	std::uint64_t val = 0;
	std::memcpy(&val, chars, 3);
	constexpr uint64_t mask = 0x000000FF000000FF;
	constexpr uint64_t mul1 = (10ULL << 32ULL);
	constexpr uint64_t mul2 = 1ULL;
	val -= 0x0000000000303030ULL;
	uint8_t spare{ static_cast<uint8_t>(val >> 16 & 0xFF) };
	val = (val * 10ULL) + (val >> 8ULL);
	val = (((val & mask) * mul1) + (((val >> 16ULL) & mask) * mul2)) >> 32ULL;
	return static_cast<uint32_t>(val) + spare;
}

uint64_t parse_digits_unrolled_two_new_working(const char* chars) {
	std::uint64_t val = 0;
	std::memcpy(&val, chars, 4);
	constexpr uint64_t mask = 0x000000FF000000FF;
	constexpr uint64_t mul1 = (1ULL << 32ULL);
	val -= 0x0000000000003030ULL;
	val = (val * 10ULL) + (val >> 8ULL);
	val = ((val & mask) * mul1) >> 32ULL;
	return static_cast<uint32_t>(val);
}

uint64_t parse_digits_unrolled_one_new_working(const char* chars) {
	return chars[0] - '0';
}

static constexpr uint64_t pow10Table[] = { 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000 };

int main() {
	std::string newString{ "29810154832.915260" };
	//std::cout << "Parsed Value: " << parse_digits_unrolled_four_new_working(newString.data()) << std::endl;
	//std::cout << "Parsed Value: " << parse_digits_unrolled_three_new_working(newString.data()) << std::endl;
	//std::cout << "Parsed Value: " << parse_digits_unrolled_two_new_working(newString.data()) << std::endl;
	//std::cout << "Parsed Value: " << parse_digits_unrolled_one_new_working(newString.data()) << std::endl;

	//std::cout << "CURRENT HEX VALUE 100 + (100000ULL << 32): " << std::hex << 100 + (100000ULL << 32) << std::endl;
	//std::cout << "CURRENT HEX VALUE 10 + (1000000ULL << 32): " << std::hex << 10 + (1000000ULL << 32) << std::endl;
	//std::cout << "CURRENT HEX VALUE 10 + (100000ULL << 32): " << std::hex << 10 + (100000ULL << 32) << std::endl;
	//std::cout << "CURRENT HEX VALUE 1 + (1000ULL << 32): " << std::hex << 1 + (1000ULL << 32) << std::endl;
	//std::cout << "CURRENT HEX VALUE (10000ULL << 32): " << std::hex << (10000ULL << 32) << std::endl;
	//std::cout << "CURRENT HEX VALUE (1000ULL << 32): " << std::hex << (1000ULL << 32) << std::endl;
	runForLengthSerialize02<"constexpr-vs-consteval", "constexpr-vs-consteval">();
	std::cout << "CURRENT DIGITS COUNT: " << count_digit_bytes(*reinterpret_cast<uint64_t*>(newString.data())) << std::endl;
	runForLengthSerialize<1, 4, 4, 2, "Old-FastFloat-vs-New-FastFloat-1", "Old-FastFloat-vs-New-FastFloat-1">();
	runForLengthSerialize<8, 8, 8, 2, "Old-FastFloat-vs-New-FastFloat-8", "Old-FastFloat-vs-New-FastFloat-8">();
	runForLengthSerialize<64, 16, 16, 2, "Old-FastFloat-vs-New-FastFloat-64", "Old-FastFloat-vs-New-FastFloat-64">();
	runForLengthSerialize<512, 32, 32, 2, "Old-FastFloat-vs-New-FastFloat-512", "Old-FastFloat-vs-New-FastFloat-512">();
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