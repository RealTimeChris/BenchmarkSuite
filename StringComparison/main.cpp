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

static constexpr int64_t maxIterationCount{ 1024 };
struct test_struct02 {
	std::string testString{};
};

template<> struct jsonifier::core<test_struct02> {
	using value_type				 = test_struct02;
	static constexpr auto parseValue = createValue<&value_type::testString>();
};

double combineDigitParts(std::string intPart, std::string fracPart, std::string expPart) {
	if (fracPart != "0") {
		intPart += '.' + fracPart;
	}
	if (expPart.size() > 0) {
		intPart += 'e' + expPart;
	}
	auto endPtr = intPart.data() + intPart.size();
	return std::strtod(intPart.data(), &endPtr);
}

template<typename value_type> double generateJsonInteger(size_t maxIntDigits, size_t maxFracDigits, size_t maxExpValue, bool negative) {
	static_assert(std::is_integral<value_type>::value, "value_type must be an integral type.");

	std::random_device rd;
	std::mt19937 gen(rd());

	auto generateDigits = [&](size_t numDigits) -> value_type {
		std::uniform_int_distribution<int64_t> digitDist(0, 9);
		value_type result = 0;
		for (size_t i = 0; i < numDigits; ++i) {
			value_type digit = digitDist(gen);
			if (result > (std::numeric_limits<value_type>::max() / 10)) {
				return -1;
			}
			result = result * 10 + digit;
			if (result < 0) {
				return -1;
			}
		}
		return result;
	};

	auto generateExponent = [&](size_t maxExp) -> int64_t {
		auto newValue = std::abs(static_cast<int64_t>(maxExp));
		return newValue;
	};

	while (true) {
		try {
			size_t intDigits	   = std::uniform_int_distribution<size_t>(maxIntDigits, maxIntDigits)(gen);
			value_type integerPart = generateDigits(intDigits);

			if (integerPart == -1)
				continue;

			value_type fractionalPart = 0;
			if (maxFracDigits > 0) {
				size_t fracDigits = std::uniform_int_distribution<size_t>(maxFracDigits, maxFracDigits)(gen);
				fractionalPart	  = generateDigits(fracDigits);

				if (fractionalPart == -1)
					continue;
			}

			int64_t exponent = generateExponent(maxExpValue);

			value_type result = integerPart;

			if (fractionalPart > 0) {
				size_t fracDigits = std::to_string(fractionalPart).size();
				value_type scale  = std::pow(10, fracDigits);

				if (result > (std::numeric_limits<value_type>::max() / scale)) {
					continue;
				}

				result = result * scale + fractionalPart;
			}
			result *= negative ? -1 : 1;

			if (exponent != 0) {
				if (exponent > 0) {
					for (int64_t i = 0; i < exponent; ++i) {
						if (result > (std::numeric_limits<value_type>::max() / 10)) {
							continue;
						}
						result *= 10;
					}
				} else {
					for (int64_t i = 0; i < std::abs(exponent); ++i) {
						result /= 10;
					}
				}
			}
			return combineDigitParts(std::to_string(integerPart), std::to_string(fractionalPart), std::to_string(exponent));

		} catch (...) {
			continue;
		}
	}
}

template<typename value_type> value_type generateRandomUint64(int64_t minDigits, int64_t maxDigits) {
	if (maxDigits < minDigits || maxDigits > 20) {
		throw std::invalid_argument("Digits must be between 1 and 20, and minDigits must be <= maxDigits.");
	}
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	value_type lowerBound = static_cast<value_type>(static_cast<int64_t>(std::pow(10, minDigits - 1)));
	value_type upperBound = static_cast<value_type>(static_cast<int64_t>(std::pow(10, maxDigits - 2)));
	std::uniform_int_distribution<value_type> dist{ lowerBound, upperBound };
	return dist(gen);
}

template<typename value_type> std::string generateJsonNumber(int64_t maxIntDigits, int64_t maxFracDigits, int64_t maxExpValue, bool possiblyNegative = true) {
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::uniform_int_distribution<value_type> signDist{ 0, 1 };
	bool negative{ possiblyNegative && signDist(gen) };
	if (maxFracDigits == 0 && maxExpValue == 0) {
		return std::to_string(negative ? -1 * generateRandomUint64<value_type>(0, maxIntDigits) : generateRandomUint64<value_type>(0, maxIntDigits));
	} else {
		return std::to_string(negative ? -1 * generateJsonInteger<value_type>(maxIntDigits, maxFracDigits, maxExpValue, negative)
									   : generateJsonInteger<value_type>(maxIntDigits, maxFracDigits, maxExpValue, negative));
	}
}

template<std::floating_point value_type> auto strtoDigits(const std::string& value) {
	return std::stod(value);
}

template<std::signed_integral value_type> auto strtoDigits(const std::string& value) {
	return std::stoll(value);
}

template<std::unsigned_integral value_type> auto strtoDigits(const std::string& value) {
	return std::stoull(value);
}

template<typename value_type, typename value_type02, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLength01(size_t intLength, size_t fracLength = 0, size_t maxExpValue = 0) {
	const std::string oldNewFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "/CitmCatalogData-Prettified.json");
	auto newFile{ oldNewFile };
	newFile.reserve(oldNewFile.size() + simdjson::SIMDJSON_PADDING);
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	value_type value{};
	std::vector<std::string> stringValues{};
	std::vector<value_type> resultValuesDig{};
	std::vector<value_type> valuesDig{};
	stringValues.resize(maxIterationCount);
	resultValuesDig.resize(maxIterationCount);
	valuesDig.resize(maxIterationCount);
	for (size_t x = 0; x < maxIterationCount; ++x) {
		stringValues[x]	 = generateJsonNumber<value_type>(intLength, fracLength, maxExpValue, jsonifier::concepts::signed_type<value_type> ? true : false);
		valuesDig[x]	 = strtoDigits<value_type02>(stringValues[x]);
		const auto* iter = stringValues[x].data();
		jsonifier_internal::integer_parser<value_type>::parseInt(value, iter);
		if (value != valuesDig[x]) {
			std::cout << "Jsonifier failed to parse: " << stringValues[x] << ", ";
			std::cout << "Jsonifier failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
		const auto* iterNew = stringValues[x].data();
		value				= 0;
		glz::detail::atoi(value, iterNew);
		if (value != valuesDig[x]) {
			std::cout << "Glaze failed to parse: " << stringValues[x] << ", ";
			std::cout << "Glaze failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	std::string buffer{};
	int64_t index{};
	buffer.resize(1024 * 1024 * 16);
	simdjson::ondemand::parser parserNew01{};
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Glaze-Function", "dodgerblue">(
		[=, &resultValuesDig]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				const char* iter = stringValues[x].data();
				auto result = glz::detail::atoi(resultValuesDig[x], iter);
				bnch_swt::doNotOptimizeAway(result);
			}
		});
	for (size_t x = 0; x < 1024; ++x) {
		if (resultValuesDig[x] != valuesDig[x]) {
			std::cout << "Glaze failed to parse: " << stringValues[x] << ", ";
			std::cout << "Glaze failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
	value = 0;
	index = 0;
	simdjson::ondemand::parser parserNew02{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Jsonifier-Function", "dodgerblue">(
		[=, &resultValuesDig]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				const char* iter = stringValues[x].data();
				auto result = jsonifier_internal::integer_parser<value_type>::parseInt(resultValuesDig[x], iter);
				bnch_swt::doNotOptimizeAway(result);
			}
		});
	for (size_t x = 0; x < 1024; ++x) {
		if (resultValuesDig[x] != valuesDig[x]) {
			std::cout << "Jsonifier failed to parse: " << stringValues[x] << ", ";
			std::cout << "Jsonifier failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
}

template<size_t length> constexpr auto generateArrayOfChars() {
	jsonifier_internal::xoshiro256 rng{};
	std::array<char, length> returnValues{};
	for (size_t x = 0; x < length; ++x) {
		returnValues[x] = static_cast<char>((rng() % 95) + 32);
	}
	return returnValues;
}

template<size_t length> constexpr auto generateStringLiteral() {
	constexpr auto newArray{ generateArrayOfChars<length>() };
	jsonifier_internal::string_literal<length, char> stringLiteral{};
	std::copy(newArray.begin(), newArray.end(), stringLiteral.data());
	return stringLiteral;
}

template<size_t length, typename value_type, typename value_type02, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLength() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::string newString{};
	for (size_t x = 0; x < length; ++x) {
		if (static_cast<char>(x) == 127) {
			continue;
		} else {
			newString.push_back(static_cast<char>(x));
		}
	}
	newString[newString.size() - 1] = 127;
	bool result{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "jsonifier_internal::memchar", "dodgerblue">(
		[=, &result]() mutable {
			for (size_t x = 0; x < 1024 ; ++x) {
				result = jsonifier_internal::char_comparison<127>::memchar(newString.data(), newString.size());
				bnch_swt::doNotOptimizeAway(result);
			}
		});
	std::cout << "CURRENT RESULT: " << result << std::endl;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "std::memchr", "dodgerblue">(
		[=, &result]() mutable {
			for (size_t x = 0; x < 1024 ; ++x) {
				result = std::memchr(newString.data(), 127, newString.size());
				bnch_swt::doNotOptimizeAway(result);
			}
		});
	std::cout << "CURRENT RESULT: " << result << std::endl;
}

uint16_t packBitsFromNibbles(uint64_t nibbles) {
	return static_cast<uint16_t>((nibbles & 0x01) | ((nibbles & 0x10) >> 3) | ((nibbles & 0x100) >> 6) | ((nibbles & 0x1000) >> 9) | ((nibbles & 0x10000) >> 12) |
		((nibbles & 0x100000) >> 15) | ((nibbles & 0x1000000) >> 18) | ((nibbles & 0x10000000) >> 21) | ((nibbles & 0x100000000) >> 24) | ((nibbles & 0x1000000000) >> 27) |
		((nibbles & 0x10000000000) >> 30) | ((nibbles & 0x100000000000) >> 33) | ((nibbles & 0x1000000000000) >> 36) | ((nibbles & 0x10000000000000) >> 39) |
		((nibbles & 0x100000000000000) >> 42) | ((nibbles & 0x1000000000000000) >> 45));
}

#if defined(JSONIFIER_WIN) || defined(JSONIFIER_MAC)
jsonifier_simd_int_128 mask01{ 0xFF, 0xFF, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0, 0 };
#elif defined(JSONIFIER_LINUX)
jsonifier_simd_int_128 mask01{ 0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00 };
#endif

#if defined(JSONIFIER_WIN) || defined(JSONIFIER_MAC)
jsonifier_simd_int_128 mask02{ 0, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0xFF };
#elif defined(JSONIFIER_LINUX)
jsonifier_simd_int_128 mask02{ 0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00, 0xFF00FF00FF00FF00 };
#endif

int32_t main() {
	std::cout << "CURRENT BITS: " << std::bitset<64>{ simd_internal::opCmpEq(mask01, mask02) } << std::endl;
	std::cout << "TZCNT: " << simd_internal::postCmpTzcnt(simd_internal::opCmpEq(mask01, mask02)) << std::endl;
	/*
	uint_validation_tests::uintTests();
	int_validation_tests::intTests();
	static constexpr jsonifier_internal::string_literal newStringLiteral{ "max_id_str22222" };
	std::string newString01{ "id_str" };
	//runForLength01<uint64_t, uint64_t, "Stage-1 Parse Test", "Uint:1-Digit">(1);
	//bnch_swt::benchmark_stage<"Stage-1 Parse Test", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	auto newFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "/TwitterData-Minified.json");
	jsonifier::jsonifier_core parser{};
	twitter_message dataNew{};
	uint32_t newValue01;
	uint32_t newValue02;
	newValue01 |= 0b1111111111111111;
	newValue02 |= 0b1111111111111111;
	//parser.parseJson<jsonifier::parse_options{ .minified = true }>(dataNew, newFile);
	for (auto& value: parser.getErrors()) {
		std::cout << "Error: " << value << std::endl;
	}
	std::cout << "CURRENT DATA-01: " << dataNew.search_metadata.max_id_str << std::endl; 
	std::cout << "CURRENT DATA-01: " << jsonifier_internal::string_literal_comparitor<newStringLiteral>::impl(newString01.data()) << std::endl;
	newFile.clear();
	//parser.serializeJson<jsonifier::serialize_options{ .prettify = true }>(dataNew, newFile);
	//std::cout << "CURRENT DATA: " << newFile << std::endl;
	std::cout << "Error: " << std::bitset<32>{ newValue01 } << std::endl;
	for (auto& value: parser.getErrors()) {
		std::cout << "Error: " << value << std::endl;
	}
	runForLength01<uint64_t, uint64_t, "unsigned-integer-test", "1-Char">(1, 0, 0);
	runForLength01<uint64_t, uint64_t, "unsigned-integer-test", "4-Char">(4, 0, 0);
	runForLength01<uint64_t, uint64_t, "unsigned-integer-test", "7-Char">(7, 0, 0);
	runForLength01<uint64_t, uint64_t, "unsigned-integer-test", "10-Char">(10, 0, 0);
	runForLength01<uint64_t, uint64_t, "unsigned-integer-test", "13-Char">(13, 0, 0);
	runForLength01<uint64_t, uint64_t, "unsigned-integer-test", "16-Char">(16, 0, 0);
	runForLength01<uint64_t, uint64_t, "unsigned-integer-test", "19-Char">(19, 0, 0);
	runForLength01<int64_t, int64_t, "signed-integer-test", "1-Char">(1, 0, 0);
	runForLength01<int64_t, int64_t, "signed-integer-test", "4-Char">(4, 0, 0);
	runForLength01<int64_t, int64_t, "signed-integer-test", "7-Char">(7, 0, 0);
	runForLength01<int64_t, int64_t, "signed-integer-test", "10-Char">(10, 0, 0);
	runForLength01<int64_t, int64_t, "signed-integer-test", "13-Char">(13, 0, 0);
	runForLength01<int64_t, int64_t, "signed-integer-test", "16-Char">(16, 0, 0);
	runForLength01<int64_t, int64_t, "signed-integer-test", "19-Char">(19, 0, 0);
	bnch_swt::benchmark_stage<"unsigned-integer-test", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"signed-integer-test", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();*/
	runForLength<1, uint64_t, uint64_t, "comparison-test", "1-Char">();
	runForLength<2, uint64_t, uint64_t, "comparison-test", "2-Char">();
	runForLength<3, uint64_t, uint64_t, "comparison-test", "3-Char">();
	runForLength<4, uint64_t, uint64_t, "comparison-test", "4-Char">();
	runForLength<5, uint64_t, uint64_t, "comparison-test", "5-Char">();
	runForLength<6, uint64_t, uint64_t, "comparison-test", "6-Char">();
	runForLength<7, uint64_t, uint64_t, "comparison-test", "7-Char">();
	runForLength<8, uint64_t, uint64_t, "comparison-test", "8-Char">();
	runForLength<9, uint64_t, uint64_t, "comparison-test", "9-Char">();
	runForLength<10, uint64_t, uint64_t, "comparison-test", "10-Char">();
	runForLength<11, uint64_t, uint64_t, "comparison-test", "11-Char">();
	runForLength<12, uint64_t, uint64_t, "comparison-test", "12-Char">();
	runForLength<13, uint64_t, uint64_t, "comparison-test", "13-Char">();
	runForLength<13, uint64_t, uint64_t, "comparison-test", "14-Char">();
	runForLength<15, uint64_t, uint64_t, "comparison-test", "15-Char">();
	runForLength<16, uint64_t, uint64_t, "comparison-test", "16-Char">();
	runForLength<32, int64_t, int64_t, "comparison-test", "32-Char">();
	runForLength<64, int64_t, int64_t, "comparison-test", "64-Char">();
	bnch_swt::benchmark_stage<"comparison-test", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	return 0;
}