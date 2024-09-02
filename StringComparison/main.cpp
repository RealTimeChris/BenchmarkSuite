#include <iostream>
#include <array>
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Uint.hpp"
#include "Tests/Float.hpp"
#include "Tests/RoundTrip.hpp"
#include "Tests/Int.hpp"
#include <glaze/glaze.hpp>

static constexpr int64_t maxIterationCount{ 1024 };

std::string combineDigitParts(std::string intPart, std::string fracPart, std::string expPart) {
	if (fracPart != "0") {
		intPart += '.' + fracPart;
	}
	if (expPart.size() > 0) {
		intPart += 'e' + expPart;
	}
	return intPart;
}

template<typename value_type> std::string generateJsonInteger(size_t maxIntDigits, size_t maxFracDigits, size_t maxExpValue, bool negative) {
	static_assert(std::is_integral<value_type>::value, "value_type must be an integral type.");

	std::random_device rd;
	std::mt19937 gen(rd());

	auto generateDigits = [&](size_t numDigits) -> value_type {
		std::uniform_int_distribution<int> digitDist(0, 9);
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
		return result ;
	};

	auto generateExponent = [&](size_t maxExp) -> int {
		std::uniform_int_distribution<int> expDist(1, static_cast<int>(maxExp));
		return expDist(gen);
	};

	while (true) {
		try {
			size_t intDigits	   = std::uniform_int_distribution<size_t>(1, maxIntDigits)(gen);
			value_type integerPart = generateDigits(intDigits);

			if (integerPart == -1)
				continue;

			value_type fractionalPart = 0;
			if (maxFracDigits > 0) {
				size_t fracDigits = std::uniform_int_distribution<size_t>(1, maxFracDigits)(gen);
				fractionalPart	  = generateDigits(fracDigits);

				if (fractionalPart == -1)
					continue;
			}

			int exponent = generateExponent(maxExpValue);

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
					for (int i = 0; i < exponent; ++i) {
						if (result > (std::numeric_limits<value_type>::max() / 10)) {
							continue;
						}
						result *= 10;
					}
				} else {
					for (int i = 0; i < std::abs(exponent); ++i) {
						result /= 10;
					}
				}
			}

			std::ostringstream resultStream;
			resultStream << result;
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
	std::random_device rd;
	std::mt19937 gen(rd());
	value_type lowerBound = static_cast<value_type>(std::pow(10, minDigits - 1));
	value_type upperBound = static_cast<value_type>(std::pow(10, maxDigits)) - 1;
	std::uniform_int_distribution<value_type> dist(lowerBound, upperBound);
	return dist(gen);
}

template<typename value_type> std::string generateJsonNumber(int32_t maxIntDigits, int32_t maxFracDigits, int32_t maxExpValue, bool possiblyNegative = true) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<value_type> signDist(0, 1);
	bool negative{ possiblyNegative && signDist(gen) };
	if (maxFracDigits == 0 && maxExpValue == 0) {
		return std::to_string(possiblyNegative && signDist(gen) == 1 ? -1 * generateRandomUint64<value_type>(0, maxIntDigits) : generateRandomUint64<value_type>(0, maxIntDigits));
	} else {
		return generateJsonInteger<value_type>(maxIntDigits, maxFracDigits, maxExpValue, negative);
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

template<typename value_type, typename value_type02, jsonifier_internal::string_literal testStage, jsonifier_internal::string_literal testName>
JSONIFIER_ALWAYS_INLINE void runForLength(size_t intLength, size_t fracLength = 0, size_t maxExpValue = 0) {
	value_type value{};
	jsonifier_internal::integer_parser<value_type, const char> intParser{};
	std::vector<std::string> stringValues{};
	std::vector<uint64_t> resultValuesDig{};
	std::vector<value_type> valuesDig{};
	stringValues.resize(maxIterationCount);
	resultValuesDig.resize(maxIterationCount);
	valuesDig.resize(maxIterationCount);
	for (size_t x = 0; x < maxIterationCount; ++x) {
		stringValues[x]	 = generateJsonNumber<value_type>(intLength, fracLength, maxExpValue, jsonifier::concepts::signed_type<value_type> ? true : false);
		valuesDig[x]	 = strtoDigits<value_type02>(stringValues[x]);
		const auto* iter = stringValues[x].data();
		intParser.parseInt(value, iter);
		if (value != valuesDig[x]) {
			std::cout << "Jsonifier failed to parse: " << stringValues[x] << ", ";
			std::cout << "Jsonifier failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
		const auto* iterNew = stringValues[x].data();
		glz::detail::atoi(value, iterNew);
		if (value != valuesDig[x]) {
			std::cout << "Glaze failed to parse: " << stringValues[x] << ", ";
			std::cout << "Glaze failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Glaze-Parsing-Function",
		"dodgerblue">([=, &resultValuesDig, &value]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			const auto* iterNew = stringValues[x].data();
			glz::detail::atoi(value, iterNew);
			bnch_swt::doNotOptimizeAway(value);
			resultValuesDig[x] = value;
		}
	});
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName,
		"Jsonifier-Parsing-Function", "dodgerblue">([=, &resultValuesDig, &value]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			const auto* iterNew = stringValues[x].data();
			intParser.parseInt(value, iterNew);
			bnch_swt::doNotOptimizeAway(value);
			resultValuesDig[x] = value;
		}
	});
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
}

int32_t main() {
	uint_validation_tests::uintTests();
	int_validation_tests::intTests();
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:1-Digit">(1);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:4-Digit">(4);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:7-Digit">(7);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:10-Digit">(10);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:13-Digit">(13);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:16-Digit">(16);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:19-Digit">(19);
	runForLength<uint64_t, double, "Uint-Integer-Exponent-Tests", "Uint-Exponent:09-Digit">(9, 0, 5);
	runForLength<uint64_t, double, "Uint-Integer-Exponent-Tests", "Uint-Exponent:11-Digit">(11, 0, 4);
	runForLength<uint64_t, double, "Uint-Integer-Exponent-Tests", "Uint-Exponent:13-Digit">(13, 0, 3);
	runForLength<uint64_t, double, "Uint-Integer-Exponent-Tests", "Uint-Exponent:15-Digit">(15, 0, 1);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:1-Digit">(1);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:4-Digit">(4);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:7-Digit">(7);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:10-Digit">(10);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:13-Digit">(13);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:16-Digit">(16);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:19-Digit">(19);
	runForLength<int64_t, double, "Int-Integer-Exponent-Tests", "Int-Exponent:09-Digit">(9, 0, 5);
	runForLength<int64_t, double, "Int-Integer-Exponent-Tests", "Int-Exponent:11-Digit">(11, 0, 4);
	runForLength<int64_t, double, "Int-Integer-Exponent-Tests", "Int-Exponent:13-Digit">(13, 0, 3);
	runForLength<int64_t, double, "Int-Integer-Exponent-Tests", "Int-Exponent:15-Digit">(15, 0, 1);
	bnch_swt::benchmark_stage<"Uint-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Uint-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Uint-Integer-Exponent-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Exponent-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	return 0;
}