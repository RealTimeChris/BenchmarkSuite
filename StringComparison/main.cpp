#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Jsonifier.hpp"
#include "FastFloatNew.hpp"
#include "StrToDNew.hpp"

template<typename value_type, typename char_type> struct float_parser {
	static constexpr char_type decimalNew = '.';
	static constexpr char_type smallE	  = 'e';
	static constexpr char_type bigE		  = 'E';
	static constexpr char_type minusNew	  = '-';
	static constexpr char_type plusNew	  = '+';
	static constexpr char_type zeroNew	  = '0';

	struct parsing_pack {
		fast_float_new::parsed_number_string_t<char_type> answer;
		char_type const* fracPtr;
		char_type const* fracEnd; 
	};

	JSONIFIER_ALWAYS_INLINE static char_type const* parseFloatImpl(char_type const* iter, value_type& value) {
		using namespace fast_float_new;
		static_assert(is_supported_float_t<value_type>(), "only some floating-point types are supported");
		static_assert(is_supported_char_t<char_type>(), "only char, wchar_t, char16_t and char32_t are supported");
		parsing_pack pack;
		pack.answer.valid			= false;
		pack.answer.too_many_digits = false;
		pack.answer.negative		= (*iter == minusNew);
		if (pack.answer.negative) {
			++iter;

			if JSONIFIER_UNLIKELY (!is_integer(*iter)) {
				return nullptr;
			}
		}
		char_type const* const start_digits = iter;
		uint8_t digit;

		while (is_integer(*iter)) {
			digit = static_cast<uint64_t>(*iter - zeroNew);
			++iter;
			pack.answer.mantissa = 10 * pack.answer.mantissa + digit;
		}

		char_type const* const intEnd = iter;
		int64_t digit_count			  = static_cast<int64_t>(intEnd - start_digits);
		auto intPtr					  = start_digits;

		if (digit_count == 0 || (start_digits[0] == zeroNew && digit_count > 1)) {
			return nullptr;
		}

		const bool has_decimal_point = [&] {
			return (*iter == decimalNew);
		}();
		if (has_decimal_point) {
			++iter;
			char_type const* before = iter;

			while (is_integer(*iter)) {
				digit = static_cast<uint8_t>(*iter - zeroNew);
				++iter;
				pack.answer.mantissa = pack.answer.mantissa * 10 + digit;
			}
			pack.answer.exponent = before - iter;
			pack.fracEnd	 = iter;
			pack.fracPtr	 = before;
			digit_count -= pack.answer.exponent;
		}

		if (has_decimal_point && pack.answer.exponent == 0) {
			return nullptr;
		}

		int64_t exp_number = 0;

		if ((smallE == *iter) || (bigE == *iter)) {
			char_type const* location_of_e = iter;
			++iter;
			bool neg_exp = false;
			if (minusNew == *iter) {
				neg_exp = true;
				++iter;
			} else if (plusNew == *iter) {
				++iter;
			}
			if (!is_integer(*iter)) {
				iter = location_of_e;
			} else {
				while (is_integer(*iter)) {
					digit = static_cast<uint8_t>(*iter - zeroNew);
					if (exp_number < 0x10000000) {
						exp_number = 10 * exp_number + digit;
					}
					++iter;
				}
				if (neg_exp) {
					exp_number = -exp_number;
				}
				pack.answer.exponent += exp_number;
			}
		}

		pack.answer.lastmatch = iter;
		pack.answer.valid	  = true;

		if (digit_count > 19) {
			char_type const* start = start_digits;
			while ((*start == zeroNew || *start == decimalNew)) {
				if (*start == zeroNew) {
					--digit_count;
				}
				++start;
			}

			if (digit_count > 19) {
				pack.answer.too_many_digits = true;
				pack.answer.mantissa						= 0;
				static constexpr uint64_t minimal_nineteen_digit_integer{ 1000000000000000000 };
				while ((pack.answer.mantissa < minimal_nineteen_digit_integer) && (intPtr != intEnd)) {
					pack.answer.mantissa = pack.answer.mantissa * 10 + static_cast<uint64_t>(*intPtr - zeroNew);
					++intPtr;
				}
				if (pack.answer.mantissa >= minimal_nineteen_digit_integer) {
					pack.answer.exponent = intEnd - intPtr + exp_number;
				} else {
					intPtr = pack.fracPtr;
					while ((pack.answer.mantissa < minimal_nineteen_digit_integer) && (intPtr != pack.fracEnd)) {
						pack.answer.mantissa = pack.answer.mantissa * 10 + static_cast<uint64_t>(*intPtr - zeroNew);
						++intPtr;
					}
					pack.answer.exponent = pack.fracPtr - intPtr + exp_number;
				}
			}
		}
		if JSONIFIER_LIKELY (pack.answer.valid) {
			return from_chars_advanced(pack.answer, value).ptr;
		} else {
			return nullptr;
		}
	}

	JSONIFIER_ALWAYS_INLINE static bool parseFloat(char_type const*& iter, value_type& value) noexcept {
		using namespace fast_float_new;

		return parseFloatImpl(iter, value);
	}
};

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

template<size_t maxIndex, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew> JSONIFIER_ALWAYS_INLINE void runForLengthSerialize() {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	std::vector<std::string> newDoubles{};
	for (size_t x = 0; x < maxIndex; ++x) {
		std::string newValue{ std::to_string(test_generator::generateValue<double>()) };
		double newDouble{};
		const auto* iter = newValue.data();
		const auto* end	 = newValue.data() + newValue.size();
		jsonifier_internal::parseFloat(iter, end, newDouble);
		if (newDouble != 0.0 && newDouble != -0.0f) {
			newDoubles.emplace_back(newValue);
		} else {
			--x;
			continue;
		}
	}
	std::vector<double> newerDoubles01{};
	std::vector<double> newerDoubles02{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "old-parse", "dodgerblue">([&]() mutable {
		for (size_t y = 0; y < maxIndex; ++y) {
			const auto* iter = newDoubles[y].data();
			const auto* end	 = newDoubles[y].data() + newDoubles[y].size();
			double newDouble{};
			jsonifier_internal::parseFloat(iter, end, newDouble);
			newerDoubles01.emplace_back(newDouble);
			bnch_swt::doNotOptimizeAway(newDouble);
		}
	});
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "new-parse", "dodgerblue">([&]() mutable {
		for (size_t y = 0; y < maxIndex; ++y) {
			const auto* iter = newDoubles[y].data();
			double newDouble{};
			float_parser<double, char>::parseFloat(iter, newDouble);
			newerDoubles02.emplace_back(newDouble);
			bnch_swt::doNotOptimizeAway(newDouble);
		}
	});
	for (size_t x = 0; x < maxIndex; ++x) {
		if (newerDoubles01[x] != newerDoubles02[x]) {
			std::cout << "FAILED TO PARSE AT INDEX: " << x << std::endl;
			std::cout << "Input Value: " << newDoubles[x] << std::endl;
			std::cout << "Intended Value: " << newerDoubles01[x] << std::endl;
			std::cout << "Actual Value: " << newerDoubles02[x] << std::endl;
		} else {
			std::cout << "Here's the value: " << newerDoubles01[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int main() {
	runForLengthSerialize<1, "Old-FastFloat-vs-New-FastFloat-1", "Old-FastFloat-vs-New-FastFloat-1">();
	runForLengthSerialize<2, "Old-FastFloat-vs-New-FastFloat-2", "Old-FastFloat-vs-New-FastFloat-2">();
	runForLengthSerialize<4, "Old-FastFloat-vs-New-FastFloat-4", "Old-FastFloat-vs-New-FastFloat-4">();
	runForLengthSerialize<8, "Old-FastFloat-vs-New-FastFloat-8", "Old-FastFloat-vs-New-FastFloat-8">();
	runForLengthSerialize<16, "Old-FastFloat-vs-New-FastFloat-16", "Old-FastFloat-vs-New-FastFloat-16">();
	runForLengthSerialize<32, "Old-FastFloat-vs-New-FastFloat-32", "Old-FastFloat-vs-New-FastFloat-32">();
	return 0;
}