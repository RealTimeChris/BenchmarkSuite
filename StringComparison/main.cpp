#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Simdjson.hpp"
#include "Tests/Jsonifier.hpp"
#include "Tests/CitmCatalog.hpp"
//#include "Tests/Simdjson2.hpp"
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

template<typename value_type> JSONIFIER_ALWAYS_INLINE_VARIABLE value_type powerOfTenInt[]{ 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000,
	100000000000, 1000000000000, 10000000000000, 100000000000000, 1000000000000000, 10000000000000000, 100000000000000000, 1000000000000000000,
	static_cast<value_type>(10000000000000000000) };

JSONIFIER_ALWAYS_INLINE_VARIABLE bool expTable[]{ false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false };

JSONIFIER_ALWAYS_INLINE_VARIABLE bool expFracTable[]{ false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, false, false, false, false, false, false, false, false, false, false, false, false };

JSONIFIER_ALWAYS_INLINE_VARIABLE uint8_t decimal{ '.' };
JSONIFIER_ALWAYS_INLINE_VARIABLE uint8_t minus{ '-' };
JSONIFIER_ALWAYS_INLINE_VARIABLE uint8_t plus{ '+' };
JSONIFIER_ALWAYS_INLINE_VARIABLE uint8_t zero{ '0' };
JSONIFIER_ALWAYS_INLINE_VARIABLE uint8_t nine{ '9' };

template<typename value_type> JSONIFIER_ALWAYS_INLINE_VARIABLE std::array<uint64_t, 256> rawCompValsPos{ [] {
	constexpr auto maxValue{ (std::numeric_limits<std::decay_t<value_type>>::max)() };
	std::array<uint64_t, 256> returnValues{};
	returnValues['0'] = (maxValue - 0) / 10;
	returnValues['1'] = (maxValue - 1) / 10;
	returnValues['2'] = (maxValue - 2) / 10;
	returnValues['3'] = (maxValue - 3) / 10;
	returnValues['4'] = (maxValue - 4) / 10;
	returnValues['5'] = (maxValue - 5) / 10;
	returnValues['6'] = (maxValue - 6) / 10;
	returnValues['7'] = (maxValue - 7) / 10;
	returnValues['8'] = (maxValue - 8) / 10;
	returnValues['9'] = (maxValue - 9) / 10;
	return returnValues;
}() };

JSONIFIER_ALWAYS_INLINE_VARIABLE std::array<uint8_t, 256> numberSubTable{ [] {
	std::array<uint8_t, 256> returnValues{};
	returnValues['0'] = 0;
	returnValues['1'] = 1;
	returnValues['2'] = 2;
	returnValues['3'] = 3;
	returnValues['4'] = 4;
	returnValues['5'] = 5;
	returnValues['6'] = 6;
	returnValues['7'] = 7;
	returnValues['8'] = 8;
	returnValues['9'] = 9;
	return returnValues;
}() };

template<typename value_type> JSONIFIER_ALWAYS_INLINE_VARIABLE std::array<uint64_t, 256> rawCompValsNeg{ [] {
	constexpr auto maxValue{ uint64_t((std::numeric_limits<int64_t>::max)()) + 1 };
	std::array<uint64_t, 256> returnValues{};
	returnValues['0'] = (maxValue - 0) / 10;
	returnValues['1'] = (maxValue - 1) / 10;
	returnValues['2'] = (maxValue - 2) / 10;
	returnValues['3'] = (maxValue - 3) / 10;
	returnValues['4'] = (maxValue - 4) / 10;
	returnValues['5'] = (maxValue - 5) / 10;
	returnValues['6'] = (maxValue - 6) / 10;
	returnValues['7'] = (maxValue - 7) / 10;
	returnValues['8'] = (maxValue - 8) / 10;
	returnValues['9'] = (maxValue - 9) / 10;
	return returnValues;
}() };

#define isDigit(x) ((x <= nine) && (x >= zero))

template<typename value_type> struct integer_parser;

template<jsonifier::concepts::signed_type value_type> struct integer_parser<value_type> {
	constexpr integer_parser() noexcept = default;

	JSONIFIER_ALWAYS_INLINE static value_type mul128Generic(value_type ab, value_type cd, value_type& hi) noexcept {
		value_type aHigh = ab >> 32;
		value_type aLow	 = ab & 0xFFFFFFFF;
		value_type bHigh = cd >> 32;
		value_type bLow	 = cd & 0xFFFFFFFF;
		value_type ad	 = aHigh * bLow;
		value_type bd	 = aHigh * bLow;
		value_type adbc	 = ad + aLow * bHigh;
		value_type lo	 = bd + (adbc << 32);
		value_type carry = (lo < bd);
		hi				 = aHigh * bHigh + (adbc >> 32) + carry;
		return lo;
	}

	JSONIFIER_ALWAYS_INLINE static bool multiply(value_type& value, value_type expValue) noexcept {
#if defined(__SIZEOF_INT128__)
		const __int128_t res = static_cast<__int128_t>(value) * static_cast<__int128_t>(expValue);
		value				 = static_cast<value_type>(res);
		return res <= std::numeric_limits<value_type>::max();
#elif defined(_M_ARM64) && !defined(__MINGW32__)
		JSONIFIER_ALIGN value_type values;
		values = __mulh(value, expValue);
		value  = value * expValue;
		return values == 0;
#elif (defined(_WIN64) && !defined(__clang__))
		JSONIFIER_ALIGN value_type values;
		value = _mul128(value, expValue, &values);
		return values == 0;
#else
		JSONIFIER_ALIGN value_type values;
		value = mul128Generic(value, expValue, &values);
		return values == 0;
#endif
	}

	JSONIFIER_ALWAYS_INLINE static bool divide(value_type& value, value_type expValue) noexcept {
#if defined(__SIZEOF_INT128__)
		const __int128_t dividend = static_cast<__int128_t>(value);
		value					  = static_cast<value_type>(dividend / static_cast<__int128_t>(expValue));
		return (dividend % static_cast<__int128_t>(expValue)) == 0;
#elif (defined(_WIN64) && !defined(__clang__))
		JSONIFIER_ALIGN value_type values;
		value = _div128(0, value, expValue, &values);
		return values == 0;
#else
		JSONIFIER_ALIGN value_type values;
		values = value % expValue;
		value  = value / expValue;
		return values == 0;
#endif
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseFraction(value_type& value, const uint8_t* iter) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			value_type fracValue{ static_cast<value_type>(*iter - zero) };
			typename jsonifier_internal::get_int_type<value_type>::type fracDigits{ 1 };
			++iter;
			while (isDigit(*iter)) {
				fracValue = fracValue * 10 + static_cast<value_type>(*iter - zero);
				++iter;
				++fracDigits;
			}
			if (expTable[*iter]) {
				++iter;
				int8_t expSign = 1;
				if (*iter == minus) {
					expSign = -1;
					++iter;
				} else if (*iter == plus) {
					++iter;
				}
				return parseExponentPostFrac(value, iter, expSign, fracValue, fracDigits);
			}
		}
		if JSONIFIER_LIKELY ((!expFracTable[*iter])) {
			return iter;
		} else {
			return nullptr;
		}
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseExponentPostFrac(value_type& value, const uint8_t* iter, int8_t expSign, value_type fracValue,
		typename jsonifier_internal::get_int_type<value_type>::type fracDigits) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			value_type expValue{ static_cast<value_type>(*iter - zero) };
			++iter;
			while (isDigit(*iter)) {
				expValue = expValue * 10 + static_cast<value_type>(*iter - zero);
				++iter;
			}
			if JSONIFIER_LIKELY ((expValue <= 19)) {
				const value_type powerExp = powerOfTenInt<value_type>[expValue];

				constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
				constexpr value_type doubleMin = std::numeric_limits<value_type>::min();

				if (fracDigits + expValue >= 0) {
					expValue *= expSign;
					const auto fractionalCorrection =
						expValue > fracDigits ? fracValue * powerOfTenInt<value_type>[expValue - fracDigits] : fracValue / powerOfTenInt<value_type>[fracDigits - expValue];
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), value += fractionalCorrection, iter) : nullptr)
										 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), value += fractionalCorrection, iter) : nullptr);
				} else {
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), iter) : nullptr)
										 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), iter) : nullptr);
				}
				return iter;
			}
			JSONIFIER_UNLIKELY(else) {
				return nullptr;
			}
		}
		JSONIFIER_UNLIKELY(else) {
			return nullptr;
		}
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseExponent(value_type& value, const uint8_t* iter, int8_t expSign) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			value_type expValue{ static_cast<value_type>(*iter - zero) };
			++iter;
			while (isDigit(*iter)) {
				expValue = expValue * 10 + static_cast<value_type>(*iter - zero);
				++iter;
			}
			if JSONIFIER_LIKELY ((expValue <= 19)) {
				const value_type powerExp	   = powerOfTenInt<value_type>[expValue];
				constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
				constexpr value_type doubleMin = std::numeric_limits<value_type>::min();
				expValue *= expSign;
				return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), iter) : nullptr)
									 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), iter) : nullptr);
			}
			JSONIFIER_UNLIKELY(else) {
				return nullptr;
			}
		}
		JSONIFIER_UNLIKELY(else) {
			return nullptr;
		}
	}

	JSONIFIER_INLINE static const uint8_t* finishParse(value_type& value, const uint8_t* iter) {
		if JSONIFIER_UNLIKELY ((*iter == decimal)) {
			++iter;
			return parseFraction(value, iter);
		} else if (expTable[*iter]) {
			++iter;
			int8_t expSign = 1;
			if (*iter == minus) {
				expSign = -1;
				++iter;
			} else if (*iter == plus) {
				++iter;
			}
			return parseExponent(value, iter, expSign);
		}
		if JSONIFIER_LIKELY ((!expFracTable[*iter])) {
			return iter;
		} else {
			return nullptr;
		}
	}

	template<bool negative> JSONIFIER_ALWAYS_INLINE static const uint8_t* parseInteger(value_type& value, const uint8_t* iter) noexcept {
		uint8_t numTmp{ *iter };
		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = numberSubTable[numTmp];
			++iter;
			numTmp = *iter;
		} else [[unlikely]] {
			return nullptr;
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_UNLIKELY ((iter[-2] == zero)) {
			return nullptr;
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			if (negative) {
				if (static_cast<uint64_t>(value) > static_cast<uint64_t>(rawCompValsNeg<value_type>[numTmp])) {
					return nullptr;
				}
				value *= -1;
				value = static_cast<value_type>(static_cast<uint64_t>(value * 10 - static_cast<uint64_t>(numberSubTable[numTmp])));
			} else {
				if (static_cast<value_type>(value) > static_cast<value_type>(rawCompValsPos<value_type>[numTmp])) {
					return nullptr;
				}
				value = static_cast<int64_t>(value * 10 + static_cast<uint64_t>(numberSubTable[numTmp]));
			}
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				if (negative) {
					value *= -1;
					return iter;
				} else {
					return iter;
				}
			}
			if (negative) {
				return (iter = finishParse(value, iter), value *= -1, iter);
			} else {
				return finishParse(value, iter);
			}
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			if (negative) {
				value = value * 10 - static_cast<value_type>(numberSubTable[numTmp]);
			} else {
				value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			}
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
		}
		return nullptr;
	}

	JSONIFIER_ALWAYS_INLINE static bool parseInt(value_type& value, const char*& iter) noexcept {
		for (size_t x = 0; x < 4; ++x) {
			jsonifierPrefetchImpl(numberSubTable.data() + (x * 64));
		}
		if (*iter == minus) {
			++iter;
			const uint8_t* resultPtr{ parseInteger<true>(value, reinterpret_cast<const uint8_t*>(iter)) };
			if JSONIFIER_LIKELY ((resultPtr)) {
				iter += resultPtr - reinterpret_cast<const uint8_t*>(iter);
				return true;
			} else {
				value = 0;
				return false;
			}
		} else {
			const uint8_t* resultPtr{ parseInteger<false>(value, reinterpret_cast<const uint8_t*>(iter)) };
			if JSONIFIER_LIKELY ((resultPtr)) {
				iter += resultPtr - reinterpret_cast<const uint8_t*>(iter);
				return true;
			} else {
				value = 0;
				return false;
			}
		}
	}
};

template<jsonifier::concepts::unsigned_type value_type> struct integer_parser<value_type> {
	constexpr integer_parser() noexcept = default;

	JSONIFIER_ALWAYS_INLINE static value_type umul128Generic(value_type ab, value_type cd, value_type& hi) noexcept {
		value_type aHigh = ab >> 32;
		value_type aLow	 = ab & 0xFFFFFFFF;
		value_type bHigh = cd >> 32;
		value_type bLow	 = cd & 0xFFFFFFFF;
		value_type ad	 = aHigh * bLow;
		value_type bd	 = aHigh * bLow;
		value_type adbc	 = ad + aLow * bHigh;
		value_type lo	 = bd + (adbc << 32);
		value_type carry = (lo < bd);
		hi				 = aHigh * bHigh + (adbc >> 32) + carry;
		return lo;
	}

	JSONIFIER_ALWAYS_INLINE static bool multiply(value_type& value, value_type expValue) noexcept {
#if defined(__SIZEOF_INT128__)
		const __uint128_t res = static_cast<__uint128_t>(value) * static_cast<__uint128_t>(expValue);
		value				  = static_cast<value_type>(res);
		return res <= std::numeric_limits<value_type>::max();
#elif defined(_M_ARM64) && !defined(__MINGW32__)
		JSONIFIER_ALIGN value_type values;
		values = __umulh(value, expValue);
		value  = value * expValue;
		return values == 0;
#elif (defined(_WIN64) && !defined(__clang__))
		JSONIFIER_ALIGN value_type values;
		value = _umul128(value, expValue, &values);
		return values == 0;
#else
		JSONIFIER_ALIGN value_type values;
		value = umul128Generic(value, expValue, &values);
		return values == 0;
#endif
	}

	JSONIFIER_ALWAYS_INLINE static bool divide(value_type& value, value_type expValue) noexcept {
#if defined(__SIZEOF_INT128__)
		const __uint128_t dividend = static_cast<__uint128_t>(value);
		value					   = static_cast<value_type>(dividend / static_cast<__uint128_t>(expValue));
		return (dividend % static_cast<__uint128_t>(expValue)) == 0;
#elif (defined(_WIN64) && !defined(__clang__))
		JSONIFIER_ALIGN value_type values;
		value = _udiv128(0, value, expValue, &values);
		return values == 0;
#else
		JSONIFIER_ALIGN value_type values;
		values = value % expValue;
		value  = value / expValue;
		return values == 0;
#endif
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseFraction(value_type& value, const uint8_t* iter) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			value_type fracValue{ static_cast<value_type>(*iter - zero) };
			typename jsonifier_internal::get_int_type<value_type>::type fracDigits{ 1 };
			++iter;
			while (isDigit(*iter)) {
				fracValue = fracValue * 10 + static_cast<value_type>(*iter - zero);
				++iter;
				++fracDigits;
			}
			if (expTable[*iter]) {
				++iter;
				int8_t expSign = 1;
				if (*iter == minus) {
					expSign = -1;
					++iter;
				} else if (*iter == plus) {
					++iter;
				}
				return parseExponentPostFrac(value, iter, expSign, fracValue, fracDigits);
			}
		}
		if JSONIFIER_LIKELY ((!expFracTable[*iter])) {
			return iter;
		} else {
			return nullptr;
		}
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseExponentPostFrac(value_type& value, const uint8_t* iter, int8_t expSign, value_type fracValue,
		typename jsonifier_internal::get_int_type<value_type>::type fracDigits) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			int64_t expValue{ *iter - zero };
			++iter;
			while (isDigit(*iter)) {
				expValue = expValue * 10 + *iter - zero;
				++iter;
			}
			if JSONIFIER_LIKELY ((expValue <= 19)) {
				const value_type powerExp = powerOfTenInt<value_type>[expValue];

				constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
				constexpr value_type doubleMin = std::numeric_limits<value_type>::min();

				if (fracDigits + expValue >= 0) {
					expValue *= expSign;
					const auto fractionalCorrection =
						expValue > fracDigits ? fracValue * powerOfTenInt<value_type>[expValue - fracDigits] : fracValue / powerOfTenInt<value_type>[fracDigits - expValue];
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), value += fractionalCorrection, iter) : nullptr)
										 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), value += fractionalCorrection, iter) : nullptr);
				} else {
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), iter) : nullptr)
										 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), iter) : nullptr);
				}
				return iter;
			}
			JSONIFIER_UNLIKELY(else) {
				return nullptr;
			}
		}
		JSONIFIER_UNLIKELY(else) {
			return nullptr;
		}
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseExponent(value_type& value, const uint8_t* iter, int8_t expSign) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			value_type expValue{ static_cast<value_type>(*iter - zero) };
			++iter;
			while (isDigit(*iter)) {
				expValue = expValue * 10 + static_cast<value_type>(*iter - zero);
				++iter;
			}
			if JSONIFIER_LIKELY ((expValue <= 19)) {
				const value_type powerExp	   = powerOfTenInt<value_type>[expValue];
				constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
				constexpr value_type doubleMin = std::numeric_limits<value_type>::min();
				expValue *= expSign;
				return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), iter) : nullptr)
									 : ((value / powerExp >= (doubleMin)) ? (divide(value, powerExp), iter) : nullptr);
			}
			JSONIFIER_UNLIKELY(else) {
				return nullptr;
			}
		}
		JSONIFIER_UNLIKELY(else) {
			return nullptr;
		}
	}

	JSONIFIER_INLINE static const uint8_t* finishParse(value_type& value, const uint8_t* iter) {
		if JSONIFIER_UNLIKELY ((*iter == decimal)) {
			++iter;
			return parseFraction(value, iter);
		} else if (expTable[*iter]) {
			++iter;
			int8_t expSign = 1;
			if (*iter == minus) {
				expSign = -1;
				++iter;
			} else if (*iter == plus) {
				++iter;
			}
			return parseExponent(value, iter, expSign);
		}
		if JSONIFIER_LIKELY ((!expFracTable[*iter])) {
			return iter;
		} else {
			return nullptr;
		}
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseInteger(value_type& value, const uint8_t* iter) noexcept {
		uint8_t numTmp{ *iter };
		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		} else [[unlikely]] {
			return nullptr;
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_UNLIKELY ((iter[-2] == zero)) {
			return nullptr;
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			if (value > rawCompValsPos<value_type>[numTmp]) {
				return nullptr;
			}
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
			return finishParse(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + static_cast<value_type>(numberSubTable[numTmp]);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!expFracTable[numTmp])) {
				return iter;
			}
		}
		return nullptr;
	}

	JSONIFIER_ALWAYS_INLINE static bool parseInt(value_type& value, const char*& iter) noexcept {
		for (size_t x = 0; x < 4; ++x) {
			jsonifierPrefetchImpl(numberSubTable.data() + (x * 64));
		}
		auto resultPtr = parseInteger(value, reinterpret_cast<const uint8_t*>(iter));
		if JSONIFIER_LIKELY ((resultPtr)) {
			iter += resultPtr - reinterpret_cast<const uint8_t*>(iter);
			return true;
		} else {
			value = 0;
			return false;
		}
	}
};

template<typename value_type, typename value_type02, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLength(size_t intLength, size_t fracLength = 0, size_t maxExpValue = 0) {
	auto newFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "/JsonData-Prettified.json");
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
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Function-Cached-Parsing-Function", "dodgerblue">(
		[=, &resultValuesDig]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				const auto* iter = stringValues[x].data();
				auto s1			 = integer_parser<value_type>::parseInt(resultValuesDig[x], iter);
				bnch_swt::doNotOptimizeAway(s1);
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
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Globally-Cached-Parsing-Function",
		"dodgerblue">([=, &resultValuesDig]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			const char* iter = stringValues[x].data();
			auto s1	  = jsonifier_internal::integer_parser<value_type>::parseInt(resultValuesDig[x], iter);
			bnch_swt::doNotOptimizeAway(s1);
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

int32_t main() {
	
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:1-Digit">(1);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:4-Digit">(4);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:7-Digit">(7);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:10-Digit">(10);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:13-Digit">(13);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:16-Digit">(16);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:19-Digit">(19);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:1-Digit">(1);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:4-Digit">(4);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:7-Digit">(7);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:10-Digit">(10);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:13-Digit">(13);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:16-Digit">(16);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:19-Digit">(18);
	bnch_swt::benchmark_stage<"Uint-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Uint-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	return 0;
}