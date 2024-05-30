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
		return result;
	};

	auto generateExponent = [&](size_t maxExp) -> int {
		auto newValue = std::abs(static_cast<int>(maxExp));
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
	value_type upperBound = static_cast<value_type>(static_cast<int64_t>(std::pow(10, maxDigits)) - 1);
	std::uniform_int_distribution<value_type> dist{ lowerBound, upperBound };
	return dist(gen);
}

template<typename value_type> std::string generateJsonNumber(int32_t maxIntDigits, int32_t maxFracDigits, int32_t maxExpValue, bool possiblyNegative = true) {
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

JSONIFIER_ALWAYS_INLINE uint16_t parse_2_chars(const char* string) noexcept {
	uint16_t value{ *std::bit_cast<uint16_t*, const char*>(string) };
	return ((value & 0x0f000f00) >> 8) + (value & 0x000f000f) * 10;
}

JSONIFIER_ALWAYS_INLINE uint32_t parse_3_chars(const char* string) noexcept {
	uint32_t value{ *std::bit_cast<uint16_t*, const char*>(string) | static_cast<uint32_t>(string[2]) << 16 };
	value = ((value & 0x00000f00) >> 8) + (value & 0x000f000f) * 10;
	return ((((value & 0x00ff0000) >> 16) + (value & 0x000000ff) * 100)) / 10;
}

JSONIFIER_ALWAYS_INLINE uint32_t parse_4_chars(const char* string) noexcept {
	std::uint32_t value;
	std::memcpy(&value, string, sizeof(value));
	std::uint32_t lower_digits = (value & 0x0f000f00) >> 8;
	std::uint32_t upper_digits = (value & 0x000f000f) * 10;
	value					   = lower_digits + upper_digits;
	return ((value & 0x00ff0000) >> 16) + ((value & 0x000000ff) * 100);
}

JSONIFIER_ALWAYS_INLINE uint64_t parse_5_chars(const char* string) noexcept {
	uint64_t value{ *std::bit_cast<uint32_t*, const char*>(string) | static_cast<uint64_t>(string[4]) << 32 };
	value = ((value & 0x0f000f000f000f00) >> 8) + (value & 0x000f000f000f000f) * 10;
	value = ((value & 0x00ff000000ff0000) >> 16) + (value & 0x000000ff000000ff) * 100;
	return (((((value & 0x0000ffff00000000) >> 32) + (value & 0x000000000000ffff) * 10000)) / 1000);
}

JSONIFIER_ALWAYS_INLINE uint64_t parse_6_chars(const char* string) noexcept {
	uint64_t value{ *std::bit_cast<uint32_t*, const char*>(string) | static_cast<uint64_t>(string[4]) << 32 | static_cast<uint64_t>(string[5]) << 40 };
	value = ((value & 0x0f000f000f000f00) >> 8) + (value & 0x000f000f000f000f) * 10;
	value = ((value & 0x00ff000000ff0000) >> 16) + (value & 0x000000ff000000ff) * 100;
	return (((((value & 0x0000ffff00000000) >> 32) + (value & 0x000000000000ffff) * 10000)) / 100);
}

JSONIFIER_ALWAYS_INLINE uint64_t parse_7_chars(const char* string) noexcept {
	uint64_t value{ *std::bit_cast<uint32_t*, const char*>(string) | static_cast<uint64_t>(string[4]) << 32 | static_cast<uint64_t>(string[5]) << 40 |
		static_cast<uint64_t>(string[6]) << 48 };
	value = ((value & 0x0f000f000f000f00) >> 8) + (value & 0x000f000f000f000f) * 10;
	value = ((value & 0x00ff000000ff0000) >> 16) + (value & 0x000000ff000000ff) * 100;
	return (((((value & 0x0000ffff00000000) >> 32) + (value & 0x000000000000ffff) * 10000)) / 10);
};

template<std::floating_point value_type> auto strtoDigits(const std::string& value) {
	return std::stod(value);
}

template<std::signed_integral value_type> auto strtoDigits(const std::string& value) {
	return std::stoll(value);
}

template<std::unsigned_integral value_type> auto strtoDigits(const std::string& value) {
	return std::stoull(value);
}

template<typename value_type, typename value_type02, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testName>
JSONIFIER_ALWAYS_INLINE void runForLength(size_t intLength, size_t fracLength = 0, size_t maxExpValue = 0) {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	value_type value{};
	//static constexpr jsonifier_internal::integer_parser<value_type, const char> intParser{};
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
		jsonifier_internal::integer_parser<value_type, const char>::parseInt(value, iter);
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
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Glaze-Parsing-Function", "dodgerblue">(
		[=, &resultValuesDig, &value]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				const auto* iterNew = stringValues[x].data();
				glz::detail::atoi(value, iterNew);
				bnch_swt::doNotOptimizeAway(value);
				resultValuesDig[x] = value;
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
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Jsonifier-Parsing-Function",
		"dodgerblue">([=, &resultValuesDig, &value]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			const auto* iterNew = stringValues[x].data();
			jsonifier_internal::integer_parser<value_type, const char>::parseInt(value, iterNew);
			bnch_swt::doNotOptimizeAway(value);
			resultValuesDig[x] = value;
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

JSONIFIER_ALWAYS_INLINE constexpr uint32_t numbits(uint32_t x) noexcept {
	return x < 2 ? x : 1 + numbits(x >> 1);
}

template<jsonifier::concepts::float_type value_type, typename char_type> JSONIFIER_MAYBE_ALWAYS_INLINE char_type* toChars(char_type* buf, value_type val) noexcept {
	static_assert(std::numeric_limits<value_type>::is_iec559);
	static_assert(std::numeric_limits<value_type>::radix == 2);
	static_assert(std::is_same_v<float, value_type> || std::is_same_v<double, value_type>);
	static_assert(sizeof(float) == 4 && sizeof(double) == 8);
	using raw = std::conditional_t<std::is_same_v<float, value_type>, uint32_t, uint64_t>;

	raw rawVal;
	std::memcpy(&rawVal, &val, sizeof(value_type));

	constexpr bool isFloat				   = std::is_same_v<float, value_type>;
	static constexpr uint32_t exponentBits = numbits(std::numeric_limits<value_type>::max_exponent - std::numeric_limits<value_type>::min_exponent + 1);
	bool sign							   = (rawVal >> (sizeof(value_type) * 8 - 1));
	uint32_t expRaw						   = rawVal << 1 >> (sizeof(raw) * 8 - exponentBits);

	if JSONIFIER_UNLIKELY ((expRaw == (static_cast<uint32_t>(1) << exponentBits) - 1)) {
		std::memcpy(buf, "null", 4);
		return buf + 4;
	}

	*buf = '-';
	buf += sign;

	if JSONIFIER_UNLIKELY (((rawVal << 1) == 0)) {
		*buf = '0';
		return buf + 1;
	}

	if constexpr (isFloat) {
		const auto value = jsonifier_jkj::dragonbox::to_decimal(val, jsonifier_jkj::dragonbox::policy::sign::ignore, jsonifier_jkj::dragonbox::policy::trailing_zero::remove);

		uint32_t sigDec			= static_cast<uint32_t>(value.significand);
		int32_t expDec			= value.exponent;
		const int32_t numDigits = static_cast<int32_t>(jsonifier_internal::fastDigitCount(sigDec));
		int32_t dotPos			= numDigits + expDec;

		if (-6 < dotPos && dotPos <= 9) {
			if (dotPos <= 0) {
				*buf++ = '0';
				*buf++ = '.';
				while (dotPos < 0) {
					*buf++ = '0';
					++dotPos;
				}
				return jsonifier_internal::writeU32Len1To9(buf, sigDec);
			} else {
				auto numEnd			  = jsonifier_internal::writeU32Len1To9(buf, sigDec);
				int32_t digitsWritten = static_cast<int32_t>(numEnd - buf);
				if (dotPos < digitsWritten) {
					std::memmove(buf + dotPos + 1, buf + dotPos, digitsWritten - dotPos);
					buf[dotPos] = '.';
					return numEnd + 1;
				} else {
					if (dotPos > digitsWritten) {
						std::memset(numEnd, '0', dotPos - digitsWritten);
						return buf + dotPos;
					} else {
						return numEnd;
					}
				}
			}
		} else {
			auto end = jsonifier_internal::writeU32Len1To9(buf + 1, sigDec);
			expDec += static_cast<int32_t>(end - (buf + 1)) - 1;
			buf[0] = buf[1];
			buf[1] = '.';
			if (end == buf + 2) {
				buf[2] = '0';
				++end;
			}
			*end = 'E';
			buf	 = end + 1;
			if (expDec < 0) {
				*buf = '-';
				++buf;
				expDec = -expDec;
			}
			expDec		= std::abs(expDec);
			uint32_t lz = expDec < 10;
			std::memcpy(buf, jsonifier_internal::charTable + (expDec * 2 + lz), 2);
			return buf + 2 - lz;
		}
	} else {
		const auto value = jsonifier_jkj::dragonbox::to_decimal(val, jsonifier_jkj::dragonbox::policy::sign::ignore, jsonifier_jkj::dragonbox::policy::trailing_zero::ignore);

		uint64_t sigDec = value.significand;
		int32_t expDec	= value.exponent;

		int32_t sigLen = 17;
		sigLen -= (sigDec < 100000000ull * 100000000ull);
		sigLen -= (sigDec < 100000000ull * 10000000ull);
		int32_t dotPos = sigLen + expDec;

		if (-6 < dotPos && dotPos <= 21) {
			if (dotPos <= 0) {
				auto numHdr = buf + (2 - dotPos);
				auto numEnd = jsonifier_internal::writeU64Len15To17Trim(numHdr, sigDec);
				buf[0]		= '0';
				buf[1]		= '.';
				buf += 2;
				std::memset(buf, '0', static_cast<size_t>(numHdr - buf));
				return numEnd;
			} else {
				std::memset(buf, '0', 24);
				auto numHdr = buf + 1;
				auto numEnd = jsonifier_internal::writeU64Len15To17Trim(numHdr, sigDec);
				std::memmove(buf, buf + 1, static_cast<size_t>(dotPos));
				buf[dotPos] = '.';
				return ((numEnd - numHdr) <= dotPos) ? buf + dotPos : numEnd;
			}
		} else {
			auto end = jsonifier_internal::writeU64Len15To17Trim(buf + 1, sigDec);
			end -= (end == buf + 2);
			expDec += sigLen - 1;
			buf[0] = buf[1];
			buf[1] = '.';
			end[0] = 'E';
			buf	   = end + 1;
			buf[0] = '-';
			buf += expDec < 0;
			expDec = std::abs(expDec);
			if (expDec < 100) {
				uint32_t lz = expDec < 10;
				std::memcpy(buf, jsonifier_internal::charTable + (expDec * 2 + lz), 2);
				return buf + 2 - lz;
			} else {
				const uint32_t hi = (static_cast<uint32_t>(expDec) * 656) >> 16;
				const uint32_t lo = static_cast<uint32_t>(expDec) - hi * 100;
				buf[0]			  = static_cast<uint8_t>(hi) + '0';
				std::memcpy(&buf[1], jsonifier_internal::charTable + (lo * 2), 2);
				return buf + 3;
			}
		}
	}
}

template<typename value_type> inline constexpr std::array<uint64_t, 256> rawCompValsPos{ [] {
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

template<typename value_type> inline constexpr std::array<uint64_t, 256> rawCompValsNeg{ [] {
	constexpr auto maxValue{ uint64_t((std::numeric_limits<std::decay_t<value_type>>::max)()) + 1 };
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

#define isDigit(x) ((x <= jsonifier_internal::nine) && (x >= jsonifier_internal::zero))

template<typename value_type, typename char_type> struct integer_parser {
	JSONIFIER_ALWAYS_INLINE constexpr integer_parser() noexcept = default;

	JSONIFIER_ALWAYS_INLINE static value_type umul128Generic(value_type ab, value_type cd, value_type& hi) noexcept {
		value_type aHigh = ab >> 32;
		value_type aLow	 = ab & 0xFFFFFFFF;
		value_type bHigh = cd >> 32;
		value_type bLow	 = cd & 0xFFFFFFFF;
		value_type ad	 = aHigh * static_cast<value_type>(bLow);
		value_type bd	 = aHigh * static_cast<value_type>(bLow);
		value_type adbc	 = ad + aLow * static_cast<value_type>(bHigh);
		value_type lo	 = bd + (adbc << 32);
		value_type carry = (lo < bd);
		hi				 = aHigh * static_cast<value_type>(bHigh) + (adbc >> 32) + carry;
		return lo;
	}

	JSONIFIER_ALWAYS_INLINE static bool multiply(value_type& value, int64_t expValue) noexcept {
		JSONIFIER_ALIGN uint64_t values;
#if defined(_M_ARM64) && !defined(__MINGW32__)
		values = __umulh(value, expValue);
		value  = value * expValue;
#elif defined(FASTFLOAT_32BIT) || (defined(_WIN64) && !defined(__clang__) && !defined(_M_ARM64))
		value = _umul128(value, expValue, &values);
#elif defined(FASTFLOAT_64BIT) && defined(__SIZEOF_INT128__)
		__uint128_t r = (( __uint128_t )value) * expValue;
		value		  = static_cast<value_type>(r);
		values		  = static_cast<value_type>(r >> 64);
#else
		value = umul128Generic(value, expValue, values);
#endif
		return values == 0;
	};

	JSONIFIER_ALWAYS_INLINE static bool divide(value_type& value, int64_t expValue) noexcept {
		JSONIFIER_ALIGN uint64_t values;
#if defined(FASTFLOAT_32BIT) || (defined(_WIN64) && !defined(__clang__))
		value = _udiv128(0, value, expValue, &values);
#elif defined(FASTFLOAT_64BIT) && defined(__SIZEOF_INT128__)
		__uint128_t dividend = __uint128_t(value);
		value				 = static_cast<value_type>(dividend / expValue);
		values				 = static_cast<value_type>(dividend % expValue);
#else
		values = value % expValue;
		value  = value / expValue;
#endif
		return values == 0;
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseFraction(value_type& value, const uint8_t* iter) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			value_type fracValue{ static_cast<value_type>(*iter - jsonifier_internal::zero) };
			int8_t fracDigits{ 1 };
			++iter;
			while (isDigit(*iter)) {
				fracValue = fracValue * 10 + (*iter - jsonifier_internal::zero);
				++iter;
				++fracDigits;
			}
			if (jsonifier_internal::expTable[*iter]) {
				++iter;
				int8_t expSign = 1;
				if (*iter == jsonifier_internal::minus) {
					expSign = -1;
					++iter;
				} else if (*iter == jsonifier_internal::plus) {
					++iter;
				}
				return parseExponentPostFrac(value, iter, expSign, fracValue, fracDigits);
			} else {
				return iter;
			}
			return iter;
		}
		JSONIFIER_UNLIKELY(else) {
			return nullptr;
		}
	}

	JSONIFIER_ALWAYS_INLINE static const uint8_t* parseExponentPostFrac(value_type& value, const uint8_t* iter, int8_t expSign, value_type fracValue, int8_t fracDigits) noexcept {
		if JSONIFIER_LIKELY ((isDigit(*iter))) {
			int64_t expValue{ *iter - jsonifier_internal::zero };
			++iter;
			while (isDigit(*iter)) {
				expValue = expValue * 10 + (*iter - jsonifier_internal::zero);
				++iter;
			}
			if JSONIFIER_LIKELY ((expValue <= 19)) {
				const value_type powerExp = jsonifier_internal::powerOfTenInt<value_type>[expValue];

				constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
				constexpr value_type doubleMin = std::numeric_limits<value_type>::min();

				if (fracDigits + expValue >= 0) {
					expValue *= expSign;
					const auto fractionalCorrection =
						expValue > fracDigits ? fracValue * jsonifier_internal::powerOfTenInt<value_type>[expValue - fracDigits] : fracValue / jsonifier_internal::powerOfTenInt<value_type>[fracDigits - expValue];
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), value += fractionalCorrection, iter) : nullptr)
										 : ((value >= (doubleMin * powerExp)) ? (divide(value, powerExp), value += fractionalCorrection, iter) : nullptr);
				} else {
					return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), iter) : nullptr)
										 : ((value >= (doubleMin * powerExp)) ? (divide(value, powerExp), iter) : nullptr);
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
			int64_t expValue{ *iter - jsonifier_internal::zero };
			++iter;
			while (isDigit(*iter)) {
				expValue = expValue * 10 + (*iter - jsonifier_internal::zero);
				++iter;
			}
			if JSONIFIER_LIKELY ((expValue <= 19)) {
				const value_type powerExp	   = jsonifier_internal::powerOfTenInt<value_type>[expValue];
				constexpr value_type doubleMax = std::numeric_limits<value_type>::max();
				constexpr value_type doubleMin = std::numeric_limits<value_type>::min();
				expValue *= expSign;
				return (expSign > 0) ? ((value <= (doubleMax / powerExp)) ? (multiply(value, powerExp), iter) : nullptr)
									 : ((value >= (doubleMin * powerExp)) ? (divide(value, powerExp), iter) : nullptr);
			}
			JSONIFIER_UNLIKELY(else) {
				return nullptr;
			}
		}
		JSONIFIER_UNLIKELY(else) {
			return nullptr;
		}
	}

	template<bool negative> JSONIFIER_INLINE static const uint8_t* finishParse(value_type& value, const uint8_t* iter) {
		if JSONIFIER_UNLIKELY ((*iter == jsonifier_internal::decimal)) {
			++iter;
			if constexpr (negative) {
				return (iter = parseFraction(value, iter), value *= -1, iter);
			} else {
				return parseFraction(value, iter);
			}
		} else if (jsonifier_internal::expTable[*iter]) {
			++iter;
			int8_t expSign = 1;
			if (*iter == jsonifier_internal::minus) {
				expSign = -1;
				++iter;
			} else if (*iter == jsonifier_internal::plus) {
				++iter;
			}
			if constexpr (negative) {
				return (iter = parseExponent(value, iter, expSign), value *= -1, iter);
			} else {
				return parseExponent(value, iter, expSign);
			}
		} else {
			if constexpr (negative) {
				value *= -1;
			}
			return iter;
		}
	};

	template<bool negative> JSONIFIER_ALWAYS_INLINE static const uint8_t* parseInteger(value_type& value, const uint8_t* iter) noexcept {
		uint8_t numTmp{ *iter };
		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = numTmp - jsonifier_internal::zero;
			++iter;
			numTmp = *iter;
		} else [[unlikely]] {
			return nullptr;
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_UNLIKELY ((iter[-2] == jsonifier_internal::zero)) {
			return nullptr;
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			value = value * 10 + (numTmp - jsonifier_internal::zero);
			++iter;
			numTmp = *iter;
		}
		JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if JSONIFIER_LIKELY ((isDigit(numTmp))) {
			if constexpr (negative) {
				value *= -1;
				value = value * 10 - (numTmp - jsonifier_internal::zero);
			} else {
				value = value * 10 + (numTmp - jsonifier_internal::zero);
			}
			++iter;
			numTmp = *iter;
		}	JSONIFIER_UNLIKELY(else) {
			if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
				if constexpr (negative) {
					value *= -1;
				}
				return iter;
			}
			return finishParse<negative>(value, iter);
		}

		if constexpr (negative) {
			if (value > rawCompValsNeg<value_type>[numTmp]) [[unlikely]] {
				return nullptr;
			}
		} else {
			if (value > rawCompValsPos<value_type>[numTmp]) [[unlikely]] {
				return nullptr;
			}
		}

		if constexpr (jsonifier::concepts::unsigned_type<value_type>) {
			if JSONIFIER_LIKELY ((isDigit(numTmp))) {
				value = value * 10 + (numTmp - jsonifier_internal::zero);
				++iter;
			}
			JSONIFIER_UNLIKELY(else) {
				if JSONIFIER_LIKELY ((!jsonifier_internal::expFracTable[numTmp])) {
					return iter;
				}
			}
		}

		return jsonifier_internal::expFracTable[numTmp] ? nullptr : iter;
	}

	JSONIFIER_ALWAYS_INLINE static bool parseInt(value_type& value, char_type*& iter) noexcept {
		if constexpr (jsonifier::concepts::signed_type<value_type>) {
			if (*iter == jsonifier_internal::minus) {
				++iter;
				const uint8_t* resultPtr{ parseInteger<true>(value, reinterpret_cast<const uint8_t*>(iter)) };
				if JSONIFIER_LIKELY ((resultPtr)) {
					iter += resultPtr - reinterpret_cast<const uint8_t*>(iter);
					return true;
				} else {
					return false;
				}
			} else {
				const uint8_t* resultPtr{ parseInteger<false>(value, reinterpret_cast<const uint8_t*>(iter)) };
				if JSONIFIER_LIKELY ((resultPtr)) {
					iter += resultPtr - reinterpret_cast<const uint8_t*>(iter);
					return true;
				} else {
					return false;
				}
			}

		} else {
			auto resultPtr = parseInteger<false>(value, reinterpret_cast<const uint8_t*>(iter));
			if JSONIFIER_LIKELY ((resultPtr)) {
				iter += resultPtr - reinterpret_cast<const uint8_t*>(iter);
				return true;
			} else {
				return false;
			}
		}
	}
};

template<size_t intLength, typename value_type, typename value_type02> JSONIFIER_ALWAYS_INLINE void runForLength03() {
	std::vector<double> doubles{};
	for (size_t x = 0; x < 1024; ++x) {
		auto newString = generateJsonNumber<uint64_t>(8, 2, 4, false);
		auto newIter   = newString.data() + newString.size();
		doubles.emplace_back(std::strtod(newString.data(), &newIter));
	}
	static constexpr jsonifier_internal::integer_parser<uint64_t, const char> intParser{};
	std::vector<std::string> newResults{};

	std::vector<std::string> oldResults{};
	for (size_t x = 0; x < 1024; ++x) {
		std::string newString{};
		newString.resize(1024);
		newResults.emplace_back(newString);
		oldResults.emplace_back(newString);
	}
	std::basic_string<char> newerString{ "320222323" };
	static constexpr jsonifier_internal::string_literal testName{ "Testing-D-To-Str" + jsonifier_internal::toStringLiteral<intLength>() };
	bnch_swt::benchmark_stage<testName, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<"TestFunction", "TestFunction01-Parsing-Function",
		"dodgerblue">([&]() mutable {
		size_t value[2];
		for (size_t x = 0; x < 1024; ++x) {
			value[0] = *std::bit_cast<size_t*>(newerString.data());
			value[1] = *std::bit_cast<size_t*>(newerString.data());
			bnch_swt::doNotOptimizeAway(value);
		}
	});
	//std::cout << "Current TestFunction01 Value: " << newString << std::endl;
	bnch_swt::benchmark_stage<testName, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<"TestFunction", "TestFunction02-Parsing-Function",
		"dodgerblue">([&]() mutable {
		size_t value[2];
		for (size_t x = 0; x < 1024; ++x) {
			std::memcpy(&value[0], newerString.data(), 8);
			std::memcpy(&value[1], newerString.data(), 8);
			bnch_swt::doNotOptimizeAway(value);
		}
	});
	for (size_t x = 0; x < 1024; ++x) {
		if (newResults[x] != oldResults[x]) {
			std::cout << "FAILED TO SERIALIZE: " << doubles[x] << std::endl;
			std::cout << "INSTEAD SERIALIZED: " << newResults[x] << std::endl;
		} else {
			//std::cout << "SUCCESFULLY SERIALIZED: " << doubles[x] << std::endl;//
		}
	}
	//std::cout << "Current TestFunction02 Value: " << newString02 << std::endl;
	bnch_swt::benchmark_stage<testName, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
}

int32_t main() {
	std::string newerString{ "9223372036854775808" };
	jsonifier_internal::string_literal testLiteral{ "TEsting" };
	jsonifier_internal::string_literal newLiteral{ testLiteral };
	std::cout << "NWE LITERAL: " << newLiteral << std::endl;
	int64_t newValue{};
	auto newIter = newerString.data();
	integer_parser<int64_t, char>::parseInt(newValue, newIter);
	newerString = "-9223372036854775809";
	integer_parser<int64_t, char>::parseInt(newValue, newIter);
	runForLength03<1, uint64_t, uint64_t>();
	uint_validation_tests::uintTests();
	int_validation_tests::intTests();
	//conformance_tests::conformanceTests();
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
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:19-Digit">(19);
	bnch_swt::benchmark_stage<"Uint-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Uint-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	return 0;
}