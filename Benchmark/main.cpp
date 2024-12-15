#include <cstdint>
#include <thread>
#include <jsonifier/Index.hpp>
#include <iostream>
#include <type_traits>
#include <utility>
#include <array>
#include "BnchSwt/BenchmarkSuite.hpp"
#include <BnchSwt/StringLiteral.hpp>
#include "RandomGenerators.hpp"
#include <charconv>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

alignas(2) JSONIFIER_INLINE_VARIABLE char charTable1[]{ 0x30u, 0x31u, 0x32u, 0x33u, 0x34u, 0x35u, 0x36u, 0x37u, 0x38u, 0x39u };

JSONIFIER_INLINE_VARIABLE uint16_t charTable02[]{ 0x3030u, 0x3130u, 0x3230u, 0x3330u, 0x3430u, 0x3530u, 0x3630u, 0x3730u, 0x3830u, 0x3930u, 0x3031u, 0x3131u, 0x3231u, 0x3331u,
	0x3431u, 0x3531u, 0x3631u, 0x3731u, 0x3831u, 0x3931u, 0x3032u, 0x3132u, 0x3232u, 0x3332u, 0x3432u, 0x3532u, 0x3632u, 0x3732u, 0x3832u, 0x3932u, 0x3033u, 0x3133u, 0x3233u,
	0x3333u, 0x3433u, 0x3533u, 0x3633u, 0x3733u, 0x3833u, 0x3933u, 0x3034u, 0x3134u, 0x3234u, 0x3334u, 0x3434u, 0x3534u, 0x3634u, 0x3734u, 0x3834u, 0x3934u, 0x3035u, 0x3135u,
	0x3235u, 0x3335u, 0x3435u, 0x3535u, 0x3635u, 0x3735u, 0x3835u, 0x3935u, 0x3036u, 0x3136u, 0x3236u, 0x3336u, 0x3436u, 0x3536u, 0x3636u, 0x3736u, 0x3836u, 0x3936u, 0x3037u,
	0x3137u, 0x3237u, 0x3337u, 0x3437u, 0x3537u, 0x3637u, 0x3737u, 0x3837u, 0x3937u, 0x3038u, 0x3138u, 0x3238u, 0x3338u, 0x3438u, 0x3538u, 0x3638u, 0x3738u, 0x3838u, 0x3938u,
	0x3039u, 0x3139u, 0x3239u, 0x3339u, 0x3439u, 0x3539u, 0x3639u, 0x3739u, 0x3839u, 0x3939u };

constexpr const uint64_t mask24	  = (1ull << 24) - 1ull;
constexpr const uint64_t mask32	  = (1ull << 32) - 1ull;
constexpr const uint64_t mask57	  = (1ull << 57) - 1ull;
constexpr const uint64_t mult1_3  = 10ull * (1 << 24) / 1000 + 1;
constexpr const uint64_t mult5_6  = 10ull * (1ull << 32ull) / 100000 + 1;
constexpr const uint64_t mult7_8  = 10ull * (1ull << 48ull) / 10000000 + 1;
constexpr const uint64_t mult9_10 = (1ull << 48ull) / 1000000 + 1;

struct int_serializing_package_2 {
	mutable uint64_t value01;
	mutable uint64_t value02;
};

template<typename value_type> JSONIFIER_INLINE char* lengthNew1(char* buf, value_type value) {
	buf[0] = charTable1[value];
	return buf + 1ull;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew2(char* buf, value_type value) {
	std::memcpy(buf, charTable02 + value, 2);
	return buf + 2;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew3(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = mult1_3 * value;
	buf[0]			   = charTable1[intPackage.value01 >> 24];
	intPackage.value02 = (intPackage.value01 & mask24) * 100ull;
	std::memcpy(buf + 1, charTable02 + (intPackage.value02 >> 24), 2);
	return buf + 3;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew4(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = mult1_3 * value;
	std::memcpy(buf, charTable02 + (intPackage.value01 >> 24), 2);
	intPackage.value02 = (intPackage.value01 & mask24) * 100ull;
	std::memcpy(buf + 2, charTable02 + (intPackage.value02 >> 24), 2);
	return buf + 4;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew5(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = mult5_6 * value;
	buf[0]			   = charTable1[intPackage.value01 >> 32];
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 1, charTable02 + (intPackage.value02 >> 32), 2);
	intPackage.value01 = (intPackage.value02 & mask32) * 100ull;
	std::memcpy(buf + 3, charTable02 + (intPackage.value01 >> 32), 2);
	return buf + 5;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew6(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = mult5_6 * value;
	std::memcpy(buf, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 2, charTable02 + (intPackage.value02 >> 32), 2);
	intPackage.value01 = (intPackage.value02 & mask32) * 100ull;
	std::memcpy(buf + 4, charTable02 + (intPackage.value01 >> 32), 2);
	return buf + 6;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew7(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = mult7_8 * value >> 16;
	buf[0]			   = charTable1[intPackage.value01 >> 32];
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 1, charTable02 + (intPackage.value02 >> 32), 2);
	intPackage.value01 = (intPackage.value02 & mask32) * 100ull;
	std::memcpy(buf + 3, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 5, charTable02 + (intPackage.value02 >> 32), 2);
	return buf + 7;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew8(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = mult7_8 * value >> 16;
	std::memcpy(buf, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 2, charTable02 + (intPackage.value02 >> 32), 2);
	intPackage.value01 = (intPackage.value02 & mask32) * 100ull;
	std::memcpy(buf + 4, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 6, charTable02 + (intPackage.value02 >> 32), 2);
	return buf + 8;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew9(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = value * 1441151880ull >> 57;
	intPackage.value02 = value - intPackage.value01 * 100000000ull;
	buf[0]			   = charTable1[intPackage.value01];
	intPackage.value01 = (mult9_10 * intPackage.value02 >> 16) + 1ull;
	std::memcpy(buf + 1, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 3, charTable02 + (intPackage.value02 >> 32), 2);
	intPackage.value01 = (intPackage.value02 & mask32) * 100ull;
	std::memcpy(buf + 5, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 7, charTable02 + (intPackage.value02 >> 32), 2);
	return buf + 9;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew10(char* buf, value_type value) {
	constexpr int_serializing_package_2 intPackage{};
	intPackage.value01 = value * 1441151880ull >> 57;
	intPackage.value02 = value - intPackage.value01 * 100000000ull;
	std::memcpy(buf, charTable02 + intPackage.value01, 2);
	intPackage.value01 = (mult9_10 * intPackage.value02 >> 16) + 1;
	std::memcpy(buf + 2, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 4, charTable02 + (intPackage.value02 >> 32), 2);
	intPackage.value01 = (intPackage.value02 & mask32) * 100ull;
	std::memcpy(buf + 6, charTable02 + (intPackage.value01 >> 32), 2);
	intPackage.value02 = (intPackage.value01 & mask32) * 100ull;
	std::memcpy(buf + 8, charTable02 + (intPackage.value02 >> 32), 2);
	return buf + 10ull;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew11(char* buf, value_type value) {
	const uint64_t z = value / 10ull;
	const uint64_t u = value - z * 10ull;
	buf				 = lengthNew10(buf, z);
	return lengthNew1(buf, u);
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew12(char* buf, value_type value) {
	const uint64_t z = value / 100ull;
	const uint64_t u = value - z * 100ull;
	buf				 = lengthNew10(buf, z);
	return lengthNew2(buf, u);
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew13(char* buf, value_type value) {
	const uint64_t z = value / 1000ull;
	const uint64_t u = value - z * 1000ull;
	buf				 = lengthNew10(buf, z);
	return lengthNew3(buf, u);
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew14(char* buf, value_type value) {
	const uint64_t z = value / 10000ull;
	const uint64_t u = value - z * 10000ull;
	buf				 = lengthNew10(buf, z);
	return lengthNew4(buf, u);
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew15(char* buf, value_type value) {
	const uint64_t z = value / 100000ull;
	const uint64_t u = value - z * 100000ull;
	buf				 = lengthNew10(buf, z);
	return lengthNew5(buf, u);
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew16(char* buf, value_type value) {
	const uint64_t z = value / 1000000ull;
	const uint64_t u = value - z * 1000000ull;
	buf				 = lengthNew10(buf, z);
	return lengthNew6(buf, u);
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew17(char* buf, value_type value) {
	uint64_t tmp = value / 10000000000ull;
	uint64_t f0	 = mult7_8 * tmp >> 16;
	buf[0]		 = charTable1[f0 >> 32];
	uint64_t f2	 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 1, charTable02 + (f2 >> 32), 2);
	uint64_t f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 3, charTable02 + (f4 >> 32), 2);
	uint64_t f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 5, charTable02 + (f6 >> 32), 2);
	tmp				 = value - tmp * 10000000000ull;
	const uint64_t u = (tmp) * 1441151880ull >> 57;
	const uint64_t z = ( tmp )-u * 100000000ull;
	std::memcpy(buf + 7, charTable02 + u, 2);
	f0 = (mult9_10 * z >> 16) + 1;
	std::memcpy(buf + 9, charTable02 + (f0 >> 32), 2);
	f2 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 11, charTable02 + (f2 >> 32), 2);
	f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 13, charTable02 + (f4 >> 32), 2);
	f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 15, charTable02 + (f6 >> 32), 2);
	return buf + 17;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew18(char* buf, value_type value) {
	uint64_t tmp = value / 10000000000ull;
	uint64_t f0	 = mult7_8 * tmp >> 16;
	std::memcpy(buf, charTable02 + (f0 >> 32), 2);
	uint64_t f2 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 2, charTable02 + (f2 >> 32), 2);
	uint64_t f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 4, charTable02 + (f4 >> 32), 2);
	uint64_t f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 6, charTable02 + (f6 >> 32), 2);
	tmp				 = value - tmp * 10000000000ull;
	const uint64_t u = (tmp) * 1441151880ull >> 57;
	const uint64_t z = ( tmp )-u * 100000000ull;
	std::memcpy(buf + 8, charTable02 + u, 2);
	f0 = (mult9_10 * z >> 16) + 1;
	std::memcpy(buf + 10, charTable02 + (f0 >> 32), 2);
	f2 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 12, charTable02 + (f2 >> 32), 2);
	f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 14, charTable02 + (f4 >> 32), 2);
	f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 16, charTable02 + (f6 >> 32), 2);
	return buf + 18;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew19(char* buf, value_type value) {
	uint64_t u		 = value / 100000000ull;
	const uint64_t z = value - u * 100000000ull;

	const uint64_t uOld{ u };
	u /= 100000000ull;
	const uint64_t y = uOld - u * 100000000ull;

	uint64_t f0 = mult1_3 * u;
	buf[0]		= charTable1[f0 >> 24];
	uint64_t f2 = (f0 & mask24) * 100ull;
	std::memcpy(buf + 1, charTable02 + (f2 >> 24), 2);
	f0 = (mult9_10 * y >> 16) + 1ull;
	std::memcpy(buf + 3, charTable02 + (f0 >> 32), 2);
	f2 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 5, charTable02 + (f2 >> 32), 2);
	uint64_t f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 7, charTable02 + (f4 >> 32), 2);
	uint64_t f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 9, charTable02 + (f6 >> 32), 2);
	f0 = (mult9_10 * z >> 16) + 1ull;
	std::memcpy(buf + 11, charTable02 + (f0 >> 32), 2);
	f2 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 13, charTable02 + (f2 >> 32), 2);
	f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 15, charTable02 + (f4 >> 32), 2);
	f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 17, charTable02 + (f6 >> 32), 2);
	return buf + 19;
}

template<typename value_type> JSONIFIER_INLINE char* lengthNew20(char* buf, value_type value) {
	uint64_t u		 = value / 100000000ull;
	const uint64_t z = value - u * 100000000ull;

	const uint64_t uOld{ u };
	u /= 100000000ull;
	const uint64_t y = uOld - u * 100000000ull;

	uint64_t f0 = mult1_3 * u;
	std::memcpy(buf, charTable02 + (f0 >> 24), 2);
	uint64_t f2 = (f0 & mask24) * 100ull;
	std::memcpy(buf + 2, charTable02 + (f2 >> 24), 2);
	f0 = (mult9_10 * y >> 16) + 1ull;
	std::memcpy(buf + 4, charTable02 + (f0 >> 32), 2);
	f2 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 6, charTable02 + (f2 >> 32), 2);
	uint64_t f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 8, charTable02 + (f4 >> 32), 2);
	uint64_t f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 10, charTable02 + (f6 >> 32), 2);
	f0 = (mult9_10 * z >> 16) + 1ull;
	std::memcpy(buf + 12, charTable02 + (f0 >> 32), 2);
	f2 = (f0 & mask32) * 100ull;
	std::memcpy(buf + 14, charTable02 + (f2 >> 32), 2);
	f4 = (f2 & mask32) * 100ull;
	std::memcpy(buf + 16, charTable02 + (f4 >> 32), 2);
	f6 = (f4 & mask32) * 100ull;
	std::memcpy(buf + 18, charTable02 + (f6 >> 32), 2);
	return buf + 20ull;
}

template<typename value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, uint64_t>
JSONIFIER_INLINE char* to_text_from_integer(char* buf, value_type value) {
	const uint64_t index{ jsonifier::internal::fastDigitCount(value) };
	switch (index) {
		case 1: {
			return lengthNew1(buf, value);
		}
		case 2: {
			return lengthNew2(buf, value);
		}
		case 3: {
			return lengthNew3(buf, value);
		}
		case 4: {
			return lengthNew4(buf, value);
		}
		case 5: {
			return lengthNew5(buf, value);
		}
		case 6: {
			return lengthNew6(buf, value);
		}
		case 7: {
			return lengthNew7(buf, value);
		}
		case 8: {
			return lengthNew8(buf, value);
		}
		case 9: {
			return lengthNew9(buf, value);
		}
		case 10: {
			return lengthNew10(buf, value);
		}
		case 11: {
			return lengthNew11(buf, value);
		}
		case 12: {
			return lengthNew12(buf, value);
		}
		case 13: {
			return lengthNew13(buf, value);
		}
		case 14: {
			return lengthNew14(buf, value);
		}
		case 15: {
			return lengthNew15(buf, value);
		}
		case 16: {
			return lengthNew16(buf, value);
		}
		case 17: {
			return lengthNew17(buf, value);
		}
		case 18: {
			return lengthNew18(buf, value);
		}
		case 19: {
			return lengthNew19(buf, value);
		}
		default: {
			return lengthNew20(buf, value);
		}
	}
}

template<typename value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, int64_t>
auto* to_text_from_integer(auto* buf, value_type x) noexcept {
	*buf = '-';
	return to_text_from_integer(buf + (x < 0), uint64_t(x ^ (x >> 63)) - (x >> 63));
}

constexpr char char_table[200] = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3', '1',
	'4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', '3', '0', '3', '1',
	'3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8', '4',
	'9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6',
	'6', '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8', '3', '8',
	'4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9' };

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
JSONIFIER_INLINE auto* to_chars_u64_len_8(auto* buf, value_type value) noexcept {
	/* 8 digits: aabbccdd */
	const uint32_t aabb = uint32_t((uint64_t(value) * 109951163) >> 40); /* (value / 10000) */
	const uint32_t ccdd = value - aabb * 10000; /* (value % 10000) */
	const uint32_t aa	= (aabb * 5243) >> 19; /* (aabb / 100) */
	const uint32_t cc	= (ccdd * 5243) >> 19; /* (ccdd / 100) */
	const uint32_t bb	= aabb - aa * 100; /* (aabb % 100) */
	const uint32_t dd	= ccdd - cc * 100; /* (ccdd % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, char_table + bb * 2, 2);
	std::memcpy(buf + 4, char_table + cc * 2, 2);
	std::memcpy(buf + 6, char_table + dd * 2, 2);
	return buf + 8;
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
JSONIFIER_INLINE auto* to_chars_u64_len_4(auto* buf, value_type value) noexcept {
	/* 4 digits: aabb */
	const uint32_t aa = (value * 5243) >> 19; /* (value / 100) */
	const uint32_t bb = value - aa * 100; /* (value % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, char_table + bb * 2, 2);
	return buf + 4;
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
inline auto* to_chars_u64_len_1_8(auto* buf, value_type value) noexcept {
	uint32_t aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

	if (value < 100) { /* 1-2 digits: aa */
		lz = value < 10;
		std::memcpy(buf, char_table + value * 2 + lz, 2);
		buf -= lz;
		return buf + 2;
	} else if (value < 10000) { /* 3-4 digits: aabb */
		aa = (value * 5243) >> 19; /* (value / 100) */
		bb = value - aa * 100; /* (value % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		return buf + 4;
	} else if (value < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(value) * 429497) >> 32); /* (value / 10000) */
		bbcc = value - aa * 10000; /* (value % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (value / 10000) */
		aabb = uint32_t((uint64_t(value) * 109951163) >> 40);
		ccdd = value - aabb * 10000; /* (value % 10000) */
		aa	 = (aabb * 5243) >> 19; /* (aabb / 100) */
		cc	 = (ccdd * 5243) >> 19; /* (ccdd / 100) */
		bb	 = aabb - aa * 100; /* (aabb % 100) */
		dd	 = ccdd - cc * 100; /* (ccdd % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		std::memcpy(buf + 6, char_table + dd * 2, 2);
		return buf + 8;
	}
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
auto* to_chars_u64_len_5_8(auto* buf, value_type value) noexcept {
	if (value < 1000000) { /* 5-6 digits: aabbcc */
		const uint32_t aa	= uint32_t((uint64_t(value) * 429497) >> 32); /* (value / 10000) */
		const uint32_t bbcc = value - aa * 10000; /* (value % 10000) */
		const uint32_t bb	= (bbcc * 5243) >> 19; /* (bbcc / 100) */
		const uint32_t cc	= bbcc - bb * 100; /* (bbcc % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (value / 10000) */
		const uint32_t aabb = uint32_t((uint64_t(value) * 109951163) >> 40);
		const uint32_t ccdd = value - aabb * 10000; /* (value % 10000) */
		const uint32_t aa	= (aabb * 5243) >> 19; /* (aabb / 100) */
		const uint32_t cc	= (ccdd * 5243) >> 19; /* (ccdd / 100) */
		const uint32_t bb	= aabb - aa * 100; /* (aabb % 100) */
		const uint32_t dd	= ccdd - cc * 100; /* (ccdd % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		std::memcpy(buf + 6, char_table + dd * 2, 2);
		return buf + 8;
	}
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint64_t>)
auto* to_chars(auto* buf, value_type value) noexcept {
	if (value < 100000000) { /* 1-8 digits */
		buf = to_chars_u64_len_1_8(buf, uint32_t(value));
		return buf;
	} else if (value < 100000000ull * 100000000ull) { /* 9-16 digits */
		const uint64_t hgh = value / 100000000;
		const auto low	   = uint32_t(value - hgh * 100000000); /* (value % 100000000) */
		buf				   = to_chars_u64_len_1_8(buf, uint32_t(hgh));
		buf				   = to_chars_u64_len_8(buf, low);
		return buf;
	} else { /* 17-20 digits */
		const uint64_t tmp = value / 100000000;
		const auto low	   = uint32_t(value - tmp * 100000000); /* (value % 100000000) */
		const auto hgh	   = uint32_t(tmp / 10000);
		const auto mid	   = uint32_t(tmp - hgh * 10000); /* (tmp % 10000) */
		buf				   = to_chars_u64_len_5_8(buf, hgh);
		buf				   = to_chars_u64_len_4(buf, mid);
		buf				   = to_chars_u64_len_8(buf, low);
		return buf;
	}
}

template<typename value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, int64_t>
auto* to_chars(auto* buf, value_type x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int64_t>::min case
	return to_chars(buf + (x < 0), uint64_t(x ^ (x >> 63)) - (x >> 63));
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
JSONIFIER_INLINE string_buffer_ptr to_chars_u64_len_8_new(string_buffer_ptr buf, value_type value) noexcept {
	/* 8 digits: aabbccdd */
	const uint32_t aabb = uint32_t((uint64_t(value) * 109951163) >> 40); /* (value / 10000) */
	const uint32_t ccdd = value - aabb * 10000; /* (value % 10000) */
	const uint32_t aa	= (aabb * 5243) >> 19; /* (aabb / 100) */
	const uint32_t cc	= (ccdd * 5243) >> 19; /* (ccdd / 100) */
	const uint32_t bb	= aabb - aa * 100; /* (aabb % 100) */
	const uint32_t dd	= ccdd - cc * 100; /* (ccdd % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, jsonifier::internal::int_tables<>::charTable02 + bb, 2);
	std::memcpy(buf + 4, jsonifier::internal::int_tables<>::charTable02 + cc, 2);
	std::memcpy(buf + 6, jsonifier::internal::int_tables<>::charTable02 + dd, 2);
	return buf + 8;
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
JSONIFIER_INLINE string_buffer_ptr to_chars_u64_len_4_new(string_buffer_ptr buf, value_type value) noexcept {
	/* 4 digits: aabb */
	const uint32_t aa = (value * 5243) >> 19; /* (value / 100) */
	const uint32_t bb = value - aa * 100; /* (value % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, jsonifier::internal::int_tables<>::charTable02 + bb, 2);
	return buf + 4;
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
JSONIFIER_INLINE string_buffer_ptr to_chars_u64_len_1_8_new(string_buffer_ptr buf, value_type value) noexcept {
	uint32_t aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

	if (value < 100) { /* 1-2 digits: aa */
		lz = value < 10;
		std::memcpy(buf, char_table + value * 2 + lz, 2);
		buf -= lz;
		return buf + 2;
	} else if (value < 10000) { /* 3-4 digits: aabb */
		aa = (value * 5243) >> 19; /* (value / 100) */
		bb = value - aa * 100; /* (value % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, jsonifier::internal::int_tables<>::charTable02 + bb, 2);
		return buf + 4;
	} else if (value < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(value) * 429497) >> 32); /* (value / 10000) */
		bbcc = value - aa * 10000; /* (value % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, jsonifier::internal::int_tables<>::charTable02 + bb, 2);
		std::memcpy(buf + 4, jsonifier::internal::int_tables<>::charTable02 + cc, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (value / 10000) */
		aabb = uint32_t((uint64_t(value) * 109951163) >> 40);
		ccdd = value - aabb * 10000; /* (value % 10000) */
		aa	 = (aabb * 5243) >> 19; /* (aabb / 100) */
		cc	 = (ccdd * 5243) >> 19; /* (ccdd / 100) */
		bb	 = aabb - aa * 100; /* (aabb % 100) */
		dd	 = ccdd - cc * 100; /* (ccdd % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, jsonifier::internal::int_tables<>::charTable02 + bb, 2);
		std::memcpy(buf + 4, jsonifier::internal::int_tables<>::charTable02 + cc, 2);
		std::memcpy(buf + 6, jsonifier::internal::int_tables<>::charTable02 + dd, 2);
		return buf + 8;
	}
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint32_t>)
JSONIFIER_INLINE string_buffer_ptr to_chars_u64_len_5_8_new(string_buffer_ptr buf, value_type value) noexcept {
	if (value < 1000000) { /* 5-6 digits: aabbcc */
		const uint32_t aa	= uint32_t((uint64_t(value) * 429497) >> 32); /* (value / 10000) */
		const uint32_t bbcc = value - aa * 10000; /* (value % 10000) */
		const uint32_t bb	= (bbcc * 5243) >> 19; /* (bbcc / 100) */
		const uint32_t cc	= bbcc - bb * 100; /* (bbcc % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, jsonifier::internal::int_tables<>::charTable02 + bb, 2);
		std::memcpy(buf + 4, jsonifier::internal::int_tables<>::charTable02 + cc, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (value / 10000) */
		const uint32_t aabb = uint32_t((uint64_t(value) * 109951163) >> 40);
		const uint32_t ccdd = value - aabb * 10000; /* (value % 10000) */
		const uint32_t aa	= (aabb * 5243) >> 19; /* (aabb / 100) */
		const uint32_t cc	= (ccdd * 5243) >> 19; /* (ccdd / 100) */
		const uint32_t bb	= aabb - aa * 100; /* (aabb % 100) */
		const uint32_t dd	= ccdd - cc * 100; /* (ccdd % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, jsonifier::internal::int_tables<>::charTable02 + bb, 2);
		std::memcpy(buf + 4, jsonifier::internal::int_tables<>::charTable02 + cc, 2);
		std::memcpy(buf + 6, jsonifier::internal::int_tables<>::charTable02 + dd, 2);
		return buf + 8;
	}
}

template<typename value_type>
	requires(std::same_as<std::remove_cvref_t<value_type>, uint64_t>)
JSONIFIER_INLINE string_buffer_ptr to_chars_new(string_buffer_ptr buf, value_type value) noexcept {
	if (value < 100000000) { /* 1-8 digits */
		buf = to_chars_u64_len_1_8(buf, uint32_t(value));
		return buf;
	} else if (value < 100000000ull * 100000000ull) { /* 9-16 digits */
		const uint64_t hgh = value / 100000000;
		const auto low	   = uint32_t(value - hgh * 100000000); /* (value % 100000000) */
		buf				   = to_chars_u64_len_1_8_new(buf, uint32_t(hgh));
		buf				   = to_chars_u64_len_8_new(buf, low);
		return buf;
	} else { /* 17-20 digits */
		const uint64_t tmp = value / 100000000;
		const auto low	   = uint32_t(value - tmp * 100000000); /* (value % 100000000) */
		const auto hgh	   = uint32_t(tmp / 10000);
		const auto mid	   = uint32_t(tmp - hgh * 10000); /* (tmp % 10000) */
		buf				   = to_chars_u64_len_5_8_new(buf, hgh);
		buf				   = to_chars_u64_len_4_new(buf, mid);
		buf				   = to_chars_u64_len_8_new(buf, low);
		return buf;
	}
}

static constexpr char radix_100_table[] = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1',
	'3', '1', '4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', '3', '0',
	'3', '1', '3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4',
	'8', '4', '9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5',
	'6', '6', '6', '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8',
	'3', '8', '4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9' };

char* itoa_always_10_digits(std::uint64_t n, char* buffer) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= n * std::uint64_t(1441151881);
	std::memcpy(buffer + 0, radix_100_table + int(y >> 57) * 2, 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, radix_100_table + int(y >> 57) * 2, 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 4, radix_100_table + int(y >> 57) * 2, 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 6, radix_100_table + int(y >> 57) * 2, 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 8, radix_100_table + int(y >> 57) * 2, 2);

	return buffer + 10;
}

template<typename value_type>
	requires std::same_as<std::remove_cvref_t<value_type>, int64_t>
JSONIFIER_INLINE string_buffer_ptr to_chars_new(string_buffer_ptr buf, value_type x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int64_t>::min case
	return to_chars(buf + (x < 0), uint64_t(x ^ (x >> 63)) - (x >> 63));
}

uint64_t generateRandomIntegerByLength(uint32_t digitLength) {
	std::uniform_int_distribution<uint64_t> distLength(1, digitLength);
	std::uniform_int_distribution<uint64_t> dist01(1, 9);
	std::uniform_int_distribution<uint64_t> dist02(0, 9);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	//digitLength = distLength(gen);

	char buffer[22]{};
	buffer[0] = static_cast<char>(dist01(gen) + '0');

	for (uint64_t x = 1ull; x < digitLength; ++x) {
		buffer[x] = static_cast<char>(dist02(gen) + '0');
	}

	buffer[digitLength] = '\0';
	return std::strtoull(buffer, nullptr, 10);
}

template<typename value_type> std::vector<value_type> generateRandomIntegers(uint64_t count, uint64_t maxLength = 0) {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<value_type> lengthNewGen(1, 20);
	std::vector<value_type> randomNumbers;

	for (uint64_t value = 0ull; value < count; ++value) {
		uint64_t newValue{ generateRandomIntegerByLength(maxLength == 0 ? lengthNewGen(gen) : maxLength) };
		randomNumbers.push_back(newValue);
	}

	return randomNumbers;
}

template<typename value_type> auto generateVectorOfVectors(uint64_t count01, uint64_t count02, uint64_t lengthNew) {
	std::vector<std::vector<value_type>> returnValues{};
	for (uint64_t x = 0ull; x < count01; ++x) {
		returnValues.emplace_back(generateRandomIntegers<value_type>(count02, lengthNew));
	}
	return returnValues;
}

static constexpr auto maxIterations{ 300 };
static constexpr auto measuredIterations{ 20 };

JSONIFIER_INLINE_VARIABLE uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10,
	10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

JSONIFIER_INLINE_VARIABLE uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull,
	99999999999ull, 999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull,
	9999999999999999999ull };

JSONIFIER_INLINE uint64_t fastDigitCount(const uint64_t inputValue) {
	const uint64_t originalDigitCount{ digitCounts[jsonifier::simd::lzcnt(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > digitCountThresholds[originalDigitCount]);
}

template<uint64_t count, uint64_t lengthNew, typename value_type, bnch_swt::string_literal testName> BNCH_SWT_INLINE void testFunction() {
	std::vector<std::vector<value_type>> testValues{ generateVectorOfVectors<value_type>(maxIterations * measuredIterations, count, lengthNew) };
	std::vector<std::vector<std::string>> testValues00{};
	std::vector<std::vector<std::string>> testValues01{};
	testValues01.resize(maxIterations * measuredIterations);
	for (uint64_t x = 0ull; x < maxIterations * measuredIterations; ++x) {
		testValues01[x].resize(count);
	}
	testValues00.resize(maxIterations * measuredIterations);
	testValues01.resize(maxIterations * measuredIterations);
	for (uint64_t x = 0ull; x < maxIterations * measuredIterations; ++x) {
		for (uint64_t y = 0ull; y < count; ++y) {
			testValues00[x].emplace_back(std::to_string(testValues[x][y]));
		}
	}
	uint64_t currentIteration{};
	std::vector<std::array<char, 30>> newerStrings{};
	newerStrings.resize(maxIterations * measuredIterations);
	srand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"glz::to_chars", "CYAN">([&] {
		uint64_t bytesProcessed{};
		for (uint64_t x = 0ull; x < count; ++x) {
			auto newPtr = to_chars(newerStrings[currentIteration].data(), testValues[currentIteration][x]);
			bytesProcessed += testValues00[currentIteration][x].size();
			testValues01[currentIteration][x] = std::string{ newerStrings[currentIteration].data(), static_cast<uint64_t>(newPtr - newerStrings[currentIteration].data()) };
		}
		bnch_swt::doNotOptimizeAway(bytesProcessed);
		++currentIteration;
		return bytesProcessed;
	});
	std::cout << "TOTAL ITERATIONS: " << currentIteration << std::endl;
	for (uint64_t x = 0ull; x < currentIteration; ++x) {
		for (uint64_t y = 0ull; y < count; ++y) {
			if (testValues00[x][y] != testValues01[x][y]) {
				std::cout << "GLZ FAILED TO SERIALIZE THIS VALUE: " << testValues00[x][y] << std::endl;
				std::cout << "GLZ FAILED TO SERIALIZE THIS VALUE (RAW): " << testValues[x][y] << std::endl;
				std::cout << "GLZ FAILED TO SERIALIZE THIS VALUE-SIZE (RAW): " << testValues00[x][y].size() << std::endl;
				std::cout << "INSTEAD IT PRODUCED THIS VALUE-SIZE: " << testValues01[x][y].size() << std::endl;
				std::cout << "INSTEAD IT PRODUCED THIS VALUE: " << testValues01[x][y] << std::endl;
			}
		}
	}

	currentIteration = 0ull;
	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::template runBenchmark<"jsonifier::internal::toChars", "CYAN">([&] {
		uint64_t bytesProcessed{};
		for (uint64_t x = 0ull; x < count; ++x) {
			auto newPtr = jsonifier::internal::toChars(newerStrings[currentIteration].data(), testValues[currentIteration][x]);
			bytesProcessed += testValues00[currentIteration][x].size();
			testValues01[currentIteration][x] = std::string{ newerStrings[currentIteration].data(), static_cast<uint64_t>(newPtr - newerStrings[currentIteration].data()) };
		}
		bnch_swt::doNotOptimizeAway(bytesProcessed);
		++currentIteration;
		return bytesProcessed;
	});
	for (uint64_t x = 0ull; x < currentIteration; ++x) {
		for (uint64_t y = 0ull; y < count; ++y) {
			if (testValues00[x][y] != testValues01[x][y]) {
				std::cout << "jsonifier::internal::toChars FAILED TO SERIALIZE THIS VALUE: " << testValues00[x][y] << std::endl;
				std::cout << "jsonifier::internal::toChars FAILED TO SERIALIZE THIS VALUE (RAW): " << testValues[x][y] << std::endl;
				std::cout << "jsonifier::internal::toChars FAILED TO SERIALIZE THIS VALUE-SIZE (RAW): " << testValues00[x][y].size() << std::endl;
				std::cout << "INSTEAD IT PRODUCED THIS VALUE-SIZE: " << testValues01[x][y].size() << std::endl;
				std::cout << "INSTEAD IT PRODUCED THIS VALUE: " << testValues01[x][y] << std::endl;
			}
		}
	}

	currentIteration = 0ull;

	bnch_swt::benchmark_stage<testName, maxIterations, measuredIterations>::printResults(true, true);
}

int main() {
	testFunction<512, 1, uint64_t, "int-to-string-comparisons-1">();
	testFunction<512, 1, int64_t, "int-to-string-comparisons-1">();
	testFunction<522, 2, uint64_t, "int-to-string-comparisons-2">();
	testFunction<522, 2, int64_t, "int-to-string-comparisons-2">();
	testFunction<542, 4, uint64_t, "int-to-string-comparisons-4">();
	testFunction<542, 4, int64_t, "int-to-string-comparisons-4">();
	testFunction<582, 8, uint64_t, "int-to-string-comparisons-8">();
	testFunction<582, 8, int64_t, "int-to-string-comparisons-8">();
	testFunction<5162, 16, uint64_t, "int-to-string-comparisons-16">();
	testFunction<5162, 16, int64_t, "int-to-string-comparisons-16">();
	testFunction<512, 0, uint64_t, "int-to-string-comparisons-x">();
	testFunction<512, 0, int64_t, "int-to-string-comparisons-x">();
	return 0ull;
}