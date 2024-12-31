#include <cstdint>
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

constexpr char char_table[200] = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3', '1',
	'4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', '3', '0', '3', '1',
	'3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8', '4',
	'9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6',
	'6', '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8', '3', '8',
	'4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9' };

template<class T>
	requires std::same_as<std::remove_cvref_t<T>, uint32_t>
auto* to_chars(auto* buf, T val) noexcept {
	/* The maximum value of uint32_t is 4294967295 (10 digits), */
	/* these digits are named as 'aabbccddee' here.             */
	uint32_t aa, bb, cc, dd, ee, aabb, bbcc, ccdd, ddee, aabbcc;

	/* Leading zero count in the first pair.                    */
	uint32_t lz;

	/* Although most compilers may convert the "division by     */
	/* constant value" into "multiply and shift", manual        */
	/* conversion can still help some compilers generate        */
	/* fewer and better instructions.                           */

	if (val < 100) { /* 1-2 digits: aa */
		lz = val < 10;
		std::memcpy(buf, char_table + (val * 2 + lz), 2);
		buf -= lz;
		return buf + 2;
	} else if (val < 10000) { /* 3-4 digits: aabb */
		aa = (val * 5243) >> 19; /* (val / 100) */
		bb = val - aa * 100; /* (val % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + (aa * 2 + lz), 2);
		buf -= lz;
		std::memcpy(&buf[2], char_table + (2 * bb), 2);

		return buf + 4;
	} else if (val < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(val) * 429497) >> 32); /* (val / 10000) */
		bbcc = val - aa * 10000; /* (val % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else if (val < 100000000) { /* 7~8 digits: aabbccdd */
		/* (val / 10000) */
		aabb = uint32_t((uint64_t(val) * 109951163) >> 40);
		ccdd = val - aabb * 10000; /* (val % 10000) */
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
	} else { /* 9~10 digits: aabbccddee */
		/* (val / 10000) */
		aabbcc = uint32_t((uint64_t(val) * 3518437209ul) >> 45);
		/* (aabbcc / 10000) */
		aa	 = uint32_t((uint64_t(aabbcc) * 429497) >> 32);
		ddee = val - aabbcc * 10000; /* (val % 10000) */
		bbcc = aabbcc - aa * 10000; /* (aabbcc % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		dd	 = (ddee * 5243) >> 19; /* (ddee / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		ee	 = ddee - dd * 100; /* (ddee % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		std::memcpy(buf + 6, char_table + dd * 2, 2);
		std::memcpy(buf + 8, char_table + ee * 2, 2);
		return buf + 10;
	}
}

template<class T>
	requires std::same_as<std::remove_cvref_t<T>, int32_t>
auto* to_chars(auto* buf, T x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int32_t>::min case
	return to_chars(buf + (x < 0), uint32_t(x ^ (x >> 31)) - (x >> 31));
}

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
JSONIFIER_INLINE auto* to_chars_u64_len_8(auto* buf, T val) noexcept {
	/* 8 digits: aabbccdd */
	const uint32_t aabb = uint32_t((uint64_t(val) * 109951163) >> 40); /* (val / 10000) */
	const uint32_t ccdd = val - aabb * 10000; /* (val % 10000) */
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
JSONIFIER_INLINE auto* to_chars_u64_len_4(auto* buf, T val) noexcept {
	/* 4 digits: aabb */
	const uint32_t aa = (val * 5243) >> 19; /* (val / 100) */
	const uint32_t bb = val - aa * 100; /* (val % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, char_table + bb * 2, 2);
	return buf + 4;
}

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
inline auto* to_chars_u64_len_1_8(auto* buf, T val) noexcept {
	uint32_t aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

	if (val < 100) { /* 1-2 digits: aa */
		lz = val < 10;
		std::memcpy(buf, char_table + val * 2 + lz, 2);
		buf -= lz;
		return buf + 2;
	} else if (val < 10000) { /* 3-4 digits: aabb */
		aa = (val * 5243) >> 19; /* (val / 100) */
		bb = val - aa * 100; /* (val % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		return buf + 4;
	} else if (val < 1000000) { /* 5-6 digits: aabbcc */
		aa	 = uint32_t((uint64_t(val) * 429497) >> 32); /* (val / 10000) */
		bbcc = val - aa * 10000; /* (val % 10000) */
		bb	 = (bbcc * 5243) >> 19; /* (bbcc / 100) */
		cc	 = bbcc - bb * 100; /* (bbcc % 100) */
		lz	 = aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (val / 10000) */
		aabb = uint32_t((uint64_t(val) * 109951163) >> 40);
		ccdd = val - aabb * 10000; /* (val % 10000) */
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
auto* to_chars_u64_len_5_8(auto* buf, T val) noexcept {
	if (val < 1000000) { /* 5-6 digits: aabbcc */
		const uint32_t aa	= uint32_t((uint64_t(val) * 429497) >> 32); /* (val / 10000) */
		const uint32_t bbcc = val - aa * 10000; /* (val % 10000) */
		const uint32_t bb	= (bbcc * 5243) >> 19; /* (bbcc / 100) */
		const uint32_t cc	= bbcc - bb * 100; /* (bbcc % 100) */
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, char_table + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, char_table + bb * 2, 2);
		std::memcpy(buf + 4, char_table + cc * 2, 2);
		return buf + 6;
	} else { /* 7-8 digits: aabbccdd */
		/* (val / 10000) */
		const uint32_t aabb = uint32_t((uint64_t(val) * 109951163) >> 40);
		const uint32_t ccdd = val - aabb * 10000; /* (val % 10000) */
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint64_t>)
auto* to_chars(auto* buf, T val) noexcept {
	if (val < 100000000) { /* 1-8 digits */
		buf = to_chars_u64_len_1_8(buf, uint32_t(val));
		return buf;
	} else if (val < 100000000ull * 100000000ull) { /* 9-16 digits */
		const uint64_t hgh = val / 100000000;
		const auto low	   = uint32_t(val - hgh * 100000000); /* (val % 100000000) */
		buf				   = to_chars_u64_len_1_8(buf, uint32_t(hgh));
		buf				   = to_chars_u64_len_8(buf, low);
		return buf;
	} else { /* 17-20 digits */
		const uint64_t tmp = val / 100000000;
		const auto low	   = uint32_t(val - tmp * 100000000); /* (val % 100000000) */
		const auto hgh	   = uint32_t(tmp / 10000);
		const auto mid	   = uint32_t(tmp - hgh * 10000); /* (tmp % 10000) */
		buf				   = to_chars_u64_len_5_8(buf, hgh);
		buf				   = to_chars_u64_len_4(buf, mid);
		buf				   = to_chars_u64_len_8(buf, low);
		return buf;
	}
}

template<class T>
	requires std::same_as<std::remove_cvref_t<T>, int64_t>
auto* to_chars(auto* buf, T x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int64_t>::min case
	return to_chars(buf + (x < 0), uint64_t(x ^ (x >> 63)) - (x >> 63));
}

uint64_t generate_random_integer() {
	// Random number generator and uniform distribution for length (1 to 20 digits)
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> length_dist(1, 20);// Random length from 1 to 20 digits

	// Get the random length
	int length = length_dist(gen);

	// Calculate the lower and upper bounds for the number of digits
	uint64_t lower_bound = 1;
	uint64_t upper_bound = 9;
	for (int i = 1; i < length; ++i) {
		lower_bound *= 10;
		upper_bound *= 10;
	}
	upper_bound -= 1;

	// Generate the random number with the selected length
	std::uniform_int_distribution<uint64_t> number_dist(lower_bound, upper_bound);
	return number_dist(gen);
}

int digit_count(uint64_t x) {
	if (x == 0)
		return 1;

	return static_cast<int>(std::log10(x)) + 1;
}

template<typename value_type, bnch_swt::string_literal testName> BNCH_SWT_ALWAYS_INLINE void testFunction() {
	std::vector<value_type> testValues{};
	std::vector<std::string> testValues00{};
	std::vector<std::string> testValues01{};
	std::vector<std::string> testValues02{};
	std::vector<std::string> testValues03{};
	testValues01.resize(64);
	testValues03.resize(64);
	testValues02.resize(64);
	for (size_t x = 0; x < 64; ++x) {
		std::string newString{};
		value_type newValue{};
		if constexpr (std::is_same_v<value_type, uint64_t>) {
			newValue  = generate_random_integer();
			newString = std::to_string(newValue);
		} else {
			newValue = generate_random_integer();
			newValue *= bnch_swt::random_generator::generateBool() ? -1 : 1;
			newString = std::to_string(newValue);
		}
		if (newValue < 0) {
			newString.resize(bnch_swt::countDigits(newValue) + 1);
		} else {
			newString.resize(bnch_swt::countDigits(newValue));
		}
		auto endPtr = newString.data() + newString.size() + 1;
		testValues.emplace_back(std::strtoull(newString.data(), &endPtr, 10));
		testValues00.emplace_back(newString);
	}

	bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::template runBenchmark<"glz-I-To-Str", "CYAN">([&] {
		size_t bytesProcessed{};
		char newerString[32]{};
		size_t currentSize{};
		for (size_t x = 0; x < 64; ++x) {
			std::memset(newerString, '\0', std::size(newerString));
			auto newPtr = to_chars(newerString, testValues[x]);
			if (testValues[x] < 0) {
				currentSize = (newPtr - newerString) + 1;
			} else {
				currentSize = newPtr - newerString;
			}
			bytesProcessed += currentSize;
			testValues01[x] = std::string{ newerString, currentSize };
		}
		return bytesProcessed;
	});
	for (size_t x = 0; x < 64; ++x) {
		if (testValues00[x] != testValues01[x]) {
			std::cout << "GLZ FAILED TO SERIALIZE THIS VALUE: " << testValues00[x] << std::endl;
			std::cout << "GLZ FAILED TO SERIALIZE THIS VALUE (RAW): " << testValues[x] << std::endl;
			std::cout << "INSTEAD IT PRODUCED THIS VALUE: " << testValues01[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::template runBenchmark<"New-I-To-Str", "CYAN">([&] {
		size_t bytesProcessed{};
		char newerString[32]{};
		size_t currentSize{};
		for (size_t x = 0; x < 64; ++x) {
			std::memset(newerString, '\0', std::size(newerString));
			auto newPtr = jsonifier_internal::toChars<value_type>(newerString, testValues[x]);
			if (testValues[x] < 0) {
				currentSize = (newPtr - newerString) + 1;
			} else {
				currentSize = newPtr - newerString;
			}
			bytesProcessed += currentSize;
			testValues03[x] = std::string{ newerString, currentSize };
		}
		return bytesProcessed;
	});
	for (size_t x = 0; x < 64; ++x) {
		if (testValues00[x] != testValues03[x]) {
			std::cout << "JSONIFIER FAILED TO SERIALIZE THIS VALUE: " << testValues00[x] << std::endl;
			std::cout << "JSONIFIER FAILED TO SERIALIZE THIS VALUE (RAW): " << testValues[x] << std::endl;
			std::cout << "INSTEAD IT PRODUCED THIS VALUE: " << testValues03[x] << std::endl;
		}
	}
	bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::printResults();
}

int main() {
	//printCharTable3(charTable3);
	//std::cout << "INTEGER LOG 10 Of 23342323123: " << integerLog10(23342323123) + 1 << std::endl;
	/**/
	testFunction<uint64_t, "-uint">();
	testFunction<int64_t, "-int">();
	return 0;
}