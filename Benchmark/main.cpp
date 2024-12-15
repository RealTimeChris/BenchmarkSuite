#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include <glaze/glaze.hpp>
#include "RandomGenerators.hpp"
#include "absl/container/flat_hash_map.h"

uint64_t generateRandomIntegerByLength(uint64_t digitLength) {
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
		uint64_t newValue{ generateRandomIntegerByLength(maxLength == 0ull ? lengthNewGen(gen) : maxLength) };
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

static constexpr auto maxIterations{ 400 };

#define GLZ_PARENS ()

// binary expansion is much more compile time efficient than quaternary expansion
#define GLZ_EXPAND(...) GLZ_EXPAND32(__VA_ARGS__)
#define GLZ_EXPAND32(...) GLZ_EXPAND16(GLZ_EXPAND16(__VA_ARGS__))
#define GLZ_EXPAND16(...) GLZ_EXPAND8(GLZ_EXPAND8(__VA_ARGS__))
#define GLZ_EXPAND8(...) GLZ_EXPAND4(GLZ_EXPAND4(__VA_ARGS__))
#define GLZ_EXPAND4(...) GLZ_EXPAND2(GLZ_EXPAND2(__VA_ARGS__))
#define GLZ_EXPAND2(...) GLZ_EXPAND1(GLZ_EXPAND1(__VA_ARGS__))
#define GLZ_EXPAND1(...) __VA_ARGS__

#define GLZ_FOR_EACH(macro, ...) __VA_OPT__(GLZ_EXPAND(GLZ_FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define GLZ_FOR_EACH_HELPER(macro, a, ...) macro(a) __VA_OPT__(, ) __VA_OPT__(GLZ_FOR_EACH_AGAIN GLZ_PARENS(macro, __VA_ARGS__))
#define GLZ_FOR_EACH_AGAIN() GLZ_FOR_EACH_HELPER

#define GLZ_EVERY(macro, ...) __VA_OPT__(GLZ_EXPAND(GLZ_EVERY_HELPER(macro, __VA_ARGS__)))
#define GLZ_EVERY_HELPER(macro, a, ...) macro(a) __VA_OPT__(GLZ_EVERY_AGAIN GLZ_PARENS(macro, __VA_ARGS__))
#define GLZ_EVERY_AGAIN() GLZ_EVERY_HELPER

#define GLZ_CASE(I) \
	case I: { \
		lambda.template operator()<I>(); \
		break; \
	}

#define GLZ_SWITCH(X, ...) \
	else if constexpr (N == X) { \
		switch (index) { \
			GLZ_EVERY(GLZ_CASE, __VA_ARGS__); \
			default: { \
				std::unreachable(); \
			} \
		} \
	}

#define GLZ_10 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#define GLZ_20 GLZ_10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#define GLZ_30 GLZ_20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
#define GLZ_40 GLZ_30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
#define GLZ_50 GLZ_40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
#define GLZ_60 GLZ_50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60

template<size_t N, class Lambda> GLZ_ALWAYS_INLINE constexpr void glz_jump_table(Lambda&& lambda, size_t index) {
	if constexpr (N == 0) {
		return;
	} else if constexpr (N == 1) {
		lambda.template operator()<0>();
	}
	GLZ_SWITCH(2, 0, 1)
	GLZ_SWITCH(3, 0, 1, 2)
	GLZ_SWITCH(4, 0, 1, 2, 3)
	GLZ_SWITCH(5, 0, 1, 2, 3, 4)
	GLZ_SWITCH(6, 0, 1, 2, 3, 4, 5)
	GLZ_SWITCH(7, 0, 1, 2, 3, 4, 5, 6)
	GLZ_SWITCH(8, 0, 1, 2, 3, 4, 5, 6, 7)
	GLZ_SWITCH(9, 0, 1, 2, 3, 4, 5, 6, 7, 8)
	GLZ_SWITCH(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	GLZ_SWITCH(11, GLZ_10)
	GLZ_SWITCH(12, GLZ_10, 11)
	GLZ_SWITCH(13, GLZ_10, 11, 12)
	GLZ_SWITCH(14, GLZ_10, 11, 12, 13)
	GLZ_SWITCH(15, GLZ_10, 11, 12, 13, 14)
	GLZ_SWITCH(16, GLZ_10, 11, 12, 13, 14, 15)
	GLZ_SWITCH(17, GLZ_10, 11, 12, 13, 14, 15, 16)
	GLZ_SWITCH(18, GLZ_10, 11, 12, 13, 14, 15, 16, 17)
	GLZ_SWITCH(19, GLZ_10, 11, 12, 13, 14, 15, 16, 17, 18)
	GLZ_SWITCH(20, GLZ_10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
	GLZ_SWITCH(21, GLZ_20)
	GLZ_SWITCH(22, GLZ_20, 21)
	GLZ_SWITCH(23, GLZ_20, 21, 22)
	GLZ_SWITCH(24, GLZ_20, 21, 22, 23)
	GLZ_SWITCH(25, GLZ_20, 21, 22, 23, 24)
	GLZ_SWITCH(26, GLZ_20, 21, 22, 23, 24, 25)
	GLZ_SWITCH(27, GLZ_20, 21, 22, 23, 24, 25, 26)
	GLZ_SWITCH(28, GLZ_20, 21, 22, 23, 24, 25, 26, 27)
	GLZ_SWITCH(29, GLZ_20, 21, 22, 23, 24, 25, 26, 27, 28)
	GLZ_SWITCH(30, GLZ_20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
	GLZ_SWITCH(31, GLZ_30)
	GLZ_SWITCH(32, GLZ_30, 31)
	GLZ_SWITCH(33, GLZ_30, 31, 32)
	GLZ_SWITCH(34, GLZ_30, 31, 32, 33)
	GLZ_SWITCH(35, GLZ_30, 31, 32, 33, 34)
	GLZ_SWITCH(36, GLZ_30, 31, 32, 33, 34, 35)
	GLZ_SWITCH(37, GLZ_30, 31, 32, 33, 34, 35, 36)
	GLZ_SWITCH(38, GLZ_30, 31, 32, 33, 34, 35, 36, 37)
	GLZ_SWITCH(39, GLZ_30, 31, 32, 33, 34, 35, 36, 37, 38)
	GLZ_SWITCH(40, GLZ_30, 31, 32, 33, 34, 35, 36, 37, 38, 39)
	GLZ_SWITCH(41, GLZ_40)
	GLZ_SWITCH(42, GLZ_40, 41)
	GLZ_SWITCH(43, GLZ_40, 41, 42)
	GLZ_SWITCH(44, GLZ_40, 41, 42, 43)
	GLZ_SWITCH(45, GLZ_40, 41, 42, 43, 44)
	GLZ_SWITCH(46, GLZ_40, 41, 42, 43, 44, 45)
	GLZ_SWITCH(47, GLZ_40, 41, 42, 43, 44, 45, 46)
	GLZ_SWITCH(48, GLZ_40, 41, 42, 43, 44, 45, 46, 47)
	GLZ_SWITCH(49, GLZ_40, 41, 42, 43, 44, 45, 46, 47, 48)
	GLZ_SWITCH(50, GLZ_40, 41, 42, 43, 44, 45, 46, 47, 48, 49)
	GLZ_SWITCH(51, GLZ_50)
	GLZ_SWITCH(52, GLZ_50, 51)
	GLZ_SWITCH(53, GLZ_50, 51, 52)
	GLZ_SWITCH(54, GLZ_50, 51, 52, 53)
	GLZ_SWITCH(55, GLZ_50, 51, 52, 53, 54)
	GLZ_SWITCH(56, GLZ_50, 51, 52, 53, 54, 55)
	GLZ_SWITCH(57, GLZ_50, 51, 52, 53, 54, 55, 56)
	GLZ_SWITCH(58, GLZ_50, 51, 52, 53, 54, 55, 56, 57)
	GLZ_SWITCH(59, GLZ_50, 51, 52, 53, 54, 55, 56, 57, 58)
	GLZ_SWITCH(60, GLZ_50, 51, 52, 53, 54, 55, 56, 57, 58, 59)
	GLZ_SWITCH(61, GLZ_60)
	GLZ_SWITCH(62, GLZ_60, 61)
	GLZ_SWITCH(63, GLZ_60, 61, 62)
	GLZ_SWITCH(64, GLZ_60, 61, 62, 63)
	else {
		glz::for_each_short_circuit<N>([&](auto I) {
			if (index == I) {
				lambda.template operator()<I>();
				return true;
			}
			return false;
		});
	}
}

// Samples from Stephen Berry and his library, Glaze: https://github.com/stephenberry/Glaze
#define JSONIFIER_PARENS ()

#define JSONIFIER_EXPAND(...) JSONIFIER_EXPAND32(__VA_ARGS__)
#define JSONIFIER_EXPAND32(...) JSONIFIER_EXPAND16(JSONIFIER_EXPAND16(__VA_ARGS__))
#define JSONIFIER_EXPAND16(...) JSONIFIER_EXPAND8(JSONIFIER_EXPAND8(__VA_ARGS__))
#define JSONIFIER_EXPAND8(...) JSONIFIER_EXPAND4(JSONIFIER_EXPAND4(__VA_ARGS__))
#define JSONIFIER_EXPAND4(...) JSONIFIER_EXPAND2(JSONIFIER_EXPAND2(__VA_ARGS__))
#define JSONIFIER_EXPAND2(...) JSONIFIER_EXPAND1(JSONIFIER_EXPAND1(__VA_ARGS__))
#define JSONIFIER_EXPAND1(...) __VA_ARGS__

#define JSONIFIER_FOR_EACH(macro, ...) __VA_OPT__(JSONIFIER_EXPAND(JSONIFIER_FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define JSONIFIER_FOR_EACH_HELPER(macro, a, ...) macro(a) __VA_OPT__(, ) __VA_OPT__(JSONIFIER_FOR_EACH_AGAIN JSONIFIER_PARENS(macro, __VA_ARGS__))
#define JSONIFIER_FOR_EACH_AGAIN() JSONIFIER_FOR_EACH_HELPER

#define JSONIFIER_EVERY(macro, ...) __VA_OPT__(JSONIFIER_EXPAND(JSONIFIER_EVERY_HELPER(macro, __VA_ARGS__)))
#define JSONIFIER_EVERY_HELPER(macro, a, ...) macro(a) __VA_OPT__(JSONIFIER_EVERY_AGAIN JSONIFIER_PARENS(macro, __VA_ARGS__))
#define JSONIFIER_EVERY_AGAIN() JSONIFIER_EVERY_HELPER

#define JSONIFIER_CASE(I) \
	case I: { \
		return base_type::template processIndex<I>(); \
	}

#define JSONIFIER_SWITCH(X, ...) \
	else if constexpr (N == X) { \
		switch (index) { \
			JSONIFIER_EVERY(JSONIFIER_CASE, __VA_ARGS__); \
			default: { \
				return false; \
			} \
		} \
	}

#define JSONIFIER_10 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#define JSONIFIER_20 JSONIFIER_10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#define JSONIFIER_30 JSONIFIER_20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
#define JSONIFIER_40 JSONIFIER_30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
#define JSONIFIER_50 JSONIFIER_40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
#define JSONIFIER_60 JSONIFIER_50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60

template<typename base_type, size_t size> struct switch_statement;

template<typename base_type, size_t N>
	requires(N <= 64)
struct switch_statement<base_type, N> : public base_type {
	static uint64_t impl(size_t index) noexcept {
		if constexpr (N == 0) {
			return N;
		} else if constexpr (N == 1) {
			return base_type::template processIndex<0>();
		}
		JSONIFIER_SWITCH(2, 0, 1)
		JSONIFIER_SWITCH(3, 0, 1, 2)
		JSONIFIER_SWITCH(4, 0, 1, 2, 3)
		JSONIFIER_SWITCH(5, 0, 1, 2, 3, 4)
		JSONIFIER_SWITCH(6, 0, 1, 2, 3, 4, 5)
		JSONIFIER_SWITCH(7, 0, 1, 2, 3, 4, 5, 6)
		JSONIFIER_SWITCH(8, 0, 1, 2, 3, 4, 5, 6, 7)
		JSONIFIER_SWITCH(9, 0, 1, 2, 3, 4, 5, 6, 7, 8)
		JSONIFIER_SWITCH(10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
		JSONIFIER_SWITCH(11, JSONIFIER_10)
		JSONIFIER_SWITCH(12, JSONIFIER_10, 11)
		JSONIFIER_SWITCH(13, JSONIFIER_10, 11, 12)
		JSONIFIER_SWITCH(14, JSONIFIER_10, 11, 12, 13)
		JSONIFIER_SWITCH(15, JSONIFIER_10, 11, 12, 13, 14)
		JSONIFIER_SWITCH(16, JSONIFIER_10, 11, 12, 13, 14, 15)
		JSONIFIER_SWITCH(17, JSONIFIER_10, 11, 12, 13, 14, 15, 16)
		JSONIFIER_SWITCH(18, JSONIFIER_10, 11, 12, 13, 14, 15, 16, 17)
		JSONIFIER_SWITCH(19, JSONIFIER_10, 11, 12, 13, 14, 15, 16, 17, 18)
		JSONIFIER_SWITCH(20, JSONIFIER_10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
		JSONIFIER_SWITCH(21, JSONIFIER_20)
		JSONIFIER_SWITCH(22, JSONIFIER_20, 21)
		JSONIFIER_SWITCH(23, JSONIFIER_20, 21, 22)
		JSONIFIER_SWITCH(24, JSONIFIER_20, 21, 22, 23)
		JSONIFIER_SWITCH(25, JSONIFIER_20, 21, 22, 23, 24)
		JSONIFIER_SWITCH(26, JSONIFIER_20, 21, 22, 23, 24, 25)
		JSONIFIER_SWITCH(27, JSONIFIER_20, 21, 22, 23, 24, 25, 26)
		JSONIFIER_SWITCH(28, JSONIFIER_20, 21, 22, 23, 24, 25, 26, 27)
		JSONIFIER_SWITCH(29, JSONIFIER_20, 21, 22, 23, 24, 25, 26, 27, 28)
		JSONIFIER_SWITCH(30, JSONIFIER_20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
		JSONIFIER_SWITCH(31, JSONIFIER_30)
		JSONIFIER_SWITCH(32, JSONIFIER_30, 31)
		JSONIFIER_SWITCH(33, JSONIFIER_30, 31, 32)
		JSONIFIER_SWITCH(34, JSONIFIER_30, 31, 32, 33)
		JSONIFIER_SWITCH(35, JSONIFIER_30, 31, 32, 33, 34)
		JSONIFIER_SWITCH(36, JSONIFIER_30, 31, 32, 33, 34, 35)
		JSONIFIER_SWITCH(37, JSONIFIER_30, 31, 32, 33, 34, 35, 36)
		JSONIFIER_SWITCH(38, JSONIFIER_30, 31, 32, 33, 34, 35, 36, 37)
		JSONIFIER_SWITCH(39, JSONIFIER_30, 31, 32, 33, 34, 35, 36, 37, 38)
		JSONIFIER_SWITCH(40, JSONIFIER_30, 31, 32, 33, 34, 35, 36, 37, 38, 39)
		JSONIFIER_SWITCH(41, JSONIFIER_40)
		JSONIFIER_SWITCH(42, JSONIFIER_40, 41)
		JSONIFIER_SWITCH(43, JSONIFIER_40, 41, 42)
		JSONIFIER_SWITCH(44, JSONIFIER_40, 41, 42, 43)
		JSONIFIER_SWITCH(45, JSONIFIER_40, 41, 42, 43, 44)
		JSONIFIER_SWITCH(46, JSONIFIER_40, 41, 42, 43, 44, 45)
		JSONIFIER_SWITCH(47, JSONIFIER_40, 41, 42, 43, 44, 45, 46)
		JSONIFIER_SWITCH(48, JSONIFIER_40, 41, 42, 43, 44, 45, 46, 47)
		JSONIFIER_SWITCH(49, JSONIFIER_40, 41, 42, 43, 44, 45, 46, 47, 48)
		JSONIFIER_SWITCH(50, JSONIFIER_40, 41, 42, 43, 44, 45, 46, 47, 48, 49)
		JSONIFIER_SWITCH(51, JSONIFIER_50)
		JSONIFIER_SWITCH(52, JSONIFIER_50, 51)
		JSONIFIER_SWITCH(53, JSONIFIER_50, 51, 52)
		JSONIFIER_SWITCH(54, JSONIFIER_50, 51, 52, 53)
		JSONIFIER_SWITCH(55, JSONIFIER_50, 51, 52, 53, 54)
		JSONIFIER_SWITCH(56, JSONIFIER_50, 51, 52, 53, 54, 55)
		JSONIFIER_SWITCH(57, JSONIFIER_50, 51, 52, 53, 54, 55, 56)
		JSONIFIER_SWITCH(58, JSONIFIER_50, 51, 52, 53, 54, 55, 56, 57)
		JSONIFIER_SWITCH(59, JSONIFIER_50, 51, 52, 53, 54, 55, 56, 57, 58)
		JSONIFIER_SWITCH(60, JSONIFIER_50, 51, 52, 53, 54, 55, 56, 57, 58, 59)
		JSONIFIER_SWITCH(61, JSONIFIER_60)
		JSONIFIER_SWITCH(62, JSONIFIER_60, 61)
		JSONIFIER_SWITCH(63, JSONIFIER_60, 61, 62)
		JSONIFIER_SWITCH(64, JSONIFIER_60, 61, 62, 63)
	}
};


struct base_type {
	template<size_t index> JSONIFIER_INLINE static uint64_t processIndex() {
		size_t returnValue{ index };
		returnValue = static_cast<size_t>(static_cast<double>(index) * static_cast<double>(index)) + 1;
		return returnValue;
	}
};

template<size_t... indices> JSONIFIER_INLINE static constexpr auto generateFunctionPtrs(std::index_sequence<indices...>) noexcept {
	using function_type = decltype(&base_type::template processIndex<0>);
	return std::array<function_type, sizeof...(indices)>{ { &base_type::template processIndex<indices>... } };
}

static constexpr auto functionPtrs{ generateFunctionPtrs(std::make_index_sequence<32>{}) };

template<uint64_t count, uint64_t lengthNew, typename value_type, bnch_swt::string_literal testNameNew> BNCH_SWT_INLINE void testFunction() {
	static constexpr bnch_swt::string_literal testName{ testNameNew };
	std::vector<std::vector<value_type>> testValues{ generateVectorOfVectors<value_type>(maxIterations, count, lengthNew) };
	std::vector<std::vector<std::string>> testValues00{};
	std::vector<std::vector<std::string>> testValues01{};
	testValues01.resize(maxIterations);
	for (uint64_t x = 0ull; x < maxIterations; ++x) {
		testValues01[x].resize(count);
	}
	testValues00.resize(maxIterations);
	testValues01.resize(maxIterations);
	for (uint64_t x = 0ull; x < maxIterations; ++x) {
		for (uint64_t y = 0ull; y < count; ++y) {
			testValues00[x].emplace_back(std::to_string(testValues[x][y]));
		}
	}
	uint64_t currentIteration{};
	std::vector<std::array<char, 30>> newerStrings{};
	newerStrings.resize(maxIterations);
	srand(static_cast<uint32_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
	bnch_swt::benchmark_stage<testName>::template runBenchmark<"glz_jump_table", "CYAN">([&] {
		uint64_t bytesProcessed{};
		auto lambda = [&]<size_t index>() {
			size_t returnValue{ index };
			returnValue = static_cast<size_t>(static_cast<double>(index) * static_cast<double>(index)) + 1;
			bytesProcessed += returnValue;
		};
		for (uint64_t x = 0ull; x < count; ++x) {
			if constexpr (count == 1) {
				bytesProcessed += base_type::processIndex<0>();
			} else if constexpr (count == 2) {
				if (x == 0) {
					bytesProcessed += base_type::processIndex<0>();
				} else {
					bytesProcessed += base_type::processIndex<1>();
				}
			} else {
				glz_jump_table<count>(lambda, x);
			}
			bnch_swt::doNotOptimizeAway(bytesProcessed);
		}
		++currentIteration;
		return bytesProcessed;
	});

	currentIteration = 0ull;
	bnch_swt::benchmark_stage<testName>::template runBenchmark<"jsonifier::internal::switch_statement", "CYAN">([&] {
		uint64_t bytesProcessed{};
		for (uint64_t x = 0ull; x < count; ++x) {
			if constexpr (count == 1) {
				bytesProcessed += base_type::processIndex<0>();
			} else if constexpr (count == 2) {
				if (x == 0) {
					bytesProcessed += base_type::processIndex<0>();
				} else {
					bytesProcessed += base_type::processIndex<1>();
				}
			} else {
				bytesProcessed += switch_statement<base_type, count>::impl(x);
			}
			bnch_swt::doNotOptimizeAway(bytesProcessed);
		}
		++currentIteration;
		return bytesProcessed;
	});

	currentIteration = 0ull;
	bnch_swt::benchmark_stage<testName>::template runBenchmark<"jsonifier::internal::functionPtrs", "CYAN">([&] {
		uint64_t bytesProcessed{};
		for (uint64_t x = 0ull; x < count; ++x) {
			if constexpr (count == 1) {
				bytesProcessed += base_type::processIndex<0>();
			} else if constexpr (count == 2) {
				if (x == 0) {
					bytesProcessed += base_type::processIndex<0>();
				} else {
					bytesProcessed += base_type::processIndex<1>();
				}
			} else {
				bytesProcessed += functionPtrs[x]();
			}
			bnch_swt::doNotOptimizeAway(bytesProcessed);
		}
		++currentIteration;
		return bytesProcessed;
	});

	currentIteration = 0ull;

	bnch_swt::benchmark_stage<testName>::printResults(true, true);
}

#include <array>
#include <cstdint>
#include <iostream>

// Table for first byte in the UTF-8 sequence (determines the length of the code point)
constexpr std::array<char, 256> firstByteTable = [] {
	std::array<char, 256> table = {};
	for (uint32_t i = 0; i <= 0x7F; ++i) {
		table[i] = 1;
	}
	for (uint32_t i = 0xC0; i <= 0xDF; ++i) {
		table[i] = 2;
	}
	for (uint32_t i = 0xE0; i <= 0xEF; ++i) {
		table[i] = 3;
	}
	for (uint32_t i = 0xF0; i <= 0xF7; ++i) {
		table[i] = 4;
	}
	return table;
}();

// Table for subsequent bytes in UTF-8 sequence (starting with 0x80)
constexpr std::array<char, 256> subsequentByteTable = [] {
	std::array<char, 256> table = {};
	for (uint32_t i = 0x80; i <= 0xBF; ++i) {
		table[i] = 0x80;// Subsequent bytes always start with 0x80
	}
	return table;
}();

// Function to encode a Unicode code point to UTF-8 using lookup tables
uint64_t codePointToUtf8(uint32_t cp, char* c) noexcept {
	if (cp <= 0x7F) {
		c[0] = static_cast<char>(cp);
		return 1;
	}
	if (cp <= 0x7FF) {
		c[0] = static_cast<char>((cp >> 6) + 192);
		c[1] = static_cast<char>((cp & 63) + 128);
		return 2;
	} else if (cp <= 0xFFFF) {
		c[0] = static_cast<char>((cp >> 12) + 224);
		c[1] = static_cast<char>(((cp >> 6) & 63) + 128);
		c[2] = static_cast<char>((cp & 63) + 128);
		return 3;
	} else if (cp <= 0x10FFFF) {
		c[0] = static_cast<char>((cp >> 18) + 240);
		c[1] = static_cast<char>(((cp >> 12) & 63) + 128);
		c[2] = static_cast<char>(((cp >> 6) & 63) + 128);
		c[3] = static_cast<char>((cp & 63) + 128);
		return 4;
	}
	return 0;
}

constexpr std::array<uint8_t, 32> values01{ [] {
	std::array<uint8_t, 32> values01{};
	for (size_t x = 0; x < 32; ++x) {
		values01[x] = x + 192;
	}
	return values01;
}() };

constexpr std::array<uint8_t, 16>values02{ [] {
	std::array<uint8_t, 16> values01{};
	for (size_t x = 0; x < 16; ++x) {
		values01[x] = x + 224;
	}
	return values01;
}() };

constexpr std::array<uint8_t, 5> values03{ [] {
	std::array<uint8_t, 5> values01{};
	for (size_t x = 0; x < 5; ++x) {
		values01[x] = x + 240;
	}
	return values01;
}() };

constexpr std::array<uint8_t, 64> values04{ [] {
	std::array<uint8_t, 64> values01{};
	for (size_t x = 0; x < 64; ++x) {
		values01[x] = x + 128;
	}
	return values01;
}() };

/// Sampled from Simdjson library: https://github.com/simdjson/simdjson
JSONIFIER_INLINE uint64_t codePointToUtf8Old(uint32_t cp, string_buffer_ptr c) noexcept {
	if (cp <= 0x7F) {
		c[0] = static_cast<char>(cp);
		return 1;
	}
	if (cp <= 0x7FF) {
		c[0] = static_cast<char>(values01[cp >> 6]);
		c[1] = static_cast<char>(values04[cp & 63]);
		return 2;
	} else if (cp <= 0xFFFF) {
		c[0] = static_cast<char>(values02[cp >> 12]);
		c[1] = static_cast<char>(values04[(cp >> 6) & 63]);
		c[2] = static_cast<char>(values04[cp & 63]);
		return 3;
	} else if (cp <= 0x10FFFF) {
		c[0] = static_cast<char>(values03[cp >> 18]);
		c[1] = static_cast<char>(values04[((cp >> 12) & 63)]);
		c[2] = static_cast<char>(values04[((cp >> 6) & 63)]);
		c[3] = static_cast<char>(values04[cp & 63]);
		return 4;
	}
	return 0;
}

int main() {
	char buffer[4];
	uint32_t codePoint = 0x7Fe;// Emoji code point
	size_t currentValue{};
	size_t previousValue{};
	for (size_t x = 0x800; x < 0xFFFF; ++x) {
		currentValue = ((x >> 6) & 63);
		if (previousValue != currentValue) {
			std::cout << "CURRENT VALUE: " << ((x >> 6) & 63) << std::endl;
		}
		previousValue = currentValue;
	}

	uint64_t bytesWritten = codePointToUtf8(codePoint, buffer);
		std::cout << "UTF-8 Encoding: ";
		for (uint64_t i = 0; i < bytesWritten; ++i) {
			std::cout << std::hex << (0xFF & buffer[i]) << " ";
		}
		std::cout << std::endl;

	uint64_t bytesWrittenOld = codePointToUtf8Old(codePoint, buffer);

	std::cout << "UTF-8 Encoding-Old: ";
	for (uint64_t i = 0; i < bytesWrittenOld; ++i) {
		std::cout << std::hex << (0xFF & buffer[i]) << " ";
	}
	std::cout << std::endl;

	codePoint = 0x7Fff;// Emoji code point

	bytesWritten = codePointToUtf8(codePoint, buffer);
	std::cout << "UTF-8 Encoding: ";
	for (uint64_t i = 0; i < bytesWritten; ++i) {
		std::cout << std::hex << (0xFF & buffer[i]) << " ";
	}
	std::cout << std::endl;

	bytesWrittenOld = codePointToUtf8Old(codePoint, buffer);

	std::cout << "UTF-8 Encoding-Old: ";
	for (uint64_t i = 0; i < bytesWrittenOld; ++i) {
		std::cout << std::hex << (0xFF & buffer[i]) << " ";
	}
	std::cout << std::endl;

	codePoint = 0x7FffE;// Emoji code point

	bytesWritten = codePointToUtf8(codePoint, buffer);
	std::cout << "UTF-8 Encoding: ";
	for (uint64_t i = 0; i < bytesWritten; ++i) {
		std::cout << std::hex << (0xFF & buffer[i]) << " ";
	}
	std::cout << std::endl;

	bytesWrittenOld = codePointToUtf8Old(codePoint, buffer);

	std::cout << "UTF-8 Encoding-Old: ";
	for (uint64_t i = 0; i < bytesWrittenOld; ++i) {
		std::cout << std::hex << (0xFF & buffer[i]) << " ";
	}
	std::cout << std::endl;

	return 0;
}
