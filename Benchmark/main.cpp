#if defined(__clang__) || (defined(__GNUC__) && defined(__llvm__))
	#define JSONIFIER_CLANG 1
#elif defined(_MSC_VER)
	#pragma warning(disable : 4820)
	#pragma warning(disable : 4371)
	#define JSONIFIER_MSVC 1
#elif defined(__GNUC__) && !defined(__clang__)
	#define JSONIFIER_GNUCXX 1
#endif

#if defined(JSONIFIER_MSVC)
	#define JSONIFIER_VISUAL_STUDIO 1
	#if defined(JSONIFIER_CLANG)
		#define JSONIFIER_CLANG_VISUAL_STUDIO 1
	#else
		#define JSONIFIER_REGULAR_VISUAL_STUDIO 1
	#endif
#endif

#if (defined(__x86_64__) || defined(_M_AMD64)) && !defined(_M_ARM64EC)
	#define JSONIFIER_IS_X86_64 1
#else
	#define JSONIFIER_IS_ARM64 1
#endif

#define JSONIFIER_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)

#if defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__)) || defined(TARGET_OS_MAC)
	#define JSONIFIER_MAC 1
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define JSONIFIER_LINUX 1
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define JSONIFIER_WIN 1
#else
	#error "Undetected platform."
#endif

#if defined(__has_builtin)
	#define JSONIFIER_HAS_BUILTIN(x) __has_builtin(x)
#else
	#define JSONIFIER_HAS_BUILTIN(x) 0
#endif

#if !defined(JSONIFIER_LIKELY)
	#define JSONIFIER_LIKELY(...) (__VA_ARGS__) [[likely]]
#endif

#if !defined(JSONIFIER_UNLIKELY)
	#define JSONIFIER_UNLIKELY(...) (__VA_ARGS__) [[unlikely]]
#endif

#if !defined(JSONIFIER_ELSE_UNLIKELY)
	#define JSONIFIER_ELSE_UNLIKELY(...) __VA_ARGS__ [[unlikely]]
#endif

#if defined(JSONIFIER_GNUCXX) || defined(JSONIFIER_CLANG)
	#define JSONIFIER_ASSUME(x) \
		do { \
			if (!(x)) \
				__builtin_unreachable(); \
		} while (0)
#elif defined(JSONIFIER_MSVC)
	#include <intrin.h>
	#define JSONIFIER_ASSUME(x) __assume(x)
#else
	#define JSONIFIER_ASSUME(x) (( void )0)
#endif

#if defined(__cpp_inline_variables) && __cpp_inline_variables >= 201606L
	#define JSONIFIER_HAS_INLINE_VARIABLE 1
#elif __cplusplus >= 201703L
	#define JSONIFIER_HAS_INLINE_VARIABLE 1
#elif defined(JSONIFIER_MSVC) && JSONIFIER_MSVC >= 1912 && _MSVC_LANG >= 201703L
	#define JSONIFIER_HAS_INLINE_VARIABLE 1
#else
	#define JSONIFIER_HAS_INLINE_VARIABLE 0
#endif

#if JSONIFIER_HAS_INLINE_VARIABLE
	#define JSONIFIER_INLINE_VARIABLE inline static constexpr
#else
	#define JSONIFIER_INLINE_VARIABLE static constexpr
#endif

#if defined(NDEBUG)
	#if defined(JSONIFIER_MSVC)
		#define JSONIFIER_INLINE [[msvc::forceinline]] inline
		#define JSONIFIER_NON_GCC_INLINE [[msvc::forceinline]] inline
		#define JSONIFIER_CLANG_INLINE inline
	#elif defined(JSONIFIER_CLANG)
		#define JSONIFIER_INLINE inline __attribute__((always_inline))
		#define JSONIFIER_NON_GCC_INLINE inline __attribute__((always_inline))
		#define JSONIFIER_CLANG_INLINE inline __attribute__((always_inline))
	#elif defined(JSONIFIER_GNUCXX)
		#define JSONIFIER_INLINE inline __attribute__((always_inline))
		#define JSONIFIER_NON_GCC_INLINE inline
		#define JSONIFIER_CLANG_INLINE inline
	#endif
#else
	#define JSONIFIER_INLINE
	#define JSONIFIER_NON_GCC_INLINE
	#define JSONIFIER_CLANG_INLINE
#endif

#if !defined JSONIFIER_ALIGN
	#define JSONIFIER_ALIGN(b) alignas(b)
#endif

#include <benchmark/benchmark.h>
#include <iostream>
#include <random>
#include <vector>
#include <limits>
#include <bit>

uint64_t generateRandomIntegerByLength(uint32_t digitLength) {
	if (digitLength == 0) {
		throw std::invalid_argument("Digit length must be greater than 0.");
	}

	if (digitLength > 20) {
		throw std::invalid_argument("Digit length exceeds the limit for uint64_t (maximum 20 digits).");
	}

	// Directly compute the min and max values using powers of 10
	uint64_t minValue = static_cast<uint64_t>(std::pow(10, digitLength - 1));
	uint64_t maxValue = static_cast<uint64_t>(std::pow(10, digitLength) - 1);

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint64_t> dist(minValue, maxValue);

	return dist(gen);
}

template<typename value_type> std::vector<value_type> generateRandomIntegers(size_t count, size_t maxLength) {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<value_type> lengthGen(1, maxLength);
	std::vector<value_type> randomNumbers;

	for (size_t i = 0; i < count; ++i) {
		uint64_t newValue{ generateRandomIntegerByLength(lengthGen(gen)) };
		if (newValue >= std::numeric_limits<value_type>::max()) {
			newValue /= 10;
		}
		randomNumbers.push_back(newValue);
	}

	return randomNumbers;
}

template<typename typeName> struct int_tables {
	static inline constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10,
		10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

	static inline constexpr uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull,
		99999999999ull, 999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull,
		9999999999999999999ull };
};

JSONIFIER_INLINE uint64_t fastDigitCount(const uint64_t inputValue) {
	const uint64_t originalDigitCount{ int_tables<void>::digitCounts[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > int_tables<void>::digitCountThresholds[originalDigitCount]);
}

JSONIFIER_INLINE int int_log2(uint64_t x) {
	return 63 - std::countl_zero(x | 1);
}

JSONIFIER_INLINE int digit_count(uint32_t x) {
	static constexpr uint32_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999 };
	int y							  = (9 * int_log2(x)) >> 5;
	y += x > table[y];
	return y + 1;
}

JSONIFIER_INLINE int digit_count(uint64_t x) {
	static constexpr uint64_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999, 9999999999, 99999999999, 999999999999, 9999999999999, 99999999999999,
		999999999999999ULL, 9999999999999999ULL, 99999999999999999ULL, 999999999999999999ULL, 9999999999999999999ULL };
	int y							  = (19 * int_log2(x) >> 6);
	y += x > table[y];
	return y + 1;
}

JSONIFIER_INLINE int alternative_digit_count(uint32_t x) {
	static constexpr uint64_t table[] = { 4294967296, 8589934582, 8589934582, 8589934582, 12884901788, 12884901788, 12884901788, 17179868184, 17179868184, 17179868184, 21474826480,
		21474826480, 21474826480, 21474826480, 25769703776, 25769703776, 25769703776, 30063771072, 30063771072, 30063771072, 34349738368, 34349738368, 34349738368, 34349738368,
		38554705664, 38554705664, 38554705664, 41949672960, 41949672960, 41949672960, 42949672960, 42949672960 };
	return (x + table[int_log2(x)]) >> 32;
}

JSONIFIER_INLINE int alternative_digit_count(uint64_t x) {
	static constexpr uint64_t table[64][2] = {
		{ 0x01, 0xfffffffffffffff6ULL },
		{ 0x01, 0xfffffffffffffff6ULL },
		{ 0x01, 0xfffffffffffffff6ULL },
		{ 0x01, 0xfffffffffffffff6ULL },
		{ 0x02, 0xffffffffffffff9cULL },
		{ 0x02, 0xffffffffffffff9cULL },
		{ 0x02, 0xffffffffffffff9cULL },
		{ 0x03, 0xfffffffffffffc18ULL },
		{ 0x03, 0xfffffffffffffc18ULL },
		{ 0x03, 0xfffffffffffffc18ULL },
		{ 0x04, 0xffffffffffffd8f0ULL },
		{ 0x04, 0xffffffffffffd8f0ULL },
		{ 0x04, 0xffffffffffffd8f0ULL },
		{ 0x04, 0xffffffffffffd8f0ULL },
		{ 0x05, 0xfffffffffffe7960ULL },
		{ 0x05, 0xfffffffffffe7960ULL },
		{ 0x05, 0xfffffffffffe7960ULL },
		{ 0x06, 0xfffffffffff0bdc0ULL },
		{ 0x06, 0xfffffffffff0bdc0ULL },
		{ 0x06, 0xfffffffffff0bdc0ULL },
		{ 0x07, 0xffffffffff676980ULL },
		{ 0x07, 0xffffffffff676980ULL },
		{ 0x07, 0xffffffffff676980ULL },
		{ 0x07, 0xffffffffff676980ULL },
		{ 0x08, 0xfffffffffa0a1f00ULL },
		{ 0x08, 0xfffffffffa0a1f00ULL },
		{ 0x08, 0xfffffffffa0a1f00ULL },
		{ 0x09, 0xffffffffc4653600ULL },
		{ 0x09, 0xffffffffc4653600ULL },
		{ 0x09, 0xffffffffc4653600ULL },
		{ 0x0a, 0xfffffffdabf41c00ULL },
		{ 0x0a, 0xfffffffdabf41c00ULL },
		{ 0x0a, 0xfffffffdabf41c00ULL },
		{ 0x0a, 0xfffffffdabf41c00ULL },
		{ 0x0b, 0xffffffe8b7891800ULL },
		{ 0x0b, 0xffffffe8b7891800ULL },
		{ 0x0b, 0xffffffe8b7891800ULL },
		{ 0x0c, 0xffffff172b5af000ULL },
		{ 0x0c, 0xffffff172b5af000ULL },
		{ 0x0c, 0xffffff172b5af000ULL },
		{ 0x0d, 0xfffff6e7b18d6000ULL },
		{ 0x0d, 0xfffff6e7b18d6000ULL },
		{ 0x0d, 0xfffff6e7b18d6000ULL },
		{ 0x0d, 0xfffff6e7b18d6000ULL },
		{ 0x0e, 0xffffa50cef85c000ULL },
		{ 0x0e, 0xffffa50cef85c000ULL },
		{ 0x0e, 0xffffa50cef85c000ULL },
		{ 0x0f, 0xfffc72815b398000ULL },
		{ 0x0f, 0xfffc72815b398000ULL },
		{ 0x0f, 0xfffc72815b398000ULL },
		{ 0x10, 0xffdc790d903f0000ULL },
		{ 0x10, 0xffdc790d903f0000ULL },
		{ 0x10, 0xffdc790d903f0000ULL },
		{ 0x10, 0xffdc790d903f0000ULL },
		{ 0x11, 0xfe9cba87a2760000ULL },
		{ 0x11, 0xfe9cba87a2760000ULL },
		{ 0x11, 0xfe9cba87a2760000ULL },
		{ 0x12, 0xf21f494c589c0000ULL },
		{ 0x12, 0xf21f494c589c0000ULL },
		{ 0x12, 0xf21f494c589c0000ULL },
		{ 0x13, 0x7538dcfb76180000ULL },
		{ 0x13, 0x7538dcfb76180000ULL },
		{ 0x13, 0x7538dcfb76180000ULL },
		{ 0x13, 0x7538dcfb76180000ULL },
	};
	int log		  = int_log2(x);
	uint64_t low  = table[log][1];
	uint64_t high = table[log][0];
	return (x + low < x) + high;
}

JSONIFIER_INLINE int fast_digit_count(uint32_t x) {
	static constexpr uint32_t table[32] = {
		9ul,//  0
		9ul,//  1
		9ul,//  2
		9ul,//  3
		99ul,//  4
		99ul,//  5
		99ul,//  6
		999ul,//  7
		999ul,//  8
		999ul,//  9
		9999ul,// 10
		9999ul,// 11
		9999ul,// 12
		9999ul,// 13
		99999ul,// 14
		99999ul,// 15
		99999ul,// 16
		999999ul,// 17
		999999ul,// 18
		999999ul,// 19
		9999999ul,// 20
		9999999ul,// 21
		9999999ul,// 22
		9999999ul,// 23
		99999999ul,// 24
		99999999ul,// 25
		99999999ul,// 26
		999999999ul,// 27
		999999999ul,// 28
		999999999ul,// 29
		4294967295ul,// 30
		4294967295ul,// 31
	};
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

JSONIFIER_INLINE int fast_digit_count(uint64_t x) {
	static constexpr uint64_t table[64] = {
		9ull,//  0
		9ull,//  1
		9ull,//  2
		9ull,//  3
		99ull,//  4
		99ull,//  5
		99ull,//  6
		999ull,//  7
		999ull,//  8
		999ull,//  9
		9999ull,// 10
		9999ull,// 11
		9999ull,// 12
		9999ull,// 13
		99999ull,// 14
		99999ull,// 15
		99999ull,// 16
		999999ull,// 17
		999999ull,// 18
		999999ull,// 19
		9999999ull,// 20
		9999999ull,// 21
		9999999ull,// 22
		9999999ull,// 23
		99999999ull,// 24
		99999999ull,// 25
		99999999ull,// 26
		999999999ull,// 27
		999999999ull,// 28
		999999999ull,// 29
		9999999999ull,// 30
		9999999999ull,// 31
		9999999999ull,// 32
		9999999999ull,// 33
		99999999999ull,// 34
		99999999999ull,// 35
		99999999999ull,// 36
		999999999999ull,// 37
		999999999999ull,// 38
		999999999999ull,// 39
		9999999999999ull,// 40
		9999999999999ull,// 41
		9999999999999ull,// 42
		9999999999999ull,// 43
		99999999999999ull,// 44
		99999999999999ull,// 45
		99999999999999ull,// 46
		999999999999999ull,// 47
		999999999999999ull,// 48
		999999999999999ull,// 49
		9999999999999999ull,// 50
		9999999999999999ull,// 51
		9999999999999999ull,// 52
		9999999999999999ull,// 53
		99999999999999999ull,// 54
		99999999999999999ull,// 55
		99999999999999999ull,// 56
		999999999999999999ull,// 57
		999999999999999999ull,// 58
		999999999999999999ull,// 59
		9999999999999999999ull,// 60
		9999999999999999999ull,// 61
		9999999999999999999ull,// 62
		9999999999999999999ull,// 63
	};
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

// 32-bit digit count benchmark
class DigitCount32Benchmark : public benchmark::Fixture {
  public:
	void SetUp(const ::benchmark::State& state) {
		randomIntegers = generateRandomIntegers<uint32_t>(state.range(0), 10);
		counts.resize(state.range(0));
		results.resize(state.range(0));

		// Pre-compute correct counts for verification
		for (size_t x = 0; x < state.range(0); ++x) {
			counts[x] = digit_count(randomIntegers[x]);
		}
	}

	void VerifyResults(const std::string& funcName) {
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			if (results[x] != counts[x]) {
				std::cerr << funcName << " failed to count the integer of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
						  << ", when it should be: " << counts[x] << std::endl;
			}
		}
	}

	std::vector<uint32_t> randomIntegers;
	std::vector<uint32_t> counts;
	std::vector<uint32_t> results;
};

// 64-bit digit count benchmark
class DigitCount64Benchmark : public benchmark::Fixture {
  public:
	void SetUp(const ::benchmark::State& state) {
		randomIntegers = generateRandomIntegers<uint64_t>(state.range(0), 20);
		counts.resize(state.range(0));
		results.resize(state.range(0));

		// Pre-compute correct counts for verification
		for (size_t x = 0; x < state.range(0); ++x) {
			counts[x] = digit_count(randomIntegers[x]);
		}
	}

	void VerifyResults(const std::string& funcName) {
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			if (results[x] != counts[x]) {
				std::cerr << funcName << " failed to count the integer of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
						  << ", when it should be: " << counts[x] << std::endl;
			}
		}
	}

	std::vector<uint64_t> randomIntegers;
	std::vector<uint64_t> counts;
	std::vector<uint64_t> results;
};

// 32-bit benchmarks
BENCHMARK_DEFINE_F(DigitCount32Benchmark, AlternativeDigitCount)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = alternative_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("alternative_digit_count");
}

BENCHMARK_DEFINE_F(DigitCount32Benchmark, FastDigitCount)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = fast_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("fast_digit_count");
}

BENCHMARK_DEFINE_F(DigitCount32Benchmark, DigitCount)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("digit_count");
}

BENCHMARK_DEFINE_F(DigitCount32Benchmark, FastDigitCountRTC)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = fastDigitCount(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("fastDigitCount");
}

// 64-bit benchmarks
BENCHMARK_DEFINE_F(DigitCount64Benchmark, FastDigitCount)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = fast_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("fast_digit_count_64");
}

BENCHMARK_DEFINE_F(DigitCount64Benchmark, AlternativeDigitCount)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = alternative_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("alternative_digit_count_64");
}

BENCHMARK_DEFINE_F(DigitCount64Benchmark, DigitCount)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("digit_count_64");
}

BENCHMARK_DEFINE_F(DigitCount64Benchmark, FastDigitCountRTC)(benchmark::State& state) {
	for (auto _: state) {
		uint64_t currentCount = 0;
		for (size_t x = 0; x < randomIntegers.size(); ++x) {
			auto newCount = fastDigitCount(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		benchmark::DoNotOptimize(currentCount);
	}
	VerifyResults("fastDigitCount_64");
}

// Register benchmarks with 1 million iterations
BENCHMARK_REGISTER_F(DigitCount32Benchmark, AlternativeDigitCount)->Arg(1000000);
BENCHMARK_REGISTER_F(DigitCount32Benchmark, FastDigitCount)->Arg(1000000);
BENCHMARK_REGISTER_F(DigitCount32Benchmark, DigitCount)->Arg(1000000);
BENCHMARK_REGISTER_F(DigitCount32Benchmark, FastDigitCountRTC)->Arg(1000000);

BENCHMARK_REGISTER_F(DigitCount64Benchmark, FastDigitCount)->Arg(1000000);
BENCHMARK_REGISTER_F(DigitCount64Benchmark, AlternativeDigitCount)->Arg(1000000);
BENCHMARK_REGISTER_F(DigitCount64Benchmark, DigitCount)->Arg(1000000);
BENCHMARK_REGISTER_F(DigitCount64Benchmark, FastDigitCountRTC)->Arg(1000000);

BENCHMARK_MAIN();