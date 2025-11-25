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

#include <BnchSwt/BenchmarkSuite.hpp>
#include <random>

uint64_t generateRandomIntegerByLength(uint32_t digitLength) {
	if (digitLength == 0) {
		throw std::invalid_argument("Digit length must be greater than 0.");
	}

	if (digitLength > 20) {
		throw std::invalid_argument("Digit length exceeds the limit for uint64_t (maximum 20 digits).");
	}

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

JSONIFIER_INLINE uint32_t rtc_digit_count(const uint32_t inputValue) {
	static constexpr uint32_t digitCountThresholds_32[]{ 0u, 9u, 99u, 999u, 9999u, 99999u, 999999u, 9999999u, 99999999u, 999999999u, 4294967295u };
	static constexpr uint8_t digitCounts_32[]{ 10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };
	const uint32_t originalDigitCount{ digitCounts_32[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint32_t>(inputValue > digitCountThresholds_32[originalDigitCount]);
}

JSONIFIER_INLINE uint64_t rtc_digit_count(const uint64_t inputValue) {
	static constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10,
		10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };
	static constexpr uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull,
		99999999999ull, 999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull,
		9999999999999999999ull };
	const uint64_t originalDigitCount{ digitCounts[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > digitCountThresholds[originalDigitCount]);
}

JSONIFIER_INLINE int int_log2(uint64_t x) {
	return 63 - std::countl_zero(x | 1);
}

JSONIFIER_INLINE int lemire_digit_count(uint32_t x) {
	static constexpr uint32_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999 };
	int y						  = (9 * int_log2(x)) >> 5;
	y += x > table[y];
	return y + 1;
}

JSONIFIER_INLINE int lemire_digit_count(uint64_t x) {
	static constexpr uint64_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999, 9999999999, 99999999999, 999999999999, 9999999999999, 99999999999999,
		999999999999999ULL, 9999999999999999ULL, 99999999999999999ULL, 999999999999999999ULL, 9999999999999999999ULL };
	int y						  = (19 * int_log2(x) >> 6);
	y += x > table[y];
	return y + 1;
}

JSONIFIER_INLINE int fast_digit_count(uint32_t x) {
	static const uint32_t table[32] = {
		9ul,
		9ul,
		9ul,
		9ul,
		99ul,
		99ul,
		99ul,
		999ul,
		999ul,
		999ul,
		9999ul,
		9999ul,
		9999ul,
		9999ul,
		99999ul,
		99999ul,
		99999ul,
		999999ul,
		999999ul,
		999999ul,
		9999999ul,
		9999999ul,
		9999999ul,
		9999999ul,
		99999999ul,
		99999999ul,
		99999999ul,
		999999999ul,
		999999999ul,
		999999999ul,
		4294967295ul,
		4294967295ul,
	};
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

JSONIFIER_INLINE int fast_digit_count(uint64_t x) {
	static const uint64_t table[64] = {
		9ull,
		9ull,
		9ull,
		9ull,
		99ull,
		99ull,
		99ull,
		999ull,
		999ull,
		999ull,
		9999ull,
		9999ull,
		9999ull,
		9999ull,
		99999ull,
		99999ull,
		99999ull,
		999999ull,
		999999ull,
		999999ull,
		9999999ull,
		9999999ull,
		9999999ull,
		9999999ull,
		99999999ull,
		99999999ull,
		99999999ull,
		999999999ull,
		999999999ull,
		999999999ull,
		9999999999ull,
		9999999999ull,
		9999999999ull,
		9999999999ull,
		99999999999ull,
		99999999999ull,
		99999999999ull,
		999999999999ull,
		999999999999ull,
		999999999999ull,
		9999999999999ull,
		9999999999999ull,
		9999999999999ull,
		9999999999999ull,
		99999999999999ull,
		99999999999999ull,
		99999999999999ull,
		999999999999999ull,
		999999999999999ull,
		999999999999999ull,
		9999999999999999ull,
		9999999999999999ull,
		9999999999999999ull,
		9999999999999999ull,
		99999999999999999ull,
		99999999999999999ull,
		99999999999999999ull,
		999999999999999999ull,
		999999999999999999ull,
		999999999999999999ull,
		9999999999999999999ull,
		9999999999999999999ull,
		9999999999999999999ull,
		9999999999999999999ull,
	};
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

static constexpr uint64_t total_iterations{ 12 };
static constexpr uint64_t measured_iterations{ 4 };

template<size_t count, bnch_swt::string_literal name, typename value_type> JSONIFIER_INLINE void testFunction() {
	auto randomIntegers = generateRandomIntegers<value_type>(count, sizeof(value_type) == 4 ? 10 : 20);
	std::vector<value_type> counts{};
	std::vector<value_type> results{};
	counts.resize(count);
	results.resize(count);
	for (size_t x = 0; x < count; ++x) {
		counts[x] = fast_digit_count(randomIntegers[x]);
	}

	static constexpr bnch_swt::string_literal bit_size{ bnch_swt::internal::toStringLiteral<sizeof(value_type) * 8>() };

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, total_iterations, measured_iterations>::template runBenchmark<"lemire-digit-count-" + bit_size, "cyan">([&]() {
		value_type currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = lemire_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<value_type>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "lemire-digit-count-" << sizeof(value_type) * 8 << " failed to count the integers of value : " << randomIntegers[x]
					  << ",instead it counted : " << results[x] << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, total_iterations, measured_iterations>::template runBenchmark<"rtc-" + bit_size + "-bit", "cyan">(
		[&]() {
		value_type currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = rtc_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<value_type>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "rtc-" << sizeof(value_type) * 8 << "-bit failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
					  << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, total_iterations, measured_iterations>::printResults(true, true);
}

int main() {
	testFunction<10000, "uint32-test-10000", uint32_t>();
	testFunction<10000, "uint64-test-10000", uint64_t>();
	return 0;
}