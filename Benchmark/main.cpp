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

uint16_t generateRandomIntegerByLength() {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint16_t> dist(0, 2500);

	return dist(gen);
}

template<typename value_type> std::vector<value_type> generateRandomIntegers(size_t count, size_t maxLength) {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint64_t> lengthGen(1, maxLength);
	std::vector<value_type> randomNumbers;

	for (size_t i = 0; i < count; ++i) {
		uint64_t newValue{ generateRandomIntegerByLength() };
		if (newValue >= std::numeric_limits<value_type>::max()) {
			newValue /= 10;
		}
		randomNumbers.push_back(newValue);
	}

	return randomNumbers;
}

template<typename value_type> struct alignas(64) static_aligned_const {
	alignas(64) value_type value{};

	JSONIFIER_INLINE constexpr operator const value_type&() const {
		return value;
	}

	JSONIFIER_INLINE operator value_type&() & {
		return value;
	}

	JSONIFIER_INLINE operator value_type&&() && {
		return std::move(value);
	}

	JSONIFIER_INLINE constexpr const value_type& operator*() const {
		return value;
	}

	JSONIFIER_INLINE value_type& operator*() {
		return value;
	}

	JSONIFIER_INLINE constexpr bool operator==(const static_aligned_const& other) const {
		return value == other.value;
	}

	JSONIFIER_INLINE constexpr bool operator!=(const static_aligned_const& other) const {
		return value != other.value;
	}

	JSONIFIER_INLINE constexpr bool operator<(const static_aligned_const& other) const {
		return value < other.value;
	}

	JSONIFIER_INLINE constexpr bool operator>(const static_aligned_const& other) const {
		return value > other.value;
	}
};

JSONIFIER_INLINE static constexpr float fp32_from_bits(uint32_t w) noexcept {
	return std::bit_cast<float>(w);
}

JSONIFIER_INLINE static constexpr uint32_t fp32_to_bits(float f) noexcept {
	return std::bit_cast<uint32_t>(f);
}

JSONIFIER_INLINE static constexpr float compute_fp16_to_fp32(uint16_t h) noexcept {
	const uint32_t w	 = static_cast<uint32_t>(h) << 16;
	const uint32_t sign	 = w & 0x80000000u;
	const uint32_t two_w = w + w;

	constexpr uint32_t exp_offset = 0xE0u << 23;
	constexpr float exp_scale	  = fp32_from_bits(0x7800000u);
	const float normalized_value  = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

	constexpr uint32_t magic_mask  = 126u << 23;
	constexpr float magic_bias	   = 0.5f;
	const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

	constexpr uint32_t denormalized_cutoff = 1u << 27;
	const uint32_t result				   = sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
	return fp32_from_bits(result);
}

alignas(64) static static_aligned_const<float>* __restrict fp16_to_fp32_array{ []() {
	alignas(64) static std::array<static_aligned_const<float>, (1 << 16)> return_values_new{};
	for (uint64_t i = 0; i < (1 << 16); ++i) {
		return_values_new[i] = static_aligned_const<float>{ compute_fp16_to_fp32(static_cast<uint16_t>(i)) };
	}
	return return_values_new.data();
}() };

JSONIFIER_INLINE static float fp16_to_fp32(uint16_t f) {
	return fp16_to_fp32_array[f];
}
#if defined(JSONIFIER_IS_ARM64)
	#include <arm_neon.h>
JSONIFIER_INLINE static float fp16_to_fp32_simd(uint16_t h) {
	return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}
#else
#include <immintrin.h>
JSONIFIER_INLINE static float fp16_to_fp32_simd(uint16_t h) {
	return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}
#endif
static constexpr uint64_t iterationCount{ 25 };
static constexpr uint64_t measureIterationCount{ 5 };

template<size_t count, bnch_swt::string_literal name> JSONIFIER_INLINE void testFunction() {
	std::vector<std::vector<uint16_t>> randomIntegers{};
	std::vector<std::vector<float>> results01{};
	results01.resize(iterationCount);
	std::vector<std::vector<float>> results02{};
	results02.resize(iterationCount);
	for (uint64_t x = 0; x < iterationCount; ++x) {
		randomIntegers.emplace_back(generateRandomIntegers<uint16_t>(count, 20));
		results01[x].resize(count);
		results02[x].resize(count);
	}

	uint64_t current_index{};
	bnch_swt::benchmark_stage<"compare-fp16-to-fp32" + name, iterationCount, measureIterationCount>::template runBenchmark<"lookup-table", "cyan">([&]() {
		for (uint64_t x = 0; x < count; ++x) {
			results01[current_index][x] = fp16_to_fp32(randomIntegers[current_index][x]);
		}
		bnch_swt::doNotOptimizeAway(results01);
		++current_index;
		return results01.size() * results01[0].size() * sizeof(float);
	});
	current_index = 0;
	bnch_swt::benchmark_stage<"compare-fp16-to-fp32" + name, iterationCount, measureIterationCount>::template runBenchmark<"sse", "cyan">([&]() {
		for (uint64_t x = 0; x < count; ++x) {
			results02[current_index][x] = fp16_to_fp32_simd(randomIntegers[current_index][x]);
		}
		bnch_swt::doNotOptimizeAway(results02);
		++current_index;
		return results01.size() * results01[0].size() * sizeof(float);
	});
	for (uint64_t x = 0; x < iterationCount; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results01[x][y] != results02[x][y]) {
				std::cout << "fast-digit-count-64 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results01[x][y]
						  << ", when it should be: " << results02[x][y] << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<"compare-fp16-to-fp32" + name, iterationCount, measureIterationCount>::printResults(true, true);
}

int main() {
	testFunction<1, "uint64-test-1">();
	testFunction<10, "uint64-test-10">();
	testFunction<100, "uint64-test-100">();
	testFunction<1000, "uint64-test-1000">();
	testFunction<10000, "uint64-test-10000">();
	testFunction<100000, "uint64-test-100000">();
	testFunction<1000000, "uint64-test-1000000">();
	return 0;
}