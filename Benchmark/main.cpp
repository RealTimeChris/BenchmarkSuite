#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include <random>
#include <utility>

uint64_t generateRandomIntegerByLength(uint32_t digitLength) {
	if (digitLength == 0) {
		throw std::invalid_argument("Digit length must be greater than 0.");
	}

	if (digitLength > 20) {
		throw std::invalid_argument("Digit length exceeds the limit for uint64_t (maximum 20 digits).");
	}

	uint64_t minValue = 1;
	for (uint32_t i = 1; i < digitLength; ++i) {
		minValue *= 10;
	}

	uint64_t maxValue = minValue * 10 - 1;

	std::random_device rd;
	std::mt19937_64 gen(rd());
	if (minValue > maxValue) {
		std::swap(minValue, maxValue);
	}
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


JSONIFIER_INLINE uint64_t fastDigitCount(const uint64_t inputValue) {
	static constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10,
		9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

	static constexpr uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull, 99999999999ull,
		999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull, 9999999999999999999ull };
	const uint64_t originalDigitCount{ digitCounts[jsonifier::simd::lzcnt(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > digitCountThresholds[originalDigitCount]);
}

JSONIFIER_INLINE uint64_t fastDigitCountCountLZero(const uint64_t inputValue) {
	static constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10,
		9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

	static constexpr uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull, 99999999999ull,
		999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull, 9999999999999999999ull };
	const uint64_t originalDigitCount{ digitCounts[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > digitCountThresholds[originalDigitCount]);
}

#define BITMANIP_INTEGRAL_TYPENAME(T) typename T, ::std::enable_if_t<::std::is_integral_v<T>, int> = 0
#define BITMANIP_UNSIGNED_TYPENAME(T) typename T, ::std::enable_if_t<::std::is_unsigned_v<T>, int> = 0

template<BITMANIP_INTEGRAL_TYPENAME(Int)> constexpr unsigned log2bits_v = "0112222333333334"[sizeof(Int) - 1] + 3 - '0';

template<typename Int, std::enable_if_t<std::is_integral_v<Int>, int> = 0> constexpr size_t bits_v = std::numeric_limits<Int>::digits;

template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint makeMask(Uint length) noexcept {
	return (Uint{ 1 } << length) - Uint{ 1 };
}

template<typename Int> auto nextLargerUint_impl() {
	constexpr auto bits = bits_v<Int>;

	if constexpr (bits >= 64) {
		return std::uintmax_t{ 0 };
	} else if constexpr (bits >= 32) {
		return std::uint_least64_t{ 0 };
	} else if constexpr (bits >= 16) {
		return std::uint_least32_t{ 0 };
	} else if constexpr (bits >= 8) {
		return std::uint_least16_t{ 0 };
	}
}

template<BITMANIP_UNSIGNED_TYPENAME(Uint)> using nextLargerUintType = decltype(nextLargerUint_impl<Uint>());

#if defined(BITMANIP_HAS_BUILTIN_CLZ) || defined(BITMANIP_HAS_BUILTIN_MSB)
	#define BITMANIP_HAS_BUILTIN_LOG2FLOOR
namespace detail {

	template<typename Int> inline Int log2floor_builtin(Int v) {
	#ifdef BITMANIP_HAS_BUILTIN_MSB
		return builtin::msb(v);
	#elif defined(BITMANIP_HAS_BUILTIN_CLZ)
		constexpr int maxIndex = bits_v<Int> - 1;
		return (v != 0) * static_cast<Int>(maxIndex - builtin::clz(v));
	#endif
	}

}// namespace detail
#endif

// POWER OF 2 TESTING ==================================================================================================

/**
 * @brief Returns whether an unsigned integer is a power of 2 or zero.
 * Note that this test is faster than having to test if val is a power of 2.
 * @param val the parameter to test
 * @return true if val is a power of 2 or if val is zero
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr bool isPow2or0(Uint val) noexcept {
	return (val & (val - 1)) == 0;
}

/**
 * @brief Returns whether an unsigned integer is a power of 2.
 * @param val the parameter to test
 * @return true if val is a power of 2
 * @see is_pow2_or_zero
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr bool isPow2(Uint val) noexcept {
	return val != 0 && isPow2or0(val);
}

// POWER OF 2 ROUNDING =================================================================================================

namespace detail {

#ifdef BITMANIP_HAS_BUILTIN_LOG2FLOOR
	#define BITMANIP_HAS_BUILTIN_CEILPOW2
	template<typename Uint> JSONIFIER_INLINE inline Uint ceilPow2m1_builtin(Uint v) noexcept {
		Uint log = log2floor_builtin(v);
		return v | (Uint{ 1 } << log) - Uint{ 1 };
	}
#endif

	template<typename Uint> JSONIFIER_INLINE constexpr Uint ceilPow2m1_shift(Uint v) noexcept {
		constexpr std::size_t iterations = log2bits_v<Uint>;
		for (std::size_t i = 0; i < iterations; ++i) {
			// after all iterations, all bits right to the msb will be filled with 1
			v |= v >> (1 << i);
		}
		return v;
	}

}// namespace detail

/**
 * @brief Rounds up an unsigned integer to the next power of 2, minus 1.
 * 0 is not rounded up and stays zero.
 * Examples: 100 -> 127, 1 -> 1, 3 -> 3, 3000 -> 4095, 64 -> 127
 * @param v the value to round up
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint ceilPow2m1(Uint v) noexcept {
// The codegen on other platforms such as ARM is actually better for the repeated shift version.
// Each step on ARM can be performed with a flexible operand which performs a shift.
#if defined(BITMANIP_HAS_BUILTIN_CEILPOW2) && defined(BITMANIP_X86_OR_X64)
	if (not builtin::isconsteval()) {
		return detail::ceilPow2m1_builtin(v);
	}
#endif
	return detail::ceilPow2m1_shift(v);
}

/**
 * @brief Rounds up an unsigned integer to the next power of 2.
 * Powers of two are not affected.
 * 0 is not rounded and stays zero.
 * Examples: 100 -> 128, 1 -> 1, 3 -> 4, 3000 -> 4096
 * @param v the value to round up
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint ceilPow2(Uint v) noexcept {
	return ceilPow2m1(v - 1) + 1;
}

/**
 * @brief Rounds down an unsigned integer to the next power of 2.
 * Powers of 2 are not affected.
 * The result of floorPow2(0) is undefined.
 * Examples: 100 -> 64, 1 -> 1, 3 -> 2, 3000 -> 2048
 * @param v the value to round down
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint floorPow2(Uint v) noexcept {
	return ceilPow2m1(v >> 1) + 1;
}

// BASE 2 LOGARITHMS ===================================================================================================

/**
 * @brief Naive implementation of log2 using repeated single-bit rightshifting.
 */
template<typename Uint> JSONIFIER_INLINE constexpr Uint log2floor_naive(Uint val) noexcept {
	return (val != 0) * uint64_t(63 - jsonifier::simd::lzcnt(val));
}

/**
 * @brief Fast implementation of log2.
 * See https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog.
 *
 * Unrolled 32-bit version:
 * unsigned shift = (v > 0xFFFF) << 4;
    v >>= shift;
    r |= shift;

    shift = (v > 0xFF  ) << 3;
    v >>= shift;
    r |= shift;

    shift = (v > 0xF   ) << 2;
    v >>= shift;
    r |= shift;

    shift = (v > 0x3   ) << 1;
    v >>= shift;
    r |= shift;

    shift = (v > 1) << 0;
    r >>= shift;
    r |= shift;
 */
template<typename Uint> JSONIFIER_INLINE constexpr Uint log2floor_fast(Uint v) noexcept {
	return (v != 0) * uint64_t(63 - jsonifier::simd::lzcnt(v));
}

namespace detail {

	constexpr unsigned char MultiplyDeBruijnBitPosition[32] = { 0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4,
		31 };

}

/**
 * @brief log2floor implementation using De Bruijn multiplication.
 * See https://graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn.
 * @param val the value
 */
constexpr std::uint32_t log2floor_debruijn(std::uint32_t val) noexcept {
	constexpr std::uint32_t magic = 0x07C4ACDD;

	val = ceilPow2m1(val);
	val *= magic;
	val >>= 27;

	return detail::MultiplyDeBruijnBitPosition[val];
}

/**
 * @brief Computes the floored binary logarithm of a given integer.
 * Example: log2floor(123) = 6
 *
 * This templated function will choose the best available method depending on the type of the integer.
 * It is undefined for negative values.
 *
 * Unlike a traditional log function, it is defined for 0: log2floor(0) = 0
 *
 * @param v the value
 * @return the floored binary logarithm
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint log2floor(Uint v) noexcept {
	return (v != 0) * uint64_t(63 - jsonifier::simd::lzcnt(v));
}

/**
 * @brief Computes the ceiled binary logarithm of a given integer.
 * Example: log2ceil(123) = 7
 *
 * This templated function will choose the best available method depending on the type of the integer.
 * It is undefined for negative values.
 *
 * Unlike a traditional log function, it is defined for 0: log2ceil(0) = 0
 *
 * @param v the value
 * @return the floored binary logarithm
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint log2ceil(Uint val) noexcept {
	const Uint result = log2floor(val);
	return result + not isPow2or0(val);
}

/**
 * @brief Computes the number of bits required to represent a given number.
 * Examples: bitLength(0) = 1, bitLength(3) = 2, bitLength(123) = 7, bitLength(4) = 3
 */
template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint bitCount(Uint val) noexcept {
	return log2floor(val) + 1;
}

// ARBITRARY BASE LOGARITHMS ===========================================================================================

namespace detail {

	/**
 * @brief Naive implementation of log base N using repeated division.
 */
	template<BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint logFloor_naive(Uint val, unsigned base) noexcept {
		Uint result = 0;
		while (val /= base) {
			++result;
		}
		return result;
	}

}// namespace detail

/**
 * @brief The maximum possible exponent for a given base that can still be represented by a given integer type.
 * Example: maxExp<uint8_t, 10> = 2, because 10^2 is representable by an 8-bit unsigned integer but 10^3 isn't.
 */
template<unsigned BASE, BITMANIP_UNSIGNED_TYPENAME(Uint)> constexpr Uint maxExp = detail::logFloor_naive<Uint>(static_cast<Uint>(~Uint{ 0u }), BASE);

static_assert(maxExp<10, std::uint8_t> == 2);
static_assert(maxExp<10, std::uint16_t> == 4);
static_assert(maxExp<10, std::uint32_t> == 9);

namespace detail {

	/**
 * @brief Simple implementation of log base N using repeated multiplication.
 * This method is slightly more sophisticated than logFloor_naive because it avoids division.
 */
	template<unsigned BASE, BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint logFloor_simple(Uint val) noexcept {
		constexpr Uint limit = maxExp<BASE, Uint>;

		Uint i	 = 0;
		Uint pow = BASE;
		for (; i <= limit; ++i, pow *= BASE) {
			if (val < pow) {
				return i;
			}
		}
		return i;
	}

	/**
 * @brief Tiny array implementation to avoid including <array>.
 */
	template<typename T, std::size_t N> struct Table {
		static_assert(N != 0, "Can't create zero-size tables");

		using value_type						= T;
		static constexpr std::size_t size_value = N;
		T data[N];

		constexpr std::size_t size() const noexcept {
			return size_value;
		}

		constexpr T front() const noexcept {
			return data[0];
		}

		constexpr T back() const noexcept {
			return data[N - 1];
		}

		constexpr T& operator[](std::size_t i) noexcept {
			return data[i];
		}

		constexpr const T& operator[](std::size_t i) const noexcept {
			return data[i];
		}
	};

	/// Unsafe fixed point multiplication between Q32.32 and Q64.0
	/// Unsafe because for large numbers, this operation can overflow.
	constexpr std::uint64_t unsafeMulQ32o32Q64(std::uint64_t q32o32, std::uint64_t q64) noexcept {
		return q32o32 * q64 >> std::uint64_t{ 32 };
	}

	/// Unsafe fixed point multiplication between Q16.16 and Q32.0
	/// Unsafe because for large numbers, this operation can overflow.
	constexpr std::uint32_t unsafeMulQ16o16Q32(std::uint32_t q16o16, std::uint32_t q32) noexcept {
		return q16o16 * q32 >> std::uint32_t{ 16 };
	}

	/**
 * @brief Compares two integers a, b.
 * @param a the first integer
 * @param b the second integer
 * @return -1 if a is lower, 1 if a is greater, 0 otherwise (if they are equal).
 */
	constexpr int cmpU64(std::uint64_t a, std::uint64_t b) noexcept {
		return (b < a) - (a < b);
	}

	/**
 * @brief Creates a table of approximations of the base BASE logarithm of powers of two: log_BASE(2^i).
 * @tparam UInt an unsigned integer type of the numbers which's base BASE logarithm will be taken
 * @tparam BASE the base
 * @return the table of approximate logarithms
 */
	template<typename Uint, std::size_t BASE> JSONIFIER_INLINE constexpr Table<unsigned char, bits_v<Uint>> makeGuessTable() noexcept {
		Table<unsigned char, bits_v<Uint>> result{};
		for (std::size_t i = 0; i < result.size(); ++i) {
			const auto pow2 = static_cast<Uint>(Uint{ 1 } << i);
			result[i]		= static_cast<unsigned char>(logFloor_naive(pow2, BASE));
		}
		return result;
	}

	template<std::size_t SIZE>
	JSONIFIER_INLINE constexpr int compareApproximationToGuessTable(std::uint64_t approxFactor, const Table<unsigned char, SIZE>& table) noexcept {
		for (unsigned b = 0; b < SIZE; ++b) {
			std::uint64_t actualLog = table[b];
			std::uint64_t approxLog = unsafeMulQ32o32Q64(approxFactor, b);
			if (int cmp = cmpU64(approxLog, actualLog); cmp != 0) {
				return cmp;
			}
		}
		return 0;
	}

	constexpr std::uint64_t NO_APPROXIMATION = ~std::uint64_t{ 0 };

	/**
 * @brief Approximates the logarithm guess table using a Q32.32 number.
 * This exploits the fact that to convert from the logarithm base 2, to logarithm base B, we need to multiply with a
 * constant factor.
 * This constant factor is being approximated in a bit-guessing approach.
 * The result of this function is guaranteed to only set the most significant bits necessary.
 * E.g. if the result has the lowest 16 bits not set, the approximation can also be truncated to Q16.16.
 * @tparam SIZE the size of the table
 * @param table the table to approximate, where table[i] <= i
 * @return the fixed-point number approximating the table
 */
	template<std::size_t SIZE> JSONIFIER_INLINE constexpr std::uint64_t approximateGuessTable(const Table<unsigned char, SIZE>& table) noexcept {
		std::uint64_t result = 0;
		for (unsigned b = 33; b-- != 0;) {
			std::uint64_t guessedResult = result | (std::uint64_t{ 1 } << b);
			int cmp						= compareApproximationToGuessTable(guessedResult, table);
			if (cmp == 0) {
				return guessedResult;
			}
			if (cmp == -1) {
				result = guessedResult;
			}
		}
		return NO_APPROXIMATION;
	}

	template<typename Uint, std::size_t BASE> struct LogFloorGuesser {
		static constexpr Table<unsigned char, bits_v<Uint>> guessTable = makeGuessTable<Uint, BASE>();
		static constexpr std::uint64_t guessTableApproximation		   = approximateGuessTable(guessTable);

		constexpr unsigned char operator()(unsigned char log2) const noexcept {
			if constexpr (guessTableApproximation == NO_APPROXIMATION) {
				return guessTable[log2];
			} else if constexpr ((guessTableApproximation & makeMask<std::uint64_t>(16u)) == 0) {
				constexpr std::uint32_t lessPreciseApproximation = guessTableApproximation >> 16;
				return unsafeMulQ16o16Q32(lessPreciseApproximation, log2);
			} else {
				return unsafeMulQ32o32Q64(guessTableApproximation, log2);
			}
		}

		constexpr unsigned char maxGuess() const noexcept {
			return guessTable.back();
		}
	};

	template<typename Uint, std::size_t BASE, typename TableUint = nextLargerUintType<Uint>> JSONIFIER_INLINE constexpr auto makePowerTable() noexcept {
		// the size of the table is maxExp<BASE, Uint> + 2 because we need to store the maximum power
		// +1 because we need to store maxExp, which is an index, not a size
		// +1 again because for narrow integers, we would like to access one beyond the "end" of the table
		//
		// as a result, the last multiplication with BASE in this function might overflow, but this is perfectly normal
		Table<TableUint, maxExp<BASE, Uint> + 2> result{};
		std::uintmax_t x = 1;
		for (std::size_t i = 0; i < result.size(); ++i, x *= BASE) {
			result[i] = static_cast<TableUint>(x);
		}
		return result;
	}

	/// table that maps from log_N(val) -> pow(N, val + 1)
	template<typename Uint, std::size_t BASE> constexpr auto logFloor_powers = detail::makePowerTable<Uint, BASE>();

	template<typename Uint, std::size_t BASE> constexpr auto powConst_powers = detail::makePowerTable<Uint, BASE, Uint>();

}// namespace detail

/**
 * @brief Computes pow(BASE, exponent) where BASE is known at compile-time.
 */
template<std::size_t BASE, BITMANIP_UNSIGNED_TYPENAME(Uint)> JSONIFIER_INLINE constexpr Uint powConst(const Uint exponent) noexcept {
	if constexpr (isPow2(BASE)) {
		return Uint{ 1 } << (exponent * log2floor(BASE));
	} else {
		return detail::powConst_powers<Uint, BASE>[exponent];
	}
}

/**
 * @brief Computes the floored logarithm of a number with a given base.
 *
 * Examples:
 *     logFloor<10>(0) = 0
 *     logFloor<10>(5) = 0
 *     logFloor<10>(10) = 1
 *     logFloor<10>(123) = 2
 *
 * Note that unlike a traditional logarithm function, it is defined for 0 and is equal to 0.
 *
 * @see https://stackoverflow.com/q/63411054
 * @tparam BASE the base (e.g. 10)
 * @param val the input value
 * @return floor(log(val, BASE))
 */
template<std::size_t BASE = 10, typename Uint>
JSONIFIER_INLINE constexpr auto logFloor(const Uint val) noexcept -> std::enable_if_t<(std::is_unsigned_v<Uint> && BASE >= 2), Uint> {
	if constexpr (isPow2(BASE)) {
		return log2floor(val) / log2floor(BASE);
	} else {
		constexpr detail::LogFloorGuesser<Uint, BASE> guesser;
		constexpr auto& powers = detail::logFloor_powers<Uint, BASE>;
		using table_value_type = typename decltype(detail::logFloor_powers<Uint, BASE>)::value_type;

		if constexpr (sizeof(Uint) < sizeof(table_value_type) || guesser.maxGuess() + 2 < powers.size()) {
			// handle the special case where our integer is narrower than the type of the powers table,
			// or by coincidence, the greatest guessed power of BASE is representable by the powers table.
			// e.g. for base 10:
			//   greatest guess for 64-bit is floor(log10(2^63)) = 18
			//   greatest representable power of 10 for 64-bit is pow(10, 19)
			//     pow(10, 18 + 1) <= pow(10, 19) => we can always access powers[guess + 1]
			//   BUT
			//   greatest guess for 8-bit is floor(log10(2^7)) = 2
			//   greatest representable power of 10 for 8-bit is pow(10, 2) = 100
			//     pow(10, 2 + 1) > pow(10, 2)    => we can not always access powers[guess + 1]
			//     (however, the powers table is not made of 8-bit integers, so we actually can)
			const unsigned char guess = guesser(log2floor(val));
			return guess + (val >= powers[guess + 1]);
		} else {
			// ALTERNATIVE from: https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog10
			// This version is always safe from overflow for any non-powers of two.
			// However, we want zero-preservation and need an additional zero-check compared to the linked page.
			const unsigned char guess = guesser(log2floor(val) + 1);
			return (guess - (val < powers[guess])) * (val != 0);
		}
	}
}

JSONIFIER_INLINE uint64_t fastDigitCountLogFloor(const uint64_t inputValue) {
	return logFloor<10>(inputValue) + 1;
}

JSONIFIER_INLINE int int_log2(uint64_t x) {
	return 63 - jsonifier::simd::lzcnt(x | 1);
}

JSONIFIER_INLINE int digit_count(uint32_t x) {
	static const uint32_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999 };
	int y						  = (9 * int_log2(x)) >> 5;
	y += x > table[y];
	return y + 1;
}

JSONIFIER_INLINE int digit_count(uint64_t x) {
	static const uint64_t table[] = { 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999, 9999999999, 99999999999, 999999999999, 9999999999999, 99999999999999,
		999999999999999ULL, 9999999999999999ULL, 99999999999999999ULL, 999999999999999999ULL, 9999999999999999999ULL };
	int y							  = (19 * int_log2(x) >> 6);
	y += x > table[y];
	return y + 1;
}

JSONIFIER_INLINE int alternative_digit_count(uint32_t x) {
	static const uint64_t table[] = { 4294967296, 8589934582, 8589934582, 8589934582, 12884901788, 12884901788, 12884901788, 17179868184, 17179868184, 17179868184, 21474826480,
		21474826480, 21474826480, 21474826480, 25769703776, 25769703776, 25769703776, 30063771072, 30063771072, 30063771072, 34349738368, 34349738368, 34349738368, 34349738368,
		38554705664, 38554705664, 38554705664, 41949672960, 41949672960, 41949672960, 42949672960, 42949672960 };
	return (x + table[int_log2(x)]) >> 32;
}

JSONIFIER_INLINE int alternative_digit_count(uint64_t x) {
	static const uint64_t table[64][2] = {
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
	// It's also possible to reuse the table from fast_digit_count_64, since the
	// first 32 entries match, and the fact that elements are 64 instead of 32 bit
	// wide doesn't seem to affect performance.
	static const uint32_t table[32] = {
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
	// table[i] is 1 less than the smallest power of 10 greater than 2 to the power of i.
	//
	// For example:
	//
	//  2^3 =  8   ->  table[3] =  10 - 1 =  9
	//  2^4 = 16   ->  table[4] = 100 - 1 = 99
	//
	static const uint64_t table[64] = {
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
	// 77/256 = 0.30078125 is an approximation of log(2)/log(10) = 0.30102999566398114
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

static constexpr auto maxIterations{ 10 };

template<size_t count, bnch_swt::string_literal name> JSONIFIER_INLINE void testFunction32() {
	std::vector<std::vector<uint32_t>> randomIntegers{};
	for (size_t x = 0; x < maxIterations; ++x) {
		randomIntegers.emplace_back(generateRandomIntegers<uint32_t>(count, sizeof(uint32_t) == 4 ? 10 : 20));
	}
	std::vector<std::vector<uint64_t>> counts{ maxIterations };
	std::vector<std::vector<uint64_t>> results{ maxIterations };
	for (size_t x = 0; x < maxIterations; ++x) {
		counts[x].resize(count);
		results[x].resize(count);
	}
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			counts[x][y] = digit_count(randomIntegers[x][y]);
		}
	}
	size_t currentIndex{};
	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"alternative-digit-count-32", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = alternative_digit_count(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "alternative-digit-count-32 failed to count the integers of value: " << randomIntegers[x][y]
						  << ", instead it counted: " << results[x][y] << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}
	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"fast-digit-count-32", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = fast_digit_count(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "fast-digit-count-32 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"digit-count-32", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = digit_count(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "digit-count-32 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"rtc-32", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = fastDigitCount(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "rtc-32 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::printResults(true, true);
}

template<size_t count, bnch_swt::string_literal name> JSONIFIER_INLINE void testFunction64() {
	std::vector<std::vector<uint64_t>> randomIntegers{};
	for (size_t x = 0; x < maxIterations; ++x) {
		randomIntegers.emplace_back(generateRandomIntegers<uint64_t>(count, sizeof(uint64_t) == 4 ? 10 : 20));
	}
	std::vector<std::vector<uint64_t>> counts{ maxIterations };
	std::vector<std::vector<uint64_t>> results{ maxIterations };
	for (size_t x = 0; x < maxIterations; ++x) {
		counts[x].resize(count);
		results[x].resize(count);
	}
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			counts[x][y] = digit_count(randomIntegers[x][y]);
		}
	}
	size_t currentIndex{};
	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"alternative-digit-count-64", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = alternative_digit_count(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "alternative-digit-count-64 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}
	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"fast-digit-count-64", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = fast_digit_count(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "fast-digit-count-64 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"digit-count-64", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = digit_count(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "digit-count-64 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"eisenwave-logfloor", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = fastDigitCountLogFloor(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "eisenwave-logfloor failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"rtc-64-countl-zero", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = fastDigitCountCountLZero(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "rtc-64 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	currentIndex = 0;

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::template runBenchmark<"rtc-64", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount			 = fastDigitCount(randomIntegers[currentIndex][x]);
			results[currentIndex][x] = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (results[x][y] != counts[x][y]) {
				std::cout << "rtc-64 failed to count the integers of value: " << randomIntegers[x][y] << ", instead it counted: " << results[x][y]
						  << ", when it should be: " << counts[x][y] << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, maxIterations, 2>::printResults(true, true);
}

int main() {
	testFunction32<1000, "uint32-test-1000">();
	testFunction64<1000, "uint64-test-1000">();
	return 0;
}