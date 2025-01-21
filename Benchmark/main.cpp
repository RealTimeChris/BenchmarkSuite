#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include <glaze/glaze.hpp>
#include <random>

constexpr char char_table[200] = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1', '3', '1',
	'4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', '3', '0', '3', '1',
	'3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4', '8', '4',
	'9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5', '6', '6',
	'6', '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8', '3', '8',
	'4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9' };

struct pair {
	char val01{};
	char val02{};
};

template<typename typeName> struct int_tables {
	static constexpr char charTable00[]{ 0x30u, 0x31u, 0x32u, 0x33u, 0x34u, 0x35u, 0x36u, 0x37u, 0x38u, 0x39u };

	static constexpr char charTable01[200]{ 0x30, 0x30, 0x30, 0x31, 0x30, 0x32, 0x30, 0x33, 0x30, 0x34, 0x30, 0x35, 0x30, 0x36, 0x30, 0x37, 0x30, 0x38, 0x30, 0x39, 0x31, 0x30,
		0x31, 0x31, 0x31, 0x32, 0x31, 0x33, 0x31, 0x34, 0x31, 0x35, 0x31, 0x36, 0x31, 0x37, 0x31, 0x38, 0x31, 0x39, 0x32, 0x30, 0x32, 0x31, 0x32, 0x32, 0x32, 0x33, 0x32, 0x34,
		0x32, 0x35, 0x32, 0x36, 0x32, 0x37, 0x32, 0x38, 0x32, 0x39, 0x33, 0x30, 0x33, 0x31, 0x33, 0x32, 0x33, 0x33, 0x33, 0x34, 0x33, 0x35, 0x33, 0x36, 0x33, 0x37, 0x33, 0x38,
		0x33, 0x39, 0x34, 0x30, 0x34, 0x31, 0x34, 0x32, 0x34, 0x33, 0x34, 0x34, 0x34, 0x35, 0x34, 0x36, 0x34, 0x37, 0x34, 0x38, 0x34, 0x39, 0x35, 0x30, 0x35, 0x31, 0x35, 0x32,
		0x35, 0x33, 0x35, 0x34, 0x35, 0x35, 0x35, 0x36, 0x35, 0x37, 0x35, 0x38, 0x35, 0x39, 0x36, 0x30, 0x36, 0x31, 0x36, 0x32, 0x36, 0x33, 0x36, 0x34, 0x36, 0x35, 0x36, 0x36,
		0x36, 0x37, 0x36, 0x38, 0x36, 0x39, 0x37, 0x30, 0x37, 0x31, 0x37, 0x32, 0x37, 0x33, 0x37, 0x34, 0x37, 0x35, 0x37, 0x36, 0x37, 0x37, 0x37, 0x38, 0x37, 0x39, 0x38, 0x30,
		0x38, 0x31, 0x38, 0x32, 0x38, 0x33, 0x38, 0x34, 0x38, 0x35, 0x38, 0x36, 0x38, 0x37, 0x38, 0x38, 0x38, 0x39, 0x39, 0x30, 0x39, 0x31, 0x39, 0x32, 0x39, 0x33, 0x39, 0x34,
		0x39, 0x35, 0x39, 0x36, 0x39, 0x37, 0x39, 0x38, 0x39, 0x39 };

	static constexpr uint16_t charTable02[100] = { 0x3030, 0x3130, 0x3230, 0x3330, 0x3430, 0x3530, 0x3630, 0x3730, 0x3830, 0x3930, 0x3031, 0x3131, 0x3231, 0x3331, 0x3431, 0x3531,
		0x3631, 0x3731, 0x3831, 0x3931, 0x3032, 0x3132, 0x3232, 0x3332, 0x3432, 0x3532, 0x3632, 0x3732, 0x3832, 0x3932, 0x3033, 0x3133, 0x3233, 0x3333, 0x3433, 0x3533, 0x3633,
		0x3733, 0x3833, 0x3933, 0x3034, 0x3134, 0x3234, 0x3334, 0x3434, 0x3534, 0x3634, 0x3734, 0x3834, 0x3934, 0x3035, 0x3135, 0x3235, 0x3335, 0x3435, 0x3535, 0x3635, 0x3735,
		0x3835, 0x3935, 0x3036, 0x3136, 0x3236, 0x3336, 0x3436, 0x3536, 0x3636, 0x3736, 0x3836, 0x3936, 0x3037, 0x3137, 0x3237, 0x3337, 0x3437, 0x3537, 0x3637, 0x3737, 0x3837,
		0x3937, 0x3038, 0x3138, 0x3238, 0x3338, 0x3438, 0x3538, 0x3638, 0x3738, 0x3838, 0x3938, 0x3039, 0x3139, 0x3239, 0x3339, 0x3439, 0x3539, 0x3639, 0x3739, 0x3839, 0x3939 };

	static constexpr uint8_t digitCounts[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10, 10, 10,
		9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

	static constexpr uint64_t digitCountThresholds[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull, 99999999999ull,
		999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull, 9999999999999999999ull };

	static constexpr uint8_t decTrailingZeroTable[]{ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
		0 };
};

JSONIFIER_INLINE static uint64_t fastDigitCount(const uint64_t inputValue) {
	const uint64_t originalDigitCount{ int_tables<void>::digitCounts[std::countl_zero(inputValue)] };
	return originalDigitCount + static_cast<uint64_t>(inputValue > int_tables<void>::digitCountThresholds[originalDigitCount]);
}

template<typename value_type> JSONIFIER_INLINE static char* toChars3(char* buf, value_type value) noexcept {
	const uint32_t aa		   = (value * 5243ull) >> 19ull;
	const uint32_t packedValue = static_cast<uint32_t>(int_tables<void>::charTable00[aa]) | (static_cast<uint32_t>(int_tables<void>::charTable02[value - aa * 100ull]) << 8);
	std::memcpy(buf, &packedValue, 3ull);
	return buf + 3ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars4(char* buf, value_type value) noexcept {
	const uint32_t aa		   = (value * 5243ull) >> 19ull;
	const uint32_t packedValue = (static_cast<uint32_t>(int_tables<void>::charTable02[aa])) | static_cast<uint32_t>(int_tables<void>::charTable02[(value - aa * 100ull)]) << 16;
	std::memcpy(buf, &packedValue, 4ull);
	return buf + 4ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars5(char* buf, value_type value) noexcept {
	uint32_t aa = (value * 429497ull) >> 32ull;
	uint64_t packedValue = static_cast<uint64_t>(int_tables<void>::charTable00[aa]);
	aa = value - aa * 10000ull;
	uint32_t bb = (aa * 5243ull) >> 19ull;
	uint16_t secondValue = int_tables<void>::charTable02[bb];
	uint16_t thirdValue	 = int_tables<void>::charTable02[aa - bb * 100ull];
	packedValue |= static_cast<uint64_t>(secondValue) << 8;
	packedValue |= static_cast<uint64_t>(thirdValue) << 24;
	std::memcpy(buf, &packedValue, 5);
	return buf + 5;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars6(char* buf, value_type value) noexcept {
	uint32_t aa = (value * 429497ull) >> 32ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	aa				  = value - aa * 10000ull;
	const uint32_t bb = (aa * 5243ull) >> 19ull;
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + bb, 2ull);
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + (aa - bb * 100ull), 2ull);
	return buf + 6ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars7(char* buf, value_type value) noexcept {
	uint32_t aabb = (value * 109951163ull) >> 40ull;
	uint32_t aa	  = (aabb * 5243ull) >> 19ull;
	buf[0]		  = int_tables<void>::charTable00[aa];
	std::memcpy(buf + 1ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = value - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 3ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 5ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 7ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars8(char* buf, value_type value) noexcept {
	uint32_t aabb = (value * 109951163ull) >> 40ull;
	uint32_t aa	  = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = value - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 6ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 8ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars9(char* buf, value_type value) noexcept {
	uint32_t high = (value * 720575941ull) >> 56ull;
	buf[0]		  = int_tables<void>::charTable00[high];
	high		  = value - high * 100000000ull;
	uint32_t aabb = (high * 109951163ull) >> 40ull;
	uint32_t aa	  = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 1ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 3ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = high - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 5ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 7ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 9ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars10(char* buf, value_type value) noexcept {
	const uint32_t high = static_cast<uint32_t>((value * 1801439851ull) >> 54ull);
	const uint32_t low	= static_cast<uint32_t>(value - high * 10000000ull);
	uint32_t aa			= (high * 5243ull) >> 19ull;
	buf[0]				= int_tables<void>::charTable00[aa];
	std::memcpy(buf + 1ull, int_tables<void>::charTable02 + (high - aa * 100ull), 2ull);
	uint32_t aabb = (low * 109951163ull) >> 40ull;
	aa			  = (aabb * 5243ull) >> 19ull;
	buf[3]		  = int_tables<void>::charTable00[aa];
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 6ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 8ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 10;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars11(char* buf, value_type value) noexcept {
	const uint32_t high = static_cast<uint32_t>(value / 10000000ull);
	const uint32_t low	= static_cast<uint32_t>(value - high * 10000000ull);
	uint32_t aa			= (high * 5243ull) >> 19ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + (high - aa * 100ull), 2ull);
	uint32_t aabb = (low * 109951163ull) >> 40ull;
	aa			  = (aabb * 5243ull) >> 19ull;
	buf[4]		  = int_tables<void>::charTable00[aa];
	std::memcpy(buf + 5ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 7ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 9ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 11;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars12(char* buf, value_type value) noexcept {
	const uint32_t high = static_cast<uint32_t>(value / 100000000ull);
	const uint32_t low	= static_cast<uint32_t>(value - high * 100000000ull);
	uint32_t aa			= (high * 5243ull) >> 19ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + (high - aa * 100ull), 2ull);
	uint32_t aabb = (low * 109951163ull) >> 40ull;
	aa			  = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 6ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 8ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 10ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 12;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars13(char* buf, value_type value) noexcept {
	const uint32_t high = static_cast<uint32_t>(value / 100000000ull);
	const uint32_t low	= static_cast<uint32_t>(value - high * 100000000ull);
	uint32_t aa			= (high * 429497ull) >> 32ull;
	const uint32_t bbcc = high - aa * 10000ull;
	const uint32_t bb	= (bbcc * 5243ull) >> 19ull;
	uint32_t cc			= bbcc - bb * 100ull;
	buf[0]				= int_tables<void>::charTable00[aa];
	std::memcpy(buf + 1ull, int_tables<void>::charTable02 + bb, 2ull);
	std::memcpy(buf + 3ull, int_tables<void>::charTable02 + cc, 2ull);
	const uint32_t aabb = (low * 109951163ull) >> 40ull;
	const uint32_t ccdd = low - aabb * 10000ull;
	aa					= (aabb * 5243ull) >> 19ull;
	cc					= (ccdd * 5243ull) >> 19ull;
	std::memcpy(buf + 5ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 7ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	std::memcpy(buf + 9ull, int_tables<void>::charTable02 + cc, 2ull);
	std::memcpy(buf + 11ull, int_tables<void>::charTable02 + (ccdd - cc * 100ull), 2ull);
	return buf + 13ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars14(char* buf, value_type value) noexcept {
	const uint32_t high = static_cast<uint32_t>(value / 100000000ull);
	const uint32_t low	= static_cast<uint32_t>(value - high * 100000000ull);
	uint32_t aa			= (high * 429497ull) >> 32ull;
	const uint32_t bbcc = high - aa * 10000ull;
	const uint32_t bb	= (bbcc * 5243ull) >> 19ull;
	uint32_t cc			= bbcc - bb * 100ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + bb, 2ull);
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + cc, 2ull);
	const uint32_t aabb = (low * 109951163ull) >> 40ull;
	const uint32_t ccdd = low - aabb * 10000ull;
	aa					= (aabb * 5243ull) >> 19ull;
	cc					= (ccdd * 5243ull) >> 19ull;
	std::memcpy(buf + 6ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 8ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	std::memcpy(buf + 10ull, int_tables<void>::charTable02 + cc, 2ull);
	std::memcpy(buf + 12ull, int_tables<void>::charTable02 + (ccdd - cc * 100ull), 2ull);
	return buf + 14ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars15(char* buf, value_type value) noexcept {
	const uint32_t high = static_cast<uint32_t>(value / 100000000ull);
	const uint32_t low	= static_cast<uint32_t>(value - high * 100000000ull);
	uint32_t aabb		= (high * 109951163ull) >> 40ull;
	uint32_t ccdd		= high - aabb * 10000ull;
	uint32_t aa			= (aabb * 5243ull) >> 19ull;
	uint32_t cc			= (ccdd * 5243ull) >> 19ull;
	const uint32_t bb	= aabb - aa * 100ull;
	const uint32_t dd	= ccdd - cc * 100ull;
	buf[0]				= int_tables<void>::charTable00[aa];
	std::memcpy(buf + 1ull, int_tables<void>::charTable02 + bb, 2ull);
	std::memcpy(buf + 3ull, int_tables<void>::charTable02 + cc, 2ull);
	std::memcpy(buf + 5ull, int_tables<void>::charTable02 + dd, 2ull);
	aabb = (low * 109951163ull) >> 40ull;
	ccdd = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	cc	 = (ccdd * 5243ull) >> 19ull;
	std::memcpy(buf + 7ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 9ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	std::memcpy(buf + 11ull, int_tables<void>::charTable02 + cc, 2ull);
	std::memcpy(buf + 13ull, int_tables<void>::charTable02 + (ccdd - cc * 100ull), 2ull);
	return buf + 15ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars16(char* buf, value_type value) noexcept {
	const uint32_t high = static_cast<uint32_t>(value / 100000000ull);
	const uint32_t low	= static_cast<uint32_t>(value - high * 100000000ull);
	uint32_t aabb		= (high * 109951163ull) >> 40ull;
	uint32_t ccdd		= high - aabb * 10000ull;
	uint32_t aa			= (aabb * 5243ull) >> 19ull;
	uint32_t cc			= (ccdd * 5243ull) >> 19ull;
	const uint32_t bb	= aabb - aa * 100ull;
	const uint32_t dd	= ccdd - cc * 100ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + bb, 2ull);
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + cc, 2ull);
	std::memcpy(buf + 6ull, int_tables<void>::charTable02 + dd, 2ull);
	aabb = (low * 109951163ull) >> 40ull;
	ccdd = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	cc	 = (ccdd * 5243ull) >> 19ull;
	std::memcpy(buf + 8ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 10ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	std::memcpy(buf + 12ull, int_tables<void>::charTable02 + cc, 2ull);
	std::memcpy(buf + 14ull, int_tables<void>::charTable02 + (ccdd - cc * 100ull), 2ull);
	return buf + 16ull;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars17(char* buf, value_type value) noexcept {
	const uint64_t high	 = value / 100000000ull;
	const uint64_t low	 = value - high * 100000000ull;
	const uint64_t high9 = (high * 720575941ull) >> 56ull;
	const uint64_t low9	 = high - high9 * 100000000ull;
	buf[0]				 = int_tables<void>::charTable00[high9];
	uint64_t aabb		 = (low9 * 109951163ull) >> 40ull;
	uint64_t aa			 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 1ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 3ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low9 - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 5ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 7ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = (low * 109951163ull) >> 40ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 9ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 11ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 13ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 15ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 17;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars18(char* buf, value_type value) noexcept {
	const uint64_t high	  = value / 100000000ull;
	const uint64_t low	  = value - high * 100000000ull;
	const uint64_t high10 = (high * 1801439851ull) >> 54;
	const uint64_t low10  = high - high10 * 10000000ull;
	uint64_t aa			  = (high10 * 5243ull) >> 19ull;
	buf[0]				  = int_tables<void>::charTable00[aa];
	std::memcpy(buf + 1ull, int_tables<void>::charTable02 + (high10 - aa * 100ull), 2ull);
	uint64_t aabb = (low10 * 109951163ull) >> 40ull;
	aa			  = (aabb * 5243ull) >> 19ull;
	buf[3]		  = int_tables<void>::charTable00[aa];
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low10 - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 6ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 8ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = (low * 109951163ull) >> 40ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 10ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 12ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 14ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 16ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 18;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars19(char* buf, value_type value) noexcept {
	const uint64_t high	  = value / 100000000ull;
	const uint64_t low	  = value - high * 100000000ull;
	const uint64_t high11 = high / 10000000ull;
	const uint64_t low11  = high - high11 * 10000000ull;
	uint64_t aa			  = (high11 * 5243ull) >> 19ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + (high11 - aa * 100ull), 2ull);
	uint64_t aabb = (low11 * 109951163ull) >> 40ull;
	aa			  = (aabb * 5243ull) >> 19ull;
	buf[4]		  = int_tables<void>::charTable00[aa];
	std::memcpy(buf + 5ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low11 - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 7ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 9ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = (low * 109951163ull) >> 40ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 11ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 13ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 15ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 17ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 19;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars20(char* buf, value_type value) noexcept {
	const uint64_t high	  = value / 100000000ull;
	const uint64_t low	  = value - high * 100000000ull;
	const uint64_t high12 = high / 100000000ull;
	const uint64_t low12  = high - high12 * 100000000ull;
	uint64_t aa			  = (high12 * 5243ull) >> 19ull;
	std::memcpy(buf, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 2ull, int_tables<void>::charTable02 + (high12 - aa * 100ull), 2ull);
	uint64_t aabb = (low12 * 109951163ull) >> 40ull;
	aa			  = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 4ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 6ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low12 - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 8ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 10ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = (low * 109951163ull) >> 40ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 12ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 14ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	aabb = low - aabb * 10000ull;
	aa	 = (aabb * 5243ull) >> 19ull;
	std::memcpy(buf + 16ull, int_tables<void>::charTable02 + aa, 2ull);
	std::memcpy(buf + 18ull, int_tables<void>::charTable02 + (aabb - aa * 100ull), 2ull);
	return buf + 20;
}

template<jsonifier::concepts::uns64_t value_type> JSONIFIER_INLINE char* toChars(char* buf, value_type value) noexcept {
	switch (fastDigitCount(value)) {
		case 20: {
			return toChars20(buf, value);
		}
		case 19: {
			return toChars19(buf, value);
		}
		case 18: {
			return toChars18(buf, value);
		}
		case 17: {
			return toChars17(buf, value);
		}
		case 16: {
			return toChars16(buf, value);
		}
		case 15: {
			return toChars15(buf, value);
		}
		case 14: {
			return toChars14(buf, value);
		}
		case 13: {
			return toChars13(buf, value);
		}
		case 12: {
			return toChars12(buf, value);
		}
		case 11: {
			return toChars11(buf, value);
		}
		case 10: {
			return toChars10(buf, value);
		}
		case 9: {
			return toChars9(buf, static_cast<uint32_t>(value));
		}
		case 8: {
			return toChars8(buf, static_cast<uint32_t>(value));
		}
		case 7: {
			return toChars7(buf, static_cast<uint32_t>(value));
		}
		case 6: {
			return toChars6(buf, static_cast<uint32_t>(value));
		}
		case 5: {
			return toChars5(buf, static_cast<uint32_t>(value));
		}
		case 4: {
			return toChars4(buf, static_cast<uint32_t>(value));
		}
		case 3: {
			return toChars3(buf, static_cast<uint32_t>(value));
		}
		case 2: {
			std::memcpy(buf, int_tables<void>::charTable02 + value, 2ull);
			return buf + 2ull;
		}
		default: {
			buf[0] = int_tables<void>::charTable00[value];
			return buf + 1ull;
		}
	}
}

template<jsonifier::concepts::sig64_t value_type> JSONIFIER_INLINE static char* toChars(char* buf, value_type value) noexcept {
	*buf = '-';
	return toChars(buf + (value < 0), uint64_t(value ^ (value >> 63ull)) - (value >> 63ull));
}

static constexpr char radix_100_table[] = { '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', '1', '0', '1', '1', '1', '2', '1',
	'3', '1', '4', '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', '3', '0',
	'3', '1', '3', '2', '3', '3', '3', '4', '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', '4', '5', '4', '6', '4', '7', '4',
	'8', '4', '9', '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', '6', '5',
	'6', '6', '6', '7', '6', '8', '6', '9', '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', '8', '0', '8', '1', '8', '2', '8',
	'3', '8', '4', '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', '9', '5', '9', '6', '9', '7', '9', '8', '9', '9' };

JSONIFIER_INLINE char* itoa_always_1_digits(char* buffer, std::uint32_t value) {
	std::memcpy(buffer, int_tables<void>::charTable00 + value, 1);
	return buffer + 1;
}

JSONIFIER_INLINE char* itoa_always_2_digits(char* buffer, std::uint32_t value) {
	std::memcpy(buffer, char_table + value, 2);
	return buffer + 2;
}

JSONIFIER_INLINE char* itoa_always_3_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(14411518807585588);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 1);
	return buffer + 3;
}

JSONIFIER_INLINE char* itoa_always_4_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(1441151880758559);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 2);
	return buffer + 4;
}

JSONIFIER_INLINE char* itoa_always_5_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(144115188075856);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 4, char_table + int(y >> 57), 1);
	return buffer + 5;
}

JSONIFIER_INLINE char* itoa_always_6_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(14411518807586);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 4, char_table + int(y >> 57), 2);
	return buffer + 6;
}

JSONIFIER_INLINE char* itoa_always_7_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(1441151880759);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 4, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 6, char_table + int(y >> 57), 1);
	return buffer + 7;
}

JSONIFIER_INLINE char* itoa_always_8_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(144115188076);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 4, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 6, char_table + int(y >> 57), 2);
	return buffer + 8;
}

JSONIFIER_INLINE char* itoa_always_9_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(14411518808);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 4, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 6, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 8, char_table + int(y >> 57), 1);
	return buffer + 9;
}

JSONIFIER_INLINE char* itoa_always_10_digits(char* buffer, std::uint32_t value) {
	constexpr auto mask = (std::uint64_t(1) << 57) - 1;
	auto y				= value * std::uint64_t(1441151881);
	std::memcpy(buffer, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 2, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 4, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 6, char_table + int(y >> 57), 2);
	y &= mask;
	y *= 100;
	std::memcpy(buffer + 8, char_table + int(y >> 57), 2);
	return buffer + 10;
}

char* itoa_digits(char* buffer, uint32_t value) {
	switch (fastDigitCount(value)) {
		case 1: {
			return itoa_always_1_digits(buffer, value);
		}
		case 2: {
			return itoa_always_2_digits(buffer, value);
		}
		case 3: {
			return itoa_always_3_digits(buffer, value);
		}
		case 4: {
			return itoa_always_4_digits(buffer, value);
		}
		case 5: {
			return itoa_always_5_digits(buffer, value);
		}
		case 6: {
			return itoa_always_6_digits(buffer, value);
		}
		case 7: {
			return itoa_always_7_digits(buffer, value);
		}
		case 8: {
			return itoa_always_8_digits(buffer, value);
		}
		case 9: {
			return itoa_always_9_digits(buffer, value);
		}
		case 10: {
			return itoa_always_10_digits(buffer, value);
		}
	}
}

template<class T>
	requires std::same_as<std::remove_cvref_t<T>, uint32_t>
auto* to_chars(auto* buf, T value) noexcept {
	/* The maximum value of uint32_t is 4294967295 (10 digits), */
	/* these digits are named as 'aabbccddee' here.             */
	uint32_t aa, bb, cc, dd, ee, aabb, bbcc, ccdd, ddee, aabbcc;

	/* Leading zero count in the first pair.                    */
	uint32_t lz;

	/* Although most compilers may convert the "division by     */
	/* constant value" into "multiply and shift", manual        */
	/* conversion can still help some compilers generate        */
	/* fewer and better instructions.                           */

	if (value < 100) { /* 1-2 digits: aa */
		lz = value < 10;
		std::memcpy(buf, char_table + (value * 2 + lz), 2);
		buf -= lz;
		return buf + 2;
	} else if (value < 10000) { /* 3-4 digits: aabb */
		aa = (value * 5243) >> 19; /* (value / 100) */
		bb = value - aa * 100; /* (value % 100) */
		lz = aa < 10;
		std::memcpy(buf, char_table + (aa * 2 + lz), 2);
		buf -= lz;
		std::memcpy(&buf[2], char_table + (2 * bb), 2);

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
	} else if (value < 100000000) { /* 7~8 digits: aabbccdd */
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
	} else { /* 9~10 digits: aabbccddee */
		/* (value / 10000) */
		aabbcc = uint32_t((uint64_t(value) * 3518437209ul) >> 45);
		/* (aabbcc / 10000) */
		aa	 = uint32_t((uint64_t(aabbcc) * 429497) >> 32);
		ddee = value - aabbcc * 10000; /* (value % 10000) */
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
GLZ_ALWAYS_INLINE auto* to_chars_u64_len_8(auto* buf, T value) noexcept {
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
GLZ_ALWAYS_INLINE auto* to_chars_u64_len_4(auto* buf, T value) noexcept {
	/* 4 digits: aabb */
	const uint32_t aa = (value * 5243) >> 19; /* (value / 100) */
	const uint32_t bb = value - aa * 100; /* (value % 100) */
	std::memcpy(buf, char_table + aa * 2, 2);
	std::memcpy(buf + 2, char_table + bb * 2, 2);
	return buf + 4;
}

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
inline auto* to_chars_u64_len_1_8(auto* buf, T value) noexcept {
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
auto* to_chars_u64_len_5_8(auto* buf, T value) noexcept {
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint64_t>)
auto* to_chars(auto* buf, T value) noexcept {
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

template<class T>
	requires std::same_as<std::remove_cvref_t<T>, int64_t>
auto* to_chars(auto* buf, T x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int64_t>::min case
	return to_chars(buf + (x < 0), uint64_t(x ^ (x >> 63)) - (x >> 63));
}

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
GLZ_ALWAYS_INLINE auto* to_chars_new_u64_len_8(auto* buf, T value) noexcept {
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
GLZ_ALWAYS_INLINE auto* to_chars_new_u64_len_4(auto* buf, T value) noexcept {
	const uint32_t aa		   = (value * 5243ull) >> 19ull;
	const uint32_t packedValue = (static_cast<uint32_t>(int_tables<void>::charTable02[aa])) | static_cast<uint32_t>(int_tables<void>::charTable02[(value - aa * 100ull)]) << 16;
	std::memcpy(buf, &packedValue, 4ull);
	return buf + 4;
}

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
inline auto* to_chars_new_u64_len_1_8(auto* buf, T value) noexcept {
	uint32_t aa, bb, cc, dd, aabb, bbcc, ccdd, lz;
	if (value < 100) { /* 1-2 digits: aa */
		lz = value < 10;
		std::memcpy(buf, char_table + value * 2 + lz, 2);
		buf -= lz;
		return buf + 2;
	} else if (value < 10000) {
		static constexpr std::array<uint8_t, 2> shiftAmount01{ 16, 8 };
		static constexpr std::array<uint8_t, 2> shiftAmount02{ 0, 8 };
		static constexpr std::array<uint8_t, 2> pointerIncrement{ 4, 3 };
		aa						   = (value * 5243ull) >> 19ull;
		lz						   = aa < 10;
		const uint32_t packedValue = static_cast<uint32_t>(int_tables<void>::charTable02[aa]) >> shiftAmount02[lz] |
			(static_cast<uint32_t>(int_tables<void>::charTable02[value - aa * 100ull]) << shiftAmount01[lz]);
		std::memcpy(buf, &packedValue, pointerIncrement[lz]);
		return buf + pointerIncrement[lz];
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint32_t>)
auto* to_chars_new_u64_len_5_8(auto* buf, T value) noexcept {
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

template<class T>
	requires(std::same_as<std::remove_cvref_t<T>, uint64_t>)
auto* to_chars_new(auto* buf, T value) noexcept {
	if (value < 100000000) { /* 1-8 digits */
		buf = to_chars_new_u64_len_1_8(buf, uint32_t(value));
		return buf;
	} else if (value < 100000000ull * 100000000ull) { /* 9-16 digits */
		const uint64_t hgh = value / 100000000;
		const auto low	   = uint32_t(value - hgh * 100000000); /* (value % 100000000) */
		buf				   = to_chars_new_u64_len_1_8(buf, uint32_t(hgh));
		buf				   = to_chars_new_u64_len_8(buf, low);
		return buf;
	} else { /* 17-20 digits */
		const uint64_t tmp = value / 100000000;
		const auto low	   = uint32_t(value - tmp * 100000000); /* (value % 100000000) */
		const auto hgh	   = uint32_t(tmp / 10000);
		const auto mid	   = uint32_t(tmp - hgh * 10000); /* (tmp % 10000) */
		buf				   = to_chars_new_u64_len_5_8(buf, hgh);
		buf				   = to_chars_new_u64_len_4(buf, mid);
		buf				   = to_chars_new_u64_len_8(buf, low);
		return buf;
	}
}

template<class T>
	requires std::same_as<std::remove_cvref_t<T>, int64_t>
auto* to_chars_new(auto* buf, T x) noexcept {
	*buf = '-';
	// shifts are necessary to have the numeric_limits<int64_t>::min case
	return to_chars_new(buf + (x < 0), uint64_t(x ^ (x >> 63)) - (x >> 63));
}

uint64_t generateRandomIntegerByLength(size_t digitLength) {
	if (digitLength == 0) {
		throw std::invalid_argument("Digit length must be greater than 0.");
	}

	if (digitLength > 20) {
		throw std::invalid_argument("Digit length exceeds the limit for uint64_t (maximum 20 digits).");
	}

	std::string newString{};
	newString.resize(digitLength);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint16_t> dist01(1, 9);
	std::uniform_int_distribution<uint16_t> dist02(0, 9);
	newString[0] = static_cast<char>(dist01(gen)) + '0';
	for (size_t x = 1; x < digitLength; ++x) {
		newString[x] = static_cast<char>(dist02(gen)) + '0';
	}
	auto endIter = newString.data() + newString.size();
	uint64_t newValue{ std::strtoull(newString.data(), &endIter, 10) };
	return newValue;
}

template<typename value_type> std::vector<value_type> generateRandomIntegers(size_t count, size_t maxLength = 0, bool generateOnlyGivenLength = false) {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	size_t maxLengthNew{ maxLength == 0 ? 20 : maxLength };
	std::uniform_int_distribution<value_type> lengthGen(1, maxLengthNew);

	std::vector<value_type> randomNumbers;
	for (size_t value = 0; value < count; ++value) {
		uint64_t newValue{};
		if (generateOnlyGivenLength) {
			newValue = generateRandomIntegerByLength(maxLengthNew);
		} else {
			newValue = generateRandomIntegerByLength(lengthGen(gen));
		}
		randomNumbers.push_back(newValue);
	}

	return randomNumbers;
}

static constexpr auto maxIterations{ 40 };

template<size_t count, size_t length = 0, bool generateOnlyGivenLength = false, bnch_swt::string_literal name> inline void testFunction64() {
	std::vector<uint64_t> randomIntegers{};
	randomIntegers = generateRandomIntegers<uint64_t>(count, length, generateOnlyGivenLength);
	std::vector<std::string> resultsReal{};
	std::vector<std::string> resultsTest{};
	resultsTest.resize(count);
	resultsReal.resize(count);
	for (size_t y = 0; y < count; ++y) {
		resultsReal[y] = std::to_string(randomIntegers[y]);
		resultsTest[y].resize(resultsReal[y].size());
	}
	size_t currentIndex{};

	bnch_swt::benchmark_stage<name, maxIterations, 10>::template runBenchmark<"glz::to_chars_new", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			to_chars_new(resultsTest[x].data(), randomIntegers[x]);
			currentCount += resultsTest[x].size();
		}
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (resultsReal[y] != resultsTest[y]) {
				std::cout << "glz::to_chars_new failed to serialize an integer of value: " << resultsReal[y] << ", instead it serialized: " << resultsTest[y] << std::endl;
			}
		}
	}
	currentIndex = 0;

	bnch_swt::benchmark_stage<name, maxIterations, 10>::template runBenchmark<"glz::to_chars", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			to_chars(resultsTest[x].data(), randomIntegers[x]);
			currentCount += resultsTest[x].size();
		}
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (resultsReal[y] != resultsTest[y]) {
				std::cout << "glz::to_chars failed to serialize an integer of value: " << resultsReal[y] << ", instead it serialized: " << resultsTest[y] << std::endl;
			}
		}
	}
	currentIndex = 0;
	bnch_swt::benchmark_stage<name, maxIterations, 10>::template runBenchmark<"toChars", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			toChars(resultsTest[x].data(), static_cast<uint64_t>(randomIntegers[x]));
			currentCount += resultsTest[x].size();
		}
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (resultsReal[y] != resultsTest[y]) {
				std::cout << "toChars failed to serialize an integer of value: " << resultsReal[y] << ", instead it serialized: " << resultsTest[y] << std::endl;
			}
		}
	}

	currentIndex = 0;
	bnch_swt::benchmark_stage<name, maxIterations, 10>::template runBenchmark<"jsonifier::internal::toChars", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			jsonifier::internal::toChars(resultsTest[x].data(), static_cast<uint64_t>(randomIntegers[x]));
			currentCount += resultsTest[x].size();
		}
		++currentIndex;
		return currentCount;
	});
	for (size_t x = 0; x < maxIterations; ++x) {
		for (size_t y = 0; y < count; ++y) {
			if (resultsReal[y] != resultsTest[y]) {
				std::cout << "jsonifier::internal::toChars failed to serialize an integer of value: " << resultsReal[y] << ", instead it serialized: " << resultsTest[y] << std::endl;
			}
		}
	}

	bnch_swt::benchmark_stage<name, maxIterations, 10>::printResults(true, true);
	bnch_swt::benchmark_stage<name, maxIterations, 10>::saveResultsToMarkdown("C:/users/chris/desktop/markdownresults.md");
}

int main() {
	testFunction64<100, 3, true, "uint64-test-3-100">();
	testFunction64<100, 4, true, "uint64-test-4-100">();
	testFunction64<100, 0, false, "uint64-test-x-100">();
	return 0;
}