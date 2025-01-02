#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "../Include/BnchSwt/Printable.hpp"
#include <iostream>
#include <bitset>
#include <vector>

struct uint128_t {
	std::bitset<128> values{};
	constexpr uint128_t() noexcept = default;

	constexpr uint128_t(const std::string& bit_string) {
		if (bit_string.length() != 128) {
			throw std::invalid_argument("String must be exactly 128 characters long");
		}
		for (uint64_t i = 0; i < 128; ++i) {
			if (bit_string[i] == '1') {
				values[127 - i] = 1;
			} else if (bit_string[i] == '0') {
				values[127 - i] = 0;
			} else {
				throw std::invalid_argument("String must only contain '1' or '0' characters");
			}
		}
	}

	constexpr uint128_t(uint64_t val01, uint64_t val02) : values{ 0 } {
		for (uint64_t i = 0; i < 64; ++i) {
			values[i] = (val01 >> i) & 1;
		}

		for (uint64_t i = 0; i < 64; ++i) {
			values[64 + i] = (val02 >> i) & 1;
		}
	}

	constexpr uint128_t(uint64_t val01) : values{ val01 } {};

	constexpr uint64_t log2() const {
		for (uint64_t i = 127; i >= 0; --i) {
			if (values[i] == 1) {
				return i;
			}
		}
		return -1;
	}

	constexpr uint128_t log10() const {
		double newVals[2]{};
		newVals[0] = log10Uint64(low64());
		newVals[1] = log10Uint64(high64());
		uint128_t result{ ceilUint64(newVals[0]), ceilUint64(newVals[1]) };
		return result;
	}

	constexpr uint128_t ceil() const {
		uint64_t newVals[2]{};
		newVals[0] = ceilUint64(low64());
		newVals[1] = ceilUint64(high64());
		uint128_t result{ newVals[0], newVals[1] };
		return result;
	}

	constexpr uint128_t pow(uint64_t exponent) const {
		double newVals[2]{};
		newVals[0] = powUint64(low64(), exponent);
		newVals[1] = powUint64(high64(), exponent);
		uint128_t result{ static_cast<uint64_t>(newVals[0]), static_cast<uint64_t>(newVals[1]) };
		return result;
	}

	constexpr uint64_t low64() const {
		uint64_t result = 0;
		for (uint64_t i = 0; i < 64; ++i) {
			result |= (static_cast<uint64_t>(values[i]) << i);
		}
		return result;
	}

	constexpr uint64_t high64() const {
		uint64_t result = 0;
		for (uint64_t i = 0; i < 64; ++i) {
			result |= (static_cast<uint64_t>(values[64 + i]) << i);
		}
		return result;
	}

	constexpr uint128_t operator<<(uint64_t shift) const {
		uint128_t result{};
		if (shift == 0) {
			return *this;
		}

		if (shift >= 128) {
			return uint128_t(0);
		}

		for (uint64_t i = 0; i < 128 - shift; ++i) {
			result.values[i + shift] = values[i];
		}

		return result;
	}

	void print_bits() const {
		for (int i = 127; i >= 0; --i) {
			std::cout << values[i];
			if (i % 8 == 0)
				std::cout << " ";// Optional: Format for easier reading
		}
		std::cout << std::endl;
	}

	constexpr uint128_t& operator<<=(uint64_t shift) {
		*this = *this >> shift;
		return *this;
	}

	constexpr uint128_t operator>>(uint64_t shift) const {
		uint128_t result{};
		if (shift < 128) {
			for (uint64_t i = shift; i < 128; ++i) {
				result.values[i - shift] = values[i];
			}
		}
		return result;
	}

	constexpr uint128_t operator+(const uint128_t& other) const {
		return addNumbersInBase(other);
	}

	constexpr uint128_t& operator+=(const uint128_t& other) {
		*this = *this + other;
		return *this;
	}

	constexpr uint128_t operator*(const uint128_t& other) const {
		return multiplyNumbersInBase(other);
	}

	constexpr uint128_t& operator*=(const uint128_t& other) {
		*this = *this + other;
		return *this;
	}

	constexpr uint128_t operator/(const uint128_t& other) const {
		return divideNumbersInBase(other);
	}

	constexpr uint128_t& operator/=(const uint128_t& other) {
		*this = *this + other;
		return *this;
	}

	constexpr uint128_t operator-(const uint128_t& other) const {
		return subtractNumbersInBase(other);
	}

	constexpr uint128_t& operator-=(const uint128_t& other) {
		*this = *this + other;
		return *this;
	}

	constexpr bool operator==(const uint128_t& other) const {
		return this->values == other.values;
	}

	constexpr bool uint128_t::operator>(const uint128_t& rhs) const {
		if (high64() == rhs.high64()) {
			return (low64() > rhs.low64());
		}
		return (high64() > rhs.high64());
	}

	constexpr bool uint128_t::operator<(const uint128_t& rhs) const {
		if (high64() == rhs.high64()) {
			return (low64() < rhs.low64());
		}
		return (high64() < rhs.high64());
	}

	constexpr bool uint128_t::operator>=(const uint128_t& rhs) const {
		return ((*this > rhs) || (*this == rhs));
	}

	constexpr bool uint128_t::operator<=(const uint128_t& rhs) const {
		return ((*this < rhs) || (*this == rhs));
	}

  protected:
	constexpr uint128_t addNumbersInBase(const uint128_t& V) const {
		uint64_t n = 127;
		uint128_t returnValues{};
		std::bitset<128> W{};
		uint64_t k = 0;
		uint64_t j = 0;

		while (j < n) {
			uint64_t sum		   = values[j] + V.values[j] + k;
			returnValues.values[j] = sum % 2;
			k					   = sum / 2;
			j++;
		}

		returnValues.values[n] = k;

		return returnValues;
	}

	constexpr uint128_t subtractNumbersInBase(const uint128_t& other) const noexcept {
		JSONIFIER_ALIGN uint64_t valuesNew[4];
		valuesNew[0] = low64();
		valuesNew[1] = high64();
		valuesNew[2] = other.low64();
		valuesNew[3] = other.high64();
		bool carryInNew{};
		valuesNew[2]	 = valuesNew[0] - valuesNew[2] - static_cast<uint64_t>(carryInNew);
		carryInNew		 = valuesNew[2] > valuesNew[0];
		valuesNew[1 + 2] = valuesNew[1] - valuesNew[1 + 2] - static_cast<uint64_t>(carryInNew);
		carryInNew		 = valuesNew[1 + 2] > valuesNew[1];
		return uint128_t{ valuesNew[2], valuesNew[3] };
	}

	constexpr uint128_t multiplyNumbersInBase(const uint128_t& V) const {
		uint128_t result{};
		uint128_t multiplicand = *this;
		uint128_t multiplier   = V;

		for (uint64_t i = 0; i < 128; ++i) {
			if (multiplier.values[i] == 1) {
				result = result + multiplicand;
			}
			multiplicand = multiplicand << 1;

			multiplier = multiplier >> 1;
		}

		return result;
	}

	constexpr uint128_t divideNumbersInBase(const uint128_t& V) const {
		uint128_t result{};
		uint128_t remainder = *this;
		uint128_t divisor	= V;

		if (divisor == uint128_t(0)) {
			throw std::invalid_argument("Division by zero.");
		}

		for (uint64_t i = 127; i >= 0; --i) {
			remainder = remainder << 1;
			result	  = result << 1;

			if (remainder >= divisor) {
				remainder = remainder - divisor;
				result	  = result + uint128_t(1);
			}
		}

		return result;
	}

	constexpr double powUint64(uint64_t base, uint64_t exponent) const {
		double result = 1.0;

		while (exponent > 0) {
			if (exponent % 2 == 1) {
				result *= base;
			}
			base *= base;
			exponent /= 2;
		}

		return result;
	}

	constexpr double log10Uint64(uint64_t num) const {
		if (num == 0) {
			return -std::numeric_limits<double>::infinity();
		}

		double count = 0.0;
		while (num >= 10) {
			num /= 10;
			count++;
		}

		double fractional = 0.0;
		while (num < 10) {
			num *= 2;
			fractional += 0.30103;
		}

		return count + fractional;
	}

	constexpr uint64_t ceilUint64(double value) const {
		uint64_t intPart = static_cast<uint64_t>(value);
		return (value > intPart) ? (intPart + 1) : intPart;
	}
};


template<uint64_t size> std::vector<uint64_t> convertBitsetToVector(const std::bitset<size>& bitset) {
	std::vector<uint64_t> returnValues{};
	for (uint64_t x = 0; x < size; ++x) {
		returnValues.emplace_back(bitset.test(x));
	}
	return returnValues;
}

namespace cpp23 {

	constexpr uint64_t ceil_uint64_t(double value) {
		uint64_t intPart = static_cast<uint64_t>(value);// The integer part of the number
		return (value > intPart) ? (intPart + 1) : intPart;
	}

	constexpr double log10_uint64_t(uint64_t num) {
		if (num == 0) {
			return -std::numeric_limits<double>::infinity();// Logarithm of 0 is undefined (negative infinity)
		}

		// Integer part of the logarithm
		double count = 0.0;
		while (num >= 10) {
			num /= 10;
			count++;
		}

		// Calculate the fractional part using log10 approximation
		double fractional = 0.0;
		while (num < 10) {
			num *= 2;
			fractional += 0.30103;// Adding log10(2) for each multiplication by 2
		}

		return count + fractional;
	}

	uint64_t log10(uint64_t x) {
		//std::cout << std::endl;
		//std::cout << "Log10-1 INPUT: " << x << std::endl;
		auto newVal = log10_uint64_t(x);
		//std::cout << "Log10-1 result: " << newVal << std::endl;
		auto newVal02 = ceil_uint64_t(newVal);
		//std::cout << "Log10-1 result-02: " << newVal02 << std::endl;
		return static_cast<uint64_t>(newVal02);
	}

	constexpr double pow_uint64(uint64_t base, uint64_t exponent) {
		double result = 1.0;

		// Exponentiation by squaring
		while (exponent > 0) {
			if (exponent % 2 == 1) {
				result *= base;
			}
			base *= base;
			exponent /= 2;
		}

		return result;
	}

	uint64_t pow(uint64_t x, uint64_t e) {
		//std::cout << "POW-1 INPUT: " << x << std::endl;
		//std::cout << "POW-1 INPUT: " << e << std::endl;
		auto newVal = pow_uint64(x, e);
		//std::cout << "POW-1 OUTPUT: " << newVal << std::endl;
		return static_cast<uint64_t>(newVal);
	}
}

uint32_t digitCountFast(uint32_t x) {
	static constexpr uint64_t table[] = { 4294967296, 8589934582, 8589934582, 8589934582, 12884901788, 12884901788, 12884901788, 17179868184, 17179868184, 17179868184, 21474826480,
		21474826480, 21474826480, 21474826480, 25769703776, 25769703776, 25769703776, 30063771072, 30063771072, 30063771072, 34349738368, 34349738368, 34349738368, 34349738368,
		38554705664, 38554705664, 38554705664, 41949672960, 41949672960, 41949672960, 42949672960, 42949672960 };
	return (x + table[31 - simd_internal::lzcnt(x | 1)]) >> 32;
}

#include <cstring>
#include <cstdint>

constexpr std::array<uint64_t, 10000> generateCharTable4() {
	std::array<uint64_t, 10000> table{};

	for (uint64_t i = 0; i < 10000; ++i) {
		uint64_t thousands = i / 1000 + '0';
		uint64_t hundreds  = (i / 100) % 10 + '0';
		uint64_t tens	   = (i / 10) % 10 + '0';
		uint64_t ones	   = i % 10 + '0';
		table[i]		   = (ones << 24) | (tens << 16) | (hundreds << 8) | thousands;
	}

	return table;
}

constexpr std::array<uint64_t, 1000> generateCharTable3() {
	std::array<uint64_t, 1000> table{};
	for (uint64_t i = 0; i < 1000; ++i) {
		uint64_t hundreds = (i / 100) % 10 + '0';
		uint64_t tens	  = (i / 10) % 10 + '0';
		uint64_t ones	  = i % 10 + '0';
		table[i]		  = (ones << 16) | (tens << 8) | (hundreds);
	}

	return table;
}

constexpr std::array<short, 100> generateCharTable2() {
	std::array<short, 100> table{};
	for (uint64_t i = 0; i < 100; ++i) {
		uint64_t tens = (i / 10) % 10 + '0';
		uint64_t ones = i % 10 + '0';

		table[i] = (ones << 8) | (tens);
	}

	return table;
}

constexpr std::array<char, 10> generateCharTable1() {
	std::array<char, 10> table{};
	for (uint64_t i = 0; i < 10; ++i) {
		uint64_t ones = i + '0';

		table[i] = ones;
	}

	return table;
}

constexpr auto charTable1 = generateCharTable1();
constexpr auto charTable2 = generateCharTable2();
constexpr auto charTable3 = generateCharTable3();
constexpr auto charTable4 = generateCharTable4();

JSONIFIER_INLINE char* length1(char* buf, uint64_t value) noexcept {
	const uint64_t aa = (value * 5243) >> 19;
	const uint64_t bb = value - aa * 100;
	std::memcpy(buf, charTable1.data() + bb, 1);
	return buf + 1;
}

JSONIFIER_INLINE char* length2(char* buf, uint64_t value) noexcept {
	const uint64_t aa = (value * 5243) >> 19;
	const uint64_t bb = value - aa * 100;
	std::memcpy(buf, charTable2.data() + bb, 2);
	return buf + 2;
}

JSONIFIER_INLINE char* length3(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	std::memcpy(buf, charTable3.data() + ccdd, 3);
	return buf + 3;
}

JSONIFIER_INLINE char* length4(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	std::memcpy(buf, charTable4.data() + ccdd, 4);
	return buf + 4;
}

JSONIFIER_INLINE char* length5(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	std::memcpy(buf, charTable1.data() + aabb, 1);
	std::memcpy(buf + 1, charTable4.data() + ccdd, 4);
	return buf + 5;
}

JSONIFIER_INLINE char* length6(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	std::memcpy(buf, charTable2.data() + aabb, 2);
	std::memcpy(buf + 2, charTable4.data() + ccdd, 4);
	return buf + 6;
}

JSONIFIER_INLINE char* length7(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	std::memcpy(buf, charTable3.data() + aabb, 3);
	std::memcpy(buf + 3, charTable4.data() + ccdd, 4);
	return buf + 7;
}

JSONIFIER_INLINE char* length8(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	std::memcpy(buf, charTable4.data() + aabb, 4);
	std::memcpy(buf + 4, charTable4.data() + ccdd, 4);
	return buf + 8;
}

JSONIFIER_INLINE char* length9(char* buf, uint64_t value) noexcept {
	const uint64_t aa = value / 100000000;
	const uint64_t bb = ( uint64_t )(value - aa * 100000000);
	buf				  = length1(buf, aa);
	buf				  = length8(buf, bb);
	return buf;
}

JSONIFIER_INLINE char* length10(char* buf, uint64_t value) noexcept {
	const uint64_t hgh = value / 100000000;
	const uint64_t low = value - hgh * 100000000;
	buf				   = length2(buf, hgh);
	buf				   = length8(buf, low);
	return buf;
}

template<typename value_type> JSONIFIER_INLINE static char* toCharsByDigitCount(char* buf, value_type value) noexcept {
	uint64_t numDigits{ digitCountFast(value) };
	switch (numDigits) {
		case 1: {
			return length1(buf, value);
		}
		case 2: {
			return length2(buf, value);
		}
		case 3: {
			return length3(buf, value);
		}
		case 4: {
			return length4(buf, value);
		}
		case 5: {
			return length5(buf, value);
		}
		case 6: {
			return length6(buf, value);
		}
		case 7: {
			return length7(buf, value);
		}
		case 8: {
			return length8(buf, value);
		}
		case 9: {
			return length9(buf, value);
		}
		case 10: {
			return length10(buf, value);
		}
	}
}

JSONIFIER_INLINE static char* lt10000000000000000(char* buf, uint64_t value) noexcept {
	const uint64_t hgh = value / 100000000;
	const uint64_t low = value - hgh * 100000000;
	buf = toCharsByDigitCount(buf, hgh);
	buf = length8(buf, low);
	return buf;
}

JSONIFIER_INLINE static char* gte10000000000000000(char* buf, uint64_t value) noexcept {
	const uint64_t tmp = value / 100000000;
	const uint64_t low = value - tmp * 100000000;
	const uint64_t hgh = tmp / 10000;
	const uint64_t mid = tmp - hgh * 10000;
	buf				   = toCharsByDigitCount(buf, hgh);
	buf				   = length4(buf, mid);
	buf				   = length8(buf, low);
	return buf;
}

template<jsonifier::concepts::uns64_t value_type> JSONIFIER_INLINE static char* toCharsOld(char* buf, value_type value) noexcept {
	static constexpr value_type hundred{ 100 };
	static constexpr value_type tenThousand{ 10000 };
	static constexpr value_type million{ 1000000 };
	static constexpr value_type tenMillion{ 100000000 };
	static constexpr value_type tenQuadrillion{ 10000000000000000 };
	if (value >= tenQuadrillion) {
		return gte10000000000000000(buf, value);
	} else if (value >= tenMillion) {
		return lt10000000000000000(buf, value);
	}
	return buf;
}

template<typename value_type> JSONIFIER_INLINE static char* toChars(char* buf, value_type value) noexcept {
	if (value < std::numeric_limits<uint32_t>::max()) {
		return toCharsByDigitCount(buf, value);
	} else {
		return toCharsOld(buf, value);
	}
}

constexpr uint64_t pow10(uint64_t exp) {
	uint64_t result = 1;
	for (uint64_t i = 0; i < exp; ++i) {
		result *= 10;
	}
	return result;
}

constexpr std::pair<uint64_t, uint64_t> calculateMinMaxForDigits(uint64_t digitCount) {
	return digitCount == 1 ? std::make_pair(0ULL, 9ULL) : std::make_pair(pow10(digitCount - 1), pow10(digitCount) - 1);
}

uint64_t randomValue(uint64_t start, uint64_t end) {
	srand(std::chrono::steady_clock::now().time_since_epoch().count());
	return static_cast<uint64_t>((static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) * (end - start)) + start;
}

template<uint64_t inputValue> void testFunction() {
	constexpr auto newPair = calculateMinMaxForDigits(inputValue);
	for (uint64_t x = 0; x < 10; ++x) {
		uint64_t newValue{ randomValue(newPair.first, newPair.second) };
		std::string outputValNew{};
		outputValNew.resize(32);
		toChars(outputValNew.data(), newValue);
		auto newIter = outputValNew.data() + outputValNew.size();
		uint64_t inputValNew{ std::strtoull(outputValNew.data(), &newIter, 10) };
		if (inputValNew != newValue) {
			std::cout << "INPUT VALUE: " << newValue << ", OUTPUT VALUE: " << outputValNew;
			std::cout << ", FOR LENGTH: " << inputValue << std::endl;
		}
		if constexpr (inputValue == 12) {
			//std::cout << "INPUT VALUE: " << x << ", OUTPUT VALUE: " << outputValNew << std::endl;
		}
	}
}

int main() {
	std::string buf{};
	buf.resize(42);
	uint64_t number = 1;
	toChars(buf.data(), number);
	std::cout << "Serialized number: ";
	testFunction<1>();
	testFunction<2>();
	testFunction<3>();
	testFunction<4>();
	testFunction<5>();
	testFunction<6>();
	testFunction<7>();
	testFunction<8>();
	testFunction<9>();
	testFunction<10>();
	testFunction<11>();
	testFunction<12>();
	testFunction<13>();
	testFunction<14>();
	testFunction<15>();
	testFunction<16>();
	testFunction<17>();
	testFunction<18>();
	testFunction<19>();
	testFunction<20>();

	std::cout << buf << std::endl;
	number = 12;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	std::cout << "Serialized number: " << buf << std::endl;
	number = 123;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	std::cout << "Serialized number: " << buf << std::endl;
	number = 1234;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	std::cout << "Serialized number: " << buf << std::endl;

	number = 23545;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	std::cout << "Serialized number: " << buf << std::endl;
	number = 123456;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	std::cout << "Serialized number: " << buf << std::endl;
	/*
	number = 123456;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 1234567;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 12345678;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 123456789;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 1234567891;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 12345678912;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 123456789123;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 1234567891234;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 12345678912345;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 123456789123456;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 1234567891234567;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 12345678912345678;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 123456789123456789;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 1234567891234567890;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);
	number = 12345678912345678901;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number);*/
	return 0;
}