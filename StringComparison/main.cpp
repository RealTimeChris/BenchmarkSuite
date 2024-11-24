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
		for (size_t i = 0; i < 128; ++i) {
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
		for (int32_t i = 0; i < 64; ++i) {
			values[i] = (val01 >> i) & 1;
		}

		for (int32_t i = 0; i < 64; ++i) {
			values[64 + i] = (val02 >> i) & 1;
		}
	}

	constexpr uint128_t(uint64_t val01) : values{ val01 } {};

	constexpr int32_t log2() const {
		for (int32_t i = 127; i >= 0; --i) {
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

	constexpr uint128_t pow(size_t exponent) const {
		double newVals[2]{};
		newVals[0] = powUint64(low64(), exponent);
		newVals[1] = powUint64(high64(), exponent);
		uint128_t result{ static_cast<uint64_t>(newVals[0]), static_cast<uint64_t>(newVals[1]) };
		return result;
	}

	constexpr uint64_t low64() const {
		uint64_t result = 0;
		for (int32_t i = 0; i < 64; ++i) {
			result |= (static_cast<uint64_t>(values[i]) << i);
		}
		return result;
	}

	constexpr uint64_t high64() const {
		uint64_t result = 0;
		for (int32_t i = 0; i < 64; ++i) {
			result |= (static_cast<uint64_t>(values[64 + i]) << i);
		}
		return result;
	}

	constexpr uint128_t operator<<(size_t shift) const {
		uint128_t result{};
		if (shift == 0) {
			return *this;
		}

		if (shift >= 128) {
			return uint128_t(0);
		}

		for (size_t i = 0; i < 128 - shift; ++i) {
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

	constexpr uint128_t& operator<<=(size_t shift) {
		*this = *this >> shift;
		return *this;
	}

	constexpr uint128_t operator>>(size_t shift) const {
		uint128_t result{};
		if (shift < 128) {
			for (size_t i = shift; i < 128; ++i) {
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
		int32_t n = 127;
		uint128_t returnValues{};
		std::bitset<128> W{};
		int32_t k = 0;
		int32_t j = 0;

		while (j < n) {
			int32_t sum			   = values[j] + V.values[j] + k;
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
		valuesNew[2]					 = valuesNew[0] - valuesNew[2] - static_cast<uint64_t>(carryInNew);
		carryInNew						 = valuesNew[2] > valuesNew[0];
		valuesNew[1 + 2]				 = valuesNew[1] - valuesNew[1 + 2] - static_cast<uint64_t>(carryInNew);
		carryInNew						 = valuesNew[1 + 2] > valuesNew[1];
		return uint128_t{ valuesNew[2], valuesNew[3] };
	}

	constexpr uint128_t multiplyNumbersInBase(const uint128_t& V) const {
		uint128_t result{};
		uint128_t multiplicand = *this;
		uint128_t multiplier   = V;

		for (int32_t i = 0; i < 128; ++i) {
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

		for (int32_t i = 127; i >= 0; --i) {
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


template<size_t size> std::vector<int32_t> convertBitsetToVector(const std::bitset<size>& bitset) {
	std::vector<int32_t> returnValues{};
	for (size_t x = 0; x < size; ++x) {
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
		return static_cast<int32_t>(newVal02);
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

int32_t digit_count(uint32_t x) {
	static std::array<std::uint64_t, 32> table = []() {
		std::array<std::uint64_t, 32> table;
		for (uint64_t i = 1; i < 33; i++) {
			const unsigned smallest = cpp23::pow(2, i - 1);
			auto newVal01 = cpp23::pow(2, 32) - cpp23::pow(10, cpp23::log10(smallest));
			auto newVal03 = (cpp23::log10(smallest) << 32);
			table[i - 1] = (i < 31 ? (newVal01) : 0) + (newVal03);
		}
		return table;
	}();
	return (x + table[31 - simd_internal::lzcnt(x | 1)]) >> 32;
}

uint32_t digitCountFast64(uint128_t x) {
	static constexpr std::array<uint128_t, 64> table = []() {
		std::array<uint128_t, 64> table{};
		for (uint64_t i = 1; i < 65; ++i) {
			uint128_t smallest = uint128_t{ 2 }.pow(i - 1);
			uint128_t newVal01 = uint128_t{ 2 }.pow(64) - uint128_t{ 10 }.pow(smallest.log10().low64());
			uint128_t newVal02 = smallest.log10() << 64;
			table[i - 1]	   = (i < 62 ? (newVal01) : 0) + (newVal02);
		}
		return table;
	}();
	return (x + (table[62 - simd_internal::lzcnt(x.low64() | 1)]) >> 64).low64() + 1;
}
#include <cstring>
#include <iostream>
#include <cstdint>

alignas(2) constexpr char charTable[]{ 0x30u, 0x30u, 0x30u, 0x31u, 0x30u, 0x32u, 0x30u, 0x33u, 0x30u, 0x34u, 0x30u, 0x35u, 0x30u, 0x36u, 0x30u, 0x37u, 0x30u, 0x38u, 0x30u, 0x39u,
	0x31u, 0x30u, 0x31u, 0x31u, 0x31u, 0x32u, 0x31u, 0x33u, 0x31u, 0x34u, 0x31u, 0x35u, 0x31u, 0x36u, 0x31u, 0x37u, 0x31u, 0x38u, 0x31u, 0x39u, 0x32u, 0x30u, 0x32u, 0x31u, 0x32u,
	0x32u, 0x32u, 0x33u, 0x32u, 0x34u, 0x32u, 0x35u, 0x32u, 0x36u, 0x32u, 0x37u, 0x32u, 0x38u, 0x32u, 0x39u, 0x33u, 0x30u, 0x33u, 0x31u, 0x33u, 0x32u, 0x33u, 0x33u, 0x33u, 0x34u,
	0x33u, 0x35u, 0x33u, 0x36u, 0x33u, 0x37u, 0x33u, 0x38u, 0x33u, 0x39u, 0x34u, 0x30u, 0x34u, 0x31u, 0x34u, 0x32u, 0x34u, 0x33u, 0x34u, 0x34u, 0x34u, 0x35u, 0x34u, 0x36u, 0x34u,
	0x37u, 0x34u, 0x38u, 0x34u, 0x39u, 0x35u, 0x30u, 0x35u, 0x31u, 0x35u, 0x32u, 0x35u, 0x33u, 0x35u, 0x34u, 0x35u, 0x35u, 0x35u, 0x36u, 0x35u, 0x37u, 0x35u, 0x38u, 0x35u, 0x39u,
	0x36u, 0x30u, 0x36u, 0x31u, 0x36u, 0x32u, 0x36u, 0x33u, 0x36u, 0x34u, 0x36u, 0x35u, 0x36u, 0x36u, 0x36u, 0x37u, 0x36u, 0x38u, 0x36u, 0x39u, 0x37u, 0x30u, 0x37u, 0x31u, 0x37u,
	0x32u, 0x37u, 0x33u, 0x37u, 0x34u, 0x37u, 0x35u, 0x37u, 0x36u, 0x37u, 0x37u, 0x37u, 0x38u, 0x37u, 0x39u, 0x38u, 0x30u, 0x38u, 0x31u, 0x38u, 0x32u, 0x38u, 0x33u, 0x38u, 0x34u,
	0x38u, 0x35u, 0x38u, 0x36u, 0x38u, 0x37u, 0x38u, 0x38u, 0x38u, 0x39u, 0x39u, 0x30u, 0x39u, 0x31u, 0x39u, 0x32u, 0x39u, 0x33u, 0x39u, 0x34u, 0x39u, 0x35u, 0x39u, 0x36u, 0x39u,
	0x37u, 0x39u, 0x38u, 0x39u, 0x39u };

JSONIFIER_INLINE char* length1(char* buf, uint64_t value) noexcept {
	const uint64_t aa = (value * 5243) >> 19;
	const uint64_t bb = value - aa * 100;
	std::memcpy(buf, charTable + bb * 2 + 1, 1);
	return buf + 1;
}

JSONIFIER_INLINE char* length2(char* buf, uint64_t value) noexcept {
	const uint64_t aa = (value * 5243) >> 19;
	const uint64_t bb = value - aa * 100;
	std::memcpy(buf, charTable + bb * 2, 2);
	return buf + 2;
}

JSONIFIER_INLINE char* length3(char* buf, uint64_t value) noexcept {
	const uint64_t aa = (value * 5243) >> 19;
	const uint64_t bb = value - aa * 100;
	std::memcpy(buf, charTable + aa * 2 + 1, 1);
	std::memcpy(buf + 1, charTable + bb * 2, 2);
	return buf + 4;
}

JSONIFIER_INLINE char* length4(char* buf, uint64_t value) noexcept {
	const uint64_t aa = (value * 5243) >> 19;
	const uint64_t bb = value - aa * 100;
	std::memcpy(buf, charTable + aa * 2, 2);
	std::memcpy(buf + 2, charTable + bb * 2, 2);
	return buf + 4;
}

JSONIFIER_INLINE char* length5(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	const uint64_t aa	= (aabb * 5243) >> 19;
	const uint64_t cc	= (ccdd * 5243) >> 19;
	const uint64_t bb	= aabb - aa * 100;
	std::memcpy(buf, charTable + aa * 2 + 3, 1);
	std::memcpy(buf + 1, charTable + bb * 2 + 1, 2);
	//std::memcpy(buf + 4, charTable + cc * 2 + 1, 2);
	return buf + 5;
}


JSONIFIER_INLINE char* length8(char* buf, uint64_t value) noexcept {
	const uint64_t aabb = (value * 109951163) >> 40;
	const uint64_t ccdd = value - aabb * 10000;
	const uint64_t aa	= (aabb * 5243) >> 19;
	const uint64_t cc	= (ccdd * 5243) >> 19;
	const uint64_t bb	= aabb - aa * 100;
	const uint64_t dd	= ccdd - cc * 100;
	std::memcpy(buf, charTable + aa * 2, 2);
	std::memcpy(buf + 2, charTable + bb * 2, 2);
	std::memcpy(buf + 4, charTable + cc * 2, 2);
	std::memcpy(buf + 6, charTable + dd * 2, 2);
	return buf + 8;
}

template<typename T> JSONIFIER_INLINE static char* toChars(char* buf, T value, int numDigits) noexcept {
	std::cout << "NUMBER OF DIGITS: " << numDigits << ", FOR VALUE: " << value << std::endl;
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
	}
}

int main() {
	std::string buf{};
	buf.resize(42);
	uint64_t number = 1;
	std::cout << "Serialized number: " << toChars(buf.data(), number, digitCountFast64(number)) << std::endl;
	std::cout << "Serialized number: ";
	std::cout << buf << std::endl;
	number = 12;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number, digitCountFast64(number));
	std::cout << "Serialized number: " << buf << std::endl;
	number = 123;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number, digitCountFast64(number));
	std::cout << "Serialized number: " << buf << std::endl;
	number = 1234;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number, digitCountFast64(number));
	std::cout << "Serialized number: " << buf << std::endl;
	number = 12345;
	buf	   = "";
	buf.resize(32);
	toChars(buf.data(), number, digitCountFast64(number));
	std::cout << "Serialized number: " << buf << std::endl;

	return 0;
}
