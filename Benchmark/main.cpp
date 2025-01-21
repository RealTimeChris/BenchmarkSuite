#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include <glaze/glaze.hpp>
#include <random>

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
	uint64_t maxValue; 
	if (digitLength == 20) {
		maxValue = std::numeric_limits<uint64_t>::max();
	} else {
		maxValue = minValue * 10 - 1;
	}

	std::random_device rd;
	std::mt19937_64 gen(rd());
	if (minValue > maxValue) {
		std::swap(minValue, maxValue);
	}
	std::uniform_int_distribution<uint64_t> dist(0, maxValue);
	return dist(gen);
}

template<typename value_type> std::vector<value_type> generateRandomIntegers(size_t count, size_t maxLength = 0) {
	std::random_device rd;
	std::mt19937_64 gen(rd());
	size_t maxLengthNew{ maxLength == 0 ? 20 : maxLength };
	std::uniform_int_distribution<value_type> lengthGen(1, maxLengthNew);
	
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

template<size_t count, size_t length = 0, bnch_swt::string_literal name> JSONIFIER_INLINE void testFunction64() {
	auto randomIntegers = generateRandomIntegers<uint64_t>(count, length);
	std::vector<std::string> resultsReal{};
	std::vector<std::string> resultsTest01{};
	std::vector<std::string> resultsTest02{};
	resultsReal.resize(count);
	resultsTest01.resize(count);
	resultsTest02.resize(count);
	for (size_t x = 0; x < count; ++x) {
		resultsReal[x] = std::to_string(randomIntegers[x]);
		resultsTest01[x].resize(resultsReal[x].size());
		resultsTest02[x].resize(resultsReal[x].size());
	}

	bnch_swt::benchmark_stage<name, 16, 4>::template runBenchmark<"glz::to_chars", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			glz::to_chars(resultsTest01[x].data(), randomIntegers[x]);
			currentCount += resultsTest01[x].size();
		}
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (resultsReal[x] != resultsTest01[x]) {
			std::cout << "glz::to_chars failed to serialize an integer of value: " << resultsReal[x] << ", instead it serialized: " << resultsTest01[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<name, 16, 4>::template runBenchmark<"jsonifier::internal::to_chars", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			jsonifier::internal::toChars(resultsTest02[x].data(), randomIntegers[x]);
			currentCount += resultsTest02[x].size();
		}
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (resultsReal[x] != resultsTest02[x]) {
			std::cout << "jsonifier::internal::toChars failed to serialize an integer of value: " << resultsReal[x] << ", instead it serialized: " << resultsTest02[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<name, 16, 4>::printResults(true, true);
}

#include <iostream>
#include <cstdint>

// Constexpr ceil division for unsigned integers
constexpr uint64_t ceil_div(uint64_t a, uint64_t b) noexcept {
    return (a + b - 1) / b;  // Proper ceil division
}


// Multiplication function
JSONIFIER_INLINE static void multiply(uint64_t& value, uint64_t expValue) noexcept {
#if defined(__SIZEOF_INT128__)
    const __uint128_t res = static_cast<__uint128_t>(value) * static_cast<__uint128_t>(expValue);
    value = static_cast<uint64_t>(res >> 64); // Keep the high 64 bits
#elif defined(_M_ARM64) && !defined(__MINGW32__)
    uint64_t high = __umulh(value, expValue); // Get high part
    value = high; // Use the high part as the result
#elif (defined(_WIN64) && !defined(__clang__))
    uint64_t high;
    _umul128(value, expValue, &high); // Multiply and get high part
    value = high; // Use the high part as the result
#else
    uint64_t high;
    umul128Generic(value, expValue, &high); // Multiply and get high part
    value = high; // Use the high part as the result
#endif
}

// Adjust the result based on the low
JSONIFIER_INLINE static uint64_t divide_by_1e8(uint64_t value) noexcept {
	static constexpr uint64_t multiplier01{ 184467440737ULL };
	uint64_t high = value;
	multiply(high, multiplier01);
	uint64_t low = value - high * 100000000ULL;
	low -= (low >= 100000000ULL) ? (++high, 100000000ULL) : 0;

    std::cout << "Found High: " << high << std::endl;
	//std::cout << "Found Mid: " << mid << std::endl;
	std::cout << "Found Low: " << low << std::endl;
    
    return high;
}

int main() {
	testFunction64<1000000, 1, "uint64-test-1-1000000">();
	testFunction64<1000000, 2, "uint64-test-2-1000000">();
	testFunction64<1000000, 3, "uint64-test-3-1000000">();
	testFunction64<1000000, 4, "uint64-test-4-1000000">();
	testFunction64<1000000, 5, "uint64-test-5-1000000">();
	testFunction64<1000000, 6, "uint64-test-6-1000000">();
	testFunction64<1000000, 7, "uint64-test-7-1000000">();
	testFunction64<1000000, 8, "uint64-test-8-1000000">();
	testFunction64<1000000, 9, "uint64-test-9-1000000">();
	testFunction64<1000000, 10, "uint64-test-10-1000000">();
	testFunction64<1000000, 11, "uint64-test-11-1000000">();
	testFunction64<1000000, 12, "uint64-test-12-1000000">();
	testFunction64<1000000, 13, "uint64-test-13-1000000">();
	testFunction64<1000000, 14, "uint64-test-14-1000000">();
	testFunction64<1000000, 15, "uint64-test-15-1000000">();
	testFunction64<1000000, 16, "uint64-test-16-1000000">();
	testFunction64<1000000, 17, "uint64-test-17-1000000">();
	testFunction64<1000000, 18, "uint64-test-18-1000000">();
	testFunction64<1000000, 19, "uint64-test-19-1000000">();
	testFunction64<1000000, 20, "uint64-test-20-1000000">();
	testFunction64<1000000, 0, "uint64-test-x-1000000">();
	return 0;
}
