#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>

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

JSONIFIER_INLINE_VARIABLE uint8_t digitCounts64[]{ 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10, 10,
	10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

JSONIFIER_INLINE_VARIABLE uint64_t digitCountThresholds64[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull, 9999999999ull,
	99999999999ull, 999999999999ull, 9999999999999ull, 99999999999999ull, 999999999999999ull, 9999999999999999ull, 99999999999999999ull, 999999999999999999ull,
	9999999999999999999ull };

JSONIFIER_INLINE uint64_t fastDigitCount(const uint64_t inputValue) {
	const uint64_t originalDigitCount{ digitCounts64[simd_internal::lzcnt(inputValue)] };
	return originalDigitCount + (inputValue > digitCountThresholds64[originalDigitCount]);
}

JSONIFIER_INLINE_VARIABLE uint8_t digitCounts32[]{ 9, 9, 9, 9, 9, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1 };

JSONIFIER_INLINE_VARIABLE uint32_t digitCountThresholds32[]{ 0ull, 9ull, 99ull, 999ull, 9999ull, 99999ull, 999999ull, 9999999ull, 99999999ull, 999999999ull };

JSONIFIER_INLINE uint32_t fastDigitCount(const uint32_t inputValue) {
	const uint32_t originalDigitCount{ digitCounts32[simd_internal::lzcnt(inputValue)] };
	return originalDigitCount + (inputValue > digitCountThresholds32[originalDigitCount]);
}

JSONIFIER_INLINE int int_log2(uint64_t x) {
	return 63 - simd_internal::lzcnt(x | 1);
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
	// It's also possible to reuse the table from fast_digit_count_64, since the
	// first 32 entries match, and the fact that elements are 64 instead of 32 bit
	// wide doesn't seem to affect performance.
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
	// table[i] is 1 less than the smallest power of 10 greater than 2 to the power of i.
	//
	// For example:
	//
	//  2^3 =  8   ->  table[3] =  10 - 1 =  9
	//  2^4 = 16   ->  table[4] = 100 - 1 = 99
	//
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
	// 77/256 = 0.30078125 is an approximation of log(2)/log(10) = 0.30102999566398114
	unsigned log = int_log2(x);
	return ((77 * log) >> 8) + 1 + (x > table[log]);
}

template<size_t count, bnch_swt::string_literal name> JSONIFIER_INLINE void testFunction32() {
	auto randomIntegers = generateRandomIntegers<uint32_t>(count, sizeof(uint32_t) == 4 ? 10 : 20);
	std::vector<uint32_t> counts{};
	std::vector<uint32_t> results{};
	counts.resize(count);
	results.resize(count);
	for (size_t x = 0; x < count; ++x) {
		counts[x] = digit_count(randomIntegers[x]);
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"alternative-digit-count-32", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = alternative_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "alternative-digit-count-32 failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
					  << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"fast-digit-count-32", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = fast_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "fast-digit-count-32 failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
					  << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"digit-count-32", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = alternative_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "digit-count-32 failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
					  << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"rtc-32-bit", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = fastDigitCount(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "rtc-32-bit failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x] << ", when it should be: " << counts[x]
					  << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::printResults(true, true);
}

template<size_t count, bnch_swt::string_literal name> JSONIFIER_INLINE void testFunction64() {
	auto randomIntegers = generateRandomIntegers<uint64_t>(count, sizeof(uint64_t) == 4 ? 10 : 20);
	std::vector<uint64_t> counts{};
	std::vector<uint64_t> results{};
	counts.resize(count);
	results.resize(count);
	for (size_t x = 0; x < count; ++x) {
		counts[x] = digit_count(randomIntegers[x]);
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"fast-digit-count-64", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = fast_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "fast-digit-count-64 failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
					  << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"alternative-digit-count-64", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = alternative_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "alternative-digit-count-64 failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
					  << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"digit-count-64", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = alternative_digit_count(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "digit-count-64 failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x]
					  << ", when it should be: " << counts[x] << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::template runBenchmark<"rtc-64-bit", "cyan">([&]() {
		uint64_t currentCount{};
		for (size_t x = 0; x < count; ++x) {
			auto newCount = fastDigitCount(randomIntegers[x]);
			results[x]	  = newCount;
			currentCount += static_cast<uint64_t>(newCount);
		}
		bnch_swt::doNotOptimizeAway(currentCount);
		return currentCount;
	});
	for (size_t x = 0; x < count; ++x) {
		if (results[x] != counts[x]) {
			std::cout << "rtc-64-bit failed to count the integers of value: " << randomIntegers[x] << ", instead it counted: " << results[x] << ", when it should be: " << counts[x]
					  << std::endl;
		}
	}

	bnch_swt::benchmark_stage<"compare-decimal-counting-functions-" + name, 4>::printResults(true, true);
}

int main() {
	testFunction32<1000000, "uint32-test-1000000">();
	testFunction64<1000000, "uint64-test-1000000">();
	return 0;
}