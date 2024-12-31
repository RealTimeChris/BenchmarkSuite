#include <cstdint>
#include <jsonifier/Index.hpp>
#include <iostream>
#include <type_traits>
#include <utility>
#include <array>
#include "BnchSwt/BenchmarkSuite.hpp"
#include "RandomGenerators.hpp"

constexpr uint8_t charTable[]{ 0x30u, 0x30u, 0x30u, 0x31u, 0x30u, 0x32u, 0x30u, 0x33u, 0x30u, 0x34u, 0x30u, 0x35u, 0x30u, 0x36u, 0x30u, 0x37u, 0x30u, 0x38u, 0x30u, 0x39u, 0x31u,
	0x30u, 0x31u, 0x31u, 0x31u, 0x32u, 0x31u, 0x33u, 0x31u, 0x34u, 0x31u, 0x35u, 0x31u, 0x36u, 0x31u, 0x37u, 0x31u, 0x38u, 0x31u, 0x39u, 0x32u, 0x30u, 0x32u, 0x31u, 0x32u, 0x32u,
	0x32u, 0x33u, 0x32u, 0x34u, 0x32u, 0x35u, 0x32u, 0x36u, 0x32u, 0x37u, 0x32u, 0x38u, 0x32u, 0x39u, 0x33u, 0x30u, 0x33u, 0x31u, 0x33u, 0x32u, 0x33u, 0x33u, 0x33u, 0x34u, 0x33u,
	0x35u, 0x33u, 0x36u, 0x33u, 0x37u, 0x33u, 0x38u, 0x33u, 0x39u, 0x34u, 0x30u, 0x34u, 0x31u, 0x34u, 0x32u, 0x34u, 0x33u, 0x34u, 0x34u, 0x34u, 0x35u, 0x34u, 0x36u, 0x34u, 0x37u,
	0x34u, 0x38u, 0x34u, 0x39u, 0x35u, 0x30u, 0x35u, 0x31u, 0x35u, 0x32u, 0x35u, 0x33u, 0x35u, 0x34u, 0x35u, 0x35u, 0x35u, 0x36u, 0x35u, 0x37u, 0x35u, 0x38u, 0x35u, 0x39u, 0x36u,
	0x30u, 0x36u, 0x31u, 0x36u, 0x32u, 0x36u, 0x33u, 0x36u, 0x34u, 0x36u, 0x35u, 0x36u, 0x36u, 0x36u, 0x37u, 0x36u, 0x38u, 0x36u, 0x39u, 0x37u, 0x30u, 0x37u, 0x31u, 0x37u, 0x32u,
	0x37u, 0x33u, 0x37u, 0x34u, 0x37u, 0x35u, 0x37u, 0x36u, 0x37u, 0x37u, 0x37u, 0x38u, 0x37u, 0x39u, 0x38u, 0x30u, 0x38u, 0x31u, 0x38u, 0x32u, 0x38u, 0x33u, 0x38u, 0x34u, 0x38u,
	0x35u, 0x38u, 0x36u, 0x38u, 0x37u, 0x38u, 0x38u, 0x38u, 0x39u, 0x39u, 0x30u, 0x39u, 0x31u, 0x39u, 0x32u, 0x39u, 0x33u, 0x39u, 0x34u, 0x39u, 0x35u, 0x39u, 0x36u, 0x39u, 0x37u,
	0x39u, 0x38u, 0x39u, 0x39u };

JSONIFIER_INLINE char* toCharsU64Len8(char* buf, uint32_t value) noexcept {
	const uint32_t aabb = static_cast<uint32_t>((static_cast<uint64_t>(value) * 109951163) >> 40);
	const uint32_t ccdd = value - aabb * 10000;
	const uint32_t aa	= (aabb * 5243) >> 19;
	const uint32_t cc	= (ccdd * 5243) >> 19;
	const uint32_t bb	= aabb - aa * 100;
	const uint32_t dd	= ccdd - cc * 100;
	std::memcpy(buf, charTable + aa * 2, 2);
	std::memcpy(buf + 2, charTable + bb * 2, 2);
	std::memcpy(buf + 4, charTable + cc * 2, 2);
	std::memcpy(buf + 6, charTable + dd * 2, 2);
	return buf + 8;
}

JSONIFIER_INLINE char* toCharsU64Len4(char* buf, uint32_t value) noexcept {
	const uint32_t aa = (value * 5243) >> 19;
	const uint32_t bb = value - aa * 100;
	std::memcpy(buf, charTable + aa * 2, 2);
	std::memcpy(buf + 2, charTable + bb * 2, 2);
	return buf + 4;
}

JSONIFIER_INLINE char* toCharsU64Len18(char* buf, uint32_t value) noexcept {
	uint32_t aa, bb, cc, dd, aabb, bbcc, ccdd, lz;

	if (value < 100) {
		lz = value < 10;
		std::memcpy(buf, charTable + value * 2 + lz, 2);
		buf -= lz;
		return buf + 2;
	} else {
		if (value < 10000) {
			aa = (value * 5243) >> 19;
			bb = value - aa * 100;
			lz = aa < 10;
			std::memcpy(buf, charTable + aa * 2 + lz, 2);
			buf -= lz;
			std::memcpy(buf + 2, charTable + bb * 2, 2);
			return buf + 4;
		} else {
			if (value < 1000000) {
				aa	 = static_cast<uint32_t>((static_cast<uint64_t>(value) * 429497) >> 32);
				bbcc = value - aa * 10000;
				bb	 = (bbcc * 5243) >> 19;
				cc	 = bbcc - bb * 100;
				lz	 = aa < 10;
				std::memcpy(buf, charTable + aa * 2 + lz, 2);
				buf -= lz;
				std::memcpy(buf + 2, charTable + bb * 2, 2);
				std::memcpy(buf + 4, charTable + cc * 2, 2);
				return buf + 6;
			} else {
				aabb = static_cast<uint32_t>((static_cast<uint64_t>(value) * 109951163) >> 40);
				ccdd = value - aabb * 10000;
				aa	 = (aabb * 5243) >> 19;
				cc	 = (ccdd * 5243) >> 19;
				bb	 = aabb - aa * 100;
				dd	 = ccdd - cc * 100;
				lz	 = aa < 10;
				std::memcpy(buf, charTable + aa * 2 + lz, 2);
				buf -= lz;
				std::memcpy(buf + 2, charTable + bb * 2, 2);
				std::memcpy(buf + 4, charTable + cc * 2, 2);
				std::memcpy(buf + 6, charTable + dd * 2, 2);
				return buf + 8;
			}
		}
	}
}

JSONIFIER_INLINE char* toCharsU64Len58(char* buf, uint32_t value) noexcept {
	if (value < 1000000) {
		const uint32_t aa	= static_cast<uint32_t>((static_cast<uint64_t>(value) * 429497) >> 32);
		const uint32_t bbcc = value - aa * 10000;
		const uint32_t bb	= (bbcc * 5243) >> 19;
		const uint32_t cc	= bbcc - bb * 100;
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, charTable + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, charTable + bb * 2, 2);
		std::memcpy(buf + 4, charTable + cc * 2, 2);
		return buf + 6;
	} else {
		const uint32_t aabb = static_cast<uint32_t>((static_cast<uint64_t>(value) * 109951163) >> 40);
		const uint32_t ccdd = value - aabb * 10000;
		const uint32_t aa	= (aabb * 5243) >> 19;
		const uint32_t cc	= (ccdd * 5243) >> 19;
		const uint32_t bb	= aabb - aa * 100;
		const uint32_t dd	= ccdd - cc * 100;
		const uint32_t lz	= aa < 10;
		std::memcpy(buf, charTable + aa * 2 + lz, 2);
		buf -= lz;
		std::memcpy(buf + 2, charTable + bb * 2, 2);
		std::memcpy(buf + 4, charTable + cc * 2, 2);
		std::memcpy(buf + 6, charTable + dd * 2, 2);
		return buf + 8;
	}
}

template<jsonifier::concepts::uns64_t value_type> JSONIFIER_INLINE static char* toChars(char* buf, value_type value) noexcept {
	if (value == 0) {
		*buf = '0';
		return buf + 1;
	} else if (value < 100000000) {
		buf = toCharsU64Len18(buf, static_cast<uint32_t>(value));
		return buf;
	} else {
		if (value < 100000000ull * 100000000ull) {
			const uint64_t hgh = value / 100000000;
			auto low		   = static_cast<uint32_t>(value - hgh * 100000000);
			buf				   = toCharsU64Len18(buf, static_cast<uint32_t>(hgh));
			buf				   = toCharsU64Len8(buf, low);
			return buf;
		} else {
			const uint64_t tmp = value / 100000000;
			auto low		   = static_cast<uint32_t>(value - tmp * 100000000);
			auto hgh		   = static_cast<uint32_t>(tmp / 10000);
			auto mid		   = static_cast<uint32_t>(tmp - hgh * 10000);
			buf				   = toCharsU64Len58(buf, hgh);
			buf				   = toCharsU64Len4(buf, mid);
			buf				   = toCharsU64Len8(buf, low);
			return buf;
		}
	}
}

template<jsonifier::concepts::sig64_t value_type> JSONIFIER_INLINE static char* toChars(char* buf, value_type value) noexcept {
	*buf = '-';
	return toChars<uint64_t>(buf + (value < 0), static_cast<uint64_t>(value ^ (value >> 63)) - (value >> 63));
}

template<typename value_type, typename context_type> struct parse_entities_for_ptrs {
	template<size_t index> BNCH_SWT_ALWAYS_INLINE static bool processIndex(value_type& value, context_type& context) noexcept {
		std::cout << "INDEX " << index << std::endl;
		return static_cast<bool>(static_cast<int32_t>(value) * static_cast<int32_t>(value) + static_cast<int32_t>(context));
	}

	template<size_t... indices>
	BNCH_SWT_ALWAYS_INLINE static constexpr auto invokeFunction(value_type& value, context_type& context, size_t index, std::index_sequence<indices...>) {
		return ((index == indices && processIndex<indices>(value, context)), ...);
	}

	BNCH_SWT_ALWAYS_INLINE static bool invokeFunction(value_type& value, context_type& context, size_t index) {
		return invokeFunction(value, context, index, std::make_index_sequence<100>{});
	}
};

template<size_t integerLength, bnch_swt::string_literal testName> BNCH_SWT_ALWAYS_INLINE void testFunction() {
	std::vector<uint64_t> testValues{};
	for (size_t x = 0; x < 1024; ++x) {
		std::string newString{ jsonifier::toString(bnch_swt::random_generator::generateUint()) };
		newString.resize(integerLength);
		testValues.emplace_back(jsonifier::strToUint64(newString));
	}
	
	bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::runBenchmark<"Old-I-To-Str", "CYAN">([&] {
		std::string newerString{};
		size_t bytesProcessed{};
		newerString.resize(integerLength);
		for (size_t x = 0; x < 1024; ++x) {
			toChars(newerString.data(), testValues[x]);
			bytesProcessed += newerString.size();
		}
		return bytesProcessed;
	});

	bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::runBenchmark<"New-I-To-Str", "CYAN">([&] {
		std::string newerString{};
		size_t bytesProcessed{};
		newerString.resize(integerLength);
		for (size_t x = 0; x < 1024; ++x) {
			jsonifier_internal::toChars(newerString.data(), testValues[x]);
			bytesProcessed += newerString.size();
		}
		return bytesProcessed;
	});
	bnch_swt::benchmark_stage<"old-vs-new-i-to-str" + testName>::printResults();
}

int main() {
	testFunction<1, "-1">();
	return 0;
}