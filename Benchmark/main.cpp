#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Jsonifier.hpp"
#include <type_traits>
#include <typeinfo>
#include <sstream>
#include <cstdint>
#include <memory>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"

template<size_t length> struct convert_length_to_int {
	static_assert(length <= 8, "Sorry, but that string is too long!");
	using type =
		std::conditional_t<length == 1, char, std::conditional_t<length <= 2, uint16_t, std::conditional_t<length <= 4, uint32_t, std::conditional_t<length <= 8, size_t, void>>>>;
};

template<size_t length> using convert_length_to_int_t = typename convert_length_to_int<length>::type;

template<size_t size, const char string[size]> constexpr convert_length_to_int_t<size> get_string_as_int() noexcept {
	convert_length_to_int_t<size> returnValue{};
	for (size_t x = 0; x < size; ++x) {
		returnValue |= static_cast<convert_length_to_int_t<size>>(string[x]) << x * 8;
	}
	return returnValue;
}

struct test_struct_new {
	constexpr test_struct_new() {
		if (!std::is_constant_evaluated()) {
			std::cout << "WERE HERE THIS IS IT!" << std::endl;
		}
	}
};

template<typename value_type> struct mutable_constexpr_int {
	mutable value_type value{};
};

template<bnch_swt::string_literal string> constexpr convert_length_to_int_t<string.size()> getStringAsInt() noexcept {
	string_view_ptr stringNew = string.data();
	convert_length_to_int_t<string.size()> returnValue{};
	for (size_t x = 0; x < string.size(); ++x) {
		returnValue |= static_cast<convert_length_to_int_t<string.size()>>(stringNew[x]) << x * 8;
	}
	return returnValue;
}

template<bnch_swt::string_literal string> JSONIFIER_ALWAYS_INLINE bool compareStringAsIntNew(const char* src) {
	static constexpr auto stringInt{ getStringAsInt<string>() };
	if constexpr (string.size() == 8) {
		constexpr mutable_constexpr_int<uint64_t> sourceVal{};
		std::memcpy(&sourceVal.value, src, string.size());
		return sourceVal.value ^ stringInt;
	} else if constexpr (string.size() == 7) {
		constexpr mutable_constexpr_int<uint64_t> sourceVal{};
		std::memcpy(&sourceVal.value, src, string.size());
		return sourceVal.value ^ stringInt;
	} else if constexpr (string.size() == 6) {
		constexpr mutable_constexpr_int<uint64_t> sourceVal{};
		std::memcpy(&sourceVal.value, src, string.size());
		return sourceVal.value ^ stringInt;
	} else if constexpr (string.size() == 5) {
		constexpr mutable_constexpr_int<uint64_t> sourceVal{};
		std::memcpy(&sourceVal.value, src, string.size());
		return sourceVal.value ^ stringInt;
	} else if constexpr (string.size() == 4) {
		constexpr mutable_constexpr_int<uint32_t> sourceVal{};
		std::memcpy(&sourceVal.value, src, string.size());
		return sourceVal.value ^ stringInt;
	} else if constexpr (string.size() == 3) {
		constexpr mutable_constexpr_int<uint32_t> sourceVal{};
		std::memcpy(&sourceVal.value, src, string.size());
		return sourceVal.value ^ stringInt;
	} else if constexpr (string.size() == 2) {
		constexpr mutable_constexpr_int<uint16_t> sourceVal{};
		std::memcpy(&sourceVal.value, src, string.size());
		return sourceVal.value ^ stringInt;
	} else {
		return src[0] ^ stringInt;
	}
}

template<bnch_swt::string_literal string> JSONIFIER_ALWAYS_INLINE bool compareStringAsInt(const char* src) {
	static constexpr auto stringInt{ getStringAsInt<string>() };
	if constexpr (string.size() == 8) {
		uint64_t sourceVal;
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else if constexpr (string.size() == 7) {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else if constexpr (string.size() == 6) {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else if constexpr (string.size() == 5) {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else if constexpr (string.size() == 4) {
		uint32_t sourceVal;
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else if constexpr (string.size() == 3) {
		uint32_t sourceVal{};
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else if constexpr (string.size() == 2) {
		uint16_t sourceVal;
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else {
		return src[0] ^ stringInt;
	}
}

template<size_t length> constexpr auto getStringLiteral() {
	char chars[length]{};
	std::copy(bnch_swt::test_generator::charset.data(), bnch_swt::test_generator::charset.data() + length, chars);
	return bnch_swt::string_literal{ chars };
}

template<size_t length> JSONIFIER_ALWAYS_INLINE void testFunction() {
	std::vector<std::string> randomStrings{};
	for (size_t x = 0; x < 101 * 100; ++x) {
		randomStrings.emplace_back(bnch_swt::test_generator::generateString(length));
	}
	static constexpr auto stringLiteral{ getStringLiteral<length>() };
	static constexpr auto lengthLiteral{ bnch_swt::toStringView<length>() };
	
	size_t currentIndex{};
	bnch_swt::benchmark_stage<"comparing-constexpr-mutable-" + bnch_swt::stringLiteralFromView<lengthLiteral.size()>(lengthLiteral), 100>::template runBenchmark<"constexpr-mutable-vs-not", "non-constexpr-mutable",
		"cyan">([&] {
		for (size_t x = 0; x < 100; ++x) {
			auto newValue = compareStringAsInt<stringLiteral>(randomStrings[currentIndex].data());
			bnch_swt::doNotOptimizeAway(newValue);
			++currentIndex;
		}
		return 8;
	});
	currentIndex = 0;
	bnch_swt::benchmark_stage<"comparing-constexpr-mutable-" + bnch_swt::stringLiteralFromView<lengthLiteral.size()>(lengthLiteral), 100>::template runBenchmark<"constexpr-mutable-vs-not",
		"constexpr-mutable", "cyan">([&] {
		for (size_t x = 0; x < 100; ++x) {
			auto newValue = compareStringAsIntNew<stringLiteral>(randomStrings[currentIndex].data());
			bnch_swt::doNotOptimizeAway(newValue);
			++currentIndex;
		}
		return 8;
	});
	bnch_swt::benchmark_stage<"comparing-constexpr-mutable-" + bnch_swt::stringLiteralFromView<lengthLiteral.size()>(lengthLiteral)>::printResults();
}

int main() {
	constexpr mutable_constexpr_int<test_struct_new> sourceVal{};
	testFunction<1>();
	testFunction<2>();
	testFunction<3>();
	testFunction<4>();
	testFunction<5>();
	testFunction<6>();
	testFunction<7>();
	testFunction<8>();
	return 0;
}
