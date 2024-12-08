#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <thread>

template<typename value_type>
concept has_value_type = requires() { typename value_type::value_type; };

template<has_value_type value_type> void testFunction(value_type value) {};

template<size_t length> struct convert_length_to_int {
	static_assert(length <= 8, "Sorry, but that string is too int64_t!");
	using type = std::conditional_t<length == 1, uint8_t,
		std::conditional_t<length <= 2, uint16_t, std::conditional_t<length <= 4, uint32_t, std::conditional_t<length <= 8, size_t, void>>>>;
};

template<size_t length> using convert_length_to_int_t = typename convert_length_to_int<length>::type;

template<bnch_swt::string_literal string> constexpr convert_length_to_int_t<string.size()> getStringAsInt() noexcept {
	string_view_ptr stringNew = string.data();
	convert_length_to_int_t<string.size()> returnValue{};
	for (size_t x = 0; x < string.size(); ++x) {
		returnValue |= static_cast<convert_length_to_int_t<string.size()>>(stringNew[x]) << x * 8;
	}
	return returnValue;
}

template<bnch_swt::string_literal string> JSONIFIER_ALWAYS_INLINE bool compareStringAsInt(const char* src) {
	static constexpr auto stringInt{ getStringAsInt<string>() };
	if constexpr (string.size() == 4) {
		uint32_t sourceVal;
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal == stringInt;
	} else {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal == stringInt;
	}
}

template<bnch_swt::string_literal string> JSONIFIER_ALWAYS_INLINE bool compareStringAsIntXor(const char* src) {
	static constexpr auto stringInt{ getStringAsInt<string>() };
	if constexpr (string.size() == 4) {
		uint32_t sourceVal;
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	} else {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, src, string.size());
		return sourceVal ^ stringInt;
	}
}

template<typename context_type, jsonifier::concepts::bool_t bool_type> JSONIFIER_ALWAYS_INLINE bool parseBool(bool_type& value, context_type& context) noexcept {
	const auto notTrue	= compareStringAsInt<"true">(context);
	const auto notFalse = compareStringAsInt<"false">(context);
	if JSONIFIER_LIKELY (!(notTrue && notFalse)) {
		value = !notTrue;
		context += 4 + notTrue;
		return true;
	} else {
		return false;
	}
}

template<typename context_type> JSONIFIER_ALWAYS_INLINE bool parseNull(context_type& context) noexcept {
	if JSONIFIER_LIKELY (!compareStringAsInt<"null">(context)) {
		context += 4;
		return true;
	} else {
		return false;
	}
}

int main() {
	static constexpr const char* falseVal{ "null" };
	static constexpr const char* trueVal{ "true" };
	std::vector<std::string> vecOfStrings{};
	for (size_t x = 0; x < 100; ++x) {
		vecOfStrings.emplace_back(bnch_swt::test_generator::generateString(8));
	}
	bnch_swt::benchmark_stage<"TEST-01">::runBenchmark<"TEST-01-01", "compareStringAsIntXor", "cyan">([=] ()mutable {
		bool value{};
		for (size_t x = 0; x < 100; ++x) {
			char* ptr = vecOfStrings[x].data();
			value	  = compareStringAsIntXor<"true">(ptr);
			bnch_swt::doNotOptimizeAway(value);
		}
		return 1024ull;
	});

	bnch_swt::benchmark_stage<"TEST-01">::runBenchmark<"TEST-01-01", "compareStringAsInt", "cyan">([=] ()mutable {
		bool value{};
		for (size_t x = 0; x < 100; ++x) {
			char* ptr = vecOfStrings[x].data();
			value	  = compareStringAsInt<"true">(ptr);
			bnch_swt::doNotOptimizeAway(value);
		}
		return 1024ull;
	});
	bnch_swt::benchmark_stage<"TEST-01">::printResults();
	testFunction(std::vector<size_t>{});

	return 0;
}