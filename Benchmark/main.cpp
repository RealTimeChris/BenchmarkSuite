#include <BnchSwt/BenchmarkSuite.hpp>
#include <jsonifier/Index.hpp>
#include "RandomGenerators.hpp"
#include <string>

template<size_t length> struct convert_length_to_int {
	static_assert(length <= 8, "Sorry, but that string is too long!");
	using type = std::conditional_t<length == 1, uint8_t,
		std::conditional_t<length <= 2, uint16_t, std::conditional_t<length <= 4, uint32_t, std::conditional_t<length <= 8, size_t, void>>>>;
};

template<size_t length> using convert_length_to_int_t = typename convert_length_to_int<length>::type;

template<bnch_swt::string_literal string> constexpr convert_length_to_int_t<string.size()> getStringAsInt() noexcept {
	const char* stringNew = string.data();
	convert_length_to_int_t<string.size()> returnValue{};
	for (size_t x = 0; x < string.size(); ++x) {
		returnValue |= static_cast<convert_length_to_int_t<string.size()>>(stringNew[x]) << x * 8;
	}
	return returnValue;
}

template<typename value_type, size_t size> struct mutable_constexpr_value {
	mutable value_type values[size];
};

template<bnch_swt::string_literal string> BNCH_SWT_ALWAYS_INLINE bool compareStringAsIntNew(const char* rhs) {
	static constexpr auto stringInt01{ getStringAsInt<string>() };
	if constexpr (string.size() == 8) {
		uint64_t sourceVal;
		std::memcpy(&sourceVal, rhs, 8);
		return !(sourceVal ^ stringInt01);
	}else if constexpr (string.size() == 7) {
		static constexpr auto stringInt02 = static_cast<uint16_t>(stringInt01 >> 32);
		static constexpr auto stringInt03 = static_cast<uint8_t>(stringInt01 >> 48);
		uint32_t sourceVal01;
		uint16_t sourceVal02;
		std::memcpy(&sourceVal01, rhs, 4);
		std::memcpy(&sourceVal02, rhs + 4, 2);
		return !((sourceVal01 ^ static_cast<uint32_t>(stringInt01)) | (sourceVal02 ^ stringInt02) | (rhs[6] ^ stringInt03));
	} else if constexpr (string.size() == 6) {
		static constexpr auto stringInt02 = static_cast<uint16_t>(stringInt01 >> 32);
		uint32_t sourceVal01;
		uint16_t sourceVal02;
		std::memcpy(&sourceVal01, rhs, 4);
		std::memcpy(&sourceVal02, rhs + 4, 2);
		return !((sourceVal01 ^ static_cast<uint32_t>(stringInt01)) | (sourceVal02 ^ stringInt02));
	} else if constexpr (string.size() == 5) {
		static constexpr auto stringInt02 = static_cast<uint8_t>(stringInt01 >> 32);
		uint32_t sourceVal;
		std::memcpy(&sourceVal, rhs, 4);
		return !((sourceVal ^ static_cast<uint32_t>(stringInt01)) | (rhs[4] ^ stringInt02));
	} else if constexpr (string.size() == 4) {
		uint32_t sourceVal;
		std::memcpy(&sourceVal, rhs, string.size());
		return !(sourceVal ^ stringInt01);
	} else {
		return (rhs[0] == stringInt01);
	}
}

 template<size_t N> consteval auto bytes_to_unsigned_type() noexcept {
	if constexpr (N == 1) {
		return uint8_t{};
	} else if constexpr (N == 2) {
		return uint16_t{};
	} else if constexpr (N == 4) {
		return uint32_t{};
	} else if constexpr (N == 8) {
		return uint64_t{};
	} else {
		return;
	}
}

template<size_t N> using unsigned_bytes_t = std::decay_t<decltype(bytes_to_unsigned_type<N>())>;

template<const std::string_view& Str, size_t N>
	requires(N <= 8)
consteval auto pack() {
	using T = unsigned_bytes_t<N>;
	T v{};
	for (size_t i = 0; i < N; ++i) {
		v |= (static_cast<T>(uint8_t(Str[i])) << ((i % 8) * 8));
	}
	return v;
}

template<const std::string_view& Str, size_t N>
	requires(N > 8)
consteval auto pack() {
	constexpr auto chunks = N / 8;
	std::array<uint64_t, ((chunks > 0) ? chunks + 1 : 1)> v{};
	for (size_t i = 0; i < N; ++i) {
		const auto chunk = i / 8;
		v[chunk] |= (static_cast<uint64_t>(uint8_t(Str[i])) << ((i % 8) * 8));
	}
	return v;
}

template<const std::string_view& Str, size_t N>
	requires(N <= 8)
consteval auto pack_buffered() {
	using T = unsigned_bytes_t<N>;
	T v{};
	for (size_t i = 0; i < Str.size(); ++i) {
		v |= (static_cast<T>(uint8_t(Str[i])) << ((i % 8) * 8));
	}
	return v;
}

template<const std::string_view& Str, size_t N = Str.size()> BNCH_SWT_ALWAYS_INLINE bool comparitor(const auto* other) noexcept {
	if constexpr (N == 8) {
		static constexpr auto packed = pack<Str, 8>();
		uint64_t in;
		std::memcpy(&in, other, 8);
		return (in == packed);
	} else if constexpr (N == 7) {
		static constexpr auto packed = pack_buffered<Str, 8>();
		uint64_t in{};
		std::memcpy(&in, other, 7);
		return (in == packed);
	} else if constexpr (N == 6) {
		static constexpr auto packed = pack_buffered<Str, 8>();
		uint64_t in{};
		std::memcpy(&in, other, 6);
		return (in == packed);
	} else if constexpr (N == 5) {
		static constexpr auto packed = pack<Str, 4>();
		uint32_t in;
		std::memcpy(&in, other, 4);
		return (in == packed) & (Str[4] == other[4]);
	} else if constexpr (N == 4) {
		static constexpr auto packed = pack<Str, 4>();
		uint32_t in;
		std::memcpy(&in, other, 4);
		return (in == packed);
	} else if constexpr (N == 3) {
		static constexpr auto packed = pack<Str, 2>();
		uint16_t in;
		std::memcpy(&in, other, 2);
		return (in == packed) & (Str[2] == other[2]);
	} else if constexpr (N == 2) {
		static constexpr auto packed = pack<Str, 2>();
		uint16_t in;
		std::memcpy(&in, other, 2);
		return (in == packed);
	} else if constexpr (N == 1) {
		return Str[0] == other[0];
	} else if constexpr (N == 0) {
		return true;
	} else {
		// Clang and GCC optimize this extremely well for constexpr std::string_view
		// Packing data can create more binary on GCC
		// The other cases probably aren't needed as compiler explorer shows them optimized equally well as memcmp
		return 0 == std::memcmp(Str.data(), other, N);
	}
}

template<bnch_swt::string_literal string> BNCH_SWT_ALWAYS_INLINE bool compareStringAsInt(const char* rhs) {
	static constexpr auto stringInt01{ getStringAsInt<string>() };
	if constexpr (string.size() == 8) {
		uint64_t sourceVal;
		std::memcpy(&sourceVal, rhs, string.size());
		return sourceVal ^ stringInt01;
	} else if constexpr (string.size() == 7) {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, rhs, string.size());
		return sourceVal ^ stringInt01;
	} else if constexpr (string.size() == 6) {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, rhs, string.size());
		return sourceVal ^ stringInt01;
	} else if constexpr (string.size() == 5) {
		uint64_t sourceVal{};
		std::memcpy(&sourceVal, rhs, string.size());
		return sourceVal ^ stringInt01;
	} else if constexpr (string.size() == 4) {
		uint32_t sourceVal;
		std::memcpy(&sourceVal, rhs, string.size());
		return sourceVal ^ stringInt01;
	} else if constexpr (string.size() == 3) {
		uint32_t sourceVal{};
		std::memcpy(&sourceVal, rhs, string.size());
		return sourceVal ^ stringInt01;
	} else if constexpr (string.size() == 2) {
		uint16_t sourceVal;
		std::memcpy(&sourceVal, rhs, string.size());
		return sourceVal ^ stringInt01;
	} else {
		return rhs[0] ^ stringInt01;
	}
}

template<size_t length> constexpr auto getStringLiteral() {
	char chars[length]{};
	std::copy(bnch_swt::test_generator::charset.data(), bnch_swt::test_generator::charset.data() + length, chars);
	return bnch_swt::string_literal{ chars };
}

template<size_t length> BNCH_SWT_ALWAYS_INLINE void testFunction() {
	std::vector<std::string> randomStrings{};
	for (size_t x = 0; x < 101 * 1001; ++x) {
		randomStrings.emplace_back(bnch_swt::test_generator::generateString(length));
	}
	static constexpr bnch_swt::string_literal stringLiteral{ getStringLiteral<length>() };
	static constexpr auto stringView{ stringLiteral.view() };
	static constexpr auto lengthLiteral{ bnch_swt::toStringView<length>() };
	static constexpr bnch_swt::string_literal testStage{ "comparing-reinterpret_cast-" + bnch_swt::stringLiteralFromView<lengthLiteral.size()>(lengthLiteral) };
	static constexpr bnch_swt::string_literal testName{ "reinterpret_cast-vs-not" };

	size_t currentIndex{};
	bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<"non-reinterpret_cast", "cyan">([&] {
		for (size_t x = 0; x < 1000; ++x) {
			auto newValue = compareStringAsInt<stringLiteral>(randomStrings[currentIndex].data());
			bnch_swt::doNotOptimizeAway(newValue);
			++currentIndex;
		}
		return length * 1000;
	});
	currentIndex = 0;
	bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<"glz-comparitor", "cyan">([&] {
		for (size_t x = 0; x < 1000; ++x) {
			auto newValue = comparitor<stringView, stringView.size()>(randomStrings[currentIndex].data());
			bnch_swt::doNotOptimizeAway(newValue);
			++currentIndex;
		}
		return length * 1000;
	});

	currentIndex = 0;
	bnch_swt::benchmark_stage<testStage, 100>::template runBenchmark<"reinterpret_cast", "cyan">([&] {
		for (size_t x = 0; x < 1000; ++x) {
			auto newValue = compareStringAsIntNew<stringLiteral>(randomStrings[currentIndex].data());
			bnch_swt::doNotOptimizeAway(newValue);
			++currentIndex;
		}
		return length * 1000;
	});
	bnch_swt::benchmark_stage<testStage>::printResults();
}

template<typename context_type, jsonifier::concepts::bool_t bool_type> JSONIFIER_ALWAYS_INLINE bool parseBool(bool_type& value, context_type& context) noexcept {
	const auto trueVal	= compareStringAsInt<"true">(context);
	const auto falseVal = compareStringAsInt<"false">(context);
	if JSONIFIER_LIKELY ((trueVal || falseVal)) {
		value = trueVal;
		context += 4 + falseVal;
		return true;
	} else {
		return false;
	}
}

consteval auto asConstant(auto value) {
	return value;
}

uint64_t testFunction01(const char* stringValues) {
	auto value{ asConstant(uint64_t{}) };
	std::memcpy(&value, stringValues, 8);
	return value;
}

uint64_t testFunction02(const char *stringValues) {
	constexpr mutable_constexpr_value<uint64_t,2> value{};
	std::memcpy(value.values, stringValues, 8);
	return *value.values;
}

uint64_t testFunction03(const char* stringValues) {
	uint64_t value{};
	std::memcpy(&value, stringValues, 8);
	return value;
}

int main() {
	//static_cast<uint64_t>(value)
	//static constexpr auto location{ std::source_location::current() };
	//static constexpr mutable_constexpr_value<test_struct_new, newSourceLocation<location>()> sourceVal{};
	//static constexpr auto newLocationLiteral{ newSourceLocation<location>() };
	//std::cout << "CURRENT LOCATION: " << newLocationLiteral << std::endl;
	std::string stringNew{ "false" };
	auto iter = stringNew.data();
	bool resultVal{};
	std::cout << "PARSING?: " << std::endl << parseBool(resultVal, iter) << " ,RESULT: " << resultVal << std::endl;
	stringNew = "true";
	iter	  = stringNew.data();
	std::cout << "PARSING?: " << std::endl << parseBool(resultVal, iter) << " ,RESULT: " << resultVal << std::endl;
	static constexpr bnch_swt::string_literal testLiteralFalse{ "false" };
	stringNew = "false";
	std::cout << "ARE THEY EQUAL?-false: " << compareStringAsIntNew<testLiteralFalse>(stringNew.data()) << std::endl;
	stringNew = "true";
	std::cout << "ARE THEY EQUAL?-false: " << compareStringAsIntNew<testLiteralFalse>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteralTrue{ "true" };
	stringNew = "true";
	std::cout << "ARE THEY EQUAL?-true: " << compareStringAsIntNew<testLiteralTrue>(stringNew.data()) << std::endl;
	stringNew = "false";
	std::cout << "ARE THEY EQUAL?-true: " << compareStringAsIntNew<testLiteralTrue>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteral8{ "12345678" };
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "02345678";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "10345678";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "12045678";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "12305678";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "12340678";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "12345078";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "12345608";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	stringNew = "12345670";
	std::cout << "ARE THEY EQUAL?-8: " << compareStringAsIntNew<testLiteral8>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteral7{ "1234567" };
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	stringNew = "0234567";
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	stringNew = "1034567";
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	stringNew = "1204567";
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	stringNew = "1230567";
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	stringNew = "1234067";
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	stringNew = "1234507";
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	stringNew = "1234560";
	std::cout << "ARE THEY EQUAL?-7: " << compareStringAsIntNew<testLiteral7>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteral6{ "123456" };
	std::cout << "ARE THEY EQUAL?-6: " << compareStringAsIntNew<testLiteral6>(stringNew.data()) << std::endl;
	stringNew = "023456";
	std::cout << "ARE THEY EQUAL?-6: " << compareStringAsIntNew<testLiteral6>(stringNew.data()) << std::endl;
	stringNew = "103456";
	std::cout << "ARE THEY EQUAL?-6: " << compareStringAsIntNew<testLiteral6>(stringNew.data()) << std::endl;
	stringNew = "120456";
	std::cout << "ARE THEY EQUAL?-6: " << compareStringAsIntNew<testLiteral6>(stringNew.data()) << std::endl;
	stringNew = "123056";
	std::cout << "ARE THEY EQUAL?-6: " << compareStringAsIntNew<testLiteral6>(stringNew.data()) << std::endl;
	stringNew = "123406";
	std::cout << "ARE THEY EQUAL?-6: " << compareStringAsIntNew<testLiteral6>(stringNew.data()) << std::endl;
	stringNew = "123450";
	std::cout << "ARE THEY EQUAL?-6: " << compareStringAsIntNew<testLiteral6>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteral5{ "12345" };
	std::cout << "ARE THEY EQUAL?-5: " << compareStringAsIntNew<testLiteral5>(stringNew.data()) << std::endl;
	stringNew = "02345";
	std::cout << "ARE THEY EQUAL?-5: " << compareStringAsIntNew<testLiteral5>(stringNew.data()) << std::endl;
	stringNew = "10345";
	std::cout << "ARE THEY EQUAL?-5: " << compareStringAsIntNew<testLiteral5>(stringNew.data()) << std::endl;
	stringNew = "12045";
	std::cout << "ARE THEY EQUAL?-5: " << compareStringAsIntNew<testLiteral5>(stringNew.data()) << std::endl;
	stringNew = "12305";
	std::cout << "ARE THEY EQUAL?-5: " << compareStringAsIntNew<testLiteral5>(stringNew.data()) << std::endl;
	stringNew = "12340";
	std::cout << "ARE THEY EQUAL?-5: " << compareStringAsIntNew<testLiteral5>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteral4{ "1234" };
	std::cout << "ARE THEY EQUAL?-4: " << compareStringAsIntNew<testLiteral4>(stringNew.data()) << std::endl;
	stringNew = "0234";
	std::cout << "ARE THEY EQUAL?-4: " << compareStringAsIntNew<testLiteral4>(stringNew.data()) << std::endl;
	stringNew = "1034";
	std::cout << "ARE THEY EQUAL?-4: " << compareStringAsIntNew<testLiteral4>(stringNew.data()) << std::endl;
	stringNew = "1204";
	std::cout << "ARE THEY EQUAL?-4: " << compareStringAsIntNew<testLiteral4>(stringNew.data()) << std::endl;
	stringNew = "1230";
	std::cout << "ARE THEY EQUAL?-4: " << compareStringAsIntNew<testLiteral4>(stringNew.data()) << std::endl;	
	static constexpr bnch_swt::string_literal testLiteral3{ "123" };
	std::cout << "ARE THEY EQUAL?-3: " << compareStringAsIntNew<testLiteral3>(stringNew.data()) << std::endl;
	stringNew = "023";
	std::cout << "ARE THEY EQUAL?-3: " << compareStringAsIntNew<testLiteral3>(stringNew.data()) << std::endl;
	stringNew = "103";
	std::cout << "ARE THEY EQUAL?-3: " << compareStringAsIntNew<testLiteral3>(stringNew.data()) << std::endl;
	stringNew = "120";
	std::cout << "ARE THEY EQUAL?-3: " << compareStringAsIntNew<testLiteral3>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteral2{ "12" };
	std::cout << "ARE THEY EQUAL?-2: " << compareStringAsIntNew<testLiteral2>(stringNew.data()) << std::endl;
	stringNew = "02";
	std::cout << "ARE THEY EQUAL?-2: " << compareStringAsIntNew<testLiteral2>(stringNew.data()) << std::endl;
	stringNew = "10";
	std::cout << "ARE THEY EQUAL?-2: " << compareStringAsIntNew<testLiteral2>(stringNew.data()) << std::endl;
	static constexpr bnch_swt::string_literal testLiteral1{ "1" };
	std::cout << "ARE THEY EQUAL?-1: " << compareStringAsIntNew<testLiteral1>(stringNew.data()) << std::endl;
	stringNew = "01";
	std::cout << "ARE THEY EQUAL?-1: " << compareStringAsIntNew<testLiteral1>(stringNew.data()) << std::endl;
	//testFunction<1>();
	//testFunction<2>();
	//testFunction<3>();
	testFunction<4>();
	testFunction<5>();
	testFunction<6>();
	testFunction<7>();
	testFunction<8>();
	return 0;
}
