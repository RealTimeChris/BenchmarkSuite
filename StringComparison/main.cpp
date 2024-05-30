#include <iostream>
#include <array>
#include <simdjson.h>
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Glaze.hpp"
#include "Tests/Simdjson.hpp"
#include "Tests/Common.hpp"
#include "Tests/Uint.hpp"
#include "Tests/Float.hpp"
#include "Tests/RoundTrip.hpp"
#include "Tests/Int.hpp"
#include "Tests/String.hpp"
#include <glaze/glaze.hpp>

static constexpr int64_t maxIterationCount{ 1024 };

double combineDigitParts(std::string intPart, std::string fracPart, std::string expPart) {
	if (fracPart != "0") {
		intPart += '.' + fracPart;
	}
	if (expPart.size() > 0) {
		intPart += 'e' + expPart;
	}
	auto endPtr = intPart.data() + intPart.size();
	return std::strtod(intPart.data(), &endPtr);
}

template<typename value_type> double generateJsonInteger(size_t maxIntDigits, size_t maxFracDigits, size_t maxExpValue, bool negative) {
	static_assert(std::is_integral<value_type>::value, "value_type must be an integral type.");

	std::random_device rd;
	std::mt19937 gen(rd());

	auto generateDigits = [&](size_t numDigits) -> value_type {
		std::uniform_int_distribution<int> digitDist(0, 9);
		value_type result = 0;
		for (size_t i = 0; i < numDigits; ++i) {
			value_type digit = digitDist(gen);
			if (result > (std::numeric_limits<value_type>::max() / 10)) {
				return -1;
			}
			result = result * 10 + digit;
			if (result < 0) {
				return -1;
			}
		}
		return result;
	};

	auto generateExponent = [&](size_t maxExp) -> int {
		auto newValue = std::abs(static_cast<int>(maxExp));
		return newValue;
	};

	while (true) {
		try {
			size_t intDigits	   = std::uniform_int_distribution<size_t>(maxIntDigits, maxIntDigits)(gen);
			value_type integerPart = generateDigits(intDigits);

			if (integerPart == -1)
				continue;

			value_type fractionalPart = 0;
			if (maxFracDigits > 0) {
				size_t fracDigits = std::uniform_int_distribution<size_t>(maxFracDigits, maxFracDigits)(gen);
				fractionalPart	  = generateDigits(fracDigits);

				if (fractionalPart == -1)
					continue;
			}

			int exponent = generateExponent(maxExpValue);

			value_type result = integerPart;

			if (fractionalPart > 0) {
				size_t fracDigits = std::to_string(fractionalPart).size();
				value_type scale  = std::pow(10, fracDigits);

				if (result > (std::numeric_limits<value_type>::max() / scale)) {
					continue;
				}

				result = result * scale + fractionalPart;
			}
			result *= negative ? -1 : 1;

			if (exponent != 0) {
				if (exponent > 0) {
					for (int i = 0; i < exponent; ++i) {
						if (result > (std::numeric_limits<value_type>::max() / 10)) {
							continue;
						}
						result *= 10;
					}
				} else {
					for (int i = 0; i < std::abs(exponent); ++i) {
						result /= 10;
					}
				}
			}
			return combineDigitParts(std::to_string(integerPart), std::to_string(fractionalPart), std::to_string(exponent));

		} catch (...) {
			continue;
		}
	}
}

template<typename value_type> value_type generateRandomUint64(int64_t minDigits, int64_t maxDigits) {
	if (maxDigits < minDigits || maxDigits > 20) {
		throw std::invalid_argument("Digits must be between 1 and 20, and minDigits must be <= maxDigits.");
	}
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	value_type lowerBound = static_cast<value_type>(static_cast<int64_t>(std::pow(10, minDigits - 1)));
	value_type upperBound = static_cast<value_type>(static_cast<int64_t>(std::pow(10, maxDigits - 2)));
	std::uniform_int_distribution<value_type> dist{ lowerBound, upperBound };
	return dist(gen);
}

template<typename value_type> std::string generateJsonNumber(int64_t maxIntDigits, int64_t maxFracDigits, int64_t maxExpValue, bool possiblyNegative = true) {
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::uniform_int_distribution<value_type> signDist{ 0, 1 };
	bool negative{ possiblyNegative && signDist(gen) };
	if (maxFracDigits == 0 && maxExpValue == 0) {
		return std::to_string(negative ? -1 * generateRandomUint64<value_type>(0, maxIntDigits) : generateRandomUint64<value_type>(0, maxIntDigits));
	} else {
		return std::to_string(negative ? -1 * generateJsonInteger<value_type>(maxIntDigits, maxFracDigits, maxExpValue, negative)
									   : generateJsonInteger<value_type>(maxIntDigits, maxFracDigits, maxExpValue, negative));
	}
}

template<std::floating_point value_type> auto strtoDigits(const std::string& value) {
	return std::stod(value);
}

template<std::signed_integral value_type> auto strtoDigits(const std::string& value) {
	return std::stoll(value);
}

template<std::unsigned_integral value_type> auto strtoDigits(const std::string& value) {
	return std::stoull(value);
}

template<typename T> class hybrid_allocator;

template<typename T> class stack_allocator {
  public:
	friend class hybrid_allocator<T>;
	using value_type = T;

	stack_allocator() noexcept {
		if (current == nullptr) {
			current = buffer;
			end		= current + (1024 * 128 / sizeof(T));
		}
	}

	template<typename U> stack_allocator(const stack_allocator<U>&) noexcept {
	}

	T* allocate(std::size_t n) {
		if (current + n > end) {
			return nullptr;
		}
		T* result = current;
		current += n;
		return result;
	}

	void deallocate(T* p, std::size_t n) noexcept {
	}

  protected:
	inline static T* current{ nullptr };
	inline static T* end{ nullptr };
	alignas(32) inline static T buffer[1024 * 512 / sizeof(T)]{};
};

template<typename value_type_new> class alloc_wrapper {
  public:
	using value_type	   = value_type_new;
	using pointer		   = value_type*;
	using size_type		   = size_t;
	using allocator_traits = std::allocator_traits<alloc_wrapper<value_type>>;

	JSONIFIER_ALWAYS_INLINE pointer allocate(size_type count) noexcept {
		if JSONIFIER_UNLIKELY ((count == 0)) {
			return nullptr;
		}
#if defined(JSONIFIER_MSVC)
		return static_cast<value_type*>(_aligned_malloc(jsonifier_internal::roundUpToMultiple<bytesPerStep>(count * sizeof(value_type)), bytesPerStep));
#else
		return static_cast<value_type*>(std::aligned_alloc(bytesPerStep, jsonifier_internal::roundUpToMultiple<bytesPerStep>(count * sizeof(value_type))));
#endif
	}

	JSONIFIER_ALWAYS_INLINE void deallocate(pointer ptr, size_t = 0) noexcept {
		if JSONIFIER_LIKELY ((ptr)) {
#if defined(JSONIFIER_MSVC)
			_aligned_free(ptr);
#else
			free(ptr);
#endif
		}
	}

	template<typename... arg_types> JSONIFIER_ALWAYS_INLINE void construct(pointer ptr, arg_types&&... args) noexcept {
		new (ptr) value_type(std::forward<arg_types>(args)...);
	}

	JSONIFIER_ALWAYS_INLINE static size_type maxSize() noexcept {
		return allocator_traits::max_size(alloc_wrapper{});
	}

	JSONIFIER_ALWAYS_INLINE void destroy(pointer ptr) noexcept {
		ptr->~value_type();
	}
};

template<typename T> class hybrid_allocator {
  public:
	using value_type = T;

	template<typename U> struct rebind {
		using other = hybrid_allocator<U>;
	};

	hybrid_allocator() = default;

	template<typename U> hybrid_allocator(const hybrid_allocator<U>&) noexcept {
	}

	T* allocate(std::size_t n) {
		T* ptr = stackAlloc.allocate(n);
		if (ptr != nullptr) {
			return ptr;
		}
		return heapAlloc.allocate(n);
	}

	void deallocate(T* p, std::size_t n) noexcept {
		if (p >= reinterpret_cast<T*>(stackAlloc.buffer) && p < reinterpret_cast<T*>(stackAlloc.buffer) + sizeof(stackAlloc.buffer)) {
			return;
		}
		heapAlloc.deallocate(p, n);
	}

  private:
	stack_allocator<T> stackAlloc;
	alloc_wrapper<T> heapAlloc;
};

template<typename value_type, typename value_type02, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLength(size_t intLength, size_t fracLength = 0, size_t maxExpValue = 0) {
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	value_type value{};
	//static constexpr jsonifier_internal::integer_parser<value_type, const char> intParser{};
	std::vector<std::string> stringValues{};
	std::vector<uint64_t> resultValuesDig{};
	std::vector<value_type> valuesDig{};
	stringValues.resize(maxIterationCount);
	resultValuesDig.resize(maxIterationCount);
	valuesDig.resize(maxIterationCount);
	for (size_t x = 0; x < maxIterationCount; ++x) {
		stringValues[x]	 = generateJsonNumber<value_type>(intLength, fracLength, maxExpValue, jsonifier::concepts::signed_type<value_type> ? true : false);
		valuesDig[x]	 = strtoDigits<value_type02>(stringValues[x]);
		const auto* iter = stringValues[x].data();
		jsonifier_internal::integer_parser<value_type, const char>::parseInt(value, iter);
		if (value != valuesDig[x]) {
			std::cout << "Jsonifier failed to parse: " << stringValues[x] << ", ";
			std::cout << "Jsonifier failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
		const auto* iterNew = stringValues[x].data();
		value				= 0;
		glz::detail::atoi(value, iterNew);
		if (value != valuesDig[x]) {
			std::cout << "Glaze failed to parse: " << stringValues[x] << ", ";
			std::cout << "Glaze failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Glaze-Parsing-Function", "dodgerblue">(
		[=, &resultValuesDig, &value]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				using StringAlloc = hybrid_allocator<char>;
				std::basic_string<char, std::char_traits<char>> s1("Stack-based string");
				s1.resize(1024 * 16);
				bnch_swt::doNotOptimizeAway(s1);
			}
		});
	for (size_t x = 0; x < 1024; ++x) {
		if (resultValuesDig[x] != valuesDig[x]) {
			std::cout << "Glaze failed to parse: " << stringValues[x] << ", ";
			std::cout << "Glaze failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Jsonifier-Parsing-Function",
		"dodgerblue">([=, &resultValuesDig, &value]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			using StringAlloc = hybrid_allocator<char>;
			std::basic_string<char, std::char_traits<char>, StringAlloc> s1("Stack-based string", StringAlloc());
			s1.resize(1024 * 16);
			bnch_swt::doNotOptimizeAway(s1);
		}
	});
	for (size_t x = 0; x < 1024; ++x) {
		if (resultValuesDig[x] != valuesDig[x]) {
			std::cout << "Jsonifier failed to parse: " << stringValues[x] << ", ";
			std::cout << "Jsonifier failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
}

struct websocket_message {
	std::optional<jsonifier::string> t{};
	std::optional<int64_t> s{};
	int64_t op{ -1 };
};

template<> struct jsonifier::core<websocket_message> {
	using value_type				 = websocket_message;
	static constexpr auto parseValue = createValue("op", &value_type::op, "s", &value_type::s, "t", &value_type::t);
};

template<> void getValue(websocket_message& value, simdjson::ondemand::value jsonData) {
	simdjson::ondemand::object obj{ getObject(jsonData) };
	getValue(value.op, obj, "op");
	getValue(value.s, obj, "s");
	getValue(value.t, obj, "t");
}

int32_t main() {
	auto newFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "/CitmCatalogData-Prettified.json");
#if defined(JSONIFIER_WIN) || defined(JSONIFIER_LINUX)
	__m256i valueNew{ uint64_t{}, uint64_t{}, uint64_t{}, uint64_t{} };
#endif
	std::variant<int32_t, uint64_t> testVariant{ 23 };
	std::string newString{
		"{\"d\":{\"activities\":[{\"created_at\":1729452628230,\"id\":\"ec0b28a579ecb4bd\",\"name\":\"TESTING\",\"type\":0}],\"broadcast\":null,\"client_status\":{\"web\":\"online\"},\"guild_id\":899144844381917254,\"status\":\"online\",\"user\":{\"id\":"
		"875908453548326922}},\"op\":0,\"s\":647,\"t\":\"PRESENCE_UPDATE\"}"
	};
	newString.reserve(newString.size() );
	simdjson::ondemand::parser parserNew{};
	jsonifier::jsonifier_core parser{};
	websocket_message newData{};
	parser.parseJson<jsonifier::parse_options{ .minified = true }>(newData, newString);
	for (auto& value: parser.getErrors()) {
		std::cout << "CURRENT ERROR: " << value << std::endl;
	}
	//parser.serializeJson(newData, newString);
	auto newDoc = parserNew.iterate(newFile);
	citm_catalog_message messageNew{};
	if (auto error = newDoc.error();error) {
		std::cout << "ERROR: " << error << std::endl;
		//return 0;
	}
	getValue(messageNew, newDoc.value());
	try {
		std::cout << "CURRENT OP: " << messageNew.venueNames.PLEYEL_PLEYEL << std::endl;
		//std::cout << "CURRENT OP: " << messageNew.s.value() << std::endl;
		//std::cout << "CURRENT OP: " << messageNew.t.value() << std::endl;
	} catch (...) {
		//std::cout << "CURRENT OP: " << messageNew.op << std::endl;
		//std::cout << "CURRENT OP: " << messageNew.s.value() << std::endl;
		//std::cout << "CURRENT OP: " << messageNew.t.value() << std::endl;
		try {
			std::rethrow_exception(std::current_exception());
		} catch (const std::exception& e) {
			std::cout << "ERROR: " << e.what() << std::endl;
		}
	}
	parser.parseJson<jsonifier::parse_options{ .minified = true }>(newData, newString);
	for (auto& value: parser.getErrors()) {
		std::cout << "CURRENT ERROR: " << value << std::endl;
	}
	parser.serializeJson(newData, newString);
	std::cout << "CURRENT OP: " << newString << std::endl;
	//std::cout << "CURRENT OP: " << newData.op << std::endl;
	//std::cout << "CURRENT S: " << newData.s.value() << std::endl;
	//std::cout << "CURRENT T: " << newData.t.value() << std::endl;
	//conformance_tests::conformanceTests();
	//round_trip_tests::roundTripTests();
	//string_validation_tests::stringTests();
	//float_validation_tests::floatTests();
	uint_validation_tests::uintTests();
	int_validation_tests::intTests();
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:1-Digit">(1);
	/*
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:4-Digit">(4);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:7-Digit">(7);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:10-Digit">(10);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:13-Digit">(13);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:16-Digit">(16);
	runForLength<uint64_t, uint64_t, "Uint-Integer-Tests", "Uint:19-Digit">(19);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:1-Digit">(1);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:4-Digit">(4);
	runForLength<int64_t, int64_t, "Int-Integer-Short-Tests", "Int:7-Digit">(7);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:10-Digit">(10);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:13-Digit">(13);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:16-Digit">(16);
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:19-Digit">(18);*/
	bnch_swt::benchmark_stage<"Uint-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Uint-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	return 0;
}