#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Simdjson.hpp"
//#include "Tests/Simdjson2.hpp"
#include "Tests/Common.hpp"
#include "Tests/Uint.hpp"
#include "Tests/Float.hpp"
#include "Tests/RoundTrip.hpp"
#include "Tests/Int.hpp"
#include "Tests/String.hpp"
#include <glaze/glaze.hpp>

static constexpr int64_t maxIterationCount{ 1024 };
struct test_struct02 {
	std::string testString{};
};

template<> struct jsonifier::core<test_struct02> {
	using value_type				 = test_struct02;
	static constexpr auto parseValue = createValue<&value_type::testString>();
};

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
		std::uniform_int_distribution<int64_t> digitDist(0, 9);
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

	auto generateExponent = [&](size_t maxExp) -> int64_t {
		auto newValue = std::abs(static_cast<int64_t>(maxExp));
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

			int64_t exponent = generateExponent(maxExpValue);

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
					for (int64_t i = 0; i < exponent; ++i) {
						if (result > (std::numeric_limits<value_type>::max() / 10)) {
							continue;
						}
						result *= 10;
					}
				} else {
					for (int64_t i = 0; i < std::abs(exponent); ++i) {
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

template<size_t nNew> JSONIFIER_ALWAYS_INLINE constexpr auto toLittleEndian(const char (&str)[nNew]) {
	constexpr auto N{ nNew - 1 };
	if constexpr (N == 1) {
		return static_cast<uint8_t>(static_cast<uint8_t>(str[0]));
	} else if constexpr (N == 2) {
		return static_cast<uint16_t>(static_cast<uint8_t>(str[0]) | (static_cast<uint8_t>(str[1]) << 8));
	} else if constexpr (N == 3) {
		return static_cast<uint32_t>(static_cast<uint8_t>(str[0]) | (static_cast<uint8_t>(str[1]) << 8) | (static_cast<uint8_t>(str[2]) << 16));
	} else if constexpr (N == 4) {
		return static_cast<uint32_t>(
			static_cast<uint8_t>(str[0]) | (static_cast<uint8_t>(str[1]) << 8) | (static_cast<uint8_t>(str[2]) << 16) | (static_cast<uint8_t>(str[3]) << 24));
	} else {
		return static_cast<uint32_t>(0);
	}
}

template<typename value_type, typename value_type02, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLength01(size_t intLength, size_t fracLength = 0, size_t maxExpValue = 0) {
	const std::string oldNewFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "/CitmCatalogData-Prettified.json");
	auto newFile{ oldNewFile };
	newFile.reserve(oldNewFile.size() + simdjson::SIMDJSON_PADDING);
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	value_type value{};
	std::vector<std::string> stringValues{};
	std::vector<value_type> resultValuesDig{};
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
	std::string buffer{};
	int64_t index{};
	buffer.resize(1024 * 1024);
	simdjson::ondemand::parser parserNew01{};
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "New-Simdjson-Function", "dodgerblue">(
		[=, &parserNew01, &newFile]() mutable {
			buffer[index] = '[';
			++index;
			buffer[index] = '\n';
			++index;
			bnch_swt::doNotOptimizeAway(buffer);
		});
	for (size_t x = 0; x < 1024; ++x) {
		if (resultValuesDig[x] != valuesDig[x]) {
			std::cout << "Glaze failed to parse: " << stringValues[x] << ", ";
			std::cout << "Glaze failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
	value = 0;
	index = 0;
	simdjson::ondemand::parser parserNew02{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Old-Simdjson-Function", "dodgerblue">(
		[=, &parserNew02, &newFile]() mutable {
			static constexpr auto newValue{ toLittleEndian("[\n") };
			std::memcpy(&buffer[index], &newValue, 2);
			index += 2;
			bnch_swt::doNotOptimizeAway(buffer);
		});
	for (size_t x = 0; x < 1024; ++x) {
		if (resultValuesDig[x] != valuesDig[x]) {
			std::cout << "Jsonifier failed to parse: " << stringValues[x] << ", ";
			std::cout << "Jsonifier failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}
	std::cout << "Current " + testName.view() + " Value: " << value << std::endl;
}

template<typename value_type, typename value_type02, jsonifier_internal::string_literal testStageNew, jsonifier_internal::string_literal testNameNew>
JSONIFIER_ALWAYS_INLINE void runForLength(size_t intLength, size_t fracLength = 0, size_t maxExpValue = 0) {
	auto newFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "/JsonData-Prettified.json");
	static constexpr jsonifier_internal::string_literal testStage{ testStageNew };
	static constexpr jsonifier_internal::string_literal testName{ testNameNew };
	value_type value{};
	std::vector<std::string> stringValues{};
	std::vector<value_type> resultValuesDig{};
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
		[=]() mutable {
			for (size_t x = 0; x < 1024; ++x) {
				const auto* iter = stringValues[x].data();
				auto s1			 = glz::detail::atoi(resultValuesDig[x], iter);
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
		"dodgerblue">([=]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			auto iter = stringValues[x].data();
			auto s1	  = jsonifier_internal::integer_parser<value_type, char>::parseInt(resultValuesDig[x], iter);
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

struct Empty {};
struct Special {
	int64_t integer;
	double real;
	double e;
	double E;
	double emptyKey;// "":  23456789012E66,
	int64_t zero;
	int64_t one;
	std::string space;
	std::string quote;
	std::string backslash;
	std::string controls;
	std::string slash;
	std::string alpha;
	std::string ALPHA;
	std::string digit;
	std::string number;// "0123456789": "digit",
	std::string special;
	std::string hex;
	bool aTrue;// "true": true,
	bool aFalse;// "false": false,
	int64_t* null;
	std::vector<int> array;
	Empty object;
	std::string address;
	std::string url;
	std::string comment;
	std::string commentKey;// "# -- --> */": " ",
	std::vector<int> spaced;// " s p a c e d " :[1,2 , 3
	std::vector<int> compact;
	std::string jsontext;
	std::string quotes;
	std::string key;// "\/\\\"\uCAFE\uBABE\uAB98\uFCDE\ubcda\uef4A\b\f\n\r\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?" : "A key can be any string"
};
using Pass01 = std::tuple<std::string, std::map<std::string, std::vector<std::string>>, Empty, std::vector<int64_t>, int64_t, bool, bool, int64_t*, Special, double, double,
	double, int64_t, double, double, double, double, double, double, std::string>;

template<typename value_type> jsonifier::vector<value_type> parseJsonArray(jsonifier::vector<jsonifier::raw_json_data> inputData) noexcept {
	jsonifier::vector<value_type> returnValues{}; 
	for (auto& value: inputData) {
		returnValues.emplace_back(static_cast<value_type>(value));
	}
	return returnValues;
}

template<const auto& options, typename buffer_type, typename serialize_context_type>
void serializeRawJson(buffer_type& buffer, const Special& rawData, serialize_context_type& serializePair) {
	buffer[serializePair.index] = '{';
	++serializePair.index;
	std::memcpy(&buffer[serializePair.index], "\"integer\":", std::size("\"integer\":") - 1);
	serializePair.index += std::size("\"integer\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.integer, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"real\":", std::size("\"real\":") - 1);
	serializePair.index += std::size("\"real\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.real, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"e\":", std::size("\"e\":") - 1);
	serializePair.index += std::size("\"e\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.e, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"E\":", std::size("\"E\":") - 1);
	serializePair.index += std::size("\"E\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.E, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"\":", std::size("\"\":") - 1);
	serializePair.index += std::size("\"\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.emptyKey, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"zero\":", std::size("\"zero\":") - 1);
	serializePair.index += std::size("\"zero\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.zero, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"one\":", std::size("\"one\":") - 1);
	serializePair.index += std::size("\"one\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.one, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"space\":", std::size("\"space\":") - 1);
	serializePair.index += std::size("\"space\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.space, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"quote\":", std::size("\"quote\":") - 1);
	serializePair.index += std::size("\"quote\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.quote, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"backslash\":", std::size("\"backslash\":") - 1);
	serializePair.index += std::size("\"backslash\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.backslash, buffer, serializePair); 

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"controls\":", std::size("\"controls\":") - 1);
	serializePair.index += std::size("\"controls\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.controls, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"slash\":", std::size("\"slash\":") - 1);
	serializePair.index += std::size("\"slash\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.slash, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"alpha\":", std::size("\"alpha\":") - 1);
	serializePair.index += std::size("\"alpha\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.alpha, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"ALPHA\":", std::size("\"ALPHA\":") - 1);
	serializePair.index += std::size("\"ALPHA\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.ALPHA, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"digit\":", std::size("\"digit\":") - 1);
	serializePair.index += std::size("\"digit\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.digit, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"0123456789\":", std::size("\"0123456789\":") - 1);
	serializePair.index += std::size("\"0123456789\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.number, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"special\":", std::size("\"special\":") - 1);
	serializePair.index += std::size("\"special\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.special, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"hex\":", std::size("\"hex\":") - 1);
	serializePair.index += std::size("\"hex\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.hex, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"true\":", std::size("\"true\":") - 1);
	serializePair.index += std::size("\"true\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.aTrue, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"false\":", std::size("\"false\":") - 1);
	serializePair.index += std::size("\"false\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.aFalse, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"null\":", std::size("\"null\":") - 1);
	serializePair.index += std::size("\"null\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.null, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"array\":", std::size("\"array\":") - 1);
	serializePair.index += std::size("\"array\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.array, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"object\":", std::size("\"object\":") - 1);
	serializePair.index += std::size("\"object\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.object, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"address\":", std::size("\"address\":") - 1);
	serializePair.index += std::size("\"address\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.address, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"url\":", std::size("\"url\":") - 1);
	serializePair.index += std::size("\"url\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.url, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"comment\":", std::size("\"comment\":") - 1);
	serializePair.index += std::size("\"comment\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.comment, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"# -- --> */\":", std::size("\"# -- --> */\":") - 1);
	serializePair.index += std::size("\"# -- --> */\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.commentKey, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\" s p a c e d \":", std::size("\" s p a c e d \":") - 1);
	serializePair.index += std::size("\" s p a c e d \":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.spaced, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"compact\":", std::size("\"compact\":") - 1);
	serializePair.index += std::size("\"compact\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.compact, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"jsontext\":", std::size("\"jsontext\":") - 1);
	serializePair.index += std::size("\"jsontext\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.jsontext, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"quotes\":", std::size("\"quotes\":") - 1);
	serializePair.index += std::size("\"quotes\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.quotes, buffer, serializePair);

	buffer[serializePair.index++] = ',';
	std::memcpy(&buffer[serializePair.index], "\"\\/\\\\\"\\uCAFE\\uBABE\\uAB98\\uFCDE\\ubcda\\uef4A\\b\\f\\n\\r\\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?\":",
		std::size("\"\\/\\\\\"\\uCAFE\\uBABE\\uAB98\\uFCDE\\ubcda\\uef4A\\b\\f\\n\\r\\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?\":") - 1);
	serializePair.index += std::size("\"\\/\\\\\"\\uCAFE\\uBABE\\uAB98\\uFCDE\\ubcda\\uef4A\\b\\f\\n\\r\\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?\":") - 1;
	jsonifier_internal::serialize<options>::impl(rawData.key, buffer, serializePair);

	buffer[serializePair.index] = '}';
	++serializePair.index;
	return;
}

Special parseRawJson(const jsonifier::raw_json_data& rawData) {
	auto specialData = rawData.operator std::unordered_map<jsonifier::string_base<char, 0Ui64>, jsonifier::raw_json_data, std::hash<jsonifier::string_base<char, 0Ui64>>,
		std::equal_to<jsonifier::string_base<char, 0Ui64>>, std::allocator<std::pair<const jsonifier::string, jsonifier::raw_json_data>>>();

	Special specialStruct;
	specialStruct.integer	 = static_cast<int64_t>(specialData["integer"]);
	specialStruct.real		 = static_cast<double>(specialData["real"]);
	specialStruct.e			 = static_cast<double>(specialData["e"]);
	specialStruct.E			 = static_cast<double>(specialData["E"]);
	specialStruct.emptyKey	 = static_cast<double>(specialData[""]);
	specialStruct.zero		 = static_cast<int64_t>(specialData["zero"]);
	specialStruct.one		 = static_cast<int64_t>(specialData["one"]);
	specialStruct.space		 = static_cast<std::string>(specialData["space"]);
	specialStruct.quote		 = static_cast<std::string>(specialData["quote"]);
	specialStruct.backslash	 = static_cast<std::string>(specialData["backslash"]);
	specialStruct.controls	 = static_cast<std::string>(specialData["controls"]);
	specialStruct.slash		 = static_cast<std::string>(specialData["slash"]);
	specialStruct.alpha		 = static_cast<std::string>(specialData["alpha"]);
	specialStruct.ALPHA		 = static_cast<std::string>(specialData["ALPHA"]);
	specialStruct.digit		 = static_cast<std::string>(specialData["digit"]);
	specialStruct.number	 = static_cast<std::string>(specialData["0123456789"]);
	specialStruct.special	 = static_cast<std::string>(specialData["special"]);
	specialStruct.hex		 = static_cast<std::string>(specialData["hex"]);
	specialStruct.aTrue		 = static_cast<bool>(specialData["true"]);
	specialStruct.aFalse	 = static_cast<bool>(specialData["false"]);
	specialStruct.null		 = nullptr;
	auto newArray			 = specialData["array"].operator jsonifier::vector<jsonifier::raw_json_data>();
	specialStruct.array		 = parseJsonArray<int>(newArray);
	specialStruct.object	 = Empty{};
	specialStruct.address	 = static_cast<std::string>(specialData["address"]);
	specialStruct.url		 = static_cast<std::string>(specialData["url"]);
	specialStruct.comment	 = static_cast<std::string>(specialData["comment"]);
	specialStruct.commentKey = static_cast<std::string>(specialData["# -- --> */"]);
	newArray				 = specialData[" s p a c e d "].operator jsonifier::vector<jsonifier::raw_json_data>();
	specialStruct.spaced	 = parseJsonArray<int>(newArray);
	newArray				 = specialData["compact"].operator jsonifier::vector<jsonifier::raw_json_data>();
	specialStruct.compact	 = parseJsonArray<int>(newArray);
	specialStruct.jsontext	 = static_cast<std::string>(specialData["jsontext"]);
	specialStruct.quotes	 = static_cast<std::string>(specialData["quotes"]);
	specialStruct.key		 = specialData["\\/\\\\\"\\uCAFE\\uBABE\\uAB98\\uFCDE\\ubcda\\uef4A\\b\\f\\n\\r\\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?"].operator jsonifier::string();
	for (auto& [key, value]: specialData) {
		if (key.find("1~!@#$%^&*()_+-=[]{}|;") != std::string::npos) {
			specialStruct.key = static_cast<std::string>(value);
		}
	}
	return specialStruct;
}

template<> struct jsonifier::core<Empty> {
	using value_type				 = Empty;
	static constexpr auto parseValue = createValue();
};

template<typename value_type>
concept special_type = std::is_same_v<Special, std::remove_cvref_t<value_type>>;

namespace jsonifier_internal {

	template<jsonifier::serialize_options options, special_type value_type, jsonifier::concepts::buffer_like buffer_type, typename serialize_context_type>
	struct serialize_impl<options, value_type, buffer_type, serialize_context_type> {
		JSONIFIER_MAYBE_ALWAYS_INLINE static void impl(value_type& value, buffer_type& buffer, serialize_context_type& serializePair) noexcept {	
			serializeRawJson<options>(buffer, value, serializePair);
		}
	};

	template<bool minified, jsonifier::parse_options options, typename parse_context_type> struct parse_impl<minified, options, Special, parse_context_type> {
		JSONIFIER_ALWAYS_INLINE static void impl(Special& value, parse_context_type& context) noexcept {
			jsonifier::raw_json_data rawData{};
			parse_impl<minified, options, jsonifier::raw_json_data, parse_context_type>::impl(rawData, context);
			value = parseRawJson(rawData);
		}
	};
}


int32_t main() {
	std::cout << "CURRENT SIZE: " << typeid(toLittleEndian("\n,")).name() << std::endl;
	char values[3]{ '\0', '\0', '\0' };
	static constexpr uint16_t commaNewline = 0x0A2C;
	std::memcpy(values, &commaNewline, 2);
	std::cout << "CURRENT VALUES: " << values << std::endl;
	auto newFile = bnch_swt::file_loader::loadFile(std::string{ JSON_TEST_PATH } + "./Tests/ConformanceTests/pass1.json");
	// Example usage
	jsonifier::raw_json_data jsonData{ newFile };
	// Initialize jsonData with the JSON jsonifier::string

	std::cout << "CURRENT VALUES: " << newFile << std::endl;
	jsonifier::jsonifier_core parser{};
	try {
		Pass01 parsedData{};
		parser.parseJson(parsedData, newFile);
		newFile.clear();
		parser.serializeJson(parsedData, newFile);
		for (auto& value: parser.getErrors()) {
			std::cout << "CURRENT ERROR: " << value << std::endl;
		}
		std::cout << "CURRENT ERROR: " << newFile << std::endl;
		std::cout << "CURRENT VALUES: " << std::get<19>(parsedData) << std::endl;
		// Further processing of parsedData...
	} catch (const std::exception& e) {
		std::cerr << "Error parsing JSON: " << e.what() << std::endl;
	}

	//conformance_tests::conformanceTests();
	//round_trip_tests::roundTripTests();
	//string_validation_tests::stringTests();
	//float_validation_tests::floatTests();
	uint_validation_tests::uintTests();
	int_validation_tests::intTests();
	runForLength01<uint64_t, uint64_t, "Stage-1 Parse Test", "Uint:1-Digit">(1);
	bnch_swt::benchmark_stage<"Stage-1 Parse Test", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	return 0;
	conformance_tests::conformanceTests();
	newFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "/JsonData-Prettified.json");
#if defined(JSONIFIER_WIN) || defined(JSONIFIER_LINUX)
	__m256i valueNew{ uint64_t{}, uint64_t{}, uint64_t{}, uint64_t{} };
#endif
	std::variant<int32_t, uint64_t> testVariant{ 23 };
	std::string newString{ "{\"d\":{\"activities\":[{\"created_at\":1729704195095,\"id\":\"ec0b28a579ecb4bd\",\"name\":\"/?≈‰?¢«?∫… | LINA "
						   "0.0.3\",\"type\":0}],\"broadcast\":null,\"client_status\":{\"web\":\"online\"},\"guild_id\":899144844381917254,\"status\":\"online\",\"user\":{\"id\":"
						   "860130066913296394}},\"op\":2424,\"s\":850,\"t\":\"PRESENCE_UPDATE\"}" };
	newString.reserve(newString.size());
	simdjson::ondemand::parser parserNew{};
	websocket_message newMessage{};
	//parser.parseJson<jsonifier::parse_options{ .minified = true }>(newData, newString);
	for (auto& value: parser.getErrors()) {
		std::cout << "CURRENT ERROR: " << value << std::endl;
	}
	//parser.serializeJson(newData, newString);
	auto newDoc = parserNew.iterate(newString);
	test<test_struct> messageNew{};
	if (auto error = newDoc.error(); error) {
		std::cout << "ERROR: " << error << std::endl;
		//return 0;
	}
	try {
		getValue(newMessage, newDoc.value());

		std::cout << "CURRENT OP: " << newMessage.op << std::endl;
		//std::cout << "CURRENT OP: " << newData.s.value() << std::endl;
		//std::cout << "CURRENT OP: " << newData.t.value() << std::endl;
	} catch (...) {
		//std::cout << "CURRENT OP: " << messageNew.op << std::endl;
		//std::cout << "CURRENT OP: " << messageNew.s.value() << std::endl;
		//std::cout << "CURRENT OP: " << messageNew.t.value() << std::endl;
		try {
			std::rethrow_exception(std::current_exception());
		} catch (const std::exception& e) {
			//std::cout << "ERROR: " << e.what() << std::endl;
		}
	}
	parser.parseJson<jsonifier::parse_options{ .minified = false }>(newMessage, newString);
	for (auto& value: parser.getErrors()) {
		std::cout << "CURRENT ERROR: " << value << std::endl;
	}
	//parser.serializeJson(newData, newString);
	//std::cout << "CURRENT OP: " << newString << std::endl;
	//std::cout << "CURRENT OP: " << newData.op << std::endl;
	//std::cout << "CURRENT S: " << newData.s.value() << std::endl;
	//std::cout << "CURRENT T: " << newData.t.value() << std::endl;
	/*
	runForLength<uint64_t, uint64_t, "Uint-Integer-Short-Tests", "Uint:1-Digit">(1);
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