#include <iostream>
#include <array>
#include <simdjson.h>
#include "Tests/Glaze.hpp"
#include <jsonifier/Index.hpp>
#include <BnchSwt/BenchmarkSuite.hpp>
#include "Tests/Conformance.hpp"
#include "Tests/Simdjson.hpp"
#include "Tests/Jsonifier.hpp"
#include "Tests/CitmCatalog.hpp"
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
	buffer.resize(1024 * 1024 * 16);
	simdjson::ondemand::parser parserNew01{};
	value = 0;
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "New-Simdjson-Function", "dodgerblue">(
		[=, &parserNew01, &newFile]() mutable {
			for (size_t x = 0; x < 128; ++x) {
				static constexpr jsonifier_internal::string_literal newString{ "TESTING1TESTING1" };
				std::memcpy(&buffer[index], newString.data(), 16);
				++index += 16;
				bnch_swt::doNotOptimizeAway(buffer);
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
	index = 0;
	simdjson::ondemand::parser parserNew02{};
	bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Old-Simdjson-Function", "dodgerblue">(
		[=, &parserNew02, &newFile]() mutable {
			for (size_t x = 0; x < 128; ++x) {
				static constexpr jsonifier_internal::string_literal newString{ "TESTING1" };
				static constexpr auto newValue{ jsonifier_internal::toLittleEndian<newString>() };
				static constexpr auto newValue02{ jsonifier_internal::toLittleEndian<newString>() };
				static constexpr std::array<uint64_t, 2> newArray{ newValue, newValue02 };
				std::memcpy(&buffer[index], newArray.data(), 16);
				index += 16;
				bnch_swt::doNotOptimizeAway(buffer);
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
		[=, &resultValuesDig]() mutable {
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
		"dodgerblue">([=, &resultValuesDig]() mutable {
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
	simdjson::ondemand::parser parserNew{};
	for (auto& value: stringValues) {
		value.reserve(value.size() + simdjson::SIMDJSON_PADDING);
	}
	/* bnch_swt::benchmark_stage<testStage, bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::template runBenchmark<testName, "Simdjson-Parsing-Function",
		"dodgerblue">([=, &parserNew, &resultValuesDig]() mutable {
		for (size_t x = 0; x < 1024; ++x) {
			auto newDoc = parserNew.iterate(stringValues[x]);
			//std::cout << "CURRENT TYPE: " << newDoc.get_number().value() << std::endl;
			resultValuesDig[x] = newDoc.get_number().value();
		}
	})
	for (size_t x = 0; x < 1024; ++x) {
		if (resultValuesDig[x] != valuesDig[x]) {
			std::cout << "Simdjson failed to parse: " << stringValues[x] << ", ";
			std::cout << "Simdjson failed to parse: " << valuesDig[x] << ", Instead it Parsed: " << resultValuesDig[x] << std::endl;
		}
	}*/
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
	int integer;
	double real;
	double e;
	double E;
	double emptyKey;// "":  23456789012E66,
	int zero;
	int one;
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
	int* null;
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
using Pass01 = std::tuple<std::string, std::map<std::string, std::vector<std::string>>, Empty, std::vector<int>, int, bool, bool, int*, Special, double, double, double, int,
	double, double, double, double, double, double, std::string>;

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
	auto specialData = rawData;

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

	template<jsonifier::serialize_options optionsNew, special_type value_type, jsonifier::concepts::buffer_like buffer_type, typename serialize_context_type>
	struct serialize_impl<optionsNew, value_type, buffer_type, serialize_context_type> {
		JSONIFIER_MAYBE_ALWAYS_INLINE static void impl(value_type& value, buffer_type& buffer, serialize_context_type& serializePair) noexcept {	
			static constexpr jsonifier::serialize_options options{ optionsNew };
			serializeRawJson<options>(buffer, value, serializePair);
		}
	};

	template<bool minified, jsonifier::parse_options optionsNew, typename parse_context_type> struct parse_impl<minified, optionsNew, Special, parse_context_type> {
		JSONIFIER_ALWAYS_INLINE static void impl(Special& value, parse_context_type& context) noexcept {
			static constexpr jsonifier::parse_options options{ optionsNew };
			jsonifier::raw_json_data rawData{};
			parse_impl<minified, options, jsonifier::raw_json_data, parse_context_type>::impl(rawData, context);
			value = parseRawJson(rawData);
		}
	};
}


int32_t main() {
	auto newFile = bnch_swt::file_loader::loadFile(std::string{ JSON_BASE_PATH } + "./CitmCatalogData-Prettified.json");
	jsonifier::jsonifier_core parser{};
	citm_catalog_message dataNew{};
	parser.parseJson(dataNew, newFile);
	for (auto& value: parser.getErrors()) {
		std::cout << "Error: " << value << std::endl;
	}
	uint_validation_tests::uintTests();
	static constexpr jsonifier_internal::string_literal newString{ "op" };
	std::string newerString{ "op" };
	//std::cout << "THE RESULT: " << (newString == newerString.data()) << std::endl;
	std::cout << "NEW STRIRNG: " << newString.size() << std::endl;
	int_validation_tests::intTests();
	//runForLength01<uint64_t, uint64_t, "Stage-1 Parse Test", "Uint:1-Digit">(1);
	//bnch_swt::benchmark_stage<"Stage-1 Parse Test", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
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
	runForLength<int64_t, int64_t, "Int-Integer-Tests", "Int:19-Digit">(18);
	bnch_swt::benchmark_stage<"Uint-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Uint-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Short-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();
	bnch_swt::benchmark_stage<"Int-Integer-Tests", bnch_swt::bench_options{ .type = bnch_swt::result_type::time }>::printResults();*/
	return 0;
}