/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/jsonifier
#include "Common.hpp"

#include <jsonifier/Index.hpp>
#include <filesystem>
#include <fstream>

namespace conformance_tests {

	struct Empty {};

	struct Special {
		int integer;
		double real;
		double e;
		double E;
		double emptyKey;
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
		std::string number;
		std::string special;
		std::string hex;
		bool aTrue;
		bool aFalse;
		int* null;
		std::vector<int> array;
		Empty object;
		std::string address;
		std::string url;
		std::string comment;
		std::string commentKey;
		std::vector<int> spaced;
		std::vector<int> compact;
		std::string jsontext;
		std::string quotes;
		std::string key;
	};

	using Pass01 = std::tuple<std::string, std::map<std::string, std::vector<std::string>>, Empty, std::vector<int>, int, bool, bool, int*, Special, double, double, double, int,
		double, double, double, double, double, double, std::string>;

	void printSpecial(const Special& s) {
		std::cout << "integer: " << s.integer << '\n';
		std::cout << "real: " << s.real << '\n';
		std::cout << "e: " << s.e << '\n';
		std::cout << "E: " << s.E << '\n';
		std::cout << "emptyKey: " << s.emptyKey << '\n';
		std::cout << "zero: " << s.zero << '\n';
		std::cout << "one: " << s.one << '\n';
		std::cout << "space: " << s.space << '\n';
		std::cout << "quote: " << s.quote << '\n';
		std::cout << "backslash: " << s.backslash << '\n';
		std::cout << "controls: " << s.controls << '\n';
		std::cout << "slash: " << s.slash << '\n';
		std::cout << "alpha: " << s.alpha << '\n';
		std::cout << "ALPHA: " << s.ALPHA << '\n';
		std::cout << "digit: " << s.digit << '\n';
		std::cout << "number: " << s.number << '\n';
		std::cout << "special: " << s.special << '\n';
		std::cout << "hex: " << s.hex << '\n';
		std::cout << "aTrue: " << std::boolalpha << s.aTrue << '\n';
		std::cout << "aFalse: " << std::boolalpha << s.aFalse << '\n';
		std::cout << "null: " << (s.null ? *s.null : 0) << '\n';

		std::cout << "array: ";
		for (const auto& elem: s.array) {
			std::cout << elem << ' ';
		}
		std::cout << '\n';

		std::cout << "address: " << s.address << '\n';
		std::cout << "url: " << s.url << '\n';
		std::cout << "comment: " << s.comment << '\n';
		std::cout << "commentKey: " << s.commentKey << '\n';

		std::cout << "spaced: ";
		for (const auto& elem: s.spaced) {
			std::cout << elem << ' ';
		}
		std::cout << '\n';

		std::cout << "compact: ";
		for (const auto& elem: s.compact) {
			std::cout << elem << ' ';
		}
		std::cout << '\n';

		std::cout << "jsontext: " << s.jsontext << '\n';
		std::cout << "quotes: " << s.quotes << '\n';
		std::cout << "key: " << s.key << '\n';
	}

}

namespace jsonifier {

	JSONIFIER_ALWAYS_INLINE conformance_tests::Special collectSpecialFromRawJsonData(raw_json_data& values) {
		conformance_tests::Special returnValues{};
		returnValues.integer	= values["integer"].get<int32_t>();
		returnValues.address	= values["address"].get<std::string>();
		returnValues.real		= values["real"].get<double>();
		returnValues.e			= values["e"].get<double>();
		returnValues.E			= values["E"].get<double>();
		returnValues.zero		= values["zero"].get<int32_t>();
		returnValues.one		= values["one"].get<int32_t>();
		returnValues.space		= values["space"].get<std::string>();
		returnValues.quote		= values["quote"].get<std::string>();
		returnValues.backslash	= values["backslash"].get<std::string>();
		returnValues.controls	= values["controls"].get<std::string>();
		returnValues.slash		= values["slash"].get<std::string>();
		returnValues.alpha		= values["alpha"].get<std::string>();
		returnValues.ALPHA		= values["ALPHA"].get<std::string>();
		returnValues.digit		= values["digit"].get<std::string>();
		returnValues.number		= values["0123456789"].get<std::string>();
		returnValues.special	= values["special"].get<std::string>();
		returnValues.hex		= values["hex"].get<std::string>();
		returnValues.aTrue		= values["true"].get<bool>();
		returnValues.aFalse		= values["false"].get<bool>();
		returnValues.null		= values["null"].get<std::nullptr_t>();
		returnValues.url		= values["url"].get<std::string>();
		returnValues.comment	= values["comment"].get<std::string>();
		returnValues.commentKey = values["# -- --> */"].get<std::string>();
		for (auto& value: values[" s p a c e d "].get<jsonifier::raw_json_data::array_type>()) {
			returnValues.spaced.emplace_back(value.get<int32_t>());
		}
		for (auto& value: values[" s p a c e d "].get<jsonifier::raw_json_data::array_type>()) {
			returnValues.compact.emplace_back(value.get<int32_t>());
		}
		returnValues.jsontext = values["jsontext"].get<std::string>();
		returnValues.quotes	  = values["quotes"].get<std::string>();
		//returnValues.key	  = values["\\/\\\\\\\"\\uCAFE\\uBABE\\uAB98\\uFCDE\\ubcda\\uef4A\\b\\f\\n\\r\\t`1~!@#$%^&*()_+-=[]{}|;:',./<>?"].get<std::string>();
		conformance_tests::printSpecial(returnValues);
		return returnValues;
	}

	template<> struct core<conformance_tests::Empty> {
		using value_type				 = conformance_tests::Empty;
		static constexpr auto parseValue = createValue();
	};

}

namespace jsonifier_internal {

	template<typename value_type>
	concept special_type = std::is_same_v<std::remove_cvref_t<value_type>, conformance_tests::Special>;

	template<bool minified, jsonifier::parse_options options, special_type value_type, typename parse_context_type>
	struct parse_impl<minified, options, value_type, parse_context_type> {
		JSONIFIER_ALWAYS_INLINE static void impl(value_type& value, parse_context_type& context) noexcept {
			jsonifier::raw_json_data dataNew{};
			parse<minified, options>::impl(dataNew, context);
			std::cout << "CURRENT DATA: " << dataNew.rawJson() << std::endl;
			value = collectSpecialFromRawJsonData(dataNew);
		}
	};

}

namespace conformance_tests {

	using S					   = std::string;
	using VS				   = std::vector<S>;
	using VVS				   = std::vector<VS>;
	using VVVS				   = std::vector<VVS>;
	using VVVVS				   = std::vector<VVVS>;
	using VVVVVS			   = std::vector<VVVVS>;
	using VVVVVVS			   = std::vector<VVVVVS>;
	using VVVVVVVS			   = std::vector<VVVVVVS>;
	using VVVVVVVVS			   = std::vector<VVVVVVVS>;
	using VVVVVVVVVS		   = std::vector<VVVVVVVVS>;
	using VVVVVVVVVVS		   = std::vector<VVVVVVVVVS>;
	using VVVVVVVVVVVS		   = std::vector<VVVVVVVVVVS>;
	using VVVVVVVVVVVVS		   = std::vector<VVVVVVVVVVVS>;
	using VVVVVVVVVVVVVS	   = std::vector<VVVVVVVVVVVVS>;
	using VVVVVVVVVVVVVVS	   = std::vector<VVVVVVVVVVVVVS>;
	using VVVVVVVVVVVVVVVS	   = std::vector<VVVVVVVVVVVVVVS>;
	using VVVVVVVVVVVVVVVVS	   = std::vector<VVVVVVVVVVVVVVVS>;
	using VVVVVVVVVVVVVVVVVS   = std::vector<VVVVVVVVVVVVVVVVS>;
	using VVVVVVVVVVVVVVVVVVS  = std::vector<VVVVVVVVVVVVVVVVVS>;
	using VVVVVVVVVVVVVVVVVVVS = std::vector<VVVVVVVVVVVVVVVVVVS>;

	template<typename test_type>
	test_type runTest(const std::string& testName, const std::string& dataToParse, jsonifier::jsonifier_core<>& parser, bool doWeFail = true) noexcept {
		std::cout << "Running Test: " << testName << std::endl;
		test_type valueNew{};
		auto result = parser.parseJson<jsonifier::parse_options{ .knownOrder = true }>(valueNew, dataToParse);
		std::cout << "Running Test: " << std::get<0>(valueNew) << std::endl;
		//std::cout << "Running Test: " << std::get<1>(valueNew) << std::endl;
		//std::cout << "Running Test: " << std::get<2>(valueNew) << std::endl;
		//std::cout << "Running Test: " << std::get<3>(valueNew) << std::endl;
		std::cout << "Running Test: " << std::get<4>(valueNew) << std::endl;
		//std::cout << "Running Test: " << std::get<9>(valueNew) << std::endl;
		Special dataNew{ std::get<9>(valueNew) };
		std::cout << "Running Test: " << dataNew.integer << std::endl;

		if ((parser.getErrors().size() == 0 && result) && !doWeFail) {
			std::cout << "Test: " << testName << " = Succeeded 01" << std::endl;
		} else if (!result && doWeFail) {
			std::cout << "Test: " << testName << " = Succeeded 02" << std::endl;
			for (auto& value: parser.getErrors()) {
				std::cout << "Jsonifier Error: " << value << std::endl;
			}
		} else {
			std::cout << "Test: " << testName << " = Failed" << std::endl;
			for (auto& value: parser.getErrors()) {
				std::cout << "Jsonifier Error: " << value << std::endl;
			}
		}
		return valueNew;
	}

	bool conformanceTests() noexcept {
		jsonifier::jsonifier_core parser{};
		std::unordered_map<std::string, test_base> jsonTests{};
		processFilesInFolder(jsonTests, "/Tests/ConformanceTests");
		std::cout << "Conformance Tests: " << std::endl;
		/*
		runTest<std::unordered_map<std::string, std::string>>("fail02.json", jsonTests["fail02.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, std::string>>("fail03.json", jsonTests["fail03.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail04.json", jsonTests["fail04.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail05.json", jsonTests["fail05.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail06.json", jsonTests["fail06.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail07.json", jsonTests["fail07.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail08.json", jsonTests["fail08.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, bool>>("fail09.json", jsonTests["fail09.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, bool>>("fail10.json", jsonTests["fail10.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, int32_t>>("fail11.json", jsonTests["fail11.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, std::string>>("fail12.json", jsonTests["fail12.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, int32_t>>("fail13.json", jsonTests["fail13.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, int32_t>>("fail14.json", jsonTests["fail14.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail15.json", jsonTests["fail15.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail16.json", jsonTests["fail16.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail17.json", jsonTests["fail17.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, int32_t*>>("fail19.json", jsonTests["fail19.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, int32_t*>>("fail20.json", jsonTests["fail20.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, int32_t*>>("fail21.json", jsonTests["fail21.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail22.json", jsonTests["fail22.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail23.json", jsonTests["fail23.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail24.json", jsonTests["fail24.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail25.json", jsonTests["fail25.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail26.json", jsonTests["fail26.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail27.json", jsonTests["fail27.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail28.json", jsonTests["fail28.json"].fileContents, parser);
		runTest<std::vector<double>>("fail29.json", jsonTests["fail29.json"].fileContents, parser);
		runTest<std::vector<double>>("fail30.json", jsonTests["fail30.json"].fileContents, parser);
		runTest<std::vector<double>>("fail31.json", jsonTests["fail31.json"].fileContents, parser);
		runTest<std::unordered_map<std::string, bool>>("fail32.json", jsonTests["fail32.json"].fileContents, parser);
		runTest<std::vector<std::string>>("fail33.json", jsonTests["fail33.json"].fileContents, parser);*/
		runTest<Pass01>("pass1.json", jsonTests["pass1.json"].fileContents, parser, false);
		//runTest<VVVVVVVVVVVVVVVVVVVS>("pass2.json", jsonTests["pass2.json"].fileContents, parser, false);
		//runTest<jsonifier::raw_json_data>("pass3.json", jsonTests["pass3.json"].fileContents, parser, false);
		return true;
	}

}