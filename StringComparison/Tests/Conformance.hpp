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

	template<typename test_type> test_type runTest(const std::string& testName, const std::string& dataToParse, jsonifier::jsonifier_core<>& parser, bool doWeFail = false) noexcept {
		std::cout << "Running Test: " << testName << std::endl;
		jsonifier::raw_json_data valueNew{};
		auto result = parser.parseJson<jsonifier::parse_options{ .knownOrder = true }>(valueNew, dataToParse);
		std::cout << "VALUE: " << valueNew.rawJson() << std::endl;
		auto& object = valueNew.getArray();
		for (auto& value: object) {
			std::cout << ", VALUE: " << value.rawJson() << std::endl;
		}
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
		processFilesInFolder(jsonTests, "/ConformanceTests");
		std::cout << "Conformance Tests: " << std::endl;
		runTest<jsonifier::raw_json_data>("pass1.json", jsonTests["pass1.json"].fileContents, parser);
		return true;
	}

}